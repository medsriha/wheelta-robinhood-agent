"""The validating proxy and its handoff with the hooks (ADR-0023; DATA_QUALITY.md).

Fakes only: an in-memory upstream, the hook-test recorder/validator, no network or database.
The real-CLI behaviour is covered by tests/e2e/test_e2e_result_boundary_cli.py.
"""

import dataclasses
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import pytest
from test_hooks import (
    NOW,
    FakeRecorder,
    FakeValidator,
    Session,
    make_deps,
    wire,
)

from wheelta_robinhood_agent.agent.model_view import is_model_view, model_view
from wheelta_robinhood_agent.agent.proxy import (
    PROXY_DEDUP_KEY,
    TOOL_USE_ID_META,
    ValidatingProxy,
    _listed_tools,
    build_proxy_server,
    upstream_timeout_seconds,
)
from wheelta_robinhood_agent.agent.proxy_dispatch import (
    CallState,
    ProxyCall,
    ProxyDispatch,
    delivered_matches,
)
from wheelta_robinhood_agent.agent.result_boundary import BoundaryValidator, mapped_evidence_of
from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.mcp_upstream import (
    UpstreamResult,
    UpstreamTimeout,
    UpstreamTool,
)
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.observability.redaction import Redactor

QUOTES = "mcp__robinhood__get_option_quotes"
RAW = "RAW-REMOTE-TEXT-must-never-be-delivered"


class FakeUpstream:
    def __init__(self, behavior: Any = None, tools: tuple[str, ...] = ("get_option_quotes",)):
        self.behavior = behavior
        self.calls: list[tuple[str, dict[str, Any], float]] = []
        self._tools = tools

    @property
    def server(self) -> str:
        return "robinhood"

    @property
    def tools(self) -> tuple[UpstreamTool, ...]:
        return tuple(UpstreamTool(n, f"d {n}", {"type": "object"}) for n in self._tools)

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_seconds: float
    ) -> UpstreamResult:
        self.calls.append((name, dict(arguments), timeout_seconds))
        if isinstance(self.behavior, Exception):
            raise self.behavior
        response = self.behavior or {
            "content": [{"type": "text", "text": json.dumps({"raw": RAW})}],
            "isError": False,
        }
        return UpstreamResult(response=response, size_bytes=1)


@dataclasses.dataclass
class Rig:
    session: Session
    proxy: ValidatingProxy
    upstream: FakeUpstream
    dispatch: ProxyDispatch

    @property
    def rec(self) -> FakeRecorder:
        return self.session.rec

    def call(self, tool: str = "get_option_quotes", args: Any = None, use_id: Any = "toolu_1"):
        meta = {TOOL_USE_ID_META: use_id} if use_id is not None else None
        params = types.CallToolRequestParams.model_validate(
            {"name": tool, "arguments": {"symbols": ["AAPL"]} if args is None else args}
            | ({"_meta": meta} if meta else {})
        )
        result: types.CallToolResult = anyio.run(self.proxy.call_tool, params)
        assert result.is_error is False  # never an isError the CLI would relay raw
        blocks = [{"type": "text", "text": c.text} for c in result.content]  # type: ignore[union-attr]
        return blocks


def rig(behavior: Any = None, validator: Any = None, **deps: Any) -> Rig:
    dispatch = ProxyDispatch(frozenset({"robinhood"}))
    hook_deps = make_deps(proxy_dispatch=dispatch, **deps)
    if validator is not None:
        hook_deps = dataclasses.replace(hook_deps, validator=validator)
    upstream = FakeUpstream(behavior)
    proxy = ValidatingProxy(
        server="robinhood",
        upstream=upstream,
        dispatch=dispatch,
        recorder=hook_deps.recorder,
        validator=hook_deps.validator,
        run_control=hook_deps.run_control,
        clock=hook_deps.clock,
        upstream_timeout_seconds=50.0,
    )
    return Rig(Session(hook_deps), proxy, upstream, dispatch)


def outcomes(r: Rig) -> list[dict[str, Any]]:
    return [kw for n, kw in r.rec.events if n == "outcome"]


# ---- the happy path ---------------------------------------------------------------------------


def test_dispatched_call_is_forwarded_once_validated_and_delivered() -> None:
    r = rig()
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    assert r.dispatch.state("toolu_1") is CallState.PENDING
    blocks = r.call()
    envelope = wire(blocks)
    assert envelope["kind"] == "validated" and envelope["data"] == {"normalized": True}
    assert RAW not in json.dumps(blocks)
    assert r.upstream.calls == [("get_option_quotes", {"symbols": ["AAPL"]}, 50.0)]
    (outcome,) = outcomes(r)
    assert outcome["status"] is ToolCallStatus.SUCCEEDED
    assert outcome["dedup_key"] == PROXY_DEDUP_KEY
    assert r.dispatch.state("toolu_1") is CallState.COMPLETED
    # PostToolUse records the delivery of exactly that output.
    out = r.session.post(QUOTES, blocks)
    assert out["hookSpecificOutput"]["updatedToolOutput"] == blocks
    assert "continue_" not in out
    delivered = r.rec.event("store_delivered")["payload"]
    assert delivered["replaced"] is True and delivered["delivery"] == "proxy"
    assert delivered["tool_output"] == envelope
    assert r.rec.names()[-1] == "delivered"
    assert len(outcomes(r)) == 1  # PostToolUse adds no second outcome


def test_invalid_result_is_missing_and_raw_kept_only_as_restricted_evidence() -> None:
    r = rig(validator=FakeValidator("invalid"))
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    envelope = wire(r.call())
    assert envelope["kind"] == "missing" and envelope["data"] is None
    assert "store_raw_invalid" in r.rec.names()
    assert outcomes(r)[0]["status"] is ToolCallStatus.FAILED


# ---- correlation ------------------------------------------------------------------------------


def test_call_without_tool_use_id_is_not_forwarded_and_stops_the_run() -> None:
    r = rig()
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    envelope = wire(r.call(use_id=None))
    assert envelope["kind"] == "error" and envelope["tool_call_id"] is None
    assert r.upstream.calls == []
    assert r.session.deps.run_control.stop_requested


def test_undispatched_or_replayed_call_is_never_forwarded() -> None:
    r = rig()
    assert wire(r.call(use_id="toolu_unknown"))["kind"] == "error"
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    r.call()
    replay = wire(r.call())  # the same tool_use_id again
    assert replay["kind"] == "error"
    assert len(r.upstream.calls) == 1


@pytest.mark.parametrize(
    ("tool", "args"),
    [("get_option_quotes", {"symbols": ["MSFT"]}), ("get_equity_quotes", {"symbols": ["AAPL"]})],
)
def test_call_differing_from_the_dispatch_is_not_forwarded(tool: str, args: Any) -> None:
    r = rig()
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    envelope = wire(r.call(tool, args))
    assert envelope["kind"] == "error" and "not forwarded" in envelope["gaps"][0]
    assert r.upstream.calls == []
    assert outcomes(r)[0]["status"] is ToolCallStatus.FAILED  # nothing reached the server
    assert r.session.deps.run_control.stop_requested


def test_tier_s_call_after_stop_is_not_forwarded() -> None:
    r = rig()
    call = ProxyCall(uuid.uuid4(), "robinhood", "create_watchlist", ToolTier.S, {"name": "x"})
    r.dispatch.register("toolu_s", call)
    r.session.deps.run_control.request_stop(StopReason.SIGTERM, NOW)
    envelope = wire(r.call("create_watchlist", {"name": "x"}, use_id="toolu_s"))
    assert "run stop requested" in envelope["gaps"][0]
    assert r.upstream.calls == []


# ---- upstream failures ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "status"), [(ToolTier.R, ToolCallStatus.FAILED), (ToolTier.X, ToolCallStatus.UNKNOWN)]
)
def test_upstream_failure_is_an_error_envelope_with_a_fixed_gap(
    tier: ToolTier, status: ToolCallStatus
) -> None:
    r = rig(UpstreamTimeout("get_option_quotes: no answer before the deadline"))
    r.dispatch.register(
        "toolu_x", ProxyCall(uuid.uuid4(), "robinhood", "get_option_quotes", tier, {"a": 1})
    )
    envelope = wire(r.call(args={"a": 1}, use_id="toolu_x"))
    assert envelope["kind"] == "error"
    assert envelope["gaps"] == ["get_option_quotes: no answer before the deadline"]
    (outcome,) = outcomes(r)
    assert outcome["status"] is status  # S/X after dispatch: unknown, never retried
    assert len(r.upstream.calls) == 1


def test_is_error_result_goes_through_the_validator() -> None:
    response = {"content": [{"type": "text", "text": RAW}], "isError": True}
    r = rig(response)
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    r.call()
    validator = r.session.deps.validator
    assert isinstance(validator, FakeValidator)
    assert validator.requests[0].tool_response == response


def test_handler_exception_returns_a_static_fallback_and_stops() -> None:
    r = rig(validator=FakeValidator("raise"))
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    blocks = r.call()
    envelope = wire(blocks)
    assert envelope["kind"] == "error"
    assert envelope["gaps"] == ["proxy failure (ValueError)"]
    assert RAW not in json.dumps(blocks)
    assert r.session.deps.run_control.stop_requested
    assert r.dispatch.delivered("toolu_1") is None


def test_clock_failure_still_sets_the_latch() -> None:
    calls = {"n": 0}

    def clock() -> Any:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("clock")
        return NOW

    r = rig(clock=clock)
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    assert wire(r.call())["kind"] == "error"
    record = r.session.deps.run_control.stop_record
    assert record is not None and record.reason is StopReason.INFRASTRUCTURE_FAILURE


# ---- PostToolUse / PostToolUseFailure for proxied calls ---------------------------------------


def test_post_with_a_foreign_result_replaces_it_and_stops() -> None:
    r = rig()
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    blocks = r.call()
    out = r.session.post(QUOTES, [{"type": "text", "text": RAW}])
    assert out["continue_"] is False
    assert out["hookSpecificOutput"]["updatedToolOutput"] == blocks
    assert "store_delivered" not in r.rec.names()


def test_post_without_proxy_output_records_an_outcome_and_stops() -> None:
    r = rig()
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    out = r.session.post(QUOTES, [{"type": "text", "text": RAW}])
    assert out["continue_"] is False
    assert wire(out["hookSpecificOutput"]["updatedToolOutput"])["kind"] == "error"
    assert outcomes(r)[0]["status"] is ToolCallStatus.FAILED


def test_post_after_the_latch_ends_the_session() -> None:
    r = rig()
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    blocks = r.call()
    r.session.deps.run_control.request_stop(StopReason.SIGTERM, NOW)
    out = r.session.post(QUOTES, blocks)
    assert out["continue_"] is False and out["stopReason"] == "run stop requested"


def test_failure_after_the_proxy_claimed_adds_no_second_outcome() -> None:
    r = rig()
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    r.call()
    r.session.fail(QUOTES)
    assert len(outcomes(r)) == 1
    assert "store_error" in r.rec.names()


def test_failure_before_the_proxy_claimed_is_failed_even_for_tier_s() -> None:
    """Nothing reached the server, so even a Tier S action has a known (failed) outcome."""
    r = rig()
    r.session.pre("mcp__robinhood__create_watchlist", {"name": "WRA · New"})
    assert r.dispatch.state("toolu_1") is CallState.PENDING
    r.session.fail("mcp__robinhood__create_watchlist")
    assert outcomes(r)[-1]["status"] is ToolCallStatus.FAILED


def test_registration_failure_denies_and_stops() -> None:
    r = rig()
    r.dispatch.register(
        "toolu_1", ProxyCall(uuid.uuid4(), "robinhood", "get_option_quotes", ToolTier.R, {})
    )
    out = r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert out["continue_"] is False


# ---- building blocks --------------------------------------------------------------------------


def test_upstream_timeout_leaves_a_margin_below_the_cli_timeout() -> None:
    assert upstream_timeout_seconds(60_000) == 50.0
    with pytest.raises(ValueError, match="MCP_TOOL_TIMEOUT"):
        upstream_timeout_seconds(10_000)


def test_settings_reject_a_tool_timeout_without_room_for_the_proxy() -> None:
    """Fail fast at startup (CLAUDE.md §6), not when the session is built."""
    from wheelta_robinhood_agent.agent.proxy import PROXY_TIMEOUT_MARGIN_SECONDS
    from wheelta_robinhood_agent.config.settings import Settings

    field = Settings.model_fields["MCP_TOOL_TIMEOUT"]
    bounds = [m.gt for m in field.metadata if hasattr(m, "gt")]
    assert bounds == [PROXY_TIMEOUT_MARGIN_SECONDS * 1000]


def test_server_lists_only_registered_tools_the_upstream_listed() -> None:
    r = rig()
    r.upstream._tools = (
        "get_option_quotes",
        "get_portfolio",
        "place_option_order",
        "not_registered_tool",
    )
    allowed = ("mcp__robinhood__get_option_quotes", "mcp__robinhood__not_registered_tool")
    # Only allowed AND registered tools are served; the disallowed order tool does not exist.
    assert [t.name for t in _listed_tools(ROBINHOOD_REGISTRY, r.upstream, allowed)] == [
        "get_option_quotes"
    ]
    server = build_proxy_server(r.proxy, ROBINHOOD_REGISTRY, allowed)
    assert server.name == "robinhood"
    with pytest.raises(ValueError, match="same server"):
        build_proxy_server(
            dataclasses.replace(r.proxy, server="wheelta"), ROBINHOOD_REGISTRY, allowed
        )


def test_dispatch_rules() -> None:
    d = ProxyDispatch(frozenset({"robinhood"}))
    call = ProxyCall(uuid.uuid4(), "robinhood", "t", ToolTier.R, {})
    with pytest.raises(ValueError, match="not proxied"):
        d.register("u", dataclasses.replace(call, server="wheelta"))
    d.register("u", call)
    with pytest.raises(ValueError, match="already registered"):
        d.register("u", call)
    with pytest.raises(ValueError, match="not claimed"):
        d.complete("u", [])
    assert d.claim("u") == call and d.claim("u") is None
    d.complete("u", [{"type": "text", "text": "x"}])
    assert d.state("u") is CallState.COMPLETED and d.state("other") is None


def test_delivered_matches_requires_the_exact_blocks() -> None:
    blocks = [{"type": "text", "text": "a"}]
    assert delivered_matches([{"type": "text", "text": "a"}], blocks)
    assert not delivered_matches({"content": blocks}, blocks)
    assert not delivered_matches([{"type": "text", "text": "b"}], blocks)
    assert not delivered_matches([{"type": "image"}], blocks)
    assert not delivered_matches([], blocks)


# ---- ADR-0030: the account placeholder -------------------------------------------------------

PORTFOLIO = "mcp__robinhood__get_portfolio"


def test_placeholder_is_substituted_only_on_the_upstream_call() -> None:
    from test_hooks import ACCOUNT, FakeValidator

    from wheelta_robinhood_agent.agent.account_scope import (
        AGENTIC_ACCOUNT_PLACEHOLDER,
        ROBINHOOD_ACCOUNT_SCOPE,
    )

    validator = FakeValidator()
    r = rig(validator=validator, account_scope_table=ROBINHOOD_ACCOUNT_SCOPE)
    placeholder = {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER}
    out = r.session.pre(PORTFOLIO, placeholder)
    # The CLI keeps the placeholder: no updatedInput, and the dispatch still matches it.
    assert "hookSpecificOutput" not in out
    blocks = r.call("get_portfolio", placeholder)
    assert r.upstream.calls == [("get_portfolio", {"account_number": ACCOUNT}, 50.0)]
    # The validator sees the real argument (mappers redact it to the last four).
    assert validator.requests[-1].effective_input == {"account_number": ACCOUNT}
    assert ACCOUNT not in json.dumps(blocks)
    assert r.rec.event("requested")["arguments_redacted"] == placeholder
    assert r.rec.event("dispatched")["effective_arguments_redacted"] == placeholder


# ---- oversized envelopes ---------------------------------------------------------------------


def test_an_oversized_envelope_is_an_error_and_the_run_continues() -> None:
    from test_hooks import FakeValidator

    from wheelta_robinhood_agent.agent.proxy import MAX_DELIVERED_CHARS

    class Big(FakeValidator):
        def __call__(self, request: Any) -> Any:
            outcome = super().__call__(request)
            data = {"rows": ["x" * 100] * (MAX_DELIVERED_CHARS // 100 + 1)}
            return outcome.model_copy(
                update={"envelope": outcome.envelope.model_copy(update={"data": data})}
            )

    r = rig(validator=Big())
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    blocks = r.call()
    envelope = wire(blocks)
    assert envelope["kind"] == "error"
    assert "too large to deliver" in envelope["gaps"][0]
    assert len(blocks[0]["text"]) < MAX_DELIVERED_CHARS
    (outcome,) = outcomes(r)
    assert outcome["status"] is ToolCallStatus.FAILED
    assert not r.session.deps.run_control.stop_requested
    assert "store_validated" not in r.rec.names()  # never recorded as validated evidence
    # PostToolUse accepts exactly the delivered error envelope: no stop.
    out = r.session.post(QUOTES, blocks)
    assert "continue_" not in out


def test_the_model_gets_the_view_and_the_ledger_keeps_the_full_envelope() -> None:
    """ADR-0037: validated rows hold the full evidence; the delivered text is its view."""
    fixture = Path(__file__).parents[2] / "fixtures" / "robinhood" / "results"
    data = json.loads((fixture / "get_option_quotes.SPY_20261016_P740.json").read_text())["data"]
    response = {
        "content": [{"type": "text", "text": json.dumps({"data": data, "guide": "prose"})}],
        "isError": False,
    }
    r = rig(behavior=response, validator=BoundaryValidator(redactor=Redactor()))
    r.session.pre(QUOTES, {"symbols": ["AAPL"]})
    blocks = r.call()
    delivered = wire(blocks)
    stored = r.rec.event("store_validated")["payload"]
    assert mapped_evidence_of(stored) is not None  # full typed evidence for facts and audit
    assert is_model_view(delivered) and delivered == model_view(stored)
    assert "quote_id" not in blocks[0]["text"]
    r.session.post(QUOTES, blocks)
    assert r.rec.event("store_delivered")["payload"]["tool_output"] == delivered
