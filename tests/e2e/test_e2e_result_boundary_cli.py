"""Result-boundary acceptance tests against the real Claude Code CLI (DATA_QUALITY.md).

`PROXY_RESULT_BOUNDARY_ACCEPTED` (agent/session.py) rests on every test here passing against
the pinned `claude-agent-sdk` and its bundled CLI (ADR-0023). Each test runs a real CLI session
through our hooks, options, and the in-process validating proxy (tests/e2e/cli_harness/), with
a fake streamable-HTTP MCP server as the proxy's upstream and a scripted recording model
endpoint, and asserts on the requests the CLI actually sent to "the model" (docs/TESTING.md
"Result boundary"): a raw payload's sentinel must never appear in any model request, and our
envelope must be what the model receives. Direct delivery (no proxy, ADR-0019) fails five of
them and is not accepted.

Skipped unless WRA_RUN_REQUIRES_CLI=1 (they spawn the CLI). TCP is allowed to 127.0.0.1 only;
the CLI's own traffic is confined by `cli_harness.session.local_env` and a blackhole proxy.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import anyio
import pytest
from claude_agent_sdk import ClaudeSDKClient
from cli_harness.mcp_server import IsError, Json, Slow, TransportFailure, oversized_payload
from cli_harness.model_server import FinalText, ToolUse
from cli_harness.session import (
    ACCOUNT_NUMBER,
    SERVER,
    Case,
    MemoryRecorder,
    SessionOutcome,
    run_cli_session,
    utc_now,
)

from wheelta_robinhood_agent.agent.hooks import ValidationOutcome, ValidationRequest
from wheelta_robinhood_agent.agent.result_boundary import (
    BoundaryValidator,
    MappedEvidence,
    MappingRequest,
)
from wheelta_robinhood_agent.agent.run_control import StopReason
from wheelta_robinhood_agent.agent.session import observe_statuses
from wheelta_robinhood_agent.agent.tool_access import build_tool_access
from wheelta_robinhood_agent.domain.enums import ExecutionMode, SourceStatus, ToolCallStatus
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.observability.redaction import Redactor

pytestmark = [pytest.mark.requires_cli, pytest.mark.allow_hosts(["127.0.0.1"])]

QUOTES = "get_option_quotes"
QUOTES_TOOL = ROBINHOOD_REGISTRY.qualified(QUOTES)
EQUITY_TOOL = ROBINHOOD_REGISTRY.qualified("get_equity_quotes")


def sentinel(case: str) -> str:
    return f"RAWSENTINEL-{case}-{uuid.uuid4().hex}"


# -- assertions over what the CLI sent to the model ---------------------------------------------


def assert_local_only(out: SessionOutcome) -> None:
    """No request tried to leave the machine and no real credential was presented."""
    assert out.proxy.attempts == [], f"non-local connection attempts: {out.proxy.attempts}"
    requests = out.model.snapshot()
    assert requests, f"the CLI never called the model endpoint (error={out.error})"
    for r in requests:
        assert r.headers.get("x-api-key") == "test-dummy"
        assert "authorization" not in r.headers


def assert_never_sent(out: SessionOutcome, *needles: str) -> None:
    for r in out.model.snapshot():
        text = r.text()
        for needle in needles:
            at = text.find(needle)
            if at >= 0:  # pytest.fail, not assert: a diff of a huge request body never ends
                excerpt = text[max(0, at - 300) : at + 200]
                pytest.fail(
                    f"raw payload reached the model ({needle}) in {r.path}: ...{excerpt}..."
                )


def result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def tool_result_for(out: SessionOutcome, tool_use_id: str) -> dict[str, Any] | None:
    """The tool_result for this tool_use as the model first saw it."""
    for r in out.model.main_loop():
        for block in r.tool_results():
            if block.get("tool_use_id") == tool_use_id:
                return block
    return None


def delivered_envelope(out: SessionOutcome, tool_use_id: str) -> dict[str, Any]:
    """Parse our JSON envelope from the model-visible tool_result (CLI reminders follow it)."""
    block = tool_result_for(out, tool_use_id)
    assert block is not None, f"no model request carried a result for {tool_use_id}"
    text = result_text(block).lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except ValueError:
        pytest.fail(f"the model did not receive our envelope; it received: {text[:400]!r}")
    assert isinstance(value, dict) and {"tool_call_id", "kind", "server", "tool"} <= set(value), (
        f"not an envelope: {text[:400]!r}"
    )
    return value


def only_call(out: SessionOutcome, tool: str) -> tuple[str, uuid.UUID]:
    calls = [c for c in out.recorder.requested_calls() if c["tool"] == tool]
    assert len(calls) == 1, f"expected one recorded {tool} call, got {len(calls)}"
    use_id = calls[0]["sdk_tool_use_id"]
    return use_id, out.recorder.ids[use_id]


def statuses(rec: MemoryRecorder, tool_call_id: uuid.UUID) -> list[ToolCallStatus]:
    return [o["status"] for o in rec.outcomes_for(tool_call_id)]


def diagnostics(out: SessionOutcome) -> str:
    return (
        f"error={out.error} timed_out={out.timed_out} elapsed={out.elapsed:.1f}s "
        f"events={[n for n, _ in out.recorder.events]} mcp={out.mcp.calls} "
        f"stderr={' | '.join(out.stderr)[-1500:]}"
    )


# -- fixture mappers / validators ---------------------------------------------------------------


def gap_mapper(marker: str) -> Callable[[MappingRequest, Callable[[], uuid.UUID]], MappedEvidence]:
    """A fixture mapper: accepts only {"ok": true} and returns evidence carrying `marker`."""

    def mapper(req: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
        payload = req.payload
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("schema violation")
        return MappedEvidence(gaps=(marker,))

    return mapper


def run(case: Case, monkeypatch: pytest.MonkeyPatch) -> SessionOutcome:
    out = run_cli_session(case, monkeypatch)
    assert not out.watchdog_fired, "the CLI wedged and the harness watchdog killed it: " + (
        diagnostics(out)
    )
    assert not out.timed_out, "the session exceeded its budget: " + diagnostics(out)
    assert_local_only(out)
    return out


# -- tests --------------------------------------------------------------------------------------


def test_updated_tool_output_replaces_a_remote_result_in_the_next_model_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PostToolUse `updatedToolOutput` envelope is what the next model request contains."""
    raw, marker = sentinel("valid"), f"ENVELOPE-{uuid.uuid4().hex}"
    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: Json({"ok": True, "raw": raw})},
        mappers={(SERVER, QUOTES): gap_mapper(marker)},
    )
    out = run(case, monkeypatch)
    use_id, tool_call_id = only_call(out, QUOTES)
    assert_never_sent(out, raw)
    envelope = delivered_envelope(out, use_id)
    assert envelope["kind"] == "validated", diagnostics(out)
    assert envelope["tool_call_id"] == str(tool_call_id)
    assert envelope["data"]["evidence_ref"] == f"evidence:{tool_call_id}"
    assert marker in envelope["gaps"]
    assert statuses(out.recorder, tool_call_id) == [ToolCallStatus.SUCCEEDED]
    assert out.recorder.of("delivered")


def test_invalid_payload_is_delivered_only_as_a_missing_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-invalid remote result never appears raw in any later model request."""
    raw = sentinel("invalid")
    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: Json({"ok": "not-a-bool", "raw": raw})},
        mappers={(SERVER, QUOTES): gap_mapper("unused")},
    )
    out = run(case, monkeypatch)
    use_id, tool_call_id = only_call(out, QUOTES)
    assert_never_sent(out, raw)
    envelope = delivered_envelope(out, use_id)
    assert envelope["kind"] == "missing", diagnostics(out)
    assert envelope["data"] is None
    assert any("schema validation" in g for g in envelope["gaps"])
    assert statuses(out.recorder, tool_call_id) == [ToolCallStatus.FAILED]
    assert any(raw in json.dumps(p) for p in out.recorder.payloads())  # restricted raw_invalid


def test_mcp_is_error_result_is_delivered_only_as_an_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MCP `isError` result goes through the same replacement path."""
    raw = sentinel("iserror")
    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: IsError(f"broker rejected the request: {raw}")},
    )
    out = run(case, monkeypatch)
    use_id, tool_call_id = only_call(out, QUOTES)
    assert_never_sent(out, raw)
    envelope = delivered_envelope(out, use_id)
    assert envelope["kind"] == "error", diagnostics(out)
    assert envelope["tool_call_id"] == str(tool_call_id)
    assert statuses(out.recorder, tool_call_id) == [ToolCallStatus.FAILED]


def test_transport_failure_is_delivered_only_as_an_error_envelope_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport error upstream: the proxy records a failed outcome and delivers an error
    envelope whose gap names only the failure class, never the response body."""
    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: TransportFailure()},
    )
    out = run(case, monkeypatch)
    use_id, tool_call_id = only_call(out, QUOTES)
    assert out.mcp.called(QUOTES) == 1  # forwarded once, never retried
    # The 500 body is raw remote output too; it must not be relayed to the model.
    assert_never_sent(out, "upstream exploded")
    outcomes = out.recorder.outcomes_for(tool_call_id)
    assert [o["status"] for o in outcomes] == [ToolCallStatus.FAILED], diagnostics(out)
    assert outcomes[0]["dedup_key"] == "proxy"
    assert outcomes[0]["error_ref"] in out.recorder.results
    envelope = delivered_envelope(out, use_id)  # the model saw the call end, not hang
    assert envelope["kind"] == "error"
    assert envelope["tool_call_id"] == str(tool_call_id)


def test_oversized_result_spilled_to_a_file_is_not_readable_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results over the SDK spill threshold follow the same path; no file-read bypass."""
    raw = sentinel("oversized")
    case = Case(steps=[], behaviors={QUOTES: Json(oversized_payload(raw))})

    def read_any_spill(body: dict[str, Any]) -> ToolUse:
        # A model trying to read the raw result back from disk: any file holding the sentinel,
        # else any file the CLI wrote, else the scratch dir itself.
        files = [p for p in case.root.rglob("*") if p.is_file()]
        spilled = [p for p in files if raw in p.read_text(errors="ignore")]
        target = (spilled or files or [case.root])[0]
        return ToolUse("Read", {"file_path": str(target)})

    case.steps = [ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), read_any_spill, FinalText("done")]
    out = run(case, monkeypatch)
    use_id, tool_call_id = only_call(out, QUOTES)
    assert_never_sent(out, raw)
    envelope = delivered_envelope(out, use_id)
    assert envelope["kind"] == "missing", diagnostics(out)
    assert envelope["tool_call_id"] == str(tool_call_id)
    # The Read attempt got an answer without the raw payload (asserted above) and was not run.
    assert len(out.model.main_loop()) >= 3, diagnostics(out)
    assert not [e for e in out.recorder.of("dispatched") if e.get("tool") == "Read"]


def test_hook_exception_interrupts_before_raw_output_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PostToolUse exception: the raw result is not sent in any later model request."""
    raw = sentinel("hookexc")
    calls = {"n": 0}

    def exploding_clock() -> datetime:
        # PreToolUse reads the clock once. Every later read fails: the proxy's (after the
        # upstream answered) and PostToolUse's first statement, which is outside its try
        # block, so the exception escapes the real hook to the SDK.
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("harness clock failure")
        return utc_now()

    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: Json({"raw": raw})},
        clock=exploding_clock,
    )
    out = run(case, monkeypatch)
    assert calls["n"] >= 3, diagnostics(out)  # PreToolUse, the proxy, and PostToolUse
    assert_never_sent(out, raw)
    use_id, _ = only_call(out, QUOTES)
    # The CLI fell back to the tool output it had: the proxy's static fallback envelope.
    assert delivered_envelope(out, use_id)["kind"] == "error", diagnostics(out)
    assert out.run_control.stop_requested


def test_hook_timeout_interrupts_before_raw_output_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PostToolUse timeout (HookMatcher.timeout) must not fall back to the raw result."""
    raw = sentinel("hooktimeout")
    timeout = 3.0
    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: Json({"raw": raw})},
        # PostToolUse records the delivery; a stalled ledger write there outlasts the timeout.
        recorder=MemoryRecorder(stall_on={"store_delivered": timeout + 5}),
        hook_timeout_seconds=timeout,
    )
    out = run(case, monkeypatch)
    assert out.recorder.of("dispatched"), diagnostics(out)
    assert_never_sent(out, raw)
    use_id, tool_call_id = only_call(out, QUOTES)
    # The CLI fell back to the tool output it had, which is the proxy's envelope.
    envelope = delivered_envelope(out, use_id)
    assert envelope["tool_call_id"] == str(tool_call_id), diagnostics(out)
    assert envelope["kind"] == "missing"


def test_stalled_validation_in_the_proxy_never_exposes_the_raw_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation stalls inside the proxy (past the hook timeout): the CLI holds no raw
    result at any point, so nothing raw can reach the model."""
    raw = sentinel("stalledvalidation")
    inner = BoundaryValidator(redactor=Redactor())
    timeout = 3.0

    def stalled_validator(request: ValidationRequest) -> ValidationOutcome:
        time.sleep(timeout + 5)  # a stalled ledger/validation call past HookMatcher.timeout
        return inner(request)

    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: Json({"raw": raw})},
        validator=stalled_validator,
        hook_timeout_seconds=timeout,
    )
    out = run(case, monkeypatch)
    assert out.recorder.of("dispatched"), diagnostics(out)
    assert_never_sent(out, raw)


def test_ledger_failure_in_post_tool_use_stops_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recording failure sets the latch, replaces output, and ends the session."""
    raw = sentinel("ledger")
    case = Case(
        steps=[
            ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}),
            ToolUse(EQUITY_TOOL, {"symbols": ["AAPL"]}),
            FinalText("done"),
        ],
        behaviors={QUOTES: Json({"raw": raw}), "get_equity_quotes": Json({"x": 1})},
        recorder=MemoryRecorder(fail_on=frozenset({"store_raw_invalid"})),
    )
    out = run(case, monkeypatch)
    use_id, _ = only_call(out, QUOTES)
    assert_never_sent(out, raw)
    assert out.run_control.stop_requested
    record = out.run_control.stop_record
    assert record is not None and record.reason is StopReason.INFRASTRUCTURE_FAILURE
    # Session ended: the model was never asked for another step, so no second call happened.
    assert out.mcp.called("get_equity_quotes") == 0, diagnostics(out)
    assert not [r for r in out.model.main_loop() if r.tool_results()], (
        "the CLI made another model request after continue=false: " + diagnostics(out)
    )
    assert out.result_message() is not None, diagnostics(out)


def test_other_account_data_never_reaches_a_model_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Account discovery strips other accounts before delivery or persistence."""
    other_number, other_name = "9ZZ11223344", sentinel("otheraccount")
    listing = {
        "accounts": [
            {"account_number": ACCOUNT_NUMBER, "type": "agentic", "nickname": "agentic"},
            {"account_number": other_number, "type": "individual", "nickname": other_name},
        ]
    }
    # An in-scope read whose (malformed) payload also carries another account's data.
    portfolio = {
        "account_number": ACCOUNT_NUMBER,
        "linked": [{"account_number": other_number, "nickname": other_name}],
    }
    case = Case(
        steps=[
            ToolUse(ROBINHOOD_REGISTRY.qualified("get_accounts"), {}),
            ToolUse(
                ROBINHOOD_REGISTRY.qualified("get_portfolio"), {"account_number": ACCOUNT_NUMBER}
            ),
            FinalText("done"),
        ],
        behaviors={"get_accounts": Json(listing), "get_portfolio": Json(portfolio)},
    )
    out = run(case, monkeypatch)
    # The model's own tool inputs never name the other account, so any hit is tool output.
    assert_never_sent(out, other_number, other_name)
    # Discovery is trusted-code-only: the model's call is denied before dispatch, so the
    # server never ran it and nothing from it was persisted.
    assert out.mcp.called("get_accounts") == 0
    _, accounts_id = only_call(out, "get_accounts")
    assert statuses(out.recorder, accounts_id) == [ToolCallStatus.DENIED], diagnostics(out)
    # The in-scope read ran; the model gets only an envelope for it.
    use_id, portfolio_id = only_call(out, "get_portfolio")
    assert out.mcp.called("get_portfolio") == 1
    envelope = delivered_envelope(out, use_id)
    assert envelope["kind"] == "missing" and envelope["data"] is None, diagnostics(out)
    # Persisted evidence never holds another account's full number (redacted to last four).
    persisted = json.dumps(out.recorder.payloads()) + json.dumps(
        [e for _, e in out.recorder.events], default=str
    )
    assert other_number not in persisted


def test_process_interruption_leaves_every_requested_call_with_an_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM mid-call: every requested call ends with a persisted or explicit unknown outcome."""
    raw = sentinel("interrupt")
    slow = Slow(seconds=30, then=Json({"raw": raw}))

    async def sigterm_mid_call(client: ClaudeSDKClient, out: SessionOutcome) -> None:
        # What the orchestrator's SIGTERM handler and session watcher do: latch, then interrupt.
        with anyio.fail_after(60):
            while not slow.started.is_set():
                await anyio.sleep(0.05)
        out.run_control.request_stop(StopReason.SIGTERM, datetime.now(UTC))
        await client.interrupt()

    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: slow},
        during=sigterm_mid_call,
        session_seconds=60,
    )
    out = run(case, monkeypatch)
    assert slow.started.is_set(), diagnostics(out)
    assert out.elapsed < 30, "the interrupt did not end the in-flight call: " + diagnostics(out)
    assert_never_sent(out, raw)
    requested = out.recorder.requested_calls()
    assert requested
    for call in requested:
        tool_call_id = out.recorder.ids[call["sdk_tool_use_id"]]
        assert statuses(out.recorder, tool_call_id), (
            f"{call['tool']} has no outcome after interruption: " + diagnostics(out)
        )


def test_init_message_and_mcp_status_shapes_match_the_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The init `mcp_servers` list and `get_mcp_status` tool names match observe_statuses.

    In-process servers (the proxy) are absent from `get_mcp_status` until the first query, so
    the status is read while a tool call is in flight; before the query it lists nothing.
    """
    slow = Slow(seconds=3, then=Json({"x": 1}))
    mid: dict[str, Any] = {}

    async def read_status_mid_call(client: ClaudeSDKClient, out: SessionOutcome) -> None:
        with anyio.fail_after(30):
            while not slow.started.is_set():
                await anyio.sleep(0.05)
        mid.update(await client.get_mcp_status())

    case = Case(
        steps=[ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), FinalText("done")],
        behaviors={QUOTES: slow},
        during=read_status_mid_call,
    )
    out = run(case, monkeypatch)
    assert out.mcp_status == {"mcpServers": []}, out.mcp_status  # before the first query
    assert mid, diagnostics(out)
    observations = observe_statuses(mid, [ROBINHOOD_REGISTRY], [SERVER], utc_now())
    assert len(observations) == 1
    obs = observations[0]
    assert obs.status is SourceStatus.CONNECTED, obs
    access = build_tool_access(
        effective_mode=ExecutionMode.OFF, workspace_writes=False, registries=(ROBINHOOD_REGISTRY,)
    )
    # The proxy serves only this run's allowed tools (in-process servers are shown to the
    # model regardless of disallowed_tools), so the status list lacks every disallowed one.
    # Discovery for proxied servers uses the upstream's full list instead (agent/session.py).
    assert obs.discovery is not None and not obs.discovery.unknown, obs.discovery
    disallowed = {n.removeprefix("mcp__robinhood__") for n in access.disallowed_tools}
    assert obs.discovery.missing <= disallowed, obs.discovery
    assert "place_option_order" in obs.discovery.missing
    init = out.init_message()
    assert init is not None, diagnostics(out)
    servers = init.data.get("mcp_servers")
    assert isinstance(servers, list)
    by_name = {s["name"]: s.get("status") for s in servers if isinstance(s, dict)}
    assert by_name.get(SERVER) == SourceStatus.CONNECTED.value, servers
    tools = init.data.get("tools")
    assert isinstance(tools, list)
    # The init list is what the model can see: exactly the allowlisted tools (layers 1-2), so
    # every disallowed tool, order tools included, is absent although the upstream lists it.
    assert set(tools) == set(access.allowed_tools), set(tools) ^ set(access.allowed_tools)
    assert not set(tools) & set(access.disallowed_tools)
