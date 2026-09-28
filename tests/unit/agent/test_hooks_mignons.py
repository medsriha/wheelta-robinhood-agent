"""Hook layer 3 for the orchestrator and research Mignons (ADR-0025, agent/mignons.py).

Role scoping, `Agent` (Tier D) spawn gating and caps, and the MignonReport boundary in
PostToolUse(Agent). Same fakes as test_hooks.py.
"""

import json
import uuid
from typing import Any

import pytest
from pydantic import JsonValue
from test_hooks import (
    ACCOUNT,
    BOARD,
    NOW,
    PLACE,
    RH,
    SCOPE,
    FakeRecorder,
    ResultEnvelope,
    Session,
    ValidationOutcome,
    ValidationRequest,
    assert_allowed,
    assert_denied,
    denied_reason,
    drive,
    make_deps,
)

from wheelta_robinhood_agent.agent.account_scope import NOT_SCOPED
from wheelta_robinhood_agent.agent.hooks import EnvelopeKind
from wheelta_robinhood_agent.agent.local_server import LOCAL_REGISTRY
from wheelta_robinhood_agent.agent.mignons import MignonLimits
from wheelta_robinhood_agent.domain.enums import ExecutionMode, ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

LIMITS = MignonLimits(max_per_run=8, max_concurrent=4, max_turns_per_mignon=40)
MARKET_ID, COMPANY_ID = "a-market-1", "a-company-1"
MARKET = {"agent_id": MARKET_ID, "agent_type": "mignon-market"}
COMPANY = {"agent_id": COMPANY_ID, "agent_type": "mignon-company"}
SPAWN = {"description": "screen", "prompt": "Screen AAPL puts.", "subagent_type": "mignon-market"}
EVIDENCE, CANDIDATE = "evidence:e-1", "candidate:c-1"
URL = "https://investor.example.com/q3"


class RefValidator:
    """Validates everything, delivering data that carries code-issued refs."""

    def __init__(self, data: JsonValue = None) -> None:
        self.data = data if data is not None else {"evidence_ref": EVIDENCE, "rows": [CANDIDATE]}

    def __call__(self, request: ValidationRequest) -> ValidationOutcome:
        return ValidationOutcome(
            envelope=ResultEnvelope(
                tool_call_id=request.tool_call_id,
                server=request.server,
                tool=request.tool,
                kind=EnvelopeKind.VALIDATED,
                data=self.data,
                retrieved_at=request.retrieved_at,
            )
        )


def session(**overrides: Any) -> Session:
    overrides.setdefault("mignon_limits", LIMITS)
    overrides.setdefault("registries", (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY, LOCAL_REGISTRY))
    overrides.setdefault("account_scope_table", {**SCOPE, "get_financials": NOT_SCOPED})
    overrides.setdefault("validator", RefValidator())
    return Session(make_deps(**overrides))


def agent_response(text: str, agent_id: str = MARKET_ID, agent_type: str = "mignon-market") -> Any:
    """The CLI's completed synchronous Agent result (real CLI 2.1.283 shape)."""
    return {
        "status": "completed",
        "prompt": SPAWN["prompt"],
        "agentId": agent_id,
        "agentType": agent_type,
        "content": [{"type": "text", "text": text}],
        "totalToolUseCount": 1,
    }


def report(*findings: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Screen AAPL puts.",
            "findings": list(findings),
            "gaps": [],
            "follow_up_questions": [],
        }
    )


def finding(claim: str, refs: list[str] | None = None, urls: list[str] | None = None) -> Any:
    return {"claim": claim, "refs": refs or [], "web_urls": urls or []}


def delivered_envelope(out: Any) -> Any:
    replaced = out["hookSpecificOutput"]["updatedToolOutput"]
    assert replaced["status"] == "completed" and replaced["agentId"]
    (block,) = replaced["content"]
    return json.loads(block["text"])


def research(s: Session, agent: dict[str, str] = MARKET, use_id: str = "toolu_q") -> None:
    """One validated call by `agent` (a Mignon, or {} for the orchestrator) that delivers
    EVIDENCE and CANDIDATE."""
    tool = RH + ("get_financials" if agent is COMPANY else "get_option_quotes")
    assert_ok(s.pre(tool, {}, use_id=use_id, **agent))
    s.post(tool, {"raw": True}, use_id=use_id)


def assert_ok(out: Any) -> None:
    assert denied_reason(out) is None, out


# ---- role scoping ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool", ["WebSearch", "WebFetch", BOARD, "mcp__wra_local__web_cache_lookup"]
)
def test_orchestrator_cannot_research_directly(tool: str) -> None:
    s = session()
    assert_denied(s, s.pre(tool, {"query": "AAPL"}), "not available to the orchestrator")


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (RH + "get_option_positions", {"account_number": ACCOUNT}),
        (RH + "get_portfolio", {"account_number": ACCOUNT}),
        ("mcp__wra_local__get_decision_facts", {}),
    ],
)
def test_mignons_never_get_account_state_or_facts(tool: str, args: dict[str, Any]) -> None:
    s = session()
    assert_denied(s, s.pre(tool, args, **MARKET), "not available to the mignon-market")


def test_mignon_never_gets_order_or_workspace_tools_even_live() -> None:
    s = session(effective_mode=ExecutionMode.LIVE)
    out = s.pre(PLACE, {"account_number": ACCOUNT}, **COMPANY)
    assert_denied(s, out, "not available to the mignon-company")
    s2 = session()
    out = s2.pre(RH + "create_watchlist", {"name": "WRA · x"}, **MARKET)
    assert_denied(s2, out, "not available to the mignon-market")


def test_mignon_types_have_their_own_tools() -> None:
    s = session()
    assert_denied(s, s.pre("WebSearch", {"query": "q"}, **MARKET), "mignon-market")
    s2 = session()
    assert_allowed(s2, s2.pre("WebSearch", {"query": "q"}, **COMPANY))


def test_mignon_call_is_attributed_when_recorded() -> None:
    s = session()
    s.pre(RH + "get_option_quotes", {}, **MARKET)
    req = s.rec.event("requested")
    assert req["agent_id"] == MARKET_ID and req["agent_type"] == "mignon-market"
    s2 = session()
    s2.pre(RH + "get_option_quotes", {})
    assert s2.rec.event("requested")["agent_id"] is None


# ---- spawning -------------------------------------------------------------------------------


def test_orchestrator_spawns_a_mignon() -> None:
    s = session()
    out = s.pre("Agent", SPAWN)
    assert_allowed(s, out)
    assert s.rec.event("requested")["tier"] is ToolTier.D


def test_mignons_cannot_spawn_mignons() -> None:
    s = session()
    assert_denied(s, s.pre("Agent", SPAWN, **MARKET), "cannot spawn")


def test_spawn_denied_when_mignons_disabled() -> None:
    s = session(mignon_limits=None)
    assert_denied(s, s.pre("Agent", SPAWN), "disabled")


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"run_in_background": True}, "not permitted"),
        ({"model": "claude-other"}, "not permitted"),
        ({"cwd": "/"}, "not permitted"),
        ({"subagent_type": "general-purpose"}, "must be a Mignon type"),
        ({"prompt": ""}, "'prompt' missing"),
        ({"description": 3}, "'description' missing"),
    ],
)
def test_spawn_inputs_are_checked(change: dict[str, Any], fragment: str) -> None:
    s = session()
    assert_denied(s, s.pre("Agent", {**SPAWN, **change}), fragment)


def test_spawn_denied_after_kill_switch_or_stop() -> None:
    s = session(kill_switch=True)
    assert_denied(s, s.pre("Agent", SPAWN), "kill switch")
    s2 = session()
    s2.deps.run_control.request_stop(StopReason.SIGTERM, NOW)
    assert_denied(s2, s2.pre("Agent", SPAWN), "run stop requested")


def test_per_run_cap_counts_every_spawn() -> None:
    s = session(mignon_limits=MignonLimits(2, 5, 40))
    for i in range(2):
        assert_ok(s.pre("Agent", SPAWN, use_id=f"toolu_a{i}"))
        s.fail("Agent", use_id=f"toolu_a{i}")
    out = s.pre("Agent", SPAWN, use_id="toolu_a2")
    assert "mignons.max_per_run=2" in (denied_reason(out) or "")


def test_concurrency_cap_frees_on_result_and_failure() -> None:
    s = session(mignon_limits=MignonLimits(8, 1, 40))
    assert_ok(s.pre("Agent", SPAWN, use_id="toolu_a0"))
    out = s.pre("Agent", SPAWN, use_id="toolu_a1")
    assert "mignons.max_concurrent=1" in (denied_reason(out) or "")
    s.post("Agent", agent_response(report()), use_id="toolu_a0")
    assert_ok(s.pre("Agent", SPAWN, use_id="toolu_a2"))
    s.fail("Agent", use_id="toolu_a2")
    assert_ok(s.pre("Agent", SPAWN, use_id="toolu_a3"))


def test_failed_spawn_is_recorded_failed_not_unknown() -> None:
    s = session()
    s.pre("Agent", SPAWN)
    s.fail("Agent")
    assert s.rec.event("outcome")["status"] is ToolCallStatus.FAILED


# ---- report boundary ------------------------------------------------------------------------


def spawn_and_research(s: Session, spawn: dict[str, Any] = SPAWN) -> None:
    assert_ok(s.pre("Agent", spawn, use_id="toolu_agent"))
    research(s)


def test_valid_report_is_delivered_as_validated_envelope() -> None:
    s = session()
    spawn_and_research(s)
    text = report(finding("The put bid is 1.20.", [EVIDENCE, CANDIDATE]))
    out = s.post("Agent", agent_response(text), use_id="toolu_agent")
    assert "continue_" not in out
    envelope = delivered_envelope(out)
    assert envelope["kind"] == "validated" and envelope["tool"] == "Agent"
    assert envelope["data"]["agent_id"] == MARKET_ID
    assert envelope["data"]["report"]["findings"][0]["refs"] == [EVIDENCE, CANDIDATE]
    outcomes = [kw for n, kw in s.rec.events if n == "outcome"]
    assert outcomes[-1]["status"] is ToolCallStatus.SUCCEEDED
    _, delivered = s.rec.results[s.rec.events[-1][1]["delivered_result_ref"]]
    assert isinstance(delivered, dict) and delivered["tool_output"] == envelope
    assert delivered["agent_id"] == MARKET_ID


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (report(finding("Unsupported.", ["evidence:never-delivered"])), "not delivered"),
        (report(finding("Bid is 1.20.", urls=[URL])), "must cite a code-issued ref"),
        (report(finding("A claim with no source.")), "at least one ref"),
        ("Here is my report: ...", "Expecting value"),
        (
            '{"task": "t", "findings": [], "gaps": [], "follow_up_questions": [], "n": 1}',
            "not allowed",
        ),
        (report(finding("Page says so.", urls=[URL])), "not fetched"),
    ],
)
def test_invalid_report_becomes_missing_envelope(text: str, fragment: str) -> None:
    s = session()
    spawn_and_research(s)
    out = s.post("Agent", agent_response(text), use_id="toolu_agent")
    assert "continue_" not in out and not s.deps.run_control.stop_requested
    envelope = delivered_envelope(out)
    assert envelope["kind"] == "missing" and envelope["data"] is None
    assert any(fragment in g for g in envelope["gaps"]), envelope["gaps"]
    assert "store_raw_invalid" in s.rec.names()
    outcomes = [kw for n, kw in s.rec.events if n == "outcome"]
    assert outcomes[-1]["status"] is ToolCallStatus.FAILED


def test_ref_delivered_to_another_mignon_is_not_citable() -> None:
    s = session()
    research(s, COMPANY, use_id="toolu_c")  # the company Mignon saw EVIDENCE
    assert_ok(s.pre("Agent", SPAWN, use_id="toolu_agent"))
    text = report(finding("Bid is 1.20.", [EVIDENCE]))
    envelope = delivered_envelope(s.post("Agent", agent_response(text), use_id="toolu_agent"))
    assert envelope["kind"] == "missing"


def test_ref_handed_over_in_the_task_is_citable() -> None:
    s = session()
    research(s, {}, use_id="toolu_o")  # the orchestrator received EVIDENCE
    spawn = {**SPAWN, "prompt": f"Follow up on {EVIDENCE}: is the spread acceptable?"}
    assert_ok(s.pre("Agent", spawn, use_id="toolu_agent"))
    text = report(finding("Spread of 0.10 per the quote.", [EVIDENCE]))
    envelope = delivered_envelope(s.post("Agent", agent_response(text), use_id="toolu_agent"))
    assert envelope["kind"] == "validated"


def test_fetched_url_is_citable_by_the_mignon_that_fetched_it() -> None:
    s = session()
    spawn = {**SPAWN, "subagent_type": "mignon-company"}
    assert_ok(s.pre("Agent", spawn, use_id="toolu_agent"))
    assert_ok(s.pre("WebFetch", {"url": URL, "prompt": "q"}, use_id="toolu_f", **COMPANY))
    s.post("WebFetch", {"text": "page"}, use_id="toolu_f")
    text = report(finding("Management reaffirmed guidance.", urls=[URL]))
    out = s.post("Agent", agent_response(text, COMPANY_ID, "mignon-company"), use_id="toolu_agent")
    assert delivered_envelope(out)["kind"] == "validated"


def test_report_from_a_different_mignon_type_is_invalid() -> None:
    s = session()
    spawn_and_research(s)
    text = report(finding("Bid is 1.20.", [EVIDENCE]))
    out = s.post("Agent", agent_response(text, MARKET_ID, "mignon-macro"), use_id="toolu_agent")
    assert any("agent type" in g for g in delivered_envelope(out)["gaps"])


@pytest.mark.parametrize(
    "response",
    [
        {"isAsync": True, "status": "async_launched", "agentId": "a-1"},
        [{"type": "text", "text": "report"}],
        {"status": "completed", "agentId": "a-1", "agentType": "mignon-market", "content": []},
    ],
)
def test_non_synchronous_or_malformed_agent_result_stops_the_run(response: Any) -> None:
    s = session()
    s.pre("Agent", SPAWN)
    out = s.post("Agent", response)
    assert out["continue_"] is False and s.deps.run_control.stop_requested
    replaced = out["hookSpecificOutput"]["updatedToolOutput"]
    text = replaced["content"][0]["text"] if isinstance(replaced, dict) else replaced[0]["text"]
    assert json.loads(text)["kind"] == "error"


def test_report_recording_failure_stops_the_run() -> None:
    s = session(recorder=FakeRecorder(frozenset({"delivered"})))
    s.pre("Agent", SPAWN)
    out = s.post("Agent", agent_response(report()))
    assert out["continue_"] is False and s.deps.run_control.stop_requested


# ---- SubagentStart --------------------------------------------------------------------------


def start(s: Session, agent_type: str) -> Any:
    cb = s.hooks["SubagentStart"][0].hooks[0]
    data = {"hook_event_name": "SubagentStart", "agent_id": "a-1", "agent_type": agent_type}
    return drive(cb(data, str(uuid.uuid4()), {"signal": None}))


def test_only_mignons_may_start() -> None:
    s = session()
    assert start(s, "mignon-macro") == {}
    assert start(s, "general-purpose")["continue_"] is False
    assert s.deps.run_control.stop_requested
    s2 = session(mignon_limits=None)
    assert start(s2, "mignon-market")["continue_"] is False
