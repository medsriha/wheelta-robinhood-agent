"""Hook layer 3 for the orchestrator and research Mignons (ADR-0025, agent/mignons.py).

Role scoping, `Agent` (Tier D) spawn gating and caps, and the MignonReport boundary in
PostToolUse(Agent). Same fakes as test_hooks.py.
"""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue, SecretStr
from test_hooks import (
    ACCOUNT,
    BOARD,
    NOW,
    PLACE,
    RH,
    SCOPE,
    TEST_MODEL,
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
    wire,
)

from wheelta_robinhood_agent.agent.account_scope import NOT_SCOPED
from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, last_assistant_text
from wheelta_robinhood_agent.agent.local_server import LOCAL_REGISTRY
from wheelta_robinhood_agent.agent.mignons import MignonLimits
from wheelta_robinhood_agent.agent.order_walk import ORDER_WORK_REGISTRY
from wheelta_robinhood_agent.agent.result_boundary import BoundaryValidator
from wheelta_robinhood_agent.domain.enums import AgentRole, ExecutionMode, ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.websearch.registry import TAVILY_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY
from wheelta_robinhood_agent.observability.redaction import Redactor

LIMITS = MignonLimits(max_per_run=8, max_concurrent=4, max_turns_per_mignon=40)
MARKET_ID, COMPANY_ID = "a-market-1", "a-company-1"
MARKET_T, COMPANY_T = f"mignon-market--{TEST_MODEL}", f"mignon-company--{TEST_MODEL}"
MACRO_T = f"mignon-macro--{TEST_MODEL}"
MARKET = {"agent_id": MARKET_ID, "agent_type": MARKET_T}
COMPANY = {"agent_id": COMPANY_ID, "agent_type": COMPANY_T}
BRIEF = {"objective": "Screen AAPL puts.", "subjects": ["AAPL"], "criteria": ["filters"]}
SPAWN = {"description": "screen", "prompt": json.dumps(BRIEF), "subagent_type": MARKET_T}
EVIDENCE, CANDIDATE = "evidence:e-1", "candidate:c-1"
URL = "https://investor.example.com/q3"
SEARCH, EXTRACT = "mcp__tavily__tavily_search", "mcp__tavily__tavily_extract"


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
    overrides.setdefault(
        "registries", (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY, TAVILY_REGISTRY, LOCAL_REGISTRY)
    )
    overrides.setdefault("account_scope_table", {**SCOPE, "get_financials": NOT_SCOPED})
    overrides.setdefault("validator", RefValidator())
    return Session(make_deps(**overrides))


def agent_response(text: str, agent_id: str = MARKET_ID, agent_type: str = MARKET_T) -> Any:
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


@pytest.mark.parametrize("tool", [SEARCH, EXTRACT, BOARD, "mcp__wra_local__web_cache_lookup"])
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


@pytest.mark.parametrize("tool", ["get_scans", "create_scan", "update_scan_filters"])
def test_close_agent_orchestrator_has_no_scan_tools(tool: str) -> None:
    """ADR-0059: scans feed discovery, the Sell Options agent's work."""
    s = session(agent=AgentRole.CLOSE)
    assert_denied(s, s.pre(RH + tool, {"title": "WRA · x"}), "not available to the orchestrator")


@pytest.mark.parametrize("agent", [AgentRole.SELL, AgentRole.WHEEL])
def test_sell_and_legacy_orchestrators_keep_scan_tools(agent: AgentRole) -> None:
    s = session(agent=agent, account_scope_table={**SCOPE, "get_scans": NOT_SCOPED})
    reason = denied_reason(s.pre(RH + "get_scans", {}))
    assert reason is None or "not available" not in reason, reason


def test_close_agent_keeps_order_tools_and_mignons_keep_their_tools() -> None:
    """ADR-0059: a roll works orders and cancels; Mignon sets do not depend on the agent.
    ADR-0066: the orchestrator starts order-work jobs; review and place are the executor's."""
    s = session(
        agent=AgentRole.CLOSE,
        effective_mode=ExecutionMode.LIVE,
        registries=(ROBINHOOD_REGISTRY, WHEELTA_REGISTRY, ORDER_WORK_REGISTRY),
    )
    for tool, args in (
        ("mcp__wra_orders__work_option_order", {}),
        (RH + "cancel_option_order", {"account_number": ACCOUNT, "order_id": "o-1"}),
    ):
        reason = denied_reason(s.pre(tool, args))
        assert reason is None or "not available to" not in reason, reason
    s = session(agent=AgentRole.CLOSE, effective_mode=ExecutionMode.LIVE)
    assert_denied(s, s.pre(PLACE, {"account_number": ACCOUNT}), "not available to the orchestrator")
    reason = denied_reason(s.pre(BOARD, {}, **MARKET))
    assert reason is None or "not available" not in reason, reason


def test_mignon_types_have_their_own_tools() -> None:
    s = session()
    assert_denied(s, s.pre(SEARCH, {"query": "q"}, **MARKET), "mignon-market")
    s2 = session()
    out = s2.pre(SEARCH, {"query": "q"}, **COMPANY)
    assert denied_reason(out) is None and "updatedInput" in out["hookSpecificOutput"]


def test_mignon_call_is_attributed_when_recorded() -> None:
    s = session()
    s.pre(RH + "get_option_quotes", {}, **MARKET)
    req = s.rec.event("requested")
    assert req["agent_id"] == MARKET_ID and req["agent_type"] == MARKET_T
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
        ({"subagent_type": "mignon-market"}, "on an allowed model"),
        ({"subagent_type": "mignon-market--claude-other"}, "on an allowed model"),
        ({"subagent_type": "mignon-market--opus"}, "on an allowed model"),
        ({"prompt": ""}, "'prompt' missing"),
        ({"prompt": "Screen AAPL puts."}, "one MignonBrief JSON object"),
        (
            {"prompt": json.dumps({"objective": "x", "criteria": ["filters.min_delta"]})},
            "no rule 'filters.min_delta'",
        ),
        ({"prompt": json.dumps({"objective": "x", "max_results": 5})}, "MignonBrief"),
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
        ("Here is my report: ...", "Expecting value"),
        (
            '{"task": "t", "findings": [], "gaps": [], "follow_up_questions": [], "n": 1}',
            "not allowed",
        ),
        ('["a", "list"]', "top level must be an object"),
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


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        (finding("Unsupported.", ["evidence:never-delivered"]), "not delivered"),
        (finding("A claim with no source."), "at least one ref"),
        (finding("Page says so.", urls=[URL]), "not fetched"),
        (finding("Bid is 1.20.", urls=[URL]), "not fetched"),
    ],
)
def test_an_invalid_finding_is_dropped_and_the_rest_is_delivered(
    bad: dict[str, Any], fragment: str
) -> None:
    """ADR-0056: only the failing finding is removed (by index, without its claim); the valid
    findings still reach the orchestrator and the original text stays on record."""
    s = session()
    spawn_and_research(s)
    text = report(finding("The put bid is 1.20.", [EVIDENCE]), bad)
    out = s.post("Agent", agent_response(text), use_id="toolu_agent")
    envelope = delivered_envelope(out)
    assert envelope["kind"] == "validated"
    assert [f["claim"] for f in envelope["data"]["report"]["findings"]] == ["The put bid is 1.20."]
    (dropped,) = envelope["data"]["dropped_findings"]
    assert dropped["index"] == 1 and any(fragment in r for r in dropped["reasons"])
    assert bad["claim"] not in json.dumps(envelope)
    assert "store_raw_invalid" in s.rec.names()
    outcomes = [kw for n, kw in s.rec.events if n == "outcome"]
    assert outcomes[-1]["status"] is ToolCallStatus.SUCCEEDED


def test_ref_delivered_to_another_mignon_is_not_citable() -> None:
    s = session()
    research(s, COMPANY, use_id="toolu_c")  # the company Mignon saw EVIDENCE
    assert_ok(s.pre("Agent", SPAWN, use_id="toolu_agent"))
    text = report(finding("Bid is 1.20.", [EVIDENCE]))
    envelope = delivered_envelope(s.post("Agent", agent_response(text), use_id="toolu_agent"))
    assert envelope["data"]["report"]["findings"] == []
    assert "not delivered" in envelope["data"]["dropped_findings"][0]["reasons"][0]


def test_ref_handed_over_in_the_task_is_citable() -> None:
    s = session()
    research(s, {}, use_id="toolu_o")  # the orchestrator received EVIDENCE
    brief = {"objective": "Is the spread acceptable?", "subjects": [EVIDENCE]}
    spawn = {**SPAWN, "prompt": json.dumps(brief)}
    assert_ok(s.pre("Agent", spawn, use_id="toolu_agent"))
    text = report(finding("Spread of 0.10 per the quote.", [EVIDENCE]))
    envelope = delivered_envelope(s.post("Agent", agent_response(text), use_id="toolu_agent"))
    assert envelope["kind"] == "validated"


def web_session(**overrides: Any) -> Session:
    """A session whose results go through the real result boundary (ADR-0058)."""
    redactor = Redactor(account_number=SecretStr(ACCOUNT))
    return session(validator=BoundaryValidator(redactor=redactor), redactor=redactor, **overrides)


def spawn_company(s: Session) -> None:
    assert_ok(s.pre("Agent", {**SPAWN, "subagent_type": COMPANY_T}, use_id="toolu_agent"))


def extract(
    s: Session,
    urls: list[str],
    *,
    ok: list[str] | None = None,
    failed: list[str] | None = None,
    agent: dict[str, str] = COMPANY,
    use_id: str = "toolu_f",
) -> Any:
    """One tavily_extract by `agent`; Tavily returns content for `ok` (default: every URL)."""
    assert_ok(s.pre(EXTRACT, {"urls": urls, "query": "q"}, use_id=use_id, **agent))
    payload = {
        "results": [{"url": u, "raw_content": "page text"} for u in (urls if ok is None else ok)],
        "failed_results": [{"url": u, "error": "Pay via x402 to continue"} for u in failed or []],
    }
    return s.post(EXTRACT, {"structuredContent": payload, "content": []}, use_id=use_id)


def cite(s: Session, url: str, claim: str = "The first round is on October 4, 2026.") -> Any:
    text = report(finding(claim, urls=[url]))
    out = s.post("Agent", agent_response(text, COMPANY_ID, COMPANY_T), use_id="toolu_agent")
    return delivered_envelope(out)


def test_a_number_from_an_extracted_page_is_kept_and_labelled_web_sourced() -> None:
    """ADR-0056: the 2026-09-30 election date, cited to an extracted article, survives."""
    s = web_session()
    spawn_company(s)
    out = extract(s, [URL])
    delivered = wire(out["hookSpecificOutput"]["updatedToolOutput"])
    assert delivered["kind"] == "validated" and delivered["data"]["untrusted_web_content"]
    envelope = cite(s, URL)
    assert envelope["kind"] == "validated"
    assert envelope["data"]["web_sourced_findings"] == [0]
    assert envelope["data"]["dropped_findings"] == []


def test_a_url_tavily_returned_no_content_for_is_not_citable() -> None:
    """ADR-0056/0058: a failed URL is not a source, and Tavily's error text is not delivered."""
    s = web_session()
    spawn_company(s)
    out = extract(s, [URL], ok=[], failed=[URL])
    delivered = wire(out["hookSpecificOutput"]["updatedToolOutput"])
    assert delivered["data"]["failed_urls"] == [URL]
    assert "x402" not in json.dumps(delivered)
    reasons = cite(s, URL)["data"]["dropped_findings"][0]["reasons"]
    assert any("not fetched" in r for r in reasons)


def test_a_search_result_is_a_lead_not_a_source() -> None:
    s = web_session()
    spawn_company(s)
    assert_ok(s.pre(SEARCH, {"query": "AAPL election"}, use_id="toolu_s", **COMPANY))
    result = {"query": "q", "results": [{"url": URL, "content": "October 4", "score": 0.9}]}
    s.post(SEARCH, {"structuredContent": result, "content": []}, use_id="toolu_s")
    reasons = cite(s, URL)["data"]["dropped_findings"][0]["reasons"]
    assert any("not fetched" in r for r in reasons)


def test_a_failed_or_timed_out_url_is_not_extracted_again_this_run() -> None:
    """ADR-0056: by any Mignon, after no content, a failed call, or a cap notice."""
    s = web_session()
    extract(s, [URL], ok=[], failed=[URL])
    out = s.pre(EXTRACT, {"urls": [URL]}, use_id="toolu_g", **COMPANY)
    assert URL in (denied_reason(out) or "") and "not extracted again" in (denied_reason(out) or "")
    slow = "https://slow.example.com/page"
    assert_ok(s.pre(EXTRACT, {"urls": [slow]}, use_id="toolu_h", **COMPANY))
    s.fail(EXTRACT, use_id="toolu_h", error="timeout of 60000ms exceeded")
    other = {"agent_id": "a-company-2", "agent_type": COMPANY_T}
    out = s.pre(EXTRACT, {"urls": [slow]}, use_id="toolu_i", **other)
    assert "not extracted again" in (denied_reason(out) or "")
    capped = "https://capped.example.com/page"
    assert_ok(s.pre(EXTRACT, {"urls": [capped]}, use_id="toolu_j", **COMPANY))
    notice = {"code": "monthly_cap_reached", "message": "Pay via x402", "next_actions": []}
    out = s.post(EXTRACT, {"structuredContent": notice, "content": []}, use_id="toolu_j")
    delivered = wire(out["hookSpecificOutput"]["updatedToolOutput"])
    assert delivered["kind"] == "missing" and "monthly_cap_reached" in delivered["gaps"][0]
    assert "x402" not in json.dumps(delivered)
    out = s.pre(EXTRACT, {"urls": [capped]}, use_id="toolu_k", **other)
    assert "not extracted again" in (denied_reason(out) or "")


def test_a_rate_limited_url_may_be_extracted_once_more_then_is_refused() -> None:
    """ADR-0060: a 429 says nothing about the page; the agent is told why and may retry once."""
    s = web_session()
    limited = {"error": "Extract failed", "detail": {"error": "blocked"}, "status": 429}
    for use_id in ("toolu_r1", "toolu_r2"):
        assert_ok(s.pre(EXTRACT, {"urls": [URL]}, use_id=use_id, **COMPANY))
        out = s.post(EXTRACT, {"structuredContent": limited, "content": []}, use_id=use_id)
        delivered = wire(out["hookSpecificOutput"]["updatedToolOutput"])
        assert delivered["kind"] == "missing" and "HTTP 429" in delivered["gaps"][0]
        assert "blocked" not in json.dumps(delivered)
    out = s.pre(EXTRACT, {"urls": [URL]}, use_id="toolu_r3", **COMPANY)
    assert "not extracted again" in (denied_reason(out) or "")


def test_a_mignon_may_extract_a_page_another_mignon_extracted_but_not_repeat_its_own() -> None:
    """ADR-0056: a page is citable only by the Mignon that extracted it, so the cache does not
    deny another Mignon's extract of the same page; a Mignon's own repeat is denied."""
    s = web_session(web_precheck=lambda tool, args: "identical fetch recorded")
    extract(s, [URL])
    out = s.pre(EXTRACT, {"urls": [URL]}, use_id="toolu_g", **COMPANY)
    assert "already extracted" in (denied_reason(out) or "")
    other = {"agent_id": "a-company-2", "agent_type": COMPANY_T}
    assert_ok(s.pre(EXTRACT, {"urls": [URL]}, use_id="toolu_h", **other))


def test_review_and_delivery_agree_on_refs_handed_over_in_the_task(tmp_path: Path) -> None:
    """ADR-0056: a run ref this Mignon's task did not name is flagged by the review (so it can
    be patched), not only dropped at delivery."""
    s = session()
    research(s, COMPANY, use_id="toolu_c")  # EVIDENCE delivered to another Mignon only
    assert_ok(s.pre("Agent", SPAWN, use_id="toolu_agent"))
    original = report(finding("Bid is 1.20.", [EVIDENCE]))
    out = stop_hook(s, transcript(tmp_path, original))
    assert out["decision"] == "block" and "findings.0.refs" in out["reason"]


def test_a_patch_cannot_keep_a_stray_field_on_a_finding(tmp_path: Path) -> None:
    """ADR-0056: a patched finding keeps only claim, refs, and web_urls."""
    s = session()
    spawn_and_research(s)
    bad = {**finding("The put bid is 1.20.", [EVIDENCE]), "note": "x"}
    original = json.dumps({"task": "t", "findings": [bad], "gaps": [], "follow_up_questions": []})
    out = stop_hook(s, transcript(tmp_path, original))
    assert out["decision"] == "block" and "findings.0" in out["reason"]
    patch = json.dumps({"patches": [{"finding": "0", "refs": [EVIDENCE]}]})
    assert stop_hook(s, transcript(tmp_path, original, patch)) == {}
    envelope = delivered_envelope(s.post("Agent", agent_response(patch), use_id="toolu_agent"))
    assert envelope["kind"] == "validated" and envelope["data"]["dropped_findings"] == []
    assert envelope["data"]["report"]["findings"][0]["claim"] == "The put bid is 1.20."


def test_dropped_reasons_never_echo_the_mignons_text() -> None:
    """ADR-0056: refs, URLs, and duplicate values are summarized, not repeated."""
    s = session()
    spawn_and_research(s)
    text = report(
        finding("Kept.", [EVIDENCE]),
        finding("X.", ["evidence:Brazil_votes_Oct_4"]),
        finding("Y.", urls=["https://news.example.com/secret-slug"]),
        finding("Z.", [EVIDENCE, EVIDENCE]),
    )
    envelope = delivered_envelope(s.post("Agent", agent_response(text), use_id="toolu_agent"))
    dumped = json.dumps(envelope["data"]["dropped_findings"])
    assert [d["index"] for d in envelope["data"]["dropped_findings"]] == [1, 2, 3]
    for echoed in ("Brazil_votes", "secret-slug", EVIDENCE):
        assert echoed not in dumped


def test_extracted_url_is_citable_by_the_mignon_that_extracted_it() -> None:
    """Both the requested spelling and the one Tavily returned are citable."""
    s = web_session()
    spawn_company(s)
    asked = "https://Investor.Example.com/q3#guidance"
    extract(s, [asked], ok=[URL])
    for url in (asked, URL):
        envelope = cite(s, url, "Management reaffirmed guidance.")
        assert envelope["kind"] == "validated" and envelope["data"]["dropped_findings"] == []
        spawn_company(s)


def test_report_from_a_different_mignon_type_is_invalid() -> None:
    s = session()
    spawn_and_research(s)
    text = report(finding("Bid is 1.20.", [EVIDENCE]))
    out = s.post("Agent", agent_response(text, MARKET_ID, MACRO_T), use_id="toolu_agent")
    assert any("agent type" in g for g in delivered_envelope(out)["gaps"])


@pytest.mark.parametrize(
    "response",
    [
        {"isAsync": True, "status": "async_launched", "agentId": "a-1"},
        [{"type": "text", "text": "report"}],
        {"status": "completed", "agentId": "a-1", "agentType": MARKET_T, "content": []},
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
    assert start(s, MACRO_T) == {}
    assert start(session(), "mignon-macro")["continue_"] is False  # no model
    assert start(session(), "mignon-macro--claude-other")["continue_"] is False
    assert start(s, "general-purpose")["continue_"] is False
    assert s.deps.run_control.stop_requested
    s2 = session(mignon_limits=None)
    assert start(s2, MARKET_T)["continue_"] is False


def test_a_mignon_on_a_model_outside_the_allowlist_is_unknown() -> None:
    s = session()
    out = s.pre(RH + "get_option_quotes", agent_id="a-1", agent_type="mignon-market--claude-x")
    assert_denied(s, out, "unknown sub-agent type")


def test_the_orchestrator_picks_among_allowed_models() -> None:
    s = session(mignon_models=(TEST_MODEL, "claude-haiku-4-5"))
    assert_ok(s.pre("Agent", {**SPAWN, "subagent_type": "mignon-company--claude-haiku-4-5"}))


# ---- report repair by patch (ADR-0047) ------------------------------------------------------


def transcript(tmp_path: Path, *texts: str) -> str:
    """A CLI transcript whose assistant messages are `texts` (the last is the final one)."""
    path = tmp_path / f"agent-{uuid.uuid4().hex}.jsonl"
    lines = [json.dumps({"type": "user", "message": {"content": "task"}})]
    for text in texts:
        message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
        lines.append(json.dumps({"type": "assistant", "message": message}))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def stop_hook(s: Session, path: str | None, agent: dict[str, str] = MARKET) -> Any:
    callback = s.hooks["SubagentStop"][0].hooks[0]
    data = {
        "hook_event_name": "SubagentStop",
        "stop_hook_active": False,
        "agent_transcript_path": path,
        **agent,
    }
    return drive(callback(data, None, {"signal": None}))


def test_uncited_finding_gets_patch_feedback_and_is_merged(tmp_path: Path) -> None:
    s = session()
    spawn_and_research(s)
    original = report(
        finding("The regime is calm.", [EVIDENCE]),
        finding("VIX is 15.9.", urls=[URL]),
        finding("No source."),
    )
    out = stop_hook(s, transcript(tmp_path, original))
    assert out["decision"] == "block"
    assert "findings.1" in out["reason"] and "findings.2" in out["reason"]
    assert "findings.0" not in out["reason"] and '"patches"' in out["reason"]
    patch = json.dumps(
        {
            "patches": [
                {"finding": "1", "refs": [EVIDENCE], "web_urls": []},
                {"finding": "2", "drop": True},
            ]
        }
    )
    assert stop_hook(s, transcript(tmp_path, original, patch)) == {}
    out = s.post("Agent", agent_response(patch), use_id="toolu_agent")
    envelope = delivered_envelope(out)
    assert envelope["kind"] == "validated"
    findings = envelope["data"]["report"]["findings"]
    assert [f["claim"] for f in findings] == ["The regime is calm.", "VIX is 15.9."]
    assert findings[1]["refs"] == [EVIDENCE]
    assert envelope["data"]["repair"] == {
        "finding_origins": [0, 1],
        "patched": [1],
        "dropped": [2],
    }
    (raw,) = [r for k, r in s.rec.results.values() if k.value == "raw_invalid"]
    assert raw["text"] == original and raw["patches"] == [patch]


def test_patch_indexes_stay_those_of_the_original_report(tmp_path: Path) -> None:
    s = session()
    spawn_and_research(s)
    original = report(finding("No source."), finding("Bid is 1.20.", urls=[URL]))
    assert stop_hook(s, transcript(tmp_path, original))["decision"] == "block"
    first = json.dumps({"patches": [{"finding": "0", "drop": True}]})
    out = stop_hook(s, transcript(tmp_path, first))
    # Finding 1 is still wrong and is named by its ORIGINAL index after finding 0 was dropped.
    assert out["decision"] == "block" and "findings.1" in out["reason"]
    second = json.dumps({"patches": [{"finding": "1", "refs": [EVIDENCE], "web_urls": []}]})
    assert stop_hook(s, transcript(tmp_path, second)) == {}
    envelope = delivered_envelope(s.post("Agent", agent_response(second), use_id="toolu_agent"))
    assert envelope["kind"] == "validated"
    assert envelope["data"]["repair"]["finding_origins"] == [1]


def test_repairs_stop_after_the_limit_and_the_finding_is_dropped(tmp_path: Path) -> None:
    s = session()
    spawn_and_research(s)
    original = report(finding("No source."))
    useless = json.dumps({"patches": [{"finding": "0", "claim": "Still no source."}]})
    assert stop_hook(s, transcript(tmp_path, original))["decision"] == "block"
    assert stop_hook(s, transcript(tmp_path, useless))["decision"] == "block"
    assert stop_hook(s, transcript(tmp_path, useless)) == {}  # MAX_MIGNON_REPAIRS reached
    envelope = delivered_envelope(s.post("Agent", agent_response(useless), use_id="toolu_agent"))
    # ADR-0056: the unsourced finding is dropped; an empty report is still a report.
    assert envelope["kind"] == "validated" and envelope["data"]["report"]["findings"] == []
    (dropped,) = envelope["data"]["dropped_findings"]
    assert dropped["index"] == 0 and "at least one ref" in dropped["reasons"][0]


def test_a_bad_patch_is_named_in_the_next_feedback(tmp_path: Path) -> None:
    s = session()
    spawn_and_research(s)
    assert stop_hook(s, transcript(tmp_path, report(finding("No source."))))["decision"] == "block"
    out = stop_hook(s, transcript(tmp_path, '{"patches": [{"finding": "7", "drop": true}]}'))
    assert out["decision"] == "block" and "no finding 7" in out["reason"]


@pytest.mark.parametrize(
    "text",
    [
        report(finding("The put bid is 1.20.", [EVIDENCE])),  # valid: nothing to repair
        "Not a report at all.",  # no object: PostToolUse reports it
        '{"task": "t", "findings": [], "gaps": [], "follow_up_questions": [], "x": "y"}',
    ],
)
def test_stop_is_allowed_when_patching_cannot_help(tmp_path: Path, text: str) -> None:
    s = session()
    spawn_and_research(s)
    assert stop_hook(s, transcript(tmp_path, text)) == {}


def test_stop_is_allowed_without_a_transcript_or_for_a_non_mignon(tmp_path: Path) -> None:
    s = session()
    spawn_and_research(s)
    assert stop_hook(s, str(tmp_path / "missing.jsonl")) == {}
    assert stop_hook(s, None) == {}
    other = {"agent_id": "x", "agent_type": "general-purpose"}
    assert stop_hook(s, transcript(tmp_path, report(finding("No source."))), other) == {}
    s.deps.run_control.request_stop(StopReason.DEADLINE, NOW)
    assert stop_hook(s, transcript(tmp_path, report(finding("No source.")))) == {}


def test_last_assistant_text_reads_the_final_message(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    tool_use = {"type": "tool_use", "id": "t", "name": "x", "input": {}}
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "assistant", "message": {"content": [tool_use]}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "text", "text": "a"},
                                {"type": "text", "text": "b"},
                            ]
                        },
                    }
                ),
                "not json",
                json.dumps({"type": "user", "message": {"content": "later"}}),
            ]
        )
    )
    assert last_assistant_text(str(path)) == "ab"
    assert last_assistant_text(str(tmp_path / "none.jsonl")) is None
    assert last_assistant_text(5) is None


def test_a_report_lists_what_it_did_not_cover_of_the_brief() -> None:
    """ADR-0061: brief subjects without a finding and `want` values a finding lacks."""
    s = session()
    research(s, {}, use_id="toolu_o")
    brief = {
        "objective": "Re-quote.",
        "subjects": ["AAPL", "MSFT"],
        "want": ["bid", "delta"],
        "notes": f"Prior quote: {EVIDENCE}",
    }
    assert_ok(s.pre("Agent", {**SPAWN, "prompt": json.dumps(brief)}, use_id="toolu_agent"))
    covered = {
        "claim": "AAPL put quoted.",
        "refs": [EVIDENCE],
        "web_urls": [],
        "subject": "AAPL",
        "values": {"bid": "1.20"},
    }
    text = json.dumps({"task": "t", "findings": [covered], "gaps": [], "follow_up_questions": []})
    envelope = delivered_envelope(s.post("Agent", agent_response(text), use_id="toolu_agent"))
    assert envelope["kind"] == "validated"
    assert envelope["data"]["coverage_gaps"] == [
        "subject MSFT: not reported",
        "subject AAPL: no value for 'delta'",
    ]
    assert envelope["data"]["report"]["findings"][0]["values"] == {"bid": "1.20"}
