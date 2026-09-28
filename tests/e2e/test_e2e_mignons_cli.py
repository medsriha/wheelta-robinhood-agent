"""Mignon acceptance tests against the real Claude Code CLI (ADR-0025, agent/mignons.py).

The orchestrator/Mignon design rests on CLI behaviour that is not documented, so each fact is
pinned here against the bundled CLI (claude-agent-sdk 0.2.160, CLI 2.1.283), through our real
options, hooks, and validating proxy (tests/e2e/cli_harness/):

- the parent's hooks see a Mignon's tool calls with `agent_id`/`agent_type`, and the proxy
  receives their `claudecode/toolUseId`;
- a Mignon is offered only its `AgentDefinition.tools`, and a call outside them is refused
  before any hook or server;
- with CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 `Agent` is synchronous, and a dict
  `updatedToolOutput` replaces the report the orchestrator's next request carries (raw
  Mignon text never reaches it);
- with CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1 no general-purpose agent type is offered.

Skipped unless WRA_RUN_REQUIRES_CLI=1. Re-run whenever the SDK or CLI is bumped.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from cli_harness.mcp_server import Json
from cli_harness.model_server import FinalText, RecordedRequest, ToolUse
from cli_harness.session import ACCOUNT_NUMBER, SERVER, Case, SessionOutcome
from test_e2e_result_boundary_cli import (
    QUOTES,
    QUOTES_TOOL,
    assert_never_sent,
    diagnostics,
    gap_mapper,
    result_text,
    run,
    sentinel,
)

from wheelta_robinhood_agent.agent.mignons import MignonLimits
from wheelta_robinhood_agent.domain.enums import ToolCallStatus
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY

pytestmark = [pytest.mark.requires_cli, pytest.mark.allow_hosts(["127.0.0.1"])]

LIMITS = MignonLimits(max_per_run=8, max_concurrent=4, max_turns_per_mignon=10)
SPAWN = ToolUse(
    "Agent",
    {"description": "screen", "prompt": "Screen AAPL puts.", "subagent_type": "mignon-market"},
)
POSITIONS_TOOL = ROBINHOOD_REGISTRY.qualified("get_option_positions")
EVIDENCE_RE = re.compile(r"evidence:[0-9a-f-]{36}")


def cite_quote(body: dict[str, Any]) -> FinalText:
    """The Mignon's final step: a MignonReport citing the evidence ref it was delivered."""
    text = json.dumps(body.get("messages", []))
    found = EVIDENCE_RE.search(text)
    refs = [found.group(0)] if found else []
    return FinalText(
        json.dumps(
            {
                "task": "Screen AAPL puts.",
                "findings": [{"claim": "The live quote came back.", "refs": refs, "web_urls": []}],
                "gaps": [],
                "follow_up_questions": [],
            }
        )
    )


def agent_result(out: SessionOutcome) -> dict[str, Any]:
    """Our envelope inside the orchestrator's Agent tool_result (after the CLI's frame)."""
    for request in out.model.main_loop():
        for block in request.tool_results():
            text = result_text(block)
            at = text.find('{"data"')
            if "[Subagent hand-back]" in text and at >= 0:
                value, _ = json.JSONDecoder().raw_decode(text[at:])
                assert isinstance(value, dict)
                return value
    pytest.fail("the orchestrator never received a Mignon hand-back: " + diagnostics(out))


def mignon_requests(out: SessionOutcome) -> list[RecordedRequest]:
    requests = out.model.mignon_loop()
    assert requests, "no Mignon ever ran: " + diagnostics(out)
    return requests


def market_case(*mignon_steps: Any, **kw: Any) -> Case:
    return Case(
        steps=[SPAWN, FinalText("done")],
        mignon_steps=list(mignon_steps),
        behaviors={QUOTES: Json({"ok": True})},
        mappers={(SERVER, QUOTES): gap_mapper("screen")},
        mignons=LIMITS,
        **kw,
    )


def test_mignon_calls_are_attributed_and_its_report_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = run(market_case(ToolUse(QUOTES_TOOL, {"symbols": ["AAPL"]}), cite_quote), monkeypatch)
    calls = {c["tool"]: c for c in out.recorder.requested_calls()}
    assert calls["Agent"].get("agent_id") is None
    quote = calls[QUOTES]
    assert quote["agent_type"] == "mignon-market" and quote["agent_id"], diagnostics(out)
    assert out.mcp.called(QUOTES) == 1  # through the proxy, correlated by toolUseId
    envelope = agent_result(out)
    assert envelope["kind"] == "validated", envelope
    assert envelope["data"]["agent_id"] == quote["agent_id"]
    assert EVIDENCE_RE.fullmatch(envelope["data"]["report"]["findings"][0]["refs"][0])
    spawn_id = out.recorder.ids[calls["Agent"]["sdk_tool_use_id"]]
    assert [o["status"] for o in out.recorder.outcomes_for(spawn_id)] == [ToolCallStatus.SUCCEEDED]


def test_raw_mignon_text_never_reaches_the_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = sentinel("mignon")
    out = run(market_case(FinalText(f"I found a great trade {raw}; buy it.")), monkeypatch)
    assert agent_result(out)["kind"] == "missing"
    for request in out.model.main_loop():
        assert raw not in request.text(), "raw Mignon text reached the orchestrator"


def test_a_mignon_sees_only_its_own_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    out = run(
        market_case(
            ToolUse(POSITIONS_TOOL, {"account_number": ACCOUNT_NUMBER}),
            ToolUse("WebSearch", {"query": "AAPL"}),
            FinalText("{}"),
        ),
        monkeypatch,
    )
    offered = set(mignon_requests(out)[0].tool_names())
    assert QUOTES_TOOL in offered
    assert not {"Agent", "WebSearch", "WebFetch", POSITIONS_TOOL} & offered, offered
    assert out.mcp.called("get_option_positions") == 0
    recorded = {c["tool"] for c in out.recorder.requested_calls()}
    assert "get_option_positions" not in recorded and "WebSearch" not in recorded


def test_only_mignon_types_are_offered_and_web_is_denied_to_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = Case(
        steps=[ToolUse("WebSearch", {"query": "AAPL"}), FinalText("done")],
        behaviors={},
        mignons=LIMITS,
    )
    out = run(case, monkeypatch)
    (first, *_) = out.model.main_loop()
    assert "Agent" in first.tool_names()
    listing = first.text().split("Available agent types for the Agent tool:", 1)[1][:2000]
    assert "- mignon-market:" in listing
    for builtin in ("- general-purpose:", "- Explore:", "- Plan:", "- claude:"):
        assert builtin not in listing, listing
    (search,) = [c for c in out.recorder.requested_calls() if c["tool"] == "WebSearch"]
    outcome = out.recorder.outcomes_for(out.recorder.ids[search["sdk_tool_use_id"]])
    assert [o["status"] for o in outcome] == [ToolCallStatus.DENIED]
    assert "orchestrator" in outcome[0]["reason"]


def test_background_is_dropped_and_the_report_still_crosses_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With background tasks disabled the CLI removes `run_in_background` before PreToolUse
    and runs the Mignon synchronously, so its report still goes through PostToolUse."""
    raw = sentinel("background")
    spawn = ToolUse("Agent", {**SPAWN.input, "run_in_background": True})
    case = Case(
        steps=[spawn, FinalText("done")],
        mignon_steps=[FinalText(raw)],
        behaviors={},
        mignons=LIMITS,
    )
    out = run(case, monkeypatch)
    (agent,) = [c for c in out.recorder.requested_calls() if c["tool"] == "Agent"]
    assert "run_in_background" not in agent["arguments_redacted"]
    assert agent_result(out)["kind"] == "missing"
    for request in out.model.main_loop():
        assert raw not in request.text()


def test_a_model_override_is_denied_and_no_mignon_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = sentinel("override")
    spawn = ToolUse("Agent", {**SPAWN.input, "model": "haiku"})
    case = Case(
        steps=[spawn, FinalText("done")],
        mignon_steps=[FinalText(raw)],
        behaviors={},
        mignons=LIMITS,
    )
    out = run(case, monkeypatch)
    assert out.model.mignon_loop() == [], diagnostics(out)
    (agent,) = [c for c in out.recorder.requested_calls() if c["tool"] == "Agent"]
    outcome = out.recorder.outcomes_for(out.recorder.ids[agent["sdk_tool_use_id"]])
    assert [o["status"] for o in outcome] == [ToolCallStatus.DENIED]
    assert_never_sent(out, raw)
