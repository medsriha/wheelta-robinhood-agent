"""Research emails use only reports accepted at the delivery boundary."""

from typing import Any
from uuid import uuid4

import pytest

from wheelta_robinhood_agent.agent import summary_loader


def test_loads_accepted_reports_but_not_rejected_or_raw_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "task": "Compare candidates",
        "findings": [{"claim": "Earnings risk", "refs": ["candidate:1"], "web_urls": []}],
        "gaps": [],
        "follow_up_questions": [],
    }
    envelopes = [
        {"kind": "validated", "tool": "Agent", "data": {"report": report}},
        {"kind": "missing", "tool": "Agent", "data": {"report": report}},
        {"kind": "error", "tool": "Agent", "data": {"report": report}},
        {"kind": "validated", "tool": "WebFetch", "data": {"report": report}},
    ]

    def delivered(*args: Any) -> Any:
        return envelopes

    def facts(*args: Any) -> tuple[()]:
        return ()

    monkeypatch.setattr(summary_loader, "_delivered_envelopes", delivered)
    monkeypatch.setattr(summary_loader.evidence, "decision_facts_for_run", facts)
    candidates, reports = summary_loader.load_summary_research(None, uuid4())  # type: ignore[arg-type]
    assert not candidates
    assert len(reports) == 1
    assert reports[0].findings[0].claim == "Earnings risk"


def test_dropped_findings_become_gaps_and_web_sourced_claims_are_labelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0056: the email shows what code dropped (index and reason, no claim) and marks a
    number that rests only on fetched pages."""
    from wheelta_robinhood_agent.observability import run_summary

    report = {
        "task": "Brazil election",
        "findings": [
            {"claim": "First round on October 4.", "refs": [], "web_urls": ["https://n.ex/a"]}
        ],
        "gaps": [],
        "follow_up_questions": [],
    }
    data = {
        "report": report,
        "dropped_findings": [{"index": 2, "reasons": ["a cited ref was not delivered"]}],
        "web_sourced_findings": [0],
    }
    envelopes = [{"kind": "validated", "tool": "Agent", "data": data}]
    monkeypatch.setattr(summary_loader, "_delivered_envelopes", lambda *a: envelopes)
    monkeypatch.setattr(summary_loader.evidence, "decision_facts_for_run", lambda *a: ())
    _, (loaded,) = summary_loader.load_summary_research(None, uuid4())  # type: ignore[arg-type]
    assert loaded.gaps == ("finding 2 dropped by code: a cited ref was not delivered",)
    summary = run_summary.RunSummaryInput.model_construct(research_reports=(loaded,))
    lines = run_summary._research_lines(summary)
    assert "- First round on October 4. (web-sourced) [https://n.ex/a]" in lines
    assert any("finding 2 dropped by code" in line for line in lines)
