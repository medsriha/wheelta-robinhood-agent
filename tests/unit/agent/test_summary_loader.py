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

    monkeypatch.setattr(summary_loader, "_delivered_envelopes", delivered)
    candidates, reports = summary_loader.load_summary_research(None, uuid4())  # type: ignore[arg-type]
    assert not candidates
    assert len(reports) == 1
    assert reports[0].findings[0].claim == "Earnings risk"
