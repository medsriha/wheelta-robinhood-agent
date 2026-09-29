"""Wheelta capture, board-query mapper, board-origin candidates, and board facts (ADR-0041)."""

import copy
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from wheelta_robinhood_agent.agent.mapped_evidence import MappingRequest
from wheelta_robinhood_agent.agent.result_boundary import (
    BoundaryValidator,
    ValidationRequest,
)
from wheelta_robinhood_agent.agent.wheelta_mappers import (
    CONTEXT_GAP,
    GROUPS_GAP,
    IDENTITY_GAP,
    map_board_query,
)
from wheelta_robinhood_agent.domain.enums import CandidateOrigin, ToolTier
from wheelta_robinhood_agent.integrations.registry import diff_discovered
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY
from wheelta_robinhood_agent.observability.redaction import Redactor

FIXTURES = Path(__file__).parents[2] / "fixtures"
CAPTURE = json.loads((FIXTURES / "wheelta" / "tools_2026-09-29.json").read_text())
BOARD = json.loads(
    (FIXTURES / "wheelta" / "results" / "board_query.default_select.json").read_text()
)
CALL = uuid.UUID("0190a0a0-0000-7000-8000-000000000009")
RETRIEVED = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)


def _ids() -> Callable[[], uuid.UUID]:
    counter = iter(range(1, 10_000))
    return lambda: uuid.UUID(int=next(counter))


def _request(payload: Any) -> MappingRequest:
    return MappingRequest(
        tool_call_id=CALL,
        server="wheelta",
        tool="wheelta_board_query",
        effective_input={},
        payload=payload,
        retrieved_at=RETRIEVED,
    )


def _payload() -> dict[str, Any]:
    return copy.deepcopy(BOARD["structuredContent"])


# ---------------------------------------------------------------------------- capture


def test_registry_matches_the_capture_and_every_tool_is_read_only() -> None:
    names = {t["name"] for t in CAPTURE["tools"]}
    assert len(names) == 12
    diff = diff_discovered(WHEELTA_REGISTRY, names)
    assert diff.unknown == frozenset() and diff.missing == frozenset()
    assert WHEELTA_REGISTRY.verified
    assert {t.tier for t in WHEELTA_REGISTRY.tools} == {ToolTier.R}


def test_captured_board_query_schema_takes_the_select_and_filter_arguments() -> None:
    (query,) = [t for t in CAPTURE["tools"] if t["name"] == "wheelta_board_query"]
    props = set(query["input_schema"]["properties"])
    assert {"filters", "select", "sort_by", "limit", "offset"} <= props


# ---------------------------------------------------------------------------- mapper


def test_board_rows_become_screens_with_the_build_identity() -> None:
    out = map_board_query(_request(_payload()), _ids())
    assert len(out.board_screens) == 3
    first = out.board_screens[0]
    assert str(first.occ_symbol) == "IWM   261016P00267000"
    assert first.bid == Decimal("1.15") and first.build_id == "df52f70ac584"
    assert first.row_id == "IWM:medium" and first.wheel_iq_score == Decimal("76.24")
    assert first.as_of == datetime(2026, 9, 28, 20, 58, 23, tzinfo=UTC)
    assert all(s.source_tool_call_ids == (CALL,) for s in out.board_screens)
    board, *rows = out.screen_context
    assert board["board"]["build_id"] == "df52f70ac584"  # type: ignore[index]
    assert rows[0]["risk.annualizedYield"] == "0.0896"  # numbers as decimal strings
    assert out.gaps == (CONTEXT_GAP,)


def test_rows_without_a_contract_identity_are_context_only() -> None:
    payload = _payload()
    del payload["rows"][1]["contract.bid"]
    out = map_board_query(_request(payload), _ids())
    assert len(out.board_screens) == 2 and len(out.screen_context) == 4
    assert IDENTITY_GAP in out.gaps


def test_grouped_results_are_context_only() -> None:
    payload = _payload()
    payload.update(mode="groups", groupedBy="sector", groups=[{"value": "Tech", "rows": 3}])
    out = map_board_query(_request(payload), _ids())
    assert out.board_screens == () and out.gaps == (GROUPS_GAP,)
    assert out.screen_context[1] == {"group": {"value": "Tech", "rows": "3"}}


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["freshness"].update(buildState="building"),
        lambda p: p["freshness"].update(buildId=""),
        lambda p: p["freshness"].update(asOf="2026-09-28T20:58:23"),  # no offset
        lambda p: p["rows"][0].update({"contract.type": "call"}),
        lambda p: p["rows"][0].update({"contract.occSymbol": "SPY   261016P00740000"}),
        lambda p: p["rows"][0].update({"contract.bid": "abc"}),
        lambda p: p["rows"][0].update({"contract.bid": True}),
        lambda p: p["rows"].append(copy.deepcopy(p["rows"][0])),  # duplicate contract
        lambda p: p.update(rows="x"),
        lambda p: p.update(rows=["x"]),
        lambda p: p.update(mode="groups", groups="x"),
    ],
)
def test_malformed_board_results_raise(change: Callable[[dict[str, Any]], None]) -> None:
    payload = _payload()
    change(payload)
    with pytest.raises(ValueError):
        map_board_query(_request(payload), _ids())


def test_a_matching_occ_symbol_is_accepted() -> None:
    payload = _payload()
    payload["rows"][0]["contract.occSymbol"] = "IWM   261016P00267000"
    assert len(map_board_query(_request(payload), _ids()).board_screens) == 3


# ------------------------------------------------------------- board-origin candidates

INSTRUMENTS = json.loads(
    (
        FIXTURES / "robinhood" / "results" / "get_option_instruments.SPY_20261016_P740.json"
    ).read_text()
)


def _validate(board_screens: Any) -> dict[str, Any]:
    validator = BoundaryValidator(
        redactor=Redactor(account_number=None), board_screens=board_screens
    )
    text = json.dumps({"data": INSTRUMENTS["data"]})
    outcome = validator(
        ValidationRequest(
            tool_call_id=CALL,
            server="robinhood",
            tool="get_option_instruments",
            tier=ToolTier.R,
            effective_input={"ids": "x"},
            tool_response={"content": [{"type": "text", "text": text}]},
            retrieved_at=RETRIEVED,
        )
    )
    data = outcome.envelope.data
    assert isinstance(data, dict)
    return data["evidence"]  # type: ignore[no-any-return]


def _screen_for_spy() -> Any:
    payload = _payload()
    payload["rows"][1]["contract.strike"] = 740.0
    screens = map_board_query(_request(payload), _ids()).board_screens
    return {str(s.occ_symbol): s for s in screens}


def test_instruments_issue_robinhood_candidates_by_default() -> None:
    (candidate,) = _validate(None)["candidates"]
    assert candidate["origin"] == CandidateOrigin.ROBINHOOD.value
    assert candidate["candidate_ref"].startswith("candidate:")
    assert _validate(lambda: {})["candidates"][0]["origin"] == "robinhood"


def test_a_contract_on_the_current_board_is_a_board_candidate() -> None:
    screens = _screen_for_spy()
    assert "SPY   261016P00740000" in screens
    (candidate,) = _validate(lambda: screens)["candidates"]
    assert candidate["origin"] == CandidateOrigin.BOARD.value


def test_only_the_runs_current_build_is_used() -> None:
    """data_quality.freshness.wheelta_board: current buildId only, never an older board."""
    from wheelta_robinhood_agent.agent.facts_tool import RunEvidence

    old = _payload()
    old["freshness"].update(buildId="old", asOf="2026-09-28T14:00:00Z")
    new = _payload()
    new["freshness"].update(buildId="new", asOf="2026-09-28T18:00:00Z")
    del new["rows"][2]  # the newer build no longer lists IWM:long
    ids = _ids()
    items = (map_board_query(_request(old), ids), map_board_query(_request(new), ids))
    screens = RunEvidence(items).board_screens()
    assert {s.build_id for s in screens.values()} == {"new"}
    assert len(screens) == 2 and "IWM   261030P00263000" not in screens
    assert RunEvidence(()).board_screens() == {}


def test_the_robinhood_scanner_fallback_is_delivered_as_context() -> None:
    """ADR-0041: preview_scan reaches the model as projected, non-citable context."""
    from wheelta_robinhood_agent.agent.result_boundary import CONTEXT_ONLY_GAP

    validator = BoundaryValidator(redactor=Redactor(account_number=None))
    rows = [{"ticker": f"T{i}", "columns": {"Symbol": f"T{i}", "RSI": "40"}} for i in range(3)]
    text = json.dumps({"data": {"result": {"results": rows, "total": 3}}, "guide": "prose"})
    outcome = validator(
        ValidationRequest(
            tool_call_id=CALL,
            server="robinhood",
            tool="preview_scan",
            tier=ToolTier.R,
            effective_input={"filters": []},
            tool_response={"content": [{"type": "text", "text": text}]},
            retrieved_at=RETRIEVED,
        )
    )
    envelope = outcome.envelope
    assert envelope.kind.value == "validated" and CONTEXT_ONLY_GAP in envelope.gaps
    data = envelope.data
    assert isinstance(data, dict) and data["context_only"] is True
    assert "evidence_ref" not in data and "Symbol" not in json.dumps(data)
