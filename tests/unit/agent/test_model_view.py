"""The model view of validated envelopes (ADR-0037): smaller, deterministic, no usable fact lost."""

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, ResultEnvelope, mcp_tool_output
from wheelta_robinhood_agent.agent.model_view import (
    EVIDENCE_VIEW_KEY,
    EVIDENCE_VIEW_VERSION,
    is_model_view,
    model_view,
)
from wheelta_robinhood_agent.agent.result_boundary import (
    CandidateEvidence,
    MappedEvidence,
    evidence_ref_for,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.run_loader import _delivered_envelopes
from wheelta_robinhood_agent.domain.enums import CandidateOrigin, OptionRight
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument, UnderlyingQuote
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger.evidence import ResultKind, StoredResult

CALL = uuid.UUID("0190a0a0-0000-7000-8000-000000000001")
OTHER_CALL = uuid.UUID("0190a0a0-0000-7000-8000-000000000002")
AT = datetime(2026, 9, 28, 18, 0, 5, tzinfo=UTC)
AT_JSON = "2026-09-28T18:00:05Z"
RUN = uuid.UUID("0190a0a0-0000-7000-8000-0000000000ff")


def _envelope(evidence: MappedEvidence, tool: str = "get_option_instruments") -> dict[str, Any]:
    return ResultEnvelope(
        tool_call_id=CALL,
        server="robinhood",
        tool=tool,
        kind=EnvelopeKind.VALIDATED,
        data={"evidence_ref": evidence_ref_for(CALL), "evidence": evidence.model_dump(mode="json")},
        retrieved_at=AT,
    ).model_dump(mode="json")


def _instrument(i: int, strike: str, *, expiration: date = date(2026, 10, 30)) -> OptionInstrument:
    return OptionInstrument(
        evidence_id=uuid.UUID(int=100 + i),
        as_of=AT,
        source_tool_call_ids=(CALL,),
        occ_symbol=OccSymbol(
            root="CCL", expiration=expiration, right=OptionRight.PUT, strike=Decimal(strike)
        ),
        broker_instrument_id=f"inst-{i}",
        underlying="CCL",
        multiplier=100,
    )


def _quote(i: int, bid: str, ask: str, as_of: datetime = AT) -> Quote:
    return Quote(
        quote_id=uuid.UUID(int=200 + i),
        broker_instrument_id=f"inst-{i}",
        bid=Decimal(bid),
        ask=Decimal(ask),
        mark=None,
        delta=Decimal("-0.037346"),
        open_interest=3451,
        as_of=as_of,
        source_tool_call_ids=(CALL,),
    )


def _rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild each row as a flat object: `common` plus the row's own columns."""
    return [
        {**table["common"], **dict(zip(table["columns"], row, strict=True))}
        for row in table["rows"]
    ]


def _size(envelope: Any) -> int:
    return len(mcp_tool_output(envelope)[0]["text"])


def test_instruments_become_a_table_with_shared_values_in_common() -> None:
    envelope = _envelope(
        MappedEvidence(instruments=(_instrument(1, "12.0000"), _instrument(2, "12.5000")))
    )
    view = model_view(envelope)
    data = view["data"]
    assert data[EVIDENCE_VIEW_KEY] == EVIDENCE_VIEW_VERSION
    assert data["evidence_ref"] == evidence_ref_for(CALL)
    assert set(data["evidence"]) == {"instruments"}  # empty categories are omitted
    table = data["evidence"]["instruments"]
    assert table["common"] == {
        "as_of": AT_JSON,
        "multiplier": 100,
        "occ_symbol.expiration": "2026-10-30",
        "occ_symbol.right": "put",
        "occ_symbol.root": "CCL",
        "tick_increment": None,
        "tick_schedule": None,
        "underlying": "CCL",
    }
    assert table["columns"] == ["broker_instrument_id", "occ_symbol.strike"]
    assert table["rows"] == [["inst-1", "12"], ["inst-2", "12.5"]]
    # The envelope around the evidence is unchanged.
    assert {k: v for k, v in view.items() if k != "data"} == {
        k: v for k, v in envelope.items() if k != "data"
    }


def test_internal_ids_and_own_call_provenance_are_dropped() -> None:
    envelope = _envelope(MappedEvidence(instruments=(_instrument(1, "12"),)))
    text = mcp_tool_output(model_view(envelope))[0]["text"]
    assert str(uuid.UUID(int=101)) not in text  # evidence_id
    assert "source_tool_call_ids" not in text
    assert "inst-1" in text  # the broker instrument ID, which order legs take


def test_provenance_naming_another_call_is_kept() -> None:
    item = _instrument(1, "12").model_copy(update={"source_tool_call_ids": (CALL, OTHER_CALL)})
    view = model_view(_envelope(MappedEvidence(instruments=(item,))))
    (row,) = view["data"]["evidence"]["instruments"]
    assert row["source_tool_call_ids"] == [str(CALL), str(OTHER_CALL)]


def test_per_row_values_stay_in_rows_and_decimals_keep_their_value() -> None:
    later = AT + timedelta(seconds=2)
    envelope = _envelope(
        MappedEvidence(
            option_quotes=(_quote(1, "0.020000", "0.040000"), _quote(2, "0.03", "0.05", later))
        ),
        tool="get_option_quotes",
    )
    table = model_view(envelope)["data"]["evidence"]["option_quotes"]
    assert "as_of" in table["columns"]  # differs per row, so it is not common
    assert table["common"]["open_interest"] == 3451 and table["common"]["mark"] is None
    rows = _rows(table)
    originals = envelope["data"]["evidence"]["option_quotes"]
    for row, original in zip(rows, originals, strict=True):
        for key in ("bid", "ask", "delta"):
            assert Decimal(row[key]) == Decimal(original[key])
    assert rows[0]["bid"] == "0.02" and rows[0]["ask"] == "0.04"


def test_candidate_refs_survive_in_the_view() -> None:
    instrument = _instrument(1, "12")
    candidates = tuple(
        CandidateEvidence(
            candidate_ref=f"candidate:{n}",
            origin=CandidateOrigin.ROBINHOOD,
            underlying="CCL",
            instrument_evidence_id=instrument.evidence_id,
            broker_instrument_id="inst-1",
            occ_symbol=instrument.occ_symbol,
        )
        for n in ("a", "b")
    )
    view = model_view(_envelope(MappedEvidence(instruments=(instrument,), candidates=candidates)))
    table = view["data"]["evidence"]["candidates"]
    assert table["columns"][0] == "candidate_ref"
    assert [r["candidate_ref"] for r in _rows(table)] == ["candidate:a", "candidate:b"]
    assert "instrument_evidence_id" not in json.dumps(view)


def test_single_item_categories_stay_objects() -> None:
    quote = UnderlyingQuote(
        evidence_id=uuid.UUID(int=9),
        as_of=AT,
        source_tool_call_ids=(CALL,),
        symbol="DVN",
        price=Decimal("46.715000"),
    )
    view = model_view(_envelope(MappedEvidence(underlying_quotes=(quote,)), "get_equity_quotes"))
    assert view["data"]["evidence"]["underlying_quotes"] == [
        {"as_of": AT_JSON, "price": "46.715", "symbol": "DVN"}
    ]


def test_views_are_marked_not_typed_evidence_and_stable() -> None:
    envelope = _envelope(MappedEvidence(instruments=(_instrument(1, "12"), _instrument(2, "13"))))
    view = model_view(envelope)
    assert is_model_view(view) and not is_model_view(envelope)
    assert mapped_evidence_of(view) is None
    assert mapped_evidence_of(envelope) is not None  # the stored envelope is still evidence
    assert model_view(view) == view
    assert model_view(json.loads(json.dumps(envelope))) == view  # ledger round trip
    assert _size(view) < _size(envelope)


def test_envelopes_without_mapped_evidence_pass_through() -> None:
    error = ResultEnvelope(
        tool_call_id=CALL,
        server="robinhood",
        tool="t",
        kind=EnvelopeKind.ERROR,
        gaps=("x",),
        retrieved_at=AT,
    ).model_dump(mode="json")
    context = {
        **error,
        "kind": "validated",
        "data": {"context_only": True, "payload": {"a": "1.50"}},
    }
    facts = {**error, "kind": "validated", "data": {"facts_ref": "facts:1", "status": "ok"}}
    for envelope in (error, context, facts):
        assert model_view(envelope) == envelope


def test_mixed_shapes_are_not_tabulated() -> None:
    envelope = _envelope(MappedEvidence(instruments=(_instrument(1, "12"), _instrument(2, "13"))))
    rows = envelope["data"]["evidence"]["instruments"]
    rows[1]["extra"] = "x"
    view = model_view(envelope)
    assert isinstance(view["data"]["evidence"]["instruments"], list)


@given(
    st.lists(
        st.tuples(
            st.decimals(min_value=Decimal("0.5"), max_value=Decimal("999"), places=2),
            st.sampled_from([date(2026, 10, 16), date(2026, 10, 30), date(2026, 11, 20)]),
        ),
        min_size=2,
        max_size=40,
    )
)
def test_every_row_rebuilds_to_its_instrument(specs: list[tuple[Decimal, date]]) -> None:
    items = tuple(
        _instrument(i, str(strike), expiration=expiration)
        for i, (strike, expiration) in enumerate(specs)
    )
    view = model_view(_envelope(MappedEvidence(instruments=items)))
    rebuilt = _rows(view["data"]["evidence"]["instruments"])
    assert len(rebuilt) == len(items)
    for row, item in zip(rebuilt, items, strict=True):
        assert row["broker_instrument_id"] == item.broker_instrument_id
        assert Decimal(row["occ_symbol.strike"]) == item.occ_symbol.strike
        assert row["occ_symbol.expiration"] == item.occ_symbol.expiration.isoformat()
        assert row["underlying"] == "CCL" and row["multiplier"] == 100
        assert row["as_of"] == AT_JSON


# ---- run_loader: a delivered view resolves to the envelope it was built from ------------------


def _stored(kind: ResultKind, payload: Any) -> StoredResult:
    return StoredResult(
        record_id=uuid.uuid4(),
        run_id=RUN,
        tool_call_id=CALL,
        kind=kind,
        payload=payload,
        corrects_id=None,
        recorded_at=AT,
    )


def _delivered(envelope: Any) -> StoredResult:
    return _stored(ResultKind.DELIVERED, {"replaced": True, "tool_output": envelope})


def _loaded(monkeypatch: pytest.MonkeyPatch, *rows: StoredResult) -> list[Any]:
    monkeypatch.setattr(ledger_evidence, "results_for_run", lambda conn, run_id: rows)
    return list(_delivered_envelopes(cast(Any, None), RUN))


def test_a_delivered_view_resolves_to_its_validated_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _envelope(MappedEvidence(instruments=(_instrument(1, "12"), _instrument(2, "13"))))
    stored = json.loads(json.dumps(full))  # as read back from jsonb
    loaded = _loaded(
        monkeypatch, _stored(ResultKind.VALIDATED, stored), _delivered(model_view(full))
    )
    assert loaded == [stored]
    assert mapped_evidence_of(loaded[0]) is not None


def test_a_view_that_does_not_match_its_envelope_counts_as_undelivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _envelope(MappedEvidence(instruments=(_instrument(1, "12"), _instrument(2, "13"))))
    tampered = model_view(full)
    tampered["data"]["evidence"]["instruments"]["rows"][0][0] = "inst-other"
    assert _loaded(monkeypatch, _stored(ResultKind.VALIDATED, full), _delivered(tampered)) == []
    assert _loaded(monkeypatch, _delivered(model_view(full))) == []  # no validated envelope


def test_full_delivered_envelopes_from_earlier_runs_still_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _envelope(MappedEvidence(instruments=(_instrument(1, "12"),)))
    assert _loaded(monkeypatch, _delivered(full)) == [full]
