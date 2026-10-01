"""Position notes (ADR-0018): the model, the display cap, and derivation from a RunRecord."""

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
    StrategyKind,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import Attempt
from wheelta_robinhood_agent.domain.position_notes import notes_from_run_record, opened_order_ids
from wheelta_robinhood_agent.domain.positions import (
    MAX_NOTES_PER_POSITION,
    PositionBook,
    PositionBookEntry,
    PositionInstrument,
    PositionNote,
    PositionNoteKind,
    latest_notes,
)
from wheelta_robinhood_agent.domain.run_record import (
    RUN_RECORD_SCHEMA_VERSION,
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    RunRecord,
    UnresolvedQuestionRecord,
)

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
RUN = UUID(int=1)
PID = UUID(int=7)
OTHER = UUID(int=8)


def _note(**kw: object) -> PositionNote:
    base: dict[str, object] = {
        "run_id": RUN,
        "noted_at": T0,
        "kind": PositionNoteKind.DECISION,
        "action": DecisionAction.HOLD,
        "decision_ref": "decision:0",
        "text": "Still out of the money; earnings are after expiry.",
    }
    base.update(kw)
    return PositionNote.model_validate(base)


def _entry(position_id: UUID = PID, **kw: object) -> PositionBookEntry:
    base: dict[str, object] = {
        "position_id": position_id,
        "position_ref": f"position:{position_id}",
        "underlying": "AAPL",
        "strategy": StrategyKind.CASH_SECURED_PUT,
        "current_instruments": (
            PositionInstrument(
                occ_symbol=OccSymbol.parse("AAPL  261016P00190000"),
                broker_instrument_id="inst-1",
                short_quantity=1,
            ),
        ),
        "entry_fill_ids": (uuid4(),),
        "entry_date": date(2026, 9, 1),
        "entry_weighted_credit": Decimal("1.20"),
        "thesis": "Durable business.",
        "roll_count": 0,
        "history_quality": DataQuality.OK,
    }
    base.update(kw)
    return PositionBookEntry.model_validate(base)


def _decision(ref: str, **kw: object) -> DecisionRecord:
    base: dict[str, object] = {
        "decision_ref": ref,
        "action": DecisionAction.HOLD,
        "priority": 0,
        "target_ref": f"position:{PID}",
        "replacement_ref": None,
        "underlying": "AAPL",
        "position_id": PID,
        "rationale": "Hold: thesis intact.",
        "thesis": None,
    }
    base.update(kw)
    return DecisionRecord.model_validate(base)


def _leg(side: OrderSide, *broker_order_ids: str | None) -> LegRecord:
    attempts = tuple(
        Attempt(
            index=i,
            place_tool_call_id=uuid4(),
            proposal_ref=None,
            requested_quantity=1,
            order_type_raw="limit",
            time_in_force_raw="gfd",
            limit_price=Decimal("1.20"),
            snapshot_ref=None,
            status=AttemptStatus.FILLED if order_id else AttemptStatus.NOT_PLACED,
            broker_order_id=order_id,
            filled_quantity=1 if order_id else None,
        )
        for i, order_id in enumerate(broker_order_ids)
    )
    occ = OccSymbol.parse("MSFT  261016P00400000")
    return LegRecord(
        leg_ref=f"leg:{side.value}:{broker_order_ids[-1]}",
        side=side,
        occ_symbol=occ,
        broker_instrument_id="inst-9",
        right=OptionRight.PUT,
        strike=occ.strike,
        expiration=occ.expiration,
        target_quantity=1,
        attempts=attempts,
    )


def _record(**kw: object) -> RunRecord:
    base: dict[str, object] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "assembler_version": "1",
        "input_hash": "sha256:x",
        "run_id": RUN,
        "environment": AppEnv.STAGING,
        "slot": T0,
        "terminated_at": T0,
        "requested_execution_mode": ExecutionMode.OFF,
        "effective_execution_mode": ExecutionMode.OFF,
        "rules_version": "5",
        "rules_hash": "sha256:r",
        "prompt_id": "wheel_agent.v6",
        "prompt_hash": "sha256:p",
        "model_id": "claude-x",
        "decision_output_status": DecisionOutputStatus.PARSED,
        "summary": "Run.",
    }
    base.update(kw)
    return RunRecord.model_validate(base)


def test_decision_note_needs_action_and_ref() -> None:
    _note()
    with pytest.raises(ValidationError, match="action and decision_ref"):
        _note(action=None)
    with pytest.raises(ValidationError, match="action and decision_ref"):
        _note(decision_ref=None)


def test_question_note_carries_only_text() -> None:
    _note(kind=PositionNoteKind.QUESTION, action=None, decision_ref=None)
    with pytest.raises(ValidationError, match="only its text"):
        _note(kind=PositionNoteKind.QUESTION, decision_ref=None)
    with pytest.raises(ValidationError, match="only its text"):
        _note(kind=PositionNoteKind.QUESTION, action=None, decision_ref=None, thesis="x")


def test_latest_notes_keeps_newest_in_order() -> None:
    notes = [_note(decision_ref=f"decision:{i}") for i in range(5)]
    kept, omitted = latest_notes(notes, 3)
    assert [n.decision_ref for n in kept] == ["decision:2", "decision:3", "decision:4"]
    assert omitted == 2
    assert latest_notes(notes[:2], 3) == (tuple(notes[:2]), 0)
    assert latest_notes([], 3) == ((), 0)
    with pytest.raises(ValueError, match="positive"):
        latest_notes(notes, 0)


def test_entry_caps_shown_notes() -> None:
    notes = tuple(_note() for _ in range(MAX_NOTES_PER_POSITION))
    assert _entry(notes=notes, notes_omitted=3).notes_omitted == 3
    with pytest.raises(ValidationError, match="at most"):
        _entry(notes=(*notes, _note()))
    assert _entry().notes == ()


def test_notes_from_record_covers_decisions_and_questions() -> None:
    book = PositionBook(as_of=T0, entries=(_entry(), _entry(OTHER)))
    record = _record(
        decisions=(
            _decision(
                "decision:0",
                action=DecisionAction.ROLL,
                replacement_ref="candidate:9",
                thesis="Same thesis, later expiry.",
                invalidation_conditions=("Guidance cut.",),
            ),
            _decision(
                "decision:1",
                action=DecisionAction.OPEN_CSP,
                target_ref="candidate:2",
                position_id=None,
                thesis="New.",
            ),
        ),
        unresolved_questions=(
            UnresolvedQuestionRecord(target_ref=f"position:{OTHER}", question="Guidance date?"),
            UnresolvedQuestionRecord(target_ref="candidate:2", question="Liquidity?"),
            UnresolvedQuestionRecord(target_ref=None, question="Macro?"),
        ),
    )
    notes = notes_from_run_record(record, book)
    assert [(n.position_id, n.dedup_key) for n in notes] == [
        (PID, f"note:{RUN}:decision:0"),
        (OTHER, f"note:{RUN}:question:0"),
    ]
    roll, question = notes[0].note, notes[1].note
    assert roll.kind is PositionNoteKind.DECISION
    assert roll.action is DecisionAction.ROLL
    assert roll.text == "Hold: thesis intact."
    assert roll.thesis == "Same thesis, later expiry."
    assert roll.invalidation_conditions == ("Guidance cut.",)
    assert roll.noted_at == T0 and roll.run_id == RUN
    assert question.kind is PositionNoteKind.QUESTION
    assert question.text == "Guidance date?"
    assert notes_from_run_record(record, book) == notes


def test_notes_skip_lineages_not_in_the_book() -> None:
    record = _record(
        decisions=(_decision("decision:0"),),
        unresolved_questions=(
            UnresolvedQuestionRecord(target_ref=f"position:{PID}", question="Still open?"),
        ),
    )
    assert notes_from_run_record(record, PositionBook(as_of=T0, entries=())) == ()


def test_entry_note_must_be_an_open_decision() -> None:
    entry_note = _note(action=DecisionAction.OPEN_CSP, thesis="Durable business.")
    assert _entry(entry_note=entry_note).entry_note == entry_note
    with pytest.raises(ValidationError, match="OPEN decision"):
        _entry(entry_note=_note())


def test_open_decision_notes_the_lineages_its_fills_created() -> None:
    """ADR-0055: an OPEN decision's rationale and thesis reach the lineage it opened."""
    new = UUID(int=9)
    stepped = UUID(int=10)
    open_csp = _decision(
        "decision:0",
        action=DecisionAction.OPEN_CSP,
        target_ref="candidate:1",
        position_id=None,
        rationale="Quality name at a 6% cushion; earnings after expiry.",
        thesis="Cloud demand holds through expiry.",
        invalidation_conditions=("Guidance cut.",),
        legs=(_leg(OrderSide.SELL_TO_OPEN, None, "b-1", "b-2", "b-3"),),
    )
    unfilled = _decision(
        "decision:1",
        action=DecisionAction.OPEN_CC,
        target_ref="candidate:2",
        position_id=None,
        thesis="x",
        legs=(_leg(OrderSide.SELL_TO_OPEN, "b-4"),),
    )
    record = _record(decisions=(open_csp, unfilled))
    assert opened_order_ids(record) == ("b-1", "b-2", "b-3", "b-4")

    # b-1 and b-2 are price steps whose fills built two lineages; b-3 never filled.
    lineages = {"b-1": new, "b-2": stepped}
    notes = notes_from_run_record(record, PositionBook(as_of=T0, entries=()), lineages)
    assert [(n.position_id, n.dedup_key) for n in notes] == [
        (new, f"note:{RUN}:decision:0"),
        (stepped, f"note:{RUN}:decision:0"),
    ]
    note = notes[0].note
    assert note.action is DecisionAction.OPEN_CSP
    assert note.text == "Quality name at a 6% cushion; earnings after expiry."
    assert note.thesis == "Cloud demand holds through expiry."
    assert note.invalidation_conditions == ("Guidance cut.",)
    assert notes_from_run_record(record, PositionBook(as_of=T0, entries=())) == ()


def test_opened_order_ids_ignore_closes_and_buy_legs() -> None:
    roll = _decision(
        "decision:0",
        action=DecisionAction.ROLL,
        replacement_ref="candidate:3",
        legs=(_leg(OrderSide.BUY_TO_CLOSE, "b-5"), _leg(OrderSide.SELL_TO_OPEN, "b-6")),
    )
    assert opened_order_ids(_record(decisions=(roll,))) == ()
