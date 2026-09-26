"""Position notes (ADR-0018): the model, the display cap, and derivation from a RunRecord."""

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    StrategyKind,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.position_notes import notes_from_run_record
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
