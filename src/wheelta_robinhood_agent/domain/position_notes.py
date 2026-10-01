"""Derive per-lineage notes from an assembled RunRecord (ADR-0018).

Pure: the orchestrator appends each result as a `note` position event, and the PositionBook
shows a lineage's notes to every later run until the lineage closes. Only references that
code already resolved to an active lineage produce a note: a decision whose `position_id`
assembly derived from the book, or an unresolved question whose `target_ref` is the
`position_ref` of a book entry. Nothing is inferred from free text.

An OPEN decision targets a candidate, not a lineage: its lineage is created when its order
fills. Its note goes to the lineages that the ledger linked its sell-to-open attempts' broker
orders to as entry fills (ADR-0055), and becomes those lineages' entry note.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from wheelta_robinhood_agent.domain.enums import OrderSide
from wheelta_robinhood_agent.domain.positions import (
    ENTRY_ACTIONS,
    PositionBook,
    PositionNote,
    PositionNoteKind,
)
from wheelta_robinhood_agent.domain.run_record import DecisionRecord, RunRecord


@dataclass(frozen=True, slots=True)
class LineageNote:
    """A note for one lineage and its per-run idempotency key."""

    position_id: UUID
    dedup_key: str
    note: PositionNote


def _entry_order_ids(decision: DecisionRecord) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            attempt.broker_order_id
            for leg in decision.legs
            if leg.side is OrderSide.SELL_TO_OPEN
            for attempt in leg.attempts
            if attempt.broker_order_id is not None
        )
    )


def opened_order_ids(record: RunRecord) -> tuple[str, ...]:
    """Broker order IDs of the sell-to-open attempts of this run's OPEN decisions."""
    return tuple(
        dict.fromkeys(
            order_id
            for decision in record.decisions
            if decision.action in ENTRY_ACTIONS
            for order_id in _entry_order_ids(decision)
        )
    )


def notes_from_run_record(
    record: RunRecord,
    book: PositionBook,
    entry_lineages: Mapping[str, UUID] | None = None,
) -> tuple[LineageNote, ...]:
    """Notes for the lineages this run's decisions and open questions targeted.

    `entry_lineages` maps a broker order ID (from `opened_order_ids`) to the lineage the
    ledger linked its entry fills to. Dedup keys are stable for a given record
    (`note:<run_id>:<decision_ref>` and `note:<run_id>:question:<index>`), so re-finalizing
    a slot does not duplicate notes.
    """
    active = {e.position_id for e in book.entries}
    by_ref = {e.position_ref: e.position_id for e in book.entries}
    lineages = entry_lineages or {}
    out: list[LineageNote] = []
    for decision in record.decisions:
        if decision.action in ENTRY_ACTIONS:
            targets = tuple(
                dict.fromkeys(
                    lineages[order_id]
                    for order_id in _entry_order_ids(decision)
                    if order_id in lineages
                )
            )
        elif decision.position_id is not None and decision.position_id in active:
            targets = (decision.position_id,)
        else:
            continue
        note = PositionNote(
            run_id=record.run_id,
            noted_at=record.terminated_at,
            kind=PositionNoteKind.DECISION,
            action=decision.action,
            decision_ref=decision.decision_ref,
            text=decision.rationale,
            thesis=decision.thesis,
            invalidation_conditions=decision.invalidation_conditions,
        )
        out.extend(
            LineageNote(
                position_id=position_id,
                dedup_key=f"note:{record.run_id}:{decision.decision_ref}",
                note=note,
            )
            for position_id in targets
        )
    for index, question in enumerate(record.unresolved_questions):
        position_id = by_ref.get(question.target_ref) if question.target_ref else None
        if position_id is None:
            continue
        out.append(
            LineageNote(
                position_id=position_id,
                dedup_key=f"note:{record.run_id}:question:{index}",
                note=PositionNote(
                    run_id=record.run_id,
                    noted_at=record.terminated_at,
                    kind=PositionNoteKind.QUESTION,
                    text=question.question,
                ),
            )
        )
    return tuple(out)


__all__ = ["LineageNote", "notes_from_run_record", "opened_order_ids"]
