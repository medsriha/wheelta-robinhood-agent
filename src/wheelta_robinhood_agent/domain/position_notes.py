"""Derive per-lineage notes from an assembled RunRecord (ADR-0018).

Pure: the orchestrator appends each result as a `note` position event, and the PositionBook
shows a lineage's notes to every later run until the lineage closes. Only references that
code already resolved to an active lineage produce a note: a decision whose `position_id`
assembly derived from the book, or an unresolved question whose `target_ref` is the
`position_ref` of a book entry. Nothing is inferred from free text.
"""

from dataclasses import dataclass
from uuid import UUID

from wheelta_robinhood_agent.domain.positions import PositionBook, PositionNote, PositionNoteKind
from wheelta_robinhood_agent.domain.run_record import RunRecord


@dataclass(frozen=True, slots=True)
class LineageNote:
    """A note for one lineage and its per-run idempotency key."""

    position_id: UUID
    dedup_key: str
    note: PositionNote


def notes_from_run_record(record: RunRecord, book: PositionBook) -> tuple[LineageNote, ...]:
    """Notes for the active lineages this run's decisions and open questions targeted.

    Dedup keys are stable for a given record (`note:<run_id>:<decision_ref>` and
    `note:<run_id>:question:<index>`), so re-finalizing a slot does not duplicate notes.
    """
    active = {e.position_id for e in book.entries}
    by_ref = {e.position_ref: e.position_id for e in book.entries}
    out: list[LineageNote] = []
    for decision in record.decisions:
        if decision.position_id is None or decision.position_id not in active:
            continue
        out.append(
            LineageNote(
                position_id=decision.position_id,
                dedup_key=f"note:{record.run_id}:{decision.decision_ref}",
                note=PositionNote(
                    run_id=record.run_id,
                    noted_at=record.terminated_at,
                    kind=PositionNoteKind.DECISION,
                    action=decision.action,
                    decision_ref=decision.decision_ref,
                    text=decision.rationale,
                    thesis=decision.thesis,
                    invalidation_conditions=decision.invalidation_conditions,
                ),
            )
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


__all__ = ["LineageNote", "notes_from_run_record"]
