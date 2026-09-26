"""PositionBook and position lineages (INTERFACES.md "PositionBook").

A lineage (`position_id`) is stable through rolls. A roll is counted once, when the
replacement order first fills, regardless of price steps or later partial fills. A short
imported without known history has null entry facts and an unknown (None) roll count, never
0. Lineages are never merged by ticker or OCC alone; ambiguity is an explicit gap.

Notes (ADR-0018) carry the agent's own earlier judgments about an active lineage (decision
rationale, thesis, open questions) into later runs. They are context, never evidence, and
leave the book with the lineage when it closes.
"""

from collections.abc import Iterable, Sequence
from datetime import date
from enum import StrEnum
from typing import Final, Self
from uuid import UUID

from pydantic import model_validator

from wheelta_robinhood_agent.domain.base import (
    Count,
    DomainModel,
    NonEmptyStr,
    PosCount,
    PosDec,
    Ref,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import DataQuality, DecisionAction, StrategyKind
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.options import OccSymbol


class PositionInstrument(DomainModel):
    """A contract currently held short in a lineage (OCC symbol + broker instrument ID)."""

    occ_symbol: OccSymbol
    broker_instrument_id: NonEmptyStr
    short_quantity: PosCount


class RollEvent(DomainModel):
    """Links close fills and the replacement's first fill to one lineage.

    Counted once, at `replacement_first_fill_id`; `replacement_order_id` identifies the
    replacement so that later partial fills or price steps are not counted again. A partial
    close without a replacement fill is not a roll event.
    """

    roll_event_id: UUID
    position_id: UUID
    close_fill_ids: tuple[UUID, ...]
    replacement_order_id: UUID
    replacement_first_fill_id: UUID
    replacement_instrument: PositionInstrument
    counted_at: UtcDatetime

    @model_validator(mode="after")
    def _check_roll(self) -> Self:
        if not self.close_fill_ids:
            raise ValueError("a roll event needs at least one close fill")
        require_unique(self.close_fill_ids, "close fill id")
        if self.replacement_first_fill_id in self.close_fill_ids:
            raise ValueError("the replacement fill cannot also be a close fill")
        return self


def count_rolls(events: Iterable[RollEvent]) -> int:
    """Number of rolls: distinct replacement orders that have first-filled.

    Duplicate observations of the same replacement (price steps, further partial fills,
    re-reads) count once (INTERFACES.md "PositionBook").
    """
    return len({event.replacement_order_id for event in events})


# ADR-0018: newest notes shown per lineage. A prompt-size bound, not a trading rule.
MAX_NOTES_PER_POSITION: Final = 24


class PositionNoteKind(StrEnum):
    DECISION = "decision"
    QUESTION = "question"


class PositionNote(DomainModel):
    """One earlier-run judgment about a lineage, kept verbatim (ADR-0018).

    A `decision` note is a management decision (HOLD/CLOSE/ROLL) that targeted the lineage:
    its rationale, plus the thesis and invalidation conditions a roll replacement supplied.
    A `question` note is an unresolved research question that targeted the lineage. The text
    is the agent's judgment from `run_id`; it supplies no financial fact to a later run.
    """

    run_id: UUID
    noted_at: UtcDatetime
    kind: PositionNoteKind
    action: DecisionAction | None = None
    decision_ref: Ref | None = None
    text: NonEmptyStr
    thesis: NonEmptyStr | None = None
    invalidation_conditions: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _check_note(self) -> Self:
        if self.kind is PositionNoteKind.DECISION:
            if self.action is None or self.decision_ref is None:
                raise ValueError("a decision note names its action and decision_ref")
        elif (
            self.action is not None
            or self.decision_ref is not None
            or self.thesis is not None
            or self.invalidation_conditions
        ):
            raise ValueError("a question note carries only its text")
        return self


def latest_notes(
    notes: Sequence[PositionNote], limit: int = MAX_NOTES_PER_POSITION
) -> tuple[tuple[PositionNote, ...], int]:
    """The newest `limit` notes in recorded order, and how many older ones were left out."""
    if limit < 1:
        raise ValueError("limit must be positive")
    kept = tuple(notes[-limit:])
    return kept, len(notes) - len(kept)


class PositionBookEntry(DomainModel):
    """One active lineage with the entry evidence management needs.

    Rules (INTERFACES.md "PositionBook"):
    - No entry fills (imported without history) means null entry date/credit and a None roll
      count, recorded as a gap.
    - A known `roll_count` equals `count_rolls(roll_events)`; roll events belong to this
      lineage.
    - `entry_weighted_credit` is the quantity-weighted entry fill price in the broker's
      option price units (per-share vs per-contract **unverified** until fixtures exist).
    - `notes` are the newest earlier-run notes, oldest first; `notes_omitted` counts older
      ones not shown (ADR-0018).
    """

    position_id: UUID
    position_ref: Ref
    underlying: NonEmptyStr
    strategy: StrategyKind
    current_instruments: tuple[PositionInstrument, ...]
    entry_fill_ids: tuple[UUID, ...] = ()
    entry_date: date | None
    entry_weighted_credit: PosDec | None
    thesis: NonEmptyStr | None
    invalidation_conditions: tuple[NonEmptyStr, ...] = ()
    entry_event_refs: tuple[Ref, ...] = ()
    share_lot_refs: tuple[NonEmptyStr, ...] = ()
    roll_events: tuple[RollEvent, ...] = ()
    roll_count: Count | None
    history_quality: DataQuality
    gaps: tuple[Gap, ...] = ()
    notes: tuple[PositionNote, ...] = ()
    notes_omitted: Count = 0

    @model_validator(mode="after")
    def _check_entry(self) -> Self:
        require_unique(self.entry_fill_ids, "entry fill id")
        if len(self.notes) > MAX_NOTES_PER_POSITION:
            raise ValueError(f"at most {MAX_NOTES_PER_POSITION} notes per lineage")
        require_unique(tuple(e.roll_event_id for e in self.roll_events), "roll event id")
        if any(e.position_id != self.position_id for e in self.roll_events):
            raise ValueError("roll events must belong to this lineage")
        if not self.entry_fill_ids:
            if self.entry_date is not None or self.entry_weighted_credit is not None:
                raise ValueError("entry facts require entry fill evidence")
            if self.roll_count is not None:
                raise ValueError("a lineage without entry history has an unknown roll count")
        if self.roll_count is None:
            if not self.gaps:
                raise ValueError("an unknown roll count must be explained by a gap")
        elif self.roll_count != count_rolls(self.roll_events):
            raise ValueError("roll_count must equal the distinct replacement first fills")
        if self.history_quality is DataQuality.OK and self.gaps:
            raise ValueError("history with gaps cannot have quality 'ok'")
        return self


class PositionBook(DomainModel):
    """All active lineages plus unresolved owned orders (INTERFACES.md "PositionBook").

    Bounded by active positions, not by the last N runs.
    """

    as_of: UtcDatetime
    entries: tuple[PositionBookEntry, ...]
    unresolved_owned_order_ids: tuple[UUID, ...] = ()
    gaps: tuple[Gap, ...] = ()

    @model_validator(mode="after")
    def _check_book(self) -> Self:
        require_unique(tuple(e.position_id for e in self.entries), "position id")
        require_unique(tuple(e.position_ref for e in self.entries), "position ref")
        require_unique(self.unresolved_owned_order_ids, "order id")
        return self
