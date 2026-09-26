"""PositionBook and position lineages (INTERFACES.md "PositionBook").

A lineage (`position_id`) is stable through rolls. A roll is counted once, when the
replacement order first fills, regardless of price steps or later partial fills. A short
imported without known history has null entry facts and an unknown (None) roll count, never
0. Lineages are never merged by ticker or OCC alone; ambiguity is an explicit gap.
"""

from collections.abc import Iterable
from datetime import date
from typing import Self
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
from wheelta_robinhood_agent.domain.enums import DataQuality, StrategyKind
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


class PositionBookEntry(DomainModel):
    """One active lineage with the entry evidence management needs.

    Rules (INTERFACES.md "PositionBook"):
    - No entry fills (imported without history) means null entry date/credit and a None roll
      count, recorded as a gap.
    - A known `roll_count` equals `count_rolls(roll_events)`; roll events belong to this
      lineage.
    - `entry_weighted_credit` is the quantity-weighted entry fill price in the broker's
      option price units (per-share vs per-contract **unverified** until fixtures exist).
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

    @model_validator(mode="after")
    def _check_entry(self) -> Self:
        require_unique(self.entry_fill_ids, "entry fill id")
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
