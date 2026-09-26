"""AccountSnapshot: the configured Agentic account only (INTERFACES.md "AccountSnapshot").

Broker field mappings and reservation semantics are **unverified** until confirmed with
captured fixtures (CLAUDE.md §9). Every cash field is therefore nullable, and a `None` value
must be explained by a gap naming it. The snapshot is never adjusted speculatively: a state
change requires a new snapshot.
"""

import re
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import model_validator

from wheelta_robinhood_agent.domain.base import (
    Count,
    Dec,
    DomainModel,
    NonEmptyStr,
    NonNegDec,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import DataQuality
from wheelta_robinhood_agent.domain.evidence import Gap

# A redacted account reference may show at most the last four digits (CLAUDE.md §7).
_LONG_DIGIT_RUN = re.compile(r"\d{5,}")


class ReservationKind(StrEnum):
    """What holds a cash/share reservation (INTERFACES.md `reservation_evidence[]`)."""

    POSITION = "position"
    WORKING_ORDER = "working_order"


class ReservationEvidence(DomainModel):
    """One position or working order the snapshot is known to include, counted once.

    `broker_ref` is the broker's opaque order/position ID. `amount_usd` is present only when
    the broker reported the reserved amount; it is never computed here.
    """

    kind: ReservationKind
    broker_ref: NonEmptyStr
    quantity: Count
    amount_usd: NonNegDec | None = None
    source_tool_call_ids: tuple[UUID, ...]

    @model_validator(mode="after")
    def _check_reservation(self) -> Self:
        if not self.source_tool_call_ids:
            raise ValueError("reservation evidence needs a source tool call")
        return self


_NULLABLE_CASH_FIELDS = (
    "account_value_usd",
    "available_settled_cash_usd",
    "csp_reserved_cash_usd",
    "csp_cash_base_usd",
)


class AccountSnapshot(DomainModel):
    """A normalized, immutable view of the configured Agentic account.

    Rules (INTERFACES.md "AccountSnapshot"; VALIDATION.md "CSP accounting"):
    - `available_settled_cash_usd` is already net of broker reservations; it excludes
      unsettled proceeds and margin credit (semantics unverified).
    - `csp_reserved_cash_usd` counts short puts plus unfilled working STO puts once.
    - `csp_cash_base_usd` equals the sum of those two when both are present, else None.
    - Each unavailable cash field is named in `gaps`; `quality=ok` requires no gaps.
    - `account_ref` is redacted (at most the last four digits).
    """

    snapshot_id: UUID
    as_of: UtcDatetime
    retrieved_at: UtcDatetime
    tool_call_ids: tuple[UUID, ...]
    account_ref: NonEmptyStr
    agentic_verified: bool
    account_value_usd: Dec | None
    available_settled_cash_usd: NonNegDec | None
    csp_reserved_cash_usd: NonNegDec | None
    csp_cash_base_usd: NonNegDec | None
    csp_cash_base_evidence_ids: tuple[UUID, ...] = ()
    positions_ref: UUID | None
    open_orders_ref: UUID | None
    tax_lots_ref: UUID | None
    reservation_evidence: tuple[ReservationEvidence, ...] = ()
    quality: DataQuality
    gaps: tuple[Gap, ...] = ()

    @model_validator(mode="after")
    def _check_snapshot(self) -> Self:
        if not self.tool_call_ids:
            raise ValueError("an account snapshot needs source tool calls")
        require_unique(self.tool_call_ids, "tool call id")
        if _LONG_DIGIT_RUN.search(self.account_ref):
            raise ValueError("account_ref must be redacted to at most the last four digits")
        c, r = self.available_settled_cash_usd, self.csp_reserved_cash_usd
        expected = c + r if c is not None and r is not None else None
        if self.csp_cash_base_usd != expected:
            raise ValueError(
                "csp_cash_base_usd must equal available_settled_cash_usd + "
                "csp_reserved_cash_usd when both are present, else None"
            )
        if self.csp_cash_base_usd is not None and not self.csp_cash_base_evidence_ids:
            raise ValueError("csp_cash_base_usd needs input evidence ids")
        gap_fields = {g.field for g in self.gaps}
        for name in _NULLABLE_CASH_FIELDS:
            if getattr(self, name) is None and name not in gap_fields:
                raise ValueError(f"{name} is None but no gap names it")
        if self.quality is DataQuality.OK and self.gaps:
            raise ValueError("a snapshot with gaps cannot have quality 'ok'")
        return self
