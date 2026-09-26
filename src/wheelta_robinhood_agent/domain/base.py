"""Shared building blocks for domain models (CLAUDE.md §4).

Every internal model is frozen and forbids extra fields. Money is `Decimal` (floats are
rejected, never coerced), timestamps are timezone-aware and normalized to UTC, collections
are tuples so a frozen model is deeply immutable.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    StringConstraints,
)


class DomainModel(BaseModel):
    """Base for every internal model: `extra="forbid"`, `frozen=True` (CLAUDE.md §4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _reject_float(value: object) -> object:
    """Money is never a binary float (CLAUDE.md §4). Decimal, int, and str are accepted."""
    if isinstance(value, float | bool):
        raise ValueError("monetary/decimal values must not be float or bool; use Decimal or str")
    return value


def _finite(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("decimal must be finite")
    return value


def _to_utc(value: datetime) -> datetime:
    """Normalize an aware datetime to UTC (CLAUDE.md §4: store UTC)."""
    return value.astimezone(UTC)


Dec = Annotated[Decimal, BeforeValidator(_reject_float), AfterValidator(_finite)]
"""A finite Decimal that rejects float input."""

NonNegDec = Annotated[Dec, Field(ge=0)]
"""A finite, non-negative Decimal (balances, reservations, quantities of money)."""

PosDec = Annotated[Dec, Field(gt=0)]
"""A finite, strictly positive Decimal (prices, strikes, ticks)."""

UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""A timezone-aware datetime, normalized to UTC. Naive datetimes are rejected."""

NonEmptyStr = Annotated[StrictStr, StringConstraints(min_length=1)]
"""A non-empty string (no coercion from other types)."""

Ref = NonEmptyStr
"""A code-issued, run-local reference string (e.g. `candidate:…`, `facts:…`)."""

Count = Annotated[int, Field(strict=True, ge=0)]
"""A non-negative integer count (contracts, shares). bool and float are rejected."""

PosCount = Annotated[int, Field(strict=True, gt=0)]
"""A strictly positive integer count."""


def require_unique(values: tuple[object, ...], what: str) -> None:
    """Raise ValueError if `values` contains duplicates (used for id/ref collections)."""
    seen: set[object] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {what}: {value!r}")
        seen.add(value)
