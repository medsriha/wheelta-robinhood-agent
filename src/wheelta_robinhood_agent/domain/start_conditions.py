"""Start conditions of the two agents of a due tick (ADR-0057). Pure: no I/O, no clock.

Trusted code reads live Robinhood state before the model connects (`agent/start_probe.py`)
and these functions decide whether the session starts:

- **Buy-to-Close agent:** at least one short option position is held. A complete
  options-positions read with no short row means there is nothing to close or roll.
- **Sell Options agent:** settled cash (`get_portfolio` `cash`, ADR-0031) of at least
  `sessions.sell_min_settled_cash_usd`, or 100 shares of one symbol not already covered by
  short options on that symbol. `get_option_positions` rows carry no call/put, so every
  short row on a symbol is counted against its shares: this can only under-count coverable
  shares, never allow a covered call the shares cannot back.

A fact that was not read, or a read that was not complete, is `None`. A condition that
cannot be shown to be met or not met is UNAVAILABLE, never a guess: the run fails closed
(it is not reported as a skip).

UNCHECKED is only for a proposal-only dry run with Robinhood served directly (ADR-0019, order
venue `none`): no trusted channel exists to read through, and no order tool is exposed, so the
session starts with the condition recorded as not checked (`unchecked_start`).
"""

from decimal import Decimal
from enum import StrEnum
from typing import Final

from wheelta_robinhood_agent.domain.base import Count, DomainModel, NonEmptyStr, PosCount
from wheelta_robinhood_agent.domain.enums import AgentRole

# One covered call needs this many shares not otherwise reserved.
SHARES_PER_CONTRACT: Final = 100


class StartOutcome(StrEnum):
    MET = "met"
    NOT_MET = "not_met"
    UNAVAILABLE = "unavailable"
    UNCHECKED = "unchecked"  # proposal-only dry run with Robinhood served directly (ADR-0019)


# Outcomes that start the session.
STARTS_SESSION: Final = frozenset({StartOutcome.MET, StartOutcome.UNCHECKED})


class ShortRow(DomainModel):
    """One short option position row: underlying, contracts, and the contract multiplier."""

    underlying: NonEmptyStr
    short_quantity: PosCount
    multiplier: PosCount


class Shares(DomainModel):
    symbol: NonEmptyStr
    quantity: Count


class StartCondition(DomainModel):
    """The decision and the facts it rests on (recorded as run metadata)."""

    role: AgentRole
    outcome: StartOutcome
    reason: NonEmptyStr
    short_positions: int | None = None
    settled_cash_usd: Decimal | None = None
    min_settled_cash_usd: Decimal | None = None
    coverable_symbols: tuple[NonEmptyStr, ...] | None = None


def evaluate_close_start(short_rows: tuple[ShortRow, ...] | None) -> StartCondition:
    """ADR-0057: the Buy-to-Close agent starts when any short option position is held.
    `None` means no complete options-positions read."""
    if short_rows is None:
        return StartCondition(
            role=AgentRole.CLOSE,
            outcome=StartOutcome.UNAVAILABLE,
            reason="no complete option positions read",
        )
    if not short_rows:
        return StartCondition(
            role=AgentRole.CLOSE,
            outcome=StartOutcome.NOT_MET,
            reason="no open short option positions",
            short_positions=0,
        )
    return StartCondition(
        role=AgentRole.CLOSE,
        outcome=StartOutcome.MET,
        reason="open short option positions held",
        short_positions=len(short_rows),
    )


def unchecked_start(role: AgentRole) -> StartCondition:
    """The condition of a proposal-only dry run with no trusted Robinhood channel."""
    return StartCondition(
        role=role,
        outcome=StartOutcome.UNCHECKED,
        reason="not checked: Robinhood is served directly in this proposal-only dry run",
    )


def coverable_symbols(
    shares: tuple[Shares, ...], short_rows: tuple[ShortRow, ...]
) -> tuple[str, ...]:
    """Symbols with at least 100 shares beyond those every short row on the symbol could
    reserve (module docstring: each short row counts as a call)."""
    reserved: dict[str, int] = {}
    for row in short_rows:
        reserved[row.underlying] = (
            reserved.get(row.underlying, 0) + row.short_quantity * row.multiplier
        )
    return tuple(
        sorted(
            h.symbol
            for h in shares
            if h.quantity - reserved.get(h.symbol, 0) >= SHARES_PER_CONTRACT
        )
    )


def evaluate_sell_start(
    *,
    settled_cash_usd: Decimal | None,
    min_settled_cash_usd: Decimal,
    shares: tuple[Shares, ...] | None,
    short_rows: tuple[ShortRow, ...] | None,
) -> StartCondition:
    """ADR-0057: the Sell Options agent starts when settled cash covers the owner's minimum,
    or when some symbol has 100 shares not covered by short options. Cash alone decides when
    it suffices; otherwise both positions reads must be complete."""

    def result(
        outcome: StartOutcome, reason: str, symbols: tuple[str, ...] | None = None
    ) -> StartCondition:
        return StartCondition(
            role=AgentRole.SELL,
            outcome=outcome,
            reason=reason,
            settled_cash_usd=settled_cash_usd,
            min_settled_cash_usd=min_settled_cash_usd,
            short_positions=len(short_rows) if short_rows is not None else None,
            coverable_symbols=symbols,
        )

    if settled_cash_usd is not None and settled_cash_usd >= min_settled_cash_usd:
        return result(StartOutcome.MET, "settled cash meets the minimum")
    if shares is None or short_rows is None:
        return result(
            StartOutcome.UNAVAILABLE,
            "settled cash below the minimum or unknown, and no complete positions read",
        )
    symbols = coverable_symbols(shares, short_rows)
    if symbols:
        return result(StartOutcome.MET, "uncovered shares can back a covered call", symbols)
    if settled_cash_usd is None:
        return result(
            StartOutcome.UNAVAILABLE, "settled cash unknown and no uncovered 100-share lot", ()
        )
    return result(
        StartOutcome.NOT_MET, "settled cash below the minimum and no uncovered 100-share lot", ()
    )


__all__ = [
    "SHARES_PER_CONTRACT",
    "STARTS_SESSION",
    "Shares",
    "ShortRow",
    "StartCondition",
    "StartOutcome",
    "coverable_symbols",
    "evaluate_close_start",
    "evaluate_sell_start",
    "unchecked_start",
]
