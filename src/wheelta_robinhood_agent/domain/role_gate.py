"""Pure check of which order sides each agent role may place (ADR-0057).

`check_role(role, legs, quantity, run)` runs before the ADR-0051 concurrency check and the
ADR-0048 leg checks on every `place_option_order`:

- **Sell Options agent:** sell-to-open legs only. A buy-to-close belongs to the Buy-to-Close
  agent, which runs first in the same tick.
- **Buy-to-Close agent:** buy-to-close legs, and a sell-to-open only as a roll's replacement
  (`roll.sequencing`): its underlying root and right must match buy-to-close fills of this
  run, and its quantity must not exceed those filled contracts less the sell-to-open
  contracts this run already placed on the same root and right. A replacement whose contract
  is unknown (no validated instrument), or a quantity that cannot be read, is denied, and so
  is any further replacement once a sell-to-open of this run has an unknown contract (its
  capacity use cannot be attributed).
- The legacy single agent (WHEEL) is not limited here.

Nothing is matched by price or time: only the recorded place arguments, the run's recorded
fills, and validated instrument identities.
"""

from wheelta_robinhood_agent.domain.base import Count, DomainModel, NonEmptyStr
from wheelta_robinhood_agent.domain.enums import AgentRole, OptionRight, OrderSide
from wheelta_robinhood_agent.domain.options import OccSymbol

ROLE_DENIAL_PREFIX = "Agent role check failed (ADR-0057); the order was NOT placed. "


class RoleLeg(DomainModel):
    """One leg of the order being placed. `occ_symbol` is known for opening legs whose
    instrument this run validated."""

    option_id: NonEmptyStr
    closing: bool  # buy to close
    opening: bool  # sell to open
    occ_symbol: OccSymbol | None = None


class RunFill(DomainModel):
    """One order this run placed: its side, contract (if known), and the contracts that
    count (filled contracts for a buy-to-close, the order quantity for a sell-to-open)."""

    side: OrderSide
    occ_symbol: OccSymbol | None
    quantity: Count


def _key(symbol: OccSymbol) -> tuple[str, OptionRight]:
    return symbol.root, symbol.right


def roll_capacity(run: tuple[RunFill, ...], root: str, right: OptionRight) -> int:
    """Contracts the Buy-to-Close agent may still open on (root, right) as replacements."""
    closed = sum(
        f.quantity
        for f in run
        if f.side is OrderSide.BUY_TO_CLOSE
        and f.occ_symbol is not None
        and _key(f.occ_symbol) == (root, right)
    )
    opened = sum(
        f.quantity
        for f in run
        if f.side is OrderSide.SELL_TO_OPEN
        and f.occ_symbol is not None
        and _key(f.occ_symbol) == (root, right)
    )
    return closed - opened


def check_role(
    role: AgentRole, legs: tuple[RoleLeg, ...], *, quantity: int | None, run: tuple[RunFill, ...]
) -> str | None:
    """A denial reason, or None when the role may place these legs (module docstring)."""
    if role is AgentRole.WHEEL:
        return None
    if role is AgentRole.SELL:
        if any(leg.closing for leg in legs):
            return ROLE_DENIAL_PREFIX + (
                "The Sell Options agent only sells to open; buy-to-close orders belong to the "
                "Buy-to-Close agent, which runs before you."
            )
        return None
    for leg in legs:
        if not leg.opening:
            continue
        if any(f.side is OrderSide.SELL_TO_OPEN and f.occ_symbol is None for f in run):
            return ROLE_DENIAL_PREFIX + (
                "An earlier sell-to-open of this run has no known contract, so the roll "
                "capacity it used cannot be attributed; no further replacement is allowed."
            )
        if leg.occ_symbol is None:
            return ROLE_DENIAL_PREFIX + (
                f"Sell-to-open {leg.option_id}: the Buy-to-Close agent opens only a roll's "
                "replacement, and this contract has no validated instrument in this run."
            )
        if quantity is None:
            return ROLE_DENIAL_PREFIX + "The order quantity must be a positive integer."
        root, right = _key(leg.occ_symbol)
        capacity = roll_capacity(run, root, right)
        if quantity > capacity:
            return ROLE_DENIAL_PREFIX + (
                f"Sell-to-open {leg.option_id} ({root} {right.value}, {quantity} contract(s)): "
                "the Buy-to-Close agent opens only a roll's replacement, after the close fills. "
                f"This run's filled buy-to-close contracts on {root} {right.value}, less "
                f"replacements already placed, allow {max(capacity, 0)}. New positions are "
                "the Sell Options agent's."
            )
    return None


__all__ = ["ROLE_DENIAL_PREFIX", "RoleLeg", "RunFill", "check_role", "roll_capacity"]
