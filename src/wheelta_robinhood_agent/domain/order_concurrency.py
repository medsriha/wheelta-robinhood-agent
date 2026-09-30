"""Pure check of when a placement may join orders already working (ADR-0051).

`check_concurrency` decides whether a new `place_option_order` may be sent while other
orders this run placed are unresolved. Only buy-to-close orders on different contracts may
work at the same time, and only when this run's fresh, Agentic-verified AccountSnapshot shows
settled cash for all of their debits at once:

1. **One placement in flight.** Another `place_option_order` of this run with no outcome yet
   denies this one: placements are sent one at a time, so each check sees the one before it.
2. **Nothing working.** With no unresolved order of this run, the placement passes. A single
   order is exactly the sequential procedure of ADR-0010 (no added check).
3. **One order per contract.** A leg on a contract with an unresolved order of this run is
   denied (orders.working); confirm or cancel that order first.
4. **Closes only.** Otherwise every leg of the new order and every unresolved order must be
   buy-to-close. A sell-to-open (a new open or a roll's replacement) waits until nothing else
   is working.
5. **Funds.** The sum of `limit price x verified multiplier x remaining quantity` over the
   unresolved closes and the new one must not exceed `available_settled_cash_usd`. Whether
   the broker already nets working buy orders out of that field is unverified, so they are
   counted again: this can deny a close that would have fit, never allow one that does not.
   A missing price, quantity, multiplier, or cash value, a stale snapshot, or one that is not
   Agentic-verified denies the concurrent close (work it after the others instead).

Nothing is estimated; every value comes from the recorded place arguments and this run's
validated evidence.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import DomainModel, NonEmptyStr, PosCount, PosDec
from wheelta_robinhood_agent.domain.facts_rules import CountSetting, FactsRuleMarker
from wheelta_robinhood_agent.domain.sanity import is_fresh, require_aware

CONCURRENCY_DENIAL_PREFIX = "Order concurrency check failed (ADR-0051); the order was NOT placed. "
_R_ACCOUNT_AGE = "data_quality.freshness.account_state_max_age_seconds"


class PlacementLeg(DomainModel):
    option_id: NonEmptyStr
    closing: bool  # buy to close


class NewPlacement(DomainModel):
    """The `place_option_order` being checked: its legs, quantity, and limit price."""

    legs: tuple[PlacementLeg, ...]
    quantity: PosCount | None
    limit_price: PosDec | None


class WorkingPlacement(DomainModel):
    """An order this run placed that the ledger projects as unresolved."""

    label: NonEmptyStr  # broker order ID, or the place call when no broker order is known
    option_id: NonEmptyStr | None
    closing: bool
    remaining_quantity: int | None
    limit_price: PosDec | None


def _debit(
    price: Decimal | None, multiplier: int | None, quantity: int | None, what: str
) -> tuple[Decimal | None, str | None]:
    if price is None:
        return None, f"{what}: limit price unknown"
    if multiplier is None:
        return None, f"{what}: contract multiplier not verified in this run"
    if quantity is None or quantity < 0:
        return None, f"{what}: remaining quantity unknown"
    return price * multiplier * quantity, None


def _cash(
    snapshot: AccountSnapshot | None, as_of: datetime, max_age: CountSetting
) -> tuple[Decimal | None, str | None]:
    if snapshot is None:
        return None, "no validated account snapshot in this run"
    if not snapshot.agentic_verified:
        return None, "the account snapshot is not Agentic-verified"
    if isinstance(max_age, FactsRuleMarker):
        if max_age is not FactsRuleMarker.NONE:
            return None, f"{_R_ACCOUNT_AGE} is not set"
    elif not is_fresh(snapshot.as_of, as_of, int(max_age)):
        return None, f"the account snapshot is older than {_R_ACCOUNT_AGE}; re-read it"
    if snapshot.available_settled_cash_usd is None:
        return None, "available settled cash is not reported"
    return snapshot.available_settled_cash_usd, None


def check_concurrency(
    new: NewPlacement,
    working: Sequence[WorkingPlacement],
    *,
    other_placements_in_flight: int,
    multipliers: Mapping[str, int | None],
    account: AccountSnapshot | None,
    account_max_age: CountSetting,
    as_of: datetime,
) -> str | None:
    """None when the placement may be sent; otherwise the denial the agent receives."""
    require_aware(as_of, "as_of")
    if other_placements_in_flight > 0:
        return CONCURRENCY_DENIAL_PREFIX + (
            "Another place_option_order is still in flight. Send placements one at a time; "
            "wait for each result before the next."
        )
    if not working:
        return None
    contracts = {leg.option_id for leg in new.legs}
    same = [w.label for w in working if w.option_id is not None and w.option_id in contracts]
    if same:
        return CONCURRENCY_DENIAL_PREFIX + (
            f"Order(s) {', '.join(same)} on the same contract are not confirmed terminal. "
            "Read get_option_orders, and cancel and confirm before placing again on that "
            "contract (orders.working)."
        )
    if not all(leg.closing for leg in new.legs) or not all(w.closing for w in working):
        listed = ", ".join(w.label for w in working)
        return CONCURRENCY_DENIAL_PREFIX + (
            f"Order(s) {listed} are still unresolved. Only buy-to-close orders on different "
            "contracts may work at the same time; place this order after they fill or are "
            "cancelled and confirmed (orders.execution_order)."
        )
    gaps: list[str] = []
    total = Decimal(0)
    for w in working:
        debit, gap = _debit(
            w.limit_price,
            multipliers.get(w.option_id) if w.option_id else None,
            w.remaining_quantity,
            f"working order {w.label}",
        )
        if gap is not None:
            gaps.append(gap)
        elif debit is not None:
            total += debit
    for leg in new.legs:
        debit, gap = _debit(
            new.limit_price, multipliers.get(leg.option_id), new.quantity, "this order"
        )
        if gap is not None:
            gaps.append(gap)
        elif debit is not None:
            total += debit
    cash, cash_gap = _cash(account, as_of, account_max_age)
    if cash_gap is not None:
        gaps.append(cash_gap)
    if gaps or cash is None:
        return CONCURRENCY_DENIAL_PREFIX + (
            "Funds for concurrent closes cannot be verified: "
            + "; ".join(gaps)
            + ". Place this close after the working ones resolve, or supply the missing value."
        )
    if total > cash:
        return CONCURRENCY_DENIAL_PREFIX + (
            f"Concurrent close debits {total} (working closes plus this one, at their limit "
            f"prices) exceed available settled cash {cash}. Place this close after the "
            "working ones resolve."
        )
    return None


__all__ = [
    "CONCURRENCY_DENIAL_PREFIX",
    "NewPlacement",
    "PlacementLeg",
    "WorkingPlacement",
    "check_concurrency",
]
