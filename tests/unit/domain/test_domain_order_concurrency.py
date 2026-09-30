"""ADR-0051: when a placement may join orders already working."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.enums import DataQuality
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts_rules import FactsRuleMarker
from wheelta_robinhood_agent.domain.order_concurrency import (
    CONCURRENCY_DENIAL_PREFIX,
    NewPlacement,
    PlacementLeg,
    WorkingPlacement,
    check_concurrency,
)

D = Decimal
T0 = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)
TC = UUID(int=900)
MULTIPLIERS = {"a": 100, "b": 100, "c": 100}


def account(cash: str | None = "1000", age: int = 30, verified: bool = True) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=UUID(int=1),
        as_of=T0 - timedelta(seconds=age),
        retrieved_at=T0 - timedelta(seconds=age),
        tool_call_ids=(TC,),
        account_ref="****1234",
        agentic_verified=verified,
        account_value_usd=D("10000"),
        available_settled_cash_usd=D(cash) if cash is not None else None,
        csp_reserved_cash_usd=D("0"),
        csp_cash_base_usd=D(cash) if cash is not None else None,
        csp_cash_base_evidence_ids=(TC,) if cash is not None else (),
        positions_ref=None,
        open_orders_ref=None,
        tax_lots_ref=None,
        quality=DataQuality.OK if cash is not None else DataQuality.MISSING,
        gaps=()
        if cash is not None
        else (
            Gap(field="available_settled_cash_usd", kind=DataQuality.MISSING, detail="x"),
            Gap(field="csp_cash_base_usd", kind=DataQuality.MISSING, detail="x"),
        ),
    )


def close(option_id: str = "b", price: str = "2.00", qty: int = 1) -> NewPlacement:
    return NewPlacement(
        legs=(PlacementLeg(option_id=option_id, closing=True),),
        quantity=qty,
        limit_price=D(price),
    )


def opening(option_id: str = "b") -> NewPlacement:
    return NewPlacement(
        legs=(PlacementLeg(option_id=option_id, closing=False),),
        quantity=1,
        limit_price=D("1.00"),
    )


def working(
    option_id: str | None = "a", closing: bool = True, price: str | None = "3.00", qty: int = 1
) -> WorkingPlacement:
    return WorkingPlacement(
        label="ord-a",
        option_id=option_id,
        closing=closing,
        remaining_quantity=qty,
        limit_price=D(price) if price is not None else None,
    )


def check(
    new: NewPlacement,
    working_orders: tuple[WorkingPlacement, ...] = (),
    in_flight: int = 0,
    snapshot: AccountSnapshot | None = None,
    max_age: int | FactsRuleMarker = 120,
    multipliers: dict[str, int | None] | None = None,
) -> str | None:
    return check_concurrency(
        new,
        working_orders,
        other_placements_in_flight=in_flight,
        multipliers=MULTIPLIERS if multipliers is None else multipliers,
        account=account() if snapshot is None else snapshot,
        account_max_age=max_age,
        as_of=T0,
    )


def test_nothing_working_passes_any_placement() -> None:
    assert check(close()) is None
    assert check(opening(), snapshot=account(cash=None)) is None  # sequential: no funds check


def test_another_placement_in_flight_is_denied() -> None:
    reason = check(close(), in_flight=1)
    assert reason is not None and reason.startswith(CONCURRENCY_DENIAL_PREFIX)
    assert "one at a time" in reason


def test_same_contract_is_denied() -> None:
    reason = check(close("a"), (working("a"),))
    assert reason is not None and "same contract" in reason and "ord-a" in reason


def test_an_open_waits_for_working_orders() -> None:
    reason = check(opening("b"), (working("a"),))
    assert reason is not None and "Only buy-to-close orders" in reason


def test_a_close_waits_for_a_working_open() -> None:
    reason = check(close("b"), (working("a", closing=False),))
    assert reason is not None and "Only buy-to-close orders" in reason


def test_funded_concurrent_closes_pass() -> None:
    # 3.00 x 100 + 2.00 x 100 x 2 = 700 <= 1000
    assert check(close("b", qty=2), (working("a"),)) is None
    # exactly equal passes: 3.00 x 100 + 7.00 x 100 = 1000
    assert check(close("b", price="7.00"), (working("a"),)) is None


def test_unfunded_concurrent_closes_are_denied_with_values() -> None:
    reason = check(close("b", price="7.01"), (working("a"),))
    assert reason is not None
    assert "1001.00" in reason and "1000" in reason


def test_unverifiable_funds_deny_the_concurrent_close() -> None:
    cases = [
        ({"snapshot": account(cash=None)}, "available settled cash is not reported"),
        ({"snapshot": account(age=121)}, "older than"),
        ({"snapshot": account(verified=False)}, "not Agentic-verified"),
        ({"max_age": FactsRuleMarker.UNSET}, "is not set"),
        ({"multipliers": {"a": 100, "b": None}}, "multiplier not verified"),
    ]
    for kwargs, text in cases:
        reason = check(close("b"), (working("a"),), **kwargs)  # type: ignore[arg-type]
        assert reason is not None and text in reason, (kwargs, reason)
    reason = check(close("b"), (working("a", price=None),))
    assert reason is not None and "limit price unknown" in reason


def test_no_limit_on_account_age_still_needs_a_snapshot() -> None:
    assert check(close("b"), (working("a"),), max_age=FactsRuleMarker.NONE) is None
