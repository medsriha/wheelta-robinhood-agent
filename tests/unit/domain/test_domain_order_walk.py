"""The code-run order walk's pure schedule (ADR-0066, domain/order_walk.py)."""

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import OrderSide
from wheelta_robinhood_agent.domain.order_walk import (
    TickSchedule,
    WalkTiming,
    can_start,
    remaining_quantity,
    round_to_tick,
    step_prices,
    within_bounds,
)

STO, BTC = OrderSide.SELL_TO_OPEN, OrderSide.BUY_TO_CLOSE
PENNY = TickSchedule(
    above_tick=Decimal("0.01"), below_tick=Decimal("0.01"), cutoff_price=Decimal("0")
)
NICKEL_ABOVE_3 = TickSchedule(
    above_tick=Decimal("0.05"), below_tick=Decimal("0.01"), cutoff_price=Decimal("3.00")
)


def D(value: str) -> Decimal:
    return Decimal(value)


@pytest.mark.parametrize(
    ("price", "side", "expected"),
    [
        ("0.834", STO, "0.84"),
        ("0.834", BTC, "0.83"),
        ("0.83", STO, "0.83"),
        ("3.02", STO, "3.05"),
        ("3.02", BTC, "3.00"),
        ("2.996", STO, "3.00"),
    ],
)
def test_round_to_tick_favors_the_trader(price: str, side: OrderSide, expected: str) -> None:
    assert round_to_tick(D(price), NICKEL_ABOVE_3, side) == D(expected)


def test_tick_validity_follows_the_cutoff() -> None:
    assert NICKEL_ABOVE_3.is_valid(D("2.99"))
    assert not NICKEL_ABOVE_3.is_valid(D("3.01"))
    assert NICKEL_ABOVE_3.is_valid(D("3.05"))
    assert not NICKEL_ABOVE_3.is_valid(D("0"))


def test_sell_walk_steps_down_evenly_to_worst() -> None:
    assert step_prices(D("0.90"), D("0.80"), 5, STO, PENNY) == (
        D("0.90"),
        D("0.88"),
        D("0.85"),
        D("0.83"),
        D("0.80"),
    )


def test_buy_walk_steps_up_to_worst() -> None:
    assert step_prices(D("0.40"), D("0.50"), 3, BTC, PENNY) == (D("0.40"), D("0.45"), D("0.50"))


def test_steps_that_round_to_a_repeat_are_dropped() -> None:
    assert step_prices(D("0.82"), D("0.80"), 5, STO, PENNY) == (D("0.82"), D("0.81"), D("0.80"))


def test_single_step_or_equal_endpoints_place_once() -> None:
    assert step_prices(D("0.83"), D("0.80"), 1, STO, PENNY) == (D("0.83"),)
    assert step_prices(D("0.83"), D("0.83"), 5, STO, PENNY) == (D("0.83"),)


@pytest.mark.parametrize(
    ("start", "worst", "side"),
    [
        ("0.80", "0.90", STO),  # a sell's worst price above its start
        ("0.90", "0.80", BTC),  # a buy's worst price below its start
        ("0.835", "0.80", STO),  # off tick
        ("3.02", "2.90", STO),  # off the nickel tick above the cutoff
    ],
)
def test_invalid_endpoints_raise(start: str, worst: str, side: OrderSide) -> None:
    with pytest.raises(ValueError):
        step_prices(D(start), D(worst), 5, side, NICKEL_ABOVE_3)


@given(
    start_cents=st.integers(min_value=1, max_value=2000),
    span_cents=st.integers(min_value=0, max_value=500),
    steps=st.integers(min_value=1, max_value=10),
    selling=st.booleans(),
)
def test_walk_is_monotone_tick_valid_and_ends_on_worst(
    start_cents: int, span_cents: int, steps: int, selling: bool
) -> None:
    side = STO if selling else BTC
    start = Decimal(start_cents + (span_cents if selling else 0)) / 100
    worst = Decimal(start_cents + (0 if selling else span_cents)) / 100
    prices = step_prices(start, worst, steps, side, PENNY)
    assert prices[0] == start and prices[-1] == (start if steps == 1 else worst)
    assert 1 <= len(prices) <= steps
    assert all(PENNY.is_valid(p) for p in prices)
    pairs = list(zip(prices, prices[1:], strict=False))
    assert all((a > b) if selling else (a < b) for a, b in pairs)


def test_bounds_quantity_and_start_window() -> None:
    assert within_bounds(D("0.80"), D("0.80"), D("0.86"))
    assert not within_bounds(D("0.79"), D("0.80"), D("0.86"))
    assert remaining_quantity(2, 1) == 1 and remaining_quantity(2, 3) == 0
    assert can_start(480.0, 300, 180.0)
    assert not can_start(479.9, 300, 180.0)


def test_timing_must_fit_the_window() -> None:
    WalkTiming(window_seconds=300, max_steps=5, step_wait_seconds=50, poll_seconds=10)
    with pytest.raises(ValidationError):
        WalkTiming(
            window_seconds=300,
            max_steps=5,
            step_wait_seconds=50,
            poll_seconds=10,
            step_overhead_seconds=11,
        )
    with pytest.raises(ValidationError):
        WalkTiming(window_seconds=300, max_steps=5, step_wait_seconds=5, poll_seconds=10)
