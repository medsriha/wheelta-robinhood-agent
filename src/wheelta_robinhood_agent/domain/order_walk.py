"""The code-run order walk's schedule (ADR-0066). Pure: no I/O, no clock.

The agent chooses a trade's contract, quantity, `start_price`, and `worst_price`; the executor
(`agent/order_walk.py`) walks between those two prices inside `orders.walk.window_seconds`.
Everything here is the deterministic part of that walk:

- `TickSchedule` / `round_to_tick`: Robinhood's `min_ticks` (`below_tick` under
  `cutoff_price`, `above_tick` at or above it; the boundary reading is ours, unverified, and
  conservative: `above_tick` is a multiple of `below_tick` on every captured chain). A price
  rounds in the trader's favor: a sell to open up, a buy to close down.
- `step_prices`: up to `max_steps` prices, linear from `start_price` to `worst_price`, each
  rounded in the trader's favor, strictly moving toward `worst_price` (a step that would repeat
  the previous price is dropped), the last step exactly on `worst_price`.
- `within_bounds`: `orders.limit_price_bounds`, the fresh quote's [bid, ask], inclusive.
- `remaining_quantity`: the target less confirmed cumulative fills, never below zero.
- `can_start`: no new walk unless the whole window plus the order wind-down still fits the
  session budget (ADR-0066 item 6).
"""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from wheelta_robinhood_agent.domain.base import DomainModel, NonNegDec, PosCount, PosDec
from wheelta_robinhood_agent.domain.enums import OrderSide


class WalkStatus(StrEnum):
    """How a walk job stands (`await_order_work`). Only `working` is not terminal."""

    WORKING = "working"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"  # ended with some, not all, contracts filled
    CANCELLED = "cancelled"  # ended unfilled: window, last step, or QUOTE_MOVED
    STOPPED = "stopped"  # a check, the review, or the stop latch ended it; outcome known
    UNKNOWN = "unknown"  # an order action's outcome is unknown; placement ends for the run


TERMINAL_WALK_STATUSES = frozenset(set(WalkStatus) - {WalkStatus.WORKING})


class TickSchedule(DomainModel):
    """A contract's price ticks: `below_tick` under `cutoff_price`, `above_tick` from it."""

    above_tick: PosDec
    below_tick: PosDec
    cutoff_price: NonNegDec

    def tick_at(self, price: Decimal) -> Decimal:
        return self.above_tick if price >= self.cutoff_price else self.below_tick

    def is_valid(self, price: Decimal) -> bool:
        """A positive price on the tick that applies at it."""
        return price > 0 and price % self.tick_at(price) == 0


def _round(price: Decimal, tick: Decimal, up: bool) -> Decimal:
    steps = (price / tick).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR)
    return steps * tick


def round_to_tick(price: Decimal, ticks: TickSchedule, side: OrderSide) -> Decimal:
    """`price` on a valid tick, rounded in the trader's favor (sell to open up, buy to close
    down). Raises ValueError when no positive tick-valid price results."""
    up = side is OrderSide.SELL_TO_OPEN
    tick = ticks.tick_at(price)
    rounded = _round(price, tick, up)
    if not ticks.is_valid(rounded):
        # Rounding crossed the cutoff: re-round on the tick that applies at the result.
        rounded = _round(price, ticks.tick_at(rounded), up)
    if not ticks.is_valid(rounded):
        raise ValueError(f"no tick-valid price near {price}")
    return rounded


class WalkTiming(DomainModel):
    """`orders.walk` timing (rules v17, ADR-0066), in whole seconds."""

    window_seconds: PosCount
    max_steps: PosCount
    step_wait_seconds: PosCount
    poll_seconds: PosCount
    # Time a step needs beyond its wait: re-quote, review, place, cancel, and confirm reads.
    step_overhead_seconds: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _fits(self) -> Self:
        if self.poll_seconds > self.step_wait_seconds:
            raise ValueError("poll_seconds exceeds step_wait_seconds")
        needed = self.max_steps * (self.step_wait_seconds + self.step_overhead_seconds)
        if needed > self.window_seconds:
            raise ValueError(
                f"max_steps x (step_wait_seconds + overhead) = {needed}s exceeds window_seconds"
            )
        return self


def step_prices(
    start_price: Decimal,
    worst_price: Decimal,
    max_steps: int,
    side: OrderSide,
    ticks: TickSchedule,
) -> tuple[Decimal, ...]:
    """The walk's limit prices, first to last (module docstring). Raises ValueError when the
    endpoints are off-tick, non-positive, or in the wrong order for the side."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    for name, price in (("start_price", start_price), ("worst_price", worst_price)):
        if not ticks.is_valid(price):
            raise ValueError(f"{name} {price} is not a positive tick-valid price")
    selling = side is OrderSide.SELL_TO_OPEN
    if (selling and worst_price > start_price) or (not selling and worst_price < start_price):
        raise ValueError(
            "worst_price must not be above start_price for a sell to open, "
            "nor below it for a buy to close"
        )
    if max_steps == 1 or start_price == worst_price:
        return (start_price,)
    span = worst_price - start_price
    prices: list[Decimal] = []
    for k in range(max_steps):
        raw = start_price + span * Decimal(k) / Decimal(max_steps - 1)
        price = worst_price if k == max_steps - 1 else round_to_tick(raw, ticks, side)
        # Never past worst_price, never a repeat or a step back toward start_price.
        price = max(price, worst_price) if selling else min(price, worst_price)
        if prices and (price >= prices[-1] if selling else price <= prices[-1]):
            continue
        prices.append(price)
    return tuple(prices)


def within_bounds(price: Decimal, bid: Decimal, ask: Decimal) -> bool:
    """`orders.limit_price_bounds`: inside the fresh quote's [bid, ask], inclusive."""
    return bid <= price <= ask


def remaining_quantity(target: int, filled: int) -> int:
    """The target less confirmed cumulative fills (`orders.working`), never below zero."""
    return max(target - filled, 0)


def can_start(
    remaining_budget_seconds: float, window_seconds: int, wind_down_seconds: float
) -> bool:
    """Whether a new walk's whole window still fits before the order wind-down."""
    return remaining_budget_seconds >= window_seconds + wind_down_seconds


__all__ = [
    "TERMINAL_WALK_STATUSES",
    "TickSchedule",
    "WalkStatus",
    "WalkTiming",
    "can_start",
    "remaining_quantity",
    "round_to_tick",
    "step_prices",
    "within_bounds",
]
