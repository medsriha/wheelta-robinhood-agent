"""Pure range and sanity checks (CLAUDE.md §12; DATA_QUALITY.md "Sanity checks", "Freshness").

Every threshold is a parameter: the caller passes values from `rules/trading_rules.toml`.
Domain never imports config and never reads the clock. Inputs that are floats, bools, or
naive datetimes raise TypeError/ValueError rather than being coerced.
"""

from datetime import datetime
from decimal import Decimal


def _dec(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def require_aware(value: datetime, name: str = "datetime") -> datetime:
    """Raise ValueError for a naive datetime (CLAUDE.md §4: timezone-aware only)."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def is_valid_bid_ask(bid: Decimal, ask: Decimal) -> bool:
    """`0 <= bid <= ask` and `ask > 0` (DATA_QUALITY.md "Sanity checks").

    A zero bid passes this check; whether it is usable for an opening is a liquidity rule
    applied elsewhere. A zero bid is never treated as a free option.
    """
    bid = _dec(bid, "bid")
    ask = _dec(ask, "ask")
    return Decimal(0) <= bid <= ask and ask > 0


def is_positive_price(price: Decimal) -> bool:
    """Prices, strikes, and premiums are > 0 (CLAUDE.md §12)."""
    return _dec(price, "price") > 0


def is_mark_within_bid_ask(mark: Decimal, bid: Decimal, ask: Decimal) -> bool:
    """A present mark lies within `[bid, ask]` (DATA_QUALITY.md "mark within bounds")."""
    mark = _dec(mark, "mark")
    return is_valid_bid_ask(bid, ask) and bid <= mark <= ask


def is_delta_plausible(delta: Decimal) -> bool:
    """`0 <= |delta| <= 1` (DATA_QUALITY.md "Sanity checks")."""
    return abs(_dec(delta, "delta")) <= 1


def is_iv_in_band(iv: Decimal, low: Decimal, high: Decimal) -> bool:
    """IV (decimal fraction) within the inclusive band from `data_quality.sanity`.

    The band is supplied by the caller; raises ValueError if `low > high`.
    """
    iv = _dec(iv, "iv")
    low = _dec(low, "low")
    high = _dec(high, "high")
    if low > high:
        raise ValueError("IV band low must not exceed high")
    return low <= iv <= high


def is_non_negative_amount(amount: Decimal) -> bool:
    """Buying power and cash balances are >= 0 (DATA_QUALITY.md "Sanity checks")."""
    return _dec(amount, "amount") >= 0


def is_integer_quantity(quantity: int | Decimal) -> bool:
    """Contracts and shares are non-negative integers (DATA_QUALITY.md "Normalization").

    An integral Decimal (e.g. a broker's `"3.0000"`) passes; floats and bools raise.
    """
    if isinstance(quantity, bool) or not isinstance(quantity, int | Decimal):
        raise TypeError(f"quantity must be int or Decimal, got {type(quantity).__name__}")
    if isinstance(quantity, Decimal):
        quantity = _dec(quantity, "quantity")
        return quantity >= 0 and quantity == quantity.to_integral_value()
    return quantity >= 0


def to_integer_quantity(quantity: int | Decimal) -> int | None:
    """Return the integer quantity, or None if it is not a non-negative integer."""
    if not is_integer_quantity(quantity):
        return None
    return int(quantity)


def is_on_tick(price: Decimal, tick: Decimal) -> bool:
    """`price` is an exact multiple of the explicit `tick` (CLAUDE.md §4, §12).

    Also used for strike increments. The tick comes from verified instrument data; it is
    never assumed. Raises ValueError for a non-positive tick.
    """
    price = _dec(price, "price")
    tick = _dec(tick, "tick")
    if tick <= 0:
        raise ValueError("tick must be positive")
    return price % tick == 0


def is_fresh(as_of: datetime | None, now: datetime, max_age_seconds: int) -> bool:
    """True iff `0 <= now - as_of <= max_age_seconds` (DATA_QUALITY.md "Freshness").

    Fails closed: an unknown `as_of` is stale, and an `as_of` later than `now` (clock skew)
    is not fresh. Both times must be timezone-aware; `max_age_seconds` comes from
    `data_quality.freshness` and must be a non-negative int.
    """
    if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, int):
        raise TypeError("max_age_seconds must be int")
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    require_aware(now, "now")
    if as_of is None:
        return False
    require_aware(as_of, "as_of")
    age = (now - as_of).total_seconds()
    return 0 <= age <= max_age_seconds
