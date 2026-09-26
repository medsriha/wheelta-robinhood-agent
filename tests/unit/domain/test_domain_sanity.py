from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wheelta_robinhood_agent.domain import sanity as s

D = Decimal
NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)


def test_bid_ask() -> None:
    assert s.is_valid_bid_ask(D("1.00"), D("1.10"))
    assert s.is_valid_bid_ask(D("0"), D("0.05"))
    assert not s.is_valid_bid_ask(D("1.20"), D("1.10"))
    assert not s.is_valid_bid_ask(D("0"), D("0"))
    assert not s.is_valid_bid_ask(D("-0.01"), D("1"))
    with pytest.raises(TypeError):
        s.is_valid_bid_ask(1.0, D("1"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        s.is_valid_bid_ask(D("NaN"), D("1"))


def test_price_mark_delta() -> None:
    assert s.is_positive_price(D("0.01"))
    assert not s.is_positive_price(D("0"))
    assert s.is_mark_within_bid_ask(D("1.05"), D("1"), D("1.1"))
    assert not s.is_mark_within_bid_ask(D("1.2"), D("1"), D("1.1"))
    assert s.is_delta_plausible(D("-0.3"))
    assert not s.is_delta_plausible(D("1.01"))


def test_iv_band() -> None:
    assert s.is_iv_in_band(D("0.3"), D("0.01"), D("5"))
    assert not s.is_iv_in_band(D("6"), D("0.01"), D("5"))
    with pytest.raises(ValueError):
        s.is_iv_in_band(D("0.3"), D("5"), D("0.01"))


def test_amount_and_quantity() -> None:
    assert s.is_non_negative_amount(D("0"))
    assert not s.is_non_negative_amount(D("-1"))
    assert s.is_integer_quantity(3)
    assert s.is_integer_quantity(D("3.0000"))
    assert not s.is_integer_quantity(D("3.5"))
    assert not s.is_integer_quantity(-1)
    assert not s.is_integer_quantity(D("-2"))
    assert s.to_integer_quantity(D("2.00")) == 2
    assert s.to_integer_quantity(D("2.5")) is None
    with pytest.raises(TypeError):
        s.is_integer_quantity(True)
    with pytest.raises(TypeError):
        s.is_integer_quantity(3.0)  # type: ignore[arg-type]


def test_tick() -> None:
    assert s.is_on_tick(D("1.25"), D("0.05"))
    assert not s.is_on_tick(D("1.27"), D("0.05"))
    assert s.is_on_tick(D("190"), D("2.5"))
    with pytest.raises(ValueError):
        s.is_on_tick(D("1"), D("0"))


def test_freshness() -> None:
    assert s.is_fresh(NOW - timedelta(seconds=60), NOW, 60)
    assert not s.is_fresh(NOW - timedelta(seconds=61), NOW, 60)
    assert not s.is_fresh(None, NOW, 60)
    assert not s.is_fresh(NOW + timedelta(seconds=1), NOW, 60)
    est = timezone(timedelta(hours=-4))
    assert s.is_fresh(datetime(2026, 9, 25, 11, 0, tzinfo=est), NOW, 0)
    with pytest.raises(ValueError):
        s.is_fresh(datetime(2026, 9, 25, 15, 0), NOW, 60)  # noqa: DTZ001
    with pytest.raises(ValueError):
        s.is_fresh(NOW, datetime(2026, 9, 25, 15, 0), 60)  # noqa: DTZ001
    with pytest.raises(ValueError):
        s.is_fresh(NOW, NOW, -1)
    with pytest.raises(TypeError):
        s.is_fresh(NOW, NOW, True)
    with pytest.raises(TypeError):
        s.is_fresh(NOW, NOW, 1.5)  # type: ignore[arg-type]


@given(
    st.decimals(allow_nan=False, allow_infinity=False, places=2, min_value=0, max_value=10**6),
    st.decimals(allow_nan=False, allow_infinity=False, places=2, min_value=0, max_value=10**6),
)
def test_bid_ask_property(a: Decimal, b: Decimal) -> None:
    lo, hi = sorted((a, b))
    assert s.is_valid_bid_ask(lo, hi) == (hi > 0)
    if lo != hi:
        assert not s.is_valid_bid_ask(hi, lo)
