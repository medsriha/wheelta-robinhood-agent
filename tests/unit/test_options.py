from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wheelta_robinhood_agent.domain.enums import OptionRight
from wheelta_robinhood_agent.domain.options import OccSymbol


def test_parse_padded_and_compact() -> None:
    padded = OccSymbol.parse("AAPL  260116C00190000")
    compact = OccSymbol.parse("AAPL260116C00190000")
    assert padded == compact
    assert padded.root == "AAPL"
    assert padded.expiration == date(2026, 1, 16)
    assert padded.right is OptionRight.CALL
    assert padded.strike == Decimal("190")
    assert str(compact) == "AAPL  260116C00190000"


def test_fractional_strike() -> None:
    s = OccSymbol.parse("F     261120P00012500")
    assert s.right is OptionRight.PUT
    assert s.strike == Decimal("12.5")


@pytest.mark.parametrize(
    "bad",
    [
        "aapl260116C00190000",
        " AAPL260116C00190000",
        "AAPL260116X00190000",
        "AAPL261316C00190000",
        "AAPL260116C0019000",
        "TOOLONGX260116C00190000",
        "",
    ],
)
def test_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        OccSymbol.parse(bad)


def test_rejects_non_positive_strike() -> None:
    with pytest.raises(ValueError):
        OccSymbol.parse("AAPL  260116C00000000")


@pytest.mark.parametrize("root", ["aapl", "1ABC", "TOOLONG", ""])
def test_direct_construction_validates_root(root: str) -> None:
    with pytest.raises(ValueError):
        OccSymbol(root=root, expiration=date(2026, 1, 16), right=OptionRight.PUT, strike=Decimal(1))


def test_str_rejects_sub_tick_strike() -> None:
    s = OccSymbol(
        root="AAPL", expiration=date(2026, 1, 16), right=OptionRight.CALL, strike=Decimal("1.0005")
    )
    with pytest.raises(ValueError):
        str(s)


@given(
    root=st.from_regex(r"[A-Z][A-Z0-9]{0,5}", fullmatch=True),
    day=st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31)),
    right=st.sampled_from(OptionRight),
    strike_units=st.integers(min_value=1, max_value=99_999_999),
)
def test_round_trip(root: str, day: date, right: OptionRight, strike_units: int) -> None:
    s = OccSymbol(root=root, expiration=day, right=right, strike=Decimal(strike_units) / 1000)
    assert OccSymbol.parse(str(s)) == s
    assert len(str(s)) == 21
