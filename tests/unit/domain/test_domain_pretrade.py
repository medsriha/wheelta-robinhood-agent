"""Pre-trade validation of sell-to-open legs (ADR-0048).

Expected values are hand-computed. The base put (strike 73, DTE 20, bid 1.00) sits exactly on
the yield floor: 1.00 x 365 / (73 x 20) = 0.25.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument, UnderlyingQuote
from wheelta_robinhood_agent.domain.facts_rules import FactsRuleMarker
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.pretrade import (
    PRETRADE_DENIAL_PREFIX,
    CheckName,
    CheckStatus,
    LegValidation,
    OpeningLeg,
    PretradeRules,
    pretrade_feedback,
    validate_opening_leg,
)
from wheelta_robinhood_agent.domain.run_record import Quote

D = Decimal
T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)  # 11:00 ET
PUT = OccSymbol.parse("XYZ   261015P00073000")  # DTE 20 from T0
CALL = OccSymbol.parse("XYZ   261015C00055000")
TC = UUID(int=900)

RULES = PretradeRules(
    min_dte=7,
    max_dte=45,
    min_abs_delta=D("0.15"),
    max_abs_delta=D("0.30"),
    min_cushion_ratio=D("0.04"),
    min_annualized_yield_ratio=D("0.25"),
    option_quote_max_age_seconds=60,
    equity_quote_max_age_seconds=60,
)


def rules(**kw: Any) -> PretradeRules:
    return RULES.model_copy(update=kw)


def instrument(occ: OccSymbol = PUT) -> OptionInstrument:
    return OptionInstrument(
        evidence_id=UUID(int=1),
        as_of=T0 - timedelta(seconds=5),
        source_tool_call_ids=(TC,),
        occ_symbol=occ,
        broker_instrument_id="inst-1",
        underlying="XYZ",
        multiplier=100,
    )


def quote(bid: str = "1.00", delta: str | None = "-0.20", age: int = 10) -> Quote:
    return Quote(
        quote_id=UUID(int=2),
        broker_instrument_id="inst-1",
        bid=D(bid),
        ask=D(bid) + D("0.10"),
        delta=D(delta) if delta is not None else None,
        as_of=T0 - timedelta(seconds=age),
        source_tool_call_ids=(TC,),
    )


def spot(price: str = "80.00", age: int = 10) -> UnderlyingQuote:
    return UnderlyingQuote(
        evidence_id=UUID(int=3),
        as_of=T0 - timedelta(seconds=age),
        source_tool_call_ids=(TC,),
        symbol="XYZ",
        price=D(price),
    )


def leg(
    inst: OptionInstrument | None = None,
    q: Quote | None = None,
    u: UnderlyingQuote | None = None,
    *,
    no_quote: bool = False,
    no_spot: bool = False,
    no_instrument: bool = False,
) -> OpeningLeg:
    return OpeningLeg(
        option_id="inst-1",
        instrument=None if no_instrument else (inst or instrument()),
        option_quote=None if no_quote else (q or quote()),
        underlying_quote=None if no_spot else (u or spot()),
    )


def check(v: LegValidation, name: CheckName) -> tuple[CheckStatus, Decimal | None, str]:
    (c,) = [c for c in v.checks if c.check is name]
    return c.status, c.value, c.detail


def run(lg: OpeningLeg, r: PretradeRules = RULES, at: datetime = T0) -> LegValidation:
    return validate_opening_leg(lg, r, at)


def test_a_leg_on_every_bound_passes() -> None:
    v = run(leg())
    assert v.passed
    assert v.contract == "XYZ 2026-10-15 put 73"
    assert check(v, CheckName.DTE)[:2] == (CheckStatus.PASS, D(20))
    assert check(v, CheckName.DELTA)[:2] == (CheckStatus.PASS, D("0.20"))
    assert check(v, CheckName.CUSHION)[:2] == (CheckStatus.PASS, D("0.0875"))  # 7 / 80
    assert check(v, CheckName.ANNUALIZED_YIELD)[:2] == (CheckStatus.PASS, D("0.25"))


def test_yield_below_the_floor_fails_with_its_inputs() -> None:
    v = run(leg(q=quote(bid="0.99")))  # 0.99 x 365 / 1460 = 0.2475
    status, value, detail = check(v, CheckName.ANNUALIZED_YIELD)
    assert status is CheckStatus.FAIL and value == D("0.2475")
    assert detail == (
        "annualized_yield 0.2475 is below filters.min_annualized_yield_ratio 0.25 "
        "(live bid 0.99, collateral strike 73, DTE 20)"
    )
    assert not v.passed


@pytest.mark.parametrize(
    ("price", "status"),
    [("76.00", CheckStatus.PASS), ("75.00", CheckStatus.FAIL)],
)
def test_put_cushion_is_strike_distance_over_spot(price: str, status: CheckStatus) -> None:
    v = run(leg(inst=instrument(OccSymbol.parse("XYZ   261015P00072960")), u=spot(price)))
    # 72.96 strike: on 76.00, 3.04 / 76 = 0.04 exactly (inclusive); on 75.00, 0.0272.
    got, _, detail = check(v, CheckName.CUSHION)
    if price == "76.00":
        assert got is status
    else:
        assert got is status and "below filters.min_cushion_ratio 0.04" in detail
        assert "(underlying 75.00, strike 72.96)" in detail


def test_in_the_money_put_has_negative_cushion() -> None:
    status, value, _ = check(run(leg(u=spot("70.00"))), CheckName.CUSHION)
    assert status is CheckStatus.FAIL and value is not None and value < 0


def test_call_cushion_and_yield_use_the_share_price() -> None:
    # Strike 55 on 50: cushion (55 - 50) / 50 = 0.10; yield 0.70 x 365 / (50 x 20) = 0.2555.
    v = run(leg(inst=instrument(CALL), q=quote(bid="0.70", delta="0.25"), u=spot("50.00")))
    assert v.passed
    assert check(v, CheckName.CUSHION)[1] == D("0.1")
    assert check(v, CheckName.ANNUALIZED_YIELD)[1] == D("0.2555")


@pytest.mark.parametrize(
    ("delta", "status"),
    [
        ("-0.15", CheckStatus.PASS),
        ("-0.30", CheckStatus.PASS),
        ("-0.14", CheckStatus.FAIL),
        ("-0.31", CheckStatus.FAIL),
    ],
)
def test_absolute_delta_bounds_are_inclusive(delta: str, status: CheckStatus) -> None:
    got, _, detail = check(run(leg(q=quote(delta=delta))), CheckName.DELTA)
    assert got is status
    if status is CheckStatus.FAIL:
        assert f"live delta {delta}" in detail


@pytest.mark.parametrize(
    ("expiration", "status", "bound"),
    [
        ("261002", CheckStatus.PASS, None),  # DTE 7
        ("261001", CheckStatus.FAIL, "below filters.min_dte 7"),  # DTE 6
        ("261109", CheckStatus.PASS, None),  # DTE 45
        ("261110", CheckStatus.FAIL, "above filters.max_dte 45"),  # DTE 46
    ],
)
def test_dte_bounds_are_inclusive(expiration: str, status: CheckStatus, bound: str | None) -> None:
    inst = instrument(OccSymbol.parse(f"XYZ   {expiration}P00073000"))
    got, _, detail = check(run(leg(inst=inst)), CheckName.DTE)
    assert got is status
    if bound is not None:
        assert bound in detail


def test_dte_uses_the_new_york_calendar_date() -> None:
    late = datetime(2026, 9, 26, 3, 0, tzinfo=UTC)  # 23:00 ET on 2026-09-25: still DTE 20
    lg = leg(
        q=quote(age=0).model_copy(update={"as_of": late}),
        u=spot().model_copy(update={"as_of": late}),
    )
    assert check(run(lg, at=late), CheckName.DTE)[1] == D(20)


def test_stale_option_quote_blocks_delta_and_yield_but_not_cushion() -> None:
    v = run(leg(q=quote(age=61)))
    assert check(v, CheckName.DELTA)[0] is CheckStatus.MISSING
    assert (
        "older than data_quality.freshness.option_quote_max_age_seconds"
        in (check(v, CheckName.DELTA)[2])
    )
    assert check(v, CheckName.ANNUALIZED_YIELD)[0] is CheckStatus.MISSING
    assert check(v, CheckName.CUSHION)[0] is CheckStatus.PASS
    assert not v.passed


def test_stale_underlying_blocks_cushion_and_call_yield() -> None:
    v = run(leg(inst=instrument(CALL), q=quote(bid="0.70", delta="0.25"), u=spot("50", age=61)))
    assert check(v, CheckName.CUSHION)[0] is CheckStatus.MISSING
    assert check(v, CheckName.ANNUALIZED_YIELD)[0] is CheckStatus.MISSING
    # A put's yield needs only the strike.
    v = run(leg(u=spot(age=61)))
    assert check(v, CheckName.ANNUALIZED_YIELD)[0] is CheckStatus.PASS


def test_a_future_quote_is_not_fresh() -> None:
    assert check(run(leg(q=quote(age=-1))), CheckName.DELTA)[0] is CheckStatus.MISSING


def test_missing_delta_blocks() -> None:
    status, _, detail = check(run(leg(q=quote(delta=None))), CheckName.DELTA)
    assert status is CheckStatus.MISSING and "reports no delta" in detail


def test_missing_quotes_block_every_dependent_check() -> None:
    v = run(leg(no_quote=True, no_spot=True))
    assert check(v, CheckName.DTE)[0] is CheckStatus.PASS
    assert {c.status for c in v.checks if c.check is not CheckName.DTE} == {CheckStatus.MISSING}
    assert "no validated option quote recorded in this run" in check(v, CheckName.DELTA)[2]
    assert "no validated XYZ quote recorded in this run" in check(v, CheckName.CUSHION)[2]


def test_unknown_instrument_blocks_everything() -> None:
    v = run(leg(no_instrument=True))
    assert v.contract is None
    assert [c.status for c in v.checks] == [CheckStatus.MISSING] * 4


def test_zero_dte_yield_is_undefined() -> None:
    lg = leg(inst=instrument(OccSymbol.parse("XYZ   260925P00073000")))
    v = run(lg, rules(min_dte=0))
    assert check(v, CheckName.DTE)[0] is CheckStatus.PASS
    assert check(v, CheckName.ANNUALIZED_YIELD)[0] is CheckStatus.MISSING


def test_tbd_bound_blocks_and_none_or_discretion_bound_is_not_enforced() -> None:
    status, _, detail = check(
        run(leg(u=spot("70.00")), rules(min_cushion_ratio=FactsRuleMarker.UNSET)),
        CheckName.CUSHION,
    )
    assert status is CheckStatus.MISSING and "filters.min_cushion_ratio is TBD" in detail
    for marker in (FactsRuleMarker.NONE, FactsRuleMarker.AGENT_DISCRETION):
        v = run(leg(u=spot("70.00")), rules(min_cushion_ratio=marker))
        assert check(v, CheckName.CUSHION)[0] is CheckStatus.PASS
    late = instrument(OccSymbol.parse("XYZ   261231P00073000"))
    assert run(leg(inst=late), rules(max_dte=FactsRuleMarker.NONE)).checks[0].status is (
        CheckStatus.PASS
    )


def test_unset_freshness_blocks_and_none_freshness_only_rejects_the_future() -> None:
    v = run(leg(), rules(option_quote_max_age_seconds=FactsRuleMarker.UNSET))
    assert check(v, CheckName.DELTA)[0] is CheckStatus.MISSING
    v = run(leg(q=quote(age=3600)), rules(option_quote_max_age_seconds=FactsRuleMarker.NONE))
    assert check(v, CheckName.DELTA)[0] is CheckStatus.PASS


def test_naive_as_of_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_opening_leg(leg(), RULES, datetime(2026, 9, 25, 15, 0))  # noqa: DTZ001


def test_feedback_is_none_when_every_leg_passes() -> None:
    assert pretrade_feedback([run(leg())]) is None
    assert pretrade_feedback([]) is None


def test_feedback_names_failed_and_passed_checks() -> None:
    failing = run(leg(q=quote(bid="0.50", delta="-0.10")))
    text = pretrade_feedback([run(leg()), failing])
    assert text is not None and text.startswith(PRETRADE_DENIAL_PREFIX)
    assert "leg XYZ 2026-10-15 put 73 (option_id inst-1): failed: " in text
    assert "abs_delta 0.10 is below filters.min_abs_delta 0.15 (live delta -0.10)" in text
    assert "annualized_yield 0.1250 is below filters.min_annualized_yield_ratio 0.25" in text
    assert "passed: dte 20, cushion 0.0875" in text
    assert text.count("leg XYZ") == 1  # the passing leg is not repeated


@given(st.integers(min_value=1, max_value=9999))
def test_cushion_passes_exactly_when_the_ratio_reaches_the_floor(cents: int) -> None:
    strike = D(cents) / 100
    price = D("100")
    inst = instrument(
        OccSymbol(root="XYZ", expiration=PUT.expiration, right=PUT.right, strike=strike)
    )
    status, _, _ = check(run(leg(inst=inst, u=spot(str(price)))), CheckName.CUSHION)
    expected = (price - strike) / price >= D("0.04")
    assert (status is CheckStatus.PASS) is expected
