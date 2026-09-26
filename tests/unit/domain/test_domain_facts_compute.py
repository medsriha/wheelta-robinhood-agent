"""compute_decision_facts: sizing, capacity, metrics, gaps (OUTPUT_ASSEMBLY.md; ADR-0014/0015).

Expected values are written out independently (hand-computed fractions), not recomputed with
the module's own helpers.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import Dec
from wheelta_robinhood_agent.domain.enums import (
    CandidateOrigin,
    DataQuality,
    OrderSide,
    StrategyKind,
)
from wheelta_robinhood_agent.domain.evidence import Gap, SourcedValue
from wheelta_robinhood_agent.domain.facts import DecisionFacts, FactsPurpose
from wheelta_robinhood_agent.domain.facts_compute import (
    CandidateProvenance,
    ConfirmedFill,
    FactInputs,
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
    ShareHolding,
    ShortOptionHolding,
    UnderlyingQuote,
    WorkingOrder,
    compute_decision_facts,
)
from wheelta_robinhood_agent.domain.facts_rules import FactsRuleMarker, FactsRules
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.positions import PositionBookEntry, PositionInstrument
from wheelta_robinhood_agent.domain.run_record import Quote

D = Decimal
T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)  # 11:00 ET, Friday
PUT = OccSymbol.parse("XYZ   261015P00050000")  # expires 2026-10-15: DTE 20 from T0
CALL = OccSymbol.parse("XYZ   261015C00050000")
OTHER_PUT = OccSymbol.parse("XYZ   261120P00045000")
TC = UUID(int=900)  # a source tool call id


def uid(n: int) -> UUID:
    return UUID(int=n)


RULES = FactsRules(
    rules_version="5",
    rules_hash="sha256:test",
    max_contracts_per_order=10,
    max_collateral_per_underlying_usd=FactsRuleMarker.NONE,
    max_collateral_per_underlying_ratio=D("1.00"),
    max_total_csp_collateral_ratio=D("1.00"),
    min_cash_reserve_usd=D("0.00"),
    cc_max_lot_coverage_ratio=D("1.00"),
    roll_dte_window=14,
    option_quote_max_age_seconds=60,
    equity_quote_max_age_seconds=60,
    account_state_max_age_seconds=120,
)


def rules(**kw: Any) -> FactsRules:
    return RULES.model_copy(update=kw)


def instrument(occ: OccSymbol = PUT, multiplier: int | None = 100) -> OptionInstrument:
    return OptionInstrument(
        evidence_id=uid(1),
        as_of=T0 - timedelta(seconds=5),
        source_tool_call_ids=(TC,),
        occ_symbol=occ,
        broker_instrument_id="inst-1",
        underlying="XYZ",
        multiplier=multiplier,
    )


def quote(bid: str = "0.40", ask: str = "0.60", age: int = 10) -> Quote:
    return Quote(
        quote_id=uid(2),
        broker_instrument_id="inst-1",
        bid=D(bid),
        ask=D(ask),
        as_of=T0 - timedelta(seconds=age),
        source_tool_call_ids=(TC,),
    )


def underlying(price: str = "40.00", age: int = 10) -> UnderlyingQuote:
    return UnderlyingQuote(
        evidence_id=uid(3),
        as_of=T0 - timedelta(seconds=age),
        source_tool_call_ids=(TC,),
        symbol="XYZ",
        price=D(price),
    )


def snapshot(
    c: str | None = "7000",
    r: str | None = "3000",
    value: str | None = "10000",
    age: int = 30,
    verified: bool = True,
    quality: DataQuality | None = None,
) -> AccountSnapshot:
    cash = {
        "available_settled_cash_usd": D(c) if c is not None else None,
        "csp_reserved_cash_usd": D(r) if r is not None else None,
        "account_value_usd": D(value) if value is not None else None,
    }
    base = D(c) + D(r) if c is not None and r is not None else None
    fields = {**cash, "csp_cash_base_usd": base}
    gaps = tuple(
        Gap(field=f, kind=DataQuality.MISSING, detail="unmapped")
        for f, v in fields.items()
        if v is None
    )
    return AccountSnapshot(
        snapshot_id=uid(4),
        as_of=T0 - timedelta(seconds=age),
        retrieved_at=T0 - timedelta(seconds=age),
        tool_call_ids=(TC,),
        account_ref="****1234",
        agentic_verified=verified,
        csp_cash_base_evidence_ids=(TC,) if base is not None else (),
        positions_ref=None,
        open_orders_ref=None,
        tax_lots_ref=None,
        quality=quality or (DataQuality.MISSING if gaps else DataQuality.OK),
        gaps=gaps,
        **fields,
    )


def positions(
    shares: tuple[ShareHolding, ...] = (),
    shorts: tuple[ShortOptionHolding, ...] = (),
    age: int = 30,
) -> PositionsRead:
    return PositionsRead(
        evidence_id=uid(5),
        as_of=T0 - timedelta(seconds=age),
        source_tool_call_ids=(TC,),
        share_holdings=shares,
        short_options=shorts,
    )


def orders(*items: WorkingOrder, age: int = 30) -> OpenOrdersRead:
    return OpenOrdersRead(
        evidence_id=uid(6),
        as_of=T0 - timedelta(seconds=age),
        source_tool_call_ids=(TC,),
        orders=items,
    )


def short(
    occ: OccSymbol = PUT, qty: int = 3, inst: str = "inst-1", mult: int | None = 100
) -> ShortOptionHolding:
    return ShortOptionHolding(
        underlying="XYZ",
        occ_symbol=occ,
        broker_instrument_id=inst,
        short_quantity=qty,
        multiplier=mult,
    )


def working(
    occ: OccSymbol = OTHER_PUT,
    side: OrderSide = OrderSide.SELL_TO_OPEN,
    qty: int = 1,
    inst: str = "inst-2",
    mult: int | None = 100,
) -> WorkingOrder:
    return WorkingOrder(
        broker_order_ref=f"ord-{inst}-{side.value}",
        underlying="XYZ",
        occ_symbol=occ,
        broker_instrument_id=inst,
        side=side,
        unfilled_quantity=qty,
        multiplier=mult,
    )


def candidate() -> CandidateProvenance:
    return CandidateProvenance(
        candidate_ref="candidate:1", origin=CandidateOrigin.ROBINHOOD, underlying="XYZ"
    )


def position(
    occ: OccSymbol = PUT,
    qty: int = 3,
    credit: str | None = "1.00",
    roll_count: int | None = 0,
) -> PositionBookEntry:
    fills = (uid(20),) if credit is not None else ()
    unknown_rolls = roll_count is None
    return PositionBookEntry(
        position_id=uid(10),
        position_ref="position:1",
        underlying="XYZ",
        strategy=StrategyKind.CASH_SECURED_PUT
        if occ.right.value == "put"
        else StrategyKind.COVERED_CALL,
        current_instruments=(
            PositionInstrument(occ_symbol=occ, broker_instrument_id="inst-1", short_quantity=qty),
        ),
        entry_fill_ids=fills,
        entry_date=date(2026, 9, 1) if credit is not None else None,
        entry_weighted_credit=D(credit) if credit is not None else None,
        thesis="t",
        roll_count=roll_count,
        history_quality=DataQuality.MISSING if unknown_rolls else DataQuality.OK,
        gaps=(Gap(field="roll_count", kind=DataQuality.MISSING, detail="imported"),)
        if unknown_rolls
        else (),
    )


def funding(value: str = "1000", age: int = 30, quality: DataQuality = DataQuality.OK) -> Any:
    return SourcedValue[Dec](
        evidence_id=uid(7),
        value=D(value),
        source="robinhood",
        as_of=T0 - timedelta(seconds=age),
        retrieved_at=T0,
        tool_call_id=TC,
        quality=quality,
        reason=None if quality is DataQuality.OK else "stale read",
    )


def open_inputs(**kw: Any) -> FactInputs:
    base: dict[str, Any] = {
        "facts_id": uid(100),
        "facts_ref": "facts:1",
        "run_id": uid(101),
        "subject_ref": "candidate:1",
        "purpose": FactsPurpose.OPEN,
        "instrument": instrument(),
        "option_quote": quote(),
        "account": snapshot(),
        "positions": positions(),
        "open_orders": orders(),
        "candidate": candidate(),
    }
    base.update(kw)
    return FactInputs(**base)


def pos_inputs(purpose: FactsPurpose = FactsPurpose.CLOSE, **kw: Any) -> FactInputs:
    occ = kw.pop("occ", PUT)
    base: dict[str, Any] = {
        "facts_id": uid(100),
        "facts_ref": "facts:1",
        "run_id": uid(101),
        "subject_ref": "position:1",
        "purpose": purpose,
        "instrument": instrument(occ),
        "option_quote": quote(),
        "underlying_quote": underlying("45.00"),
        "account": snapshot(),
        "positions": positions(shorts=(short(occ),)),
        "open_orders": orders(),
        "position": position(occ),
        "debit_funding": funding(),
    }
    base.update(kw)
    return FactInputs(**base)


def facts(inputs: FactInputs, r: FactsRules = RULES, as_of: datetime = T0) -> DecisionFacts:
    return compute_decision_facts(inputs, r, as_of)


def metric(f: DecisionFacts, name: str) -> Decimal | None:
    m = f.metric(name)
    return None if m is None else m.value.value


def gap_kinds(f: DecisionFacts) -> dict[str, DataQuality]:
    return {g.field: g.kind for g in f.gaps}


# --------------------------------------------------------------------------------------------
# CSP sizing and cash accounting
# --------------------------------------------------------------------------------------------


def test_validation_md_cash_example_permits_one_5000_reservation() -> None:
    """C=7000, R=3000, B=10000, reserve 0, ratio 1.00: a 5000 reservation fits (VALIDATION.md)."""
    f = facts(open_inputs())
    assert f.initial_quantity == 1
    assert f.remaining_quantity == 1
    assert metric(f, "collateral_per_contract_usd") == D("5000")
    assert metric(f, "capacity_cash_contracts") == D(1)  # 7000 // 5000; not (7000-3000)
    assert metric(f, "capacity_total_csp_contracts") == D(1)  # (10000-3000) // 5000
    assert metric(f, "capacity_underlying_ratio_contracts") == D(2)  # 10000 // 5000
    assert metric(f, "capacity_order_cap_contracts") == D(10)
    assert f.metric("capacity_underlying_usd_contracts") is None  # "none" does not constrain
    assert f.quality is DataQuality.OK
    assert f.gaps == ()
    assert f.snapshot_ref == uid(4)
    assert f.quote_refs == (uid(2),)
    assert f.candidate_ref == "candidate:1"
    assert f.position_id is None
    assert f.rules_version == "5"


def test_opening_metrics_use_the_live_bid() -> None:
    f = facts(open_inputs())
    assert metric(f, "dte") == D(20)
    # 0.40 x 100 x 365 / (5000 x 20) = 14600 / 100000
    assert metric(f, "annualized_yield_on_collateral_ratio") == D("0.146")
    assert metric(f, "spread_ratio_of_mid") == D("0.4")  # 0.20 / 0.50
    yield_metric = f.metric("annualized_yield_on_collateral_ratio")
    assert yield_metric is not None and yield_metric.value.derivation is not None
    assert yield_metric.value.derivation.formula == "annualized_yield_on_collateral"
    assert set(yield_metric.value.derivation.input_evidence_ids) == {uid(1), uid(2)}
    assert yield_metric.value.as_of == T0 - timedelta(seconds=10)  # oldest input


def test_zero_capacity_is_a_real_zero() -> None:
    f = facts(open_inputs(account=snapshot(c="4000", r="0", value="4000")))
    assert f.initial_quantity == 0
    assert f.remaining_quantity == 0
    assert f.quality is DataQuality.OK


def test_negative_room_floors_to_zero() -> None:
    f = facts(open_inputs(), rules(max_total_csp_collateral_ratio=D("0.20")))
    assert metric(f, "capacity_total_csp_contracts") == D(0)  # 0.2 x 10000 - 3000 < 0
    assert f.initial_quantity == 0


def test_reserve_and_total_ratio_apply() -> None:
    f = facts(
        open_inputs(account=snapshot(c="20000", r="0", value="20000")),
        rules(min_cash_reserve_usd=D("6000"), max_total_csp_collateral_ratio=D("0.5")),
    )
    assert metric(f, "capacity_cash_contracts") == D(2)  # 14000 // 5000
    assert metric(f, "capacity_total_csp_contracts") == D(2)  # 10000 // 5000
    assert f.initial_quantity == 2


def test_order_cap_binds() -> None:
    big = snapshot(c="1000000", r="0", value="1000000")
    assert facts(open_inputs(account=big)).initial_quantity == 10


def test_none_caps_do_not_constrain() -> None:
    big = snapshot(c="60000", r="0", value="60000")
    r = rules(
        max_contracts_per_order=FactsRuleMarker.NONE,
        max_collateral_per_underlying_ratio=FactsRuleMarker.NONE,
        max_total_csp_collateral_ratio=FactsRuleMarker.NONE,
        min_cash_reserve_usd=FactsRuleMarker.NONE,
    )
    f = facts(open_inputs(account=big, positions=None, open_orders=None), r)
    assert f.initial_quantity == 12  # cash only: 60000 // 5000
    assert f.gaps == ()


@pytest.mark.parametrize(
    ("key", "marker"),
    [
        ("min_cash_reserve_usd", FactsRuleMarker.UNSET),
        ("max_contracts_per_order", FactsRuleMarker.AGENT_DISCRETION),
        ("max_total_csp_collateral_ratio", FactsRuleMarker.UNSET),
        ("max_collateral_per_underlying_ratio", FactsRuleMarker.UNSET),
        ("max_collateral_per_underlying_usd", FactsRuleMarker.UNSET),
    ],
)
def test_unset_rule_makes_quantity_unavailable(key: str, marker: FactsRuleMarker) -> None:
    f = facts(open_inputs(), rules(**{key: marker}))
    assert f.initial_quantity is None
    assert f.remaining_quantity is None
    kinds = gap_kinds(f)
    assert kinds[f"limits.{key}"] is DataQuality.MISSING
    assert kinds["initial_quantity"] is DataQuality.MISSING
    assert f.quality is DataQuality.MISSING


def test_per_underlying_caps_subtract_existing_csp_collateral_once() -> None:
    held = positions(shorts=(short(OTHER_PUT, qty=1, inst="inst-9"), short(CALL, qty=5)))
    open_orders = orders(
        working(),  # STO put 45 x 100 x 1 = 4500
        working(side=OrderSide.BUY_TO_CLOSE, inst="inst-9"),  # not a reservation of collateral
        working(occ=CALL, inst="inst-3"),  # a call: not CSP collateral
    )
    r = rules(max_collateral_per_underlying_usd=D("20000"))
    acct = snapshot(c="50000", r="4500", value="30000")
    f = facts(open_inputs(positions=held, open_orders=open_orders, account=acct), r)
    assert metric(f, "existing_underlying_csp_collateral_usd") == D("9000")
    assert metric(f, "capacity_underlying_usd_contracts") == D(2)  # 11000 // 5000
    assert metric(f, "capacity_underlying_ratio_contracts") == D(4)  # 21000 // 5000
    assert f.initial_quantity == 2


def test_other_underlyings_are_ignored() -> None:
    other = ShortOptionHolding(
        underlying="ABC",
        occ_symbol=OccSymbol.parse("ABC   261015P00050000"),
        broker_instrument_id="inst-a",
        short_quantity=9,
        multiplier=100,
    )
    f = facts(open_inputs(positions=positions(shorts=(other,))))
    assert metric(f, "existing_underlying_csp_collateral_usd") == D(0)


def test_unknown_existing_multiplier_is_a_gap() -> None:
    held = positions(shorts=(short(OTHER_PUT, qty=1, inst="inst-9", mult=None),))
    f = facts(open_inputs(positions=held))
    assert f.initial_quantity is None
    assert "existing_csp_collateral_usd" in gap_kinds(f)


def test_missing_account_value_is_a_gap() -> None:
    f = facts(open_inputs(account=snapshot(value=None)))
    assert f.initial_quantity is None
    assert "account_value_usd" in gap_kinds(f)


@pytest.mark.parametrize(
    ("acct", "name"),
    [
        (snapshot(c=None), "available_settled_cash_usd"),
        (snapshot(r=None), "csp_reserved_cash_usd"),
    ],
)
def test_missing_cash_fields_are_gaps(acct: AccountSnapshot, name: str) -> None:
    f = facts(open_inputs(account=acct))
    assert f.initial_quantity is None
    assert name in gap_kinds(f)


def test_account_snapshot_problems() -> None:
    missing = facts(open_inputs(account=None))
    assert gap_kinds(missing)["account_snapshot"] is DataQuality.MISSING
    assert missing.snapshot_ref is None
    unverified = facts(open_inputs(account=snapshot(verified=False)))
    assert gap_kinds(unverified)["account_snapshot"] is DataQuality.MISSING
    contradictory = facts(open_inputs(account=snapshot(quality=DataQuality.CONTRADICTORY)))
    assert gap_kinds(contradictory)["account_snapshot"] is DataQuality.CONTRADICTORY
    assert contradictory.quality is DataQuality.CONTRADICTORY
    stale = facts(open_inputs(account=snapshot(age=121)))
    assert gap_kinds(stale)["account_snapshot"] is DataQuality.STALE
    assert stale.quality is DataQuality.STALE
    for f in (missing, unverified, contradictory, stale):
        assert f.initial_quantity is None


def test_missing_or_stale_reads() -> None:
    no_positions = facts(open_inputs(positions=None))
    assert gap_kinds(no_positions)["positions"] is DataQuality.MISSING
    assert no_positions.initial_quantity is None
    stale_orders = facts(open_inputs(open_orders=orders(age=500)))
    assert gap_kinds(stale_orders)["open_orders"] is DataQuality.STALE
    no_orders = facts(open_inputs(open_orders=None))
    assert gap_kinds(no_orders)["open_orders"] is DataQuality.MISSING


def test_unverified_multiplier() -> None:
    f = facts(open_inputs(instrument=instrument(multiplier=None)))
    assert f.initial_quantity is None
    assert "multiplier" in gap_kinds(f)
    assert f.metric("annualized_yield_on_collateral_ratio") is None
    assert f.metric("collateral_per_contract_usd") is None


def test_established_target_and_confirmed_fills() -> None:
    big = snapshot(c="25000", r="0", value="25000")
    f = facts(
        open_inputs(
            account=big,
            established_target_quantity=3,
            confirmed_fills=(ConfirmedFill(evidence_id=uid(30), quantity=1),),
        )
    )
    assert f.initial_quantity == 3
    assert f.remaining_quantity == 2  # min(3 - 1, fresh capacity 5)
    assert uid(30) in f.input_evidence_ids
    capped = facts(
        open_inputs(
            established_target_quantity=3,
            confirmed_fills=(ConfirmedFill(evidence_id=uid(30), quantity=1),),
        )
    )
    assert capped.remaining_quantity == 1  # fresh capacity 1 binds


def test_fills_beyond_target_are_contradictory() -> None:
    f = facts(
        open_inputs(
            established_target_quantity=1,
            confirmed_fills=(ConfirmedFill(evidence_id=uid(30), quantity=2),),
        )
    )
    assert f.initial_quantity == 1
    assert f.remaining_quantity is None
    assert gap_kinds(f)["remaining_quantity"] is DataQuality.CONTRADICTORY


def test_fills_without_established_target_are_rejected() -> None:
    with pytest.raises(ValidationError, match="established target"):
        open_inputs(confirmed_fills=(ConfirmedFill(evidence_id=uid(30), quantity=1),))


@given(
    c1=st.integers(min_value=0, max_value=10**7),
    extra=st.integers(min_value=0, max_value=10**7),
    r=st.integers(min_value=0, max_value=10**6),
    strike=st.integers(min_value=1, max_value=2000),
)
def test_csp_sizing_is_monotone_in_cash_and_never_overspends(
    c1: int, extra: int, r: int, strike: int
) -> None:
    occ = OccSymbol.parse(f"XYZ   261015P{strike * 1000:08d}")
    rr = rules(max_contracts_per_order=FactsRuleMarker.NONE)

    def size(c: int) -> int:
        acct = snapshot(c=str(c), r=str(r), value=str(c + r))
        f = facts(open_inputs(account=acct, instrument=instrument(occ)), rr)
        assert f.initial_quantity is not None
        return f.initial_quantity

    low, high = size(c1), size(c1 + extra)
    assert low <= high
    assert D(high) * strike * 100 <= c1 + extra  # N <= C - reserve (reserve 0)


# --------------------------------------------------------------------------------------------
# CC sizing
# --------------------------------------------------------------------------------------------


def cc_inputs(**kw: Any) -> FactInputs:
    base: dict[str, Any] = {
        "instrument": instrument(CALL),
        "underlying_quote": underlying("40.00"),
        "positions": positions(
            shares=(
                ShareHolding(symbol="XYZ", quantity=350),
                ShareHolding(symbol="ABC", quantity=900),
            ),
            shorts=(short(OccSymbol.parse("XYZ   261120C00055000"), qty=1, inst="inst-8"),),
        ),
        "open_orders": orders(working(occ=OccSymbol.parse("XYZ   261120C00060000"), inst="inst-7")),
    }
    base.update(kw)
    return open_inputs(**base)


def test_cc_capacity_counts_short_calls_and_working_calls_once() -> None:
    f = facts(cc_inputs())
    assert metric(f, "uncovered_shares") == D(150)  # 350 - 100 - 100
    assert metric(f, "capacity_coverage_contracts") == D(1)
    assert f.initial_quantity == 1
    # 0.40 x 100 x 365 / (4000 x 20) = 14600 / 80000
    assert metric(f, "annualized_yield_on_collateral_ratio") == D("0.1825")
    assert metric(f, "collateral_per_contract_usd") == D("4000")


def test_cc_coverage_ratio_limits_and_markers() -> None:
    half = facts(cc_inputs(), rules(cc_max_lot_coverage_ratio=D("0.5")))
    assert half.initial_quantity == 0  # 175 - 200 < 0
    unlimited = facts(cc_inputs(), rules(cc_max_lot_coverage_ratio=FactsRuleMarker.NONE))
    assert unlimited.initial_quantity == 1
    unset = facts(cc_inputs(), rules(cc_max_lot_coverage_ratio=FactsRuleMarker.UNSET))
    assert unset.initial_quantity is None
    assert "limits.cc_max_lot_coverage_ratio" in gap_kinds(unset)


def test_cc_without_shares_is_zero() -> None:
    f = facts(cc_inputs(positions=positions(), open_orders=orders()))
    assert f.initial_quantity == 0


def test_cc_unknown_covering_multiplier_and_missing_price() -> None:
    held = positions(
        shares=(ShareHolding(symbol="XYZ", quantity=300),),
        shorts=(short(OccSymbol.parse("XYZ   261120C00055000"), qty=1, inst="inst-8", mult=None),),
    )
    f = facts(cc_inputs(positions=held, underlying_quote=None))
    assert f.initial_quantity is None
    kinds = gap_kinds(f)
    assert "covered_shares" in kinds
    assert kinds["underlying_quote"] is DataQuality.MISSING
    assert f.metric("annualized_yield_on_collateral_ratio") is None
    assert f.metric("spread_ratio_of_mid") is not None


def test_cc_unverified_multiplier() -> None:
    f = facts(cc_inputs(instrument=instrument(CALL, multiplier=None)))
    assert f.initial_quantity is None
    assert "multiplier" in gap_kinds(f)


# --------------------------------------------------------------------------------------------
# Quotes, freshness, DTE, annualization
# --------------------------------------------------------------------------------------------


def test_stale_or_missing_quote() -> None:
    stale = facts(open_inputs(option_quote=quote(age=61)))
    assert gap_kinds(stale)["option_quote"] is DataQuality.STALE
    assert stale.metric("spread_ratio_of_mid") is None
    assert stale.metric("annualized_yield_on_collateral_ratio") is None
    assert stale.initial_quantity == 1  # CSP sizing does not depend on the quote
    missing = facts(open_inputs(option_quote=None))
    assert gap_kinds(missing)["option_quote"] is DataQuality.MISSING
    assert missing.quote_refs == ()


def test_freshness_boundary_is_inclusive() -> None:
    assert "option_quote" not in gap_kinds(facts(open_inputs(option_quote=quote(age=60))))


def test_freshness_rule_markers() -> None:
    none_rule = rules(option_quote_max_age_seconds=FactsRuleMarker.NONE)
    old = facts(open_inputs(option_quote=quote(age=100000)), none_rule)
    assert "option_quote" not in gap_kinds(old)
    future = facts(open_inputs(option_quote=quote(age=-5)), none_rule)
    assert gap_kinds(future)["option_quote"] is DataQuality.STALE
    unset = facts(open_inputs(), rules(option_quote_max_age_seconds=FactsRuleMarker.UNSET))
    kinds = gap_kinds(unset)
    assert kinds["option_quote"] is DataQuality.STALE
    assert kinds["data_quality.freshness.option_quote_max_age_seconds"] is DataQuality.MISSING


def test_dte_uses_new_york_calendar_date() -> None:
    # 2026-09-26 02:00 UTC is still 2026-09-25 in New York: DTE stays 20.
    late = datetime(2026, 9, 26, 2, 0, tzinfo=UTC)
    stale_ok = rules(
        option_quote_max_age_seconds=FactsRuleMarker.NONE,
        account_state_max_age_seconds=FactsRuleMarker.NONE,
    )
    assert metric(facts(open_inputs(), stale_ok, late), "dte") == D(20)
    # 2026-09-26 05:00 UTC is 2026-09-26 in New York: DTE 19.
    next_day = datetime(2026, 9, 26, 5, 0, tzinfo=UTC)
    assert metric(facts(open_inputs(), stale_ok, next_day), "dte") == D(19)


def test_zero_dte_has_no_annualized_yield_and_expired_has_no_dte() -> None:
    today = OccSymbol.parse("XYZ   260925P00050000")
    f = facts(open_inputs(instrument=instrument(today)))
    assert metric(f, "dte") == D(0)
    assert "annualized_yield_on_collateral_ratio" in gap_kinds(f)
    expired = OccSymbol.parse("XYZ   260924P00050000")
    g = facts(open_inputs(instrument=instrument(expired)))
    assert g.metric("dte") is None
    assert "dte" in gap_kinds(g)
    assert g.metric("annualized_yield_on_collateral_ratio") is None


def test_naive_as_of_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_decision_facts(open_inputs(), RULES, datetime(2026, 9, 25, 15, 0))  # noqa: DTZ001


def test_formula_versions_and_evidence_are_complete() -> None:
    f = facts(open_inputs())
    formulas = {fv.formula: fv.version for fv in f.formula_versions}
    assert formulas == {
        "annualized_yield_on_collateral": "1",
        "collateral_per_contract": "1",
        "csp_capacity": "1",
        "decision_quantity": "1",
        "dte_calendar_days": "1",
        "spread_ratio_of_mid": "1",
    }
    assert f.input_evidence_ids == (uid(1), uid(2), uid(4), uid(5), uid(6))
    assert f.observed_at == T0


def test_same_inputs_same_facts() -> None:
    assert facts(open_inputs()) == facts(open_inputs())


# --------------------------------------------------------------------------------------------
# CLOSE / MANAGE (ROLL close leg)
# --------------------------------------------------------------------------------------------


def test_close_with_discretionary_quantity() -> None:
    f = facts(pos_inputs(close_quantity=2))
    assert f.initial_quantity == 2
    assert f.remaining_quantity == 2
    assert f.close_quantity == 2
    assert metric(f, "available_short_quantity") == D(3)
    assert metric(f, "capacity_debit_funding_contracts") == D(16)  # 1000 // (0.60 x 100)
    assert metric(f, "close_capacity_contracts") == D(3)
    assert f.position_id == uid(10)
    assert f.candidate_ref is None
    assert f.quality is DataQuality.OK


def test_close_without_quantity_is_unavailable() -> None:
    f = facts(pos_inputs())
    assert f.initial_quantity is None
    assert f.remaining_quantity is None
    assert gap_kinds(f)["close_quantity"] is DataQuality.MISSING
    assert metric(f, "close_capacity_contracts") == D(3)  # capacity is still reported


def test_close_quantity_above_short_quantity_is_contradictory() -> None:
    f = facts(pos_inputs(close_quantity=4))
    assert f.initial_quantity is None
    assert gap_kinds(f)["close_quantity"] is DataQuality.CONTRADICTORY


def test_manage_targets_the_whole_short_quantity() -> None:
    f = facts(pos_inputs(FactsPurpose.MANAGE))
    assert f.initial_quantity == 3
    assert f.remaining_quantity == 3


def test_manage_after_partial_close_fill() -> None:
    f = facts(
        pos_inputs(
            FactsPurpose.MANAGE,
            positions=positions(shorts=(short(qty=1),)),
            established_target_quantity=3,
            confirmed_fills=(ConfirmedFill(evidence_id=uid(31), quantity=2),),
        )
    )
    assert f.initial_quantity == 3
    assert f.remaining_quantity == 1


def test_close_fills_beyond_target() -> None:
    f = facts(
        pos_inputs(
            close_quantity=1,
            positions=positions(shorts=(short(qty=1),)),
            confirmed_fills=(ConfirmedFill(evidence_id=uid(31), quantity=2),),
        )
    )
    assert gap_kinds(f)["remaining_quantity"] is DataQuality.CONTRADICTORY


def test_broker_short_quantity_disagreeing_with_book_is_contradictory() -> None:
    f = facts(pos_inputs(FactsPurpose.MANAGE, positions=positions(shorts=(short(qty=2),))))
    assert gap_kinds(f)["short_quantity"] is DataQuality.CONTRADICTORY
    assert f.initial_quantity == 3
    assert f.remaining_quantity is None
    absent = facts(pos_inputs(FactsPurpose.MANAGE, positions=positions()))
    assert gap_kinds(absent)["short_quantity"] is DataQuality.CONTRADICTORY


def test_working_buy_to_close_reduces_available_short() -> None:
    btc = working(occ=PUT, side=OrderSide.BUY_TO_CLOSE, inst="inst-1")
    other = working(occ=PUT, side=OrderSide.SELL_TO_OPEN, inst="inst-1")
    f = facts(pos_inputs(FactsPurpose.MANAGE, open_orders=orders(btc, other)))
    assert metric(f, "available_short_quantity") == D(2)
    assert f.remaining_quantity == 2


def test_debit_funding_limits_close() -> None:
    f = facts(pos_inputs(close_quantity=2, debit_funding=funding("100")))
    assert metric(f, "capacity_debit_funding_contracts") == D(1)  # 100 // 60
    assert f.remaining_quantity == 1
    at_limit = facts(
        pos_inputs(close_quantity=2, debit_funding=funding("100"), limit_price=D("0.50"))
    )
    assert metric(at_limit, "capacity_debit_funding_contracts") == D(2)  # 100 // 50
    assert at_limit.limit_price == D("0.50")


def test_debit_funding_unavailable_leaves_short_capacity_and_a_gap() -> None:
    absent = facts(pos_inputs(close_quantity=2, debit_funding=None))
    assert gap_kinds(absent)["debit_funding_usd"] is DataQuality.MISSING
    assert absent.remaining_quantity == 2
    unusable = facts(pos_inputs(close_quantity=2, debit_funding=funding(quality=DataQuality.STALE)))
    assert gap_kinds(unusable)["debit_funding_usd"] is DataQuality.STALE
    old = facts(pos_inputs(close_quantity=2, debit_funding=funding(age=121)))
    assert gap_kinds(old)["debit_funding_usd"] is DataQuality.STALE
    no_price = facts(pos_inputs(close_quantity=2, option_quote=None))
    assert "debit_funding_capacity" in gap_kinds(no_price)
    no_mult = facts(pos_inputs(close_quantity=2, instrument=instrument(multiplier=None)))
    assert "debit_funding_capacity" in gap_kinds(no_mult)
    for f in (unusable, old, no_price, no_mult):
        assert f.remaining_quantity == 2


def test_close_needs_fresh_reads() -> None:
    f = facts(pos_inputs(close_quantity=2, positions=None))
    assert f.initial_quantity == 2
    assert f.remaining_quantity is None
    assert "positions" in gap_kinds(f)


@pytest.mark.parametrize(
    ("occ", "price", "expected"),
    [
        (PUT, "50.00", D(1)),  # at the strike: tested
        (PUT, "49.99", D(1)),
        (PUT, "50.01", D(0)),
        (CALL, "50.00", D(1)),
        (CALL, "50.01", D(1)),
        (CALL, "49.99", D(0)),
    ],
)
def test_tested_boundaries(occ: OccSymbol, price: str, expected: Decimal) -> None:
    f = facts(pos_inputs(FactsPurpose.MANAGE, occ=occ, underlying_quote=underlying(price)))
    assert metric(f, "tested_flag") == expected


def test_tested_needs_a_fresh_underlying_price() -> None:
    f = facts(pos_inputs(FactsPurpose.MANAGE, underlying_quote=underlying(age=61)))
    assert f.metric("tested_flag") is None
    assert gap_kinds(f)["underlying_quote"] is DataQuality.STALE


def test_roll_window_flag() -> None:
    assert metric(facts(pos_inputs(FactsPurpose.MANAGE)), "in_roll_dte_window_flag") == D(0)
    edge = facts(pos_inputs(FactsPurpose.MANAGE), rules(roll_dte_window=20))
    assert metric(edge, "in_roll_dte_window_flag") == D(1)  # DTE 20 <= 20
    unlimited = facts(pos_inputs(FactsPurpose.MANAGE), rules(roll_dte_window=FactsRuleMarker.NONE))
    assert metric(unlimited, "in_roll_dte_window_flag") == D(1)
    unset = facts(pos_inputs(FactsPurpose.MANAGE), rules(roll_dte_window=FactsRuleMarker.UNSET))
    assert unset.metric("in_roll_dte_window_flag") is None
    assert "management.roll_dte_window" in gap_kinds(unset)


def test_captured_ratio_and_remaining_yield_for_a_put() -> None:
    f = facts(pos_inputs(FactsPurpose.MANAGE))
    assert metric(f, "captured_ratio") == D("0.4")  # (1.00 - 0.60) / 1.00
    # 0.60 x 100 x 365 / (5000 x 20) = 21900 / 100000
    assert metric(f, "remaining_yield_ratio") == D("0.219")
    captured = f.metric("captured_ratio")
    assert captured is not None and captured.value.derivation is not None
    assert uid(20) in captured.value.derivation.input_evidence_ids
    assert uid(20) in f.input_evidence_ids


def test_remaining_yield_for_a_call_uses_share_price() -> None:
    f = facts(pos_inputs(FactsPurpose.MANAGE, occ=CALL, underlying_quote=underlying("40.00")))
    # 0.60 x 100 x 365 / (4000 x 20) = 21900 / 80000
    assert metric(f, "remaining_yield_ratio") == D("0.27375")


def test_captured_ratio_gaps() -> None:
    no_history = facts(
        pos_inputs(FactsPurpose.MANAGE, position=position(credit=None, roll_count=None))
    )
    assert "entry_weighted_credit" in gap_kinds(no_history)
    assert no_history.metric("captured_ratio") is None
    unknown_rolls = facts(pos_inputs(FactsPurpose.MANAGE, position=position(roll_count=None)))
    assert "captured_ratio" in gap_kinds(unknown_rolls)
    no_quote = facts(pos_inputs(FactsPurpose.MANAGE, option_quote=None))
    assert no_quote.metric("captured_ratio") is None
    assert no_quote.metric("remaining_yield_ratio") is None
    assert no_quote.metric("spread_ratio_of_mid") is None


def test_position_formulas() -> None:
    f = facts(pos_inputs(close_quantity=1))
    assert {fv.formula for fv in f.formula_versions} == {
        "captured_ratio",
        "close_capacity",
        "collateral_per_contract",
        "decision_quantity",
        "dte_calendar_days",
        "remaining_yield",
        "roll_dte_window",
        "spread_ratio_of_mid",
        "tested",
    }


# --------------------------------------------------------------------------------------------
# FactInputs validation
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        (
            {"option_quote": quote().model_copy(update={"broker_instrument_id": "x"})},
            "option quote",
        ),
        (
            {"underlying_quote": underlying().model_copy(update={"symbol": "ABC"})},
            "underlying quote",
        ),
        ({"close_quantity": 1}, "CLOSE-purpose"),
        ({"position": position()}, "targets a candidate"),
        ({"candidate": None}, "targets a candidate"),
        ({"subject_ref": "candidate:2"}, "subject_ref"),
        (
            {"candidate": candidate().model_copy(update={"underlying": "ABC"})},
            "candidate underlying",
        ),
        ({"debit_funding": funding()}, "closing legs"),
        (
            {
                "established_target_quantity": 2,
                "confirmed_fills": (
                    ConfirmedFill(evidence_id=uid(30), quantity=1),
                    ConfirmedFill(evidence_id=uid(30), quantity=1),
                ),
            },
            "duplicate",
        ),
    ],
)
def test_open_input_validation(kw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        open_inputs(**kw)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"position": None}, "targets a position"),
        ({"candidate": candidate()}, "targets a position"),
        ({"subject_ref": "position:2"}, "subject_ref"),
        ({"position": position().model_copy(update={"underlying": "ABC"})}, "position underlying"),
        (
            {
                "instrument": instrument().model_copy(update={"broker_instrument_id": "inst-x"}),
                "option_quote": None,
            },
            "current contract",
        ),
        ({"occ": CALL, "position": position(PUT)}, "strategy"),
        ({"established_target_quantity": 2}, "discretionary close_quantity"),
    ],
)
def test_position_input_validation(kw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        pos_inputs(**kw)


def test_observation_needs_a_tool_call() -> None:
    with pytest.raises(ValidationError, match="source tool call"):
        OptionInstrument(
            evidence_id=uid(1),
            as_of=T0,
            source_tool_call_ids=(),
            occ_symbol=PUT,
            broker_instrument_id="inst-1",
            underlying="XYZ",
            multiplier=100,
        )


def test_float_prices_are_rejected() -> None:
    with pytest.raises(ValidationError):
        open_inputs(limit_price=0.5)
