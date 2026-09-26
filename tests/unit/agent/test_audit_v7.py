"""V7 Hard caps: pass / violation / unverifiable, live and dry run."""

from decimal import Decimal
from uuid import uuid4

from test_audit_builders import (
    BTC,
    NEW_PUT,
    PUT,
    STO,
    T0,
    LegSpec,
    Scenario,
    live_csp,
    off_csp,
    only,
    outcomes,
    reasons,
    rules_with,
)

from wheelta_robinhood_agent.agent.audit import DayHistory, WorkingOrderObservation, check_v7
from wheelta_robinhood_agent.config.rules import RuleMarker
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditOutcome,
    DecisionAction,
    ExecutionMode,
    OrderSide,
)
from wheelta_robinhood_agent.domain.run_record import DecisionOutputStatus

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE
TBD, NONE = RuleMarker.TBD, RuleMarker.NONE


def test_live_csp_passes_every_cap() -> None:
    s = Scenario()
    live_csp(s)
    f = check_v7(s.ctx())
    assert outcomes(f) == [P, P, P, P, P, P]
    assert only(f, "4").observed_value == "R=0 N=5000 B=10000"


def test_no_attempts_counts_zero_lineages() -> None:
    assert outcomes(check_v7(Scenario().ctx())) == [P, P]


def test_per_order_cap() -> None:
    s = Scenario()
    live_csp(s, qty=11, cash="100000")
    assert outcomes(check_v7(s.ctx()), "1") == [V]
    s.rules = rules_with(limits={"max_contracts_per_order": NONE})
    assert outcomes(check_v7(s.ctx()), "1") == [P]
    s.rules = rules_with(limits={"max_contracts_per_order": TBD})
    assert reasons(check_v7(s.ctx()), "1") == ["rule_tbd"]


def test_unknown_quantity_and_instrument() -> None:
    s = Scenario()
    s.state(90)
    call = s.place(100, "put-9", PUT, STO, 1, "1.10")
    s.orders.pop(call.identity.tool_call_id)
    s.place_unrecorded = {call.identity.tool_call_id}
    f = check_v7(s.ctx())
    assert f == () or all(x.sub_item != "1" for x in f)  # side unknown: not an STO attempt
    s2 = Scenario()
    s2.state(90)
    s2.place(100, "put-9", PUT, STO, 1, "1.10")
    f2 = check_v7(s2.ctx())
    assert reasons(f2, "3-4") == ["missing_evidence"]


def test_new_positions_per_run_counts_lineages_not_steps() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90, cash="1000000")
    for i in range(6):
        first = s.place(100 + i * 100, "put-1", PUT, STO, 2, "1.15")
        s.status(first, 110 + i * 100, AttemptStatus.CANCELLED, filled=1)
        second = s.place(150 + i * 100, "put-1", PUT, STO, 1, "1.10")
        s.status(second, 160 + i * 100, AttemptStatus.FILLED, filled=1)
        s.decide(
            DecisionAction.OPEN_CSP,
            LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second]),
        )
    f = check_v7(s.ctx())
    run = [x for x in f if x.sub_item == "2" and x.rule_key == "limits.max_new_positions_per_run"]
    assert [x.outcome for x in run] == [V]
    assert run[0].observed_value == "known=6 unknown=0"


def test_roll_replacement_does_not_count_and_unfilled_open_does_not_count() -> None:
    s = Scenario()
    s.rules = rules_with(limits={"max_new_positions_per_run": 0})
    s.inst("put-new", NEW_PUT)
    s.state(90)
    close = s.place(100, "put-1", PUT, BTC, 1, "0.50")
    repl = s.place(200, "put-new", NEW_PUT, STO, 1, "1.10")
    s.status(repl, 210, AttemptStatus.FILLED, filled=1)
    s.decide(
        DecisionAction.ROLL,
        LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1, [close], leg_ref="leg:c"),
        LegSpec(OrderSide.SELL_TO_OPEN, NEW_PUT, "put-new", 1, [repl], depends_on="leg:c"),
        target_ref="position:1",
    )
    unfilled = s.place(300, "put-new", NEW_PUT, STO, 1, "1.10")
    s.status(unfilled, 310, AttemptStatus.CANCELLED, filled=0)
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, NEW_PUT, "put-new", 1, [unfilled])
    )
    f = check_v7(s.ctx())
    run = [x for x in f if x.rule_key == "limits.max_new_positions_per_run"]
    assert [x.outcome for x in run] == [P]


def test_unattributed_fill_makes_count_unverifiable() -> None:
    s = Scenario()
    s.rules = rules_with(limits={"max_new_positions_per_run": 1})
    live_csp(s)
    s.inst("put-2", NEW_PUT)
    extra = s.place(300, "put-2", NEW_PUT, STO, 1, "1.10")
    s.status(extra, 310, AttemptStatus.FILLED, filled=1)
    f = check_v7(s.ctx())
    run = [x for x in f if x.rule_key == "limits.max_new_positions_per_run"]
    assert reasons(tuple(run)) == ["lineage_unknown"]


def test_day_cap_uses_earlier_lineages() -> None:
    s = Scenario()
    live_csp(s)
    s.day = DayHistory(
        trading_date=T0.date(), prior_new_lineage_ids=tuple(uuid4() for _ in range(10))
    )
    day = [x for x in check_v7(s.ctx()) if x.rule_key == "limits.max_new_positions_per_day"]
    assert [x.outcome for x in day] == [V]
    s.day = None
    day = [x for x in check_v7(s.ctx()) if x.rule_key == "limits.max_new_positions_per_day"]
    assert reasons(tuple(day)) == ["missing_evidence"]
    s.rules = rules_with(
        limits={"max_new_positions_per_day": NONE, "max_new_positions_per_run": TBD}
    )
    f = [x for x in check_v7(s.ctx()) if x.sub_item == "2"]
    assert [x.outcome for x in f] == [U, P]


def test_dry_run_counts_unconditional_intents() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.rules = rules_with(limits={"max_new_positions_per_run": 1})
    s.state(10, cash="100000")
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-2", conditional=True, depends_on=None))
    run = [x for x in check_v7(s.ctx()) if x.rule_key == "limits.max_new_positions_per_run"]
    assert [x.outcome for x in run] == [P]
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-3"))
    run = [x for x in check_v7(s.ctx()) if x.rule_key == "limits.max_new_positions_per_run"]
    assert [x.outcome for x in run] == [V]


def test_per_underlying_caps() -> None:
    s = Scenario()
    s.inst("put-0", PUT)
    working = WorkingOrderObservation(
        broker_order_id="w",
        broker_instrument_id="put-0",
        side_raw=STO,
        quantity=1,
        filled_quantity=0,
        owned=False,
    )
    s.inst("put-1", PUT)
    s.state(90, value="12000", shorts=(("put-0", 1),), working=(working,))
    q = s.quote("put-1", 95)
    p = s.place(100, "put-1", PUT, STO, 1, "1.10", quotes=(q,))
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [p]))
    f = only(check_v7(s.ctx()), "3")
    assert f.outcome is V and "15000" in f.detail
    s.rules = rules_with(
        limits={
            "max_collateral_per_underlying_ratio": NONE,
            "max_collateral_per_underlying_usd": Decimal("20000"),
        }
    )
    assert outcomes(check_v7(s.ctx()), "3") == [P]
    s.rules = rules_with(limits={"max_collateral_per_underlying_ratio": NONE})
    assert outcomes(check_v7(s.ctx()), "3") == [P]
    s.rules = rules_with(limits={"max_collateral_per_underlying_ratio": TBD})
    assert reasons(check_v7(s.ctx()), "3") == ["rule_tbd"]


def test_per_underlying_unknowns() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90, value=None)
    s.place(100, "put-1", PUT, STO, 1, "1.10")
    assert reasons(check_v7(s.ctx()), "3") == ["unmapped_broker_field"]
    s.state(95, shorts=None)
    assert reasons(check_v7(s.ctx()), "3") == ["unmapped_broker_field"]
    s.states.clear()
    f = check_v7(s.ctx())
    assert reasons(f, "3") == ["no_pre_order_state"]
    assert reasons(f, "4") == ["no_pre_order_state"]
    s.state(96, shorts=(("unknown", 1),))
    assert reasons(check_v7(s.ctx()), "3") == ["unmapped_broker_field"]
    working = WorkingOrderObservation(
        broker_order_id="w",
        broker_instrument_id="put-1",
        side_raw=None,
        quantity=1,
        filled_quantity=0,
        owned=False,
    )
    s.state(97, working=(working,))
    assert reasons(check_v7(s.ctx()), "3") == ["unmapped_broker_field"]
    s.state(98, working=(working.model_copy(update={"side_raw": STO, "filled_quantity": None}),))
    assert reasons(check_v7(s.ctx()), "3") == ["unmapped_broker_field"]
    s.state(99, working=(working.model_copy(update={"side_raw": BTC}),))
    assert outcomes(check_v7(s.ctx()), "3") == [P]


def test_total_ratio() -> None:
    s = Scenario()
    live_csp(s, qty=2, cash="5000", reserved="5000")
    assert outcomes(check_v7(s.ctx()), "4") == [V]
    s.rules = rules_with(limits={"max_total_csp_collateral_ratio": NONE})
    assert outcomes(check_v7(s.ctx()), "4") == [P]
    s.rules = rules_with(limits={"max_total_csp_collateral_ratio": TBD})
    assert reasons(check_v7(s.ctx()), "4") == ["rule_tbd"]
    s2 = Scenario()
    s2.inst("put-1", PUT)
    s2.state(90, reserved=None)
    s2.place(100, "put-1", PUT, STO, 1, "1.10")
    assert reasons(check_v7(s2.ctx()), "4") == ["unmapped_broker_field"]


def test_dry_run_total_ratio_two_intents_and_conditional() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.state(10, cash="7000", reserved="0", value="100000")
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-2"))
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-3", conditional=True, depends_on=None))
    f = check_v7(s.ctx())
    assert outcomes(f, "4")[:2] == [P, V]
    assert reasons(f, "4") == ["dry_run_no_execution"]
    assert reasons(f, "3") == ["dry_run_no_execution"]
    s.output_status = DecisionOutputStatus.MISSING
    assert reasons(check_v7(s.ctx())) == ["missing_final_output"]


def test_daily_drawdown_breaker() -> None:
    s = Scenario()
    live_csp(s)
    s.rules = rules_with(circuit_breakers={"max_daily_drawdown_usd": TBD})
    assert reasons(check_v7(s.ctx()), "5") == ["rule_tbd"]
    s.rules = rules_with(circuit_breakers={"max_daily_drawdown_usd": Decimal("500")})
    assert reasons(check_v7(s.ctx()), "5") == ["open_definition"]


def test_filled_is_unknown_for_dispatched_place_without_broker_order() -> None:
    import dataclasses

    from wheelta_robinhood_agent.agent.audit._common import live_attempts
    from wheelta_robinhood_agent.agent.audit.v7_caps import _filled

    s = Scenario()
    live_csp(s)
    (attempt,) = live_attempts(s.ctx())
    assert attempt.dispatched
    assert _filled(dataclasses.replace(attempt, order=None)) is None
    undispatched = dataclasses.replace(attempt, order=None, place_call=None)
    assert _filled(undispatched) == 0
