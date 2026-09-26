"""V1 Strategy identity: pass / violation / unverifiable, live and dry run."""

from decimal import Decimal

from test_audit_builders import (
    BTC,
    CALL,
    PUT,
    PUT2,
    STO,
    LegSpec,
    Scenario,
    live_csp,
    off_csp,
    only,
    outcomes,
    reasons,
    rules_with,
)

from wheelta_robinhood_agent.agent.audit import WorkingOrderObservation, check_v1
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


def test_live_csp_passes() -> None:
    s = Scenario()
    live_csp(s)
    f = check_v1(s.ctx())
    assert outcomes(f, "1") == [P]
    assert outcomes(f, "3") == [P]
    assert only(f, "3").observed_value == "N=5000 C=10000"


def test_no_attempts_live_passes() -> None:
    assert outcomes(check_v1(Scenario().ctx())) == [P]


def test_non_limit_or_bad_side_is_violation() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    s.place(100, "put-1", PUT, STO, 1, "1.10", order_type="market")
    s.place(200, "put-1", PUT, "buy_to_open", 1, "1.10")
    f = check_v1(s.ctx())
    assert outcomes(f, "1") == [V, V]


def test_csp_exceeding_cash_is_violation() -> None:
    s = Scenario()
    live_csp(s, qty=3)
    assert outcomes(check_v1(s.ctx()), "3") == [V]


def test_reserve_rule_applies() -> None:
    s = Scenario()
    s.rules = rules_with(limits={"min_cash_reserve_usd": Decimal("6000")})
    live_csp(s)
    assert outcomes(check_v1(s.ctx()), "3") == [V]
    s.rules = rules_with(limits={"min_cash_reserve_usd": RuleMarker.NONE})
    assert outcomes(check_v1(s.ctx()), "3") == [P]
    s.rules = rules_with(limits={"min_cash_reserve_usd": RuleMarker.TBD})
    assert reasons(check_v1(s.ctx()), "3") == ["rule_tbd"]
    s.rules = rules_with(limits={"min_cash_reserve_usd": RuleMarker.AGENT_DISCRETION})
    assert reasons(check_v1(s.ctx()), "3") == ["open_definition"]


def test_validation_example_without_double_subtraction() -> None:
    """C=7000, R=3000 after an earlier put: a 5000 reservation fits (VALIDATION.md example)."""
    s = Scenario()
    live_csp(s, cash="7000", reserved="3000")
    assert outcomes(check_v1(s.ctx()), "3") == [P]


def test_two_csps_against_refreshed_balances() -> None:
    s = Scenario()
    first, _ = live_csp(s)
    s.inst("put-2", PUT)
    state = s.state(290, cash="5000", reserved="5000", reflects=(first,))
    q = s.quote("put-2", 295)
    second = s.place(300, "put-2", PUT, STO, 1, "1.10", quotes=(q,), snapshot=state)
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-2", 1, [second]))
    assert outcomes(check_v1(s.ctx()), "3") == [P, P]


def test_state_not_reflecting_prior_mutation_is_unverifiable() -> None:
    s = Scenario()
    live_csp(s)
    s.inst("put-2", PUT)
    s.state(290, cash="5000", reserved="5000")  # does not reflect the first placement
    s.place(300, "put-2", PUT, STO, 1, "1.10")
    assert reasons(check_v1(s.ctx()), "3") == ["state_missing_mutation"]


def test_pre_order_state_problems_are_unverifiable() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.place(100, "put-1", PUT, STO, 1, "1.10")
    assert reasons(check_v1(s.ctx()), "3") == ["no_pre_order_state"]
    s.state(99, as_of=-500)
    assert reasons(check_v1(s.ctx()), "3") == ["stale_state"]
    s.state(99.5, verified=False)
    assert reasons(check_v1(s.ctx()), "3") == ["unverified_account"]
    s.state(99.7, cash=None)
    assert reasons(check_v1(s.ctx()), "3") == ["unmapped_broker_field"]
    s.rules = rules_with(freshness={"account_state_max_age_seconds": RuleMarker.TBD})
    assert reasons(check_v1(s.ctx()), "3") == ["rule_tbd"]
    s.rules = rules_with(freshness={"account_state_max_age_seconds": RuleMarker.NONE})
    assert reasons(check_v1(s.ctx()), "3") == ["unmapped_broker_field"]


def test_missing_instrument_and_arguments_are_unverifiable() -> None:
    s = Scenario()
    s.state(90)
    s.place(100, "put-9", PUT, STO, 1, "1.10")
    f = check_v1(s.ctx())
    assert reasons(f, "3-4") == ["missing_evidence"]
    s2 = Scenario()
    s2.inst("put-1", PUT)
    s2.state(90)
    call = s2.place(100, "put-1", PUT, STO, 1, "1.10")
    s2.orders.pop(call.identity.tool_call_id)
    assert reasons(check_v1(s2.ctx()), "1") == ["missing_evidence"]


def test_buy_to_close_quantity() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    other = WorkingOrderObservation(
        broker_order_id="o-x",
        broker_instrument_id="put-1",
        side_raw=BTC,
        quantity=1,
        filled_quantity=0,
        owned=False,
    )
    s.state(90, shorts=(("put-1", 2),), working=(other,))
    s.place(100, "put-1", PUT, BTC, 1, "0.50")
    assert outcomes(check_v1(s.ctx()), "2") == [P]
    s.place(200, "put-1", PUT, BTC, 2, "0.50")
    s.state(190, shorts=(("put-1", 2),), working=(other,), reflects=tuple(s.calls[-3:]))
    f = check_v1(s.ctx())
    assert outcomes(f, "2")[-1] in (V, U)


def test_buy_to_close_violation_and_unknowns() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90, shorts=(("put-1", 1),))
    s.place(100, "put-1", PUT, BTC, 2, "0.50")
    assert outcomes(check_v1(s.ctx()), "2") == [V]
    s.state(95, shorts=None)
    assert reasons(check_v1(s.ctx()), "2") == ["unmapped_broker_field"]
    unknown_fill = WorkingOrderObservation(
        broker_order_id="o",
        broker_instrument_id="put-1",
        side_raw=BTC,
        quantity=1,
        filled_quantity=None,
        owned=False,
    )
    s.state(96, shorts=(("put-1", 1),), working=(unknown_fill,))
    assert reasons(check_v1(s.ctx()), "2") == ["unmapped_broker_field"]
    no_side = unknown_fill.model_copy(update={"side_raw": None, "filled_quantity": 0})
    s.state(97, shorts=(("put-1", 1),), working=(no_side,))
    assert reasons(check_v1(s.ctx()), "2") == ["unmapped_broker_field"]
    other_side = unknown_fill.model_copy(update={"side_raw": STO, "filled_quantity": 0})
    other_contract = other_side.model_copy(update={"broker_instrument_id": "zzz"})
    s.state(98, shorts=(("put-1", 3),), working=(other_side, other_contract))
    assert outcomes(check_v1(s.ctx()), "2") == [P]
    s.state(99, verified=False)
    assert reasons(check_v1(s.ctx()), "2") == ["unverified_account"]


def test_covered_call_shares() -> None:
    s = Scenario()
    s.inst("call-1", CALL)
    s.inst("call-0", CALL)
    s.inst("put-1", PUT)
    s.state(90, shares=(("AAPL", 200, 0), ("MSFT", 100, 0)), shorts=(("call-0", 1), ("put-1", 1)))
    s.place(100, "call-1", CALL, STO, 1, "1.10")
    f = check_v1(s.ctx())
    assert outcomes(f, "4") == [P]
    assert only(f, "4").observed_value == "needed=100 free=100"


def test_covered_call_reservations() -> None:
    s = Scenario()
    s.inst("call-1", CALL)
    working = WorkingOrderObservation(
        broker_order_id="w",
        broker_instrument_id="call-1",
        side_raw=STO,
        quantity=1,
        filled_quantity=0,
        owned=False,
    )
    s.state(90, shares=(("AAPL", 100, 0),), working=(working,))
    s.place(100, "call-1", CALL, STO, 1, "1.10")
    assert outcomes(check_v1(s.ctx()), "4") == [V]
    s.state(91, shares=(("AAPL", 100, None),))
    assert reasons(check_v1(s.ctx()), "4") == ["unmapped_broker_field"]
    s.state(92, shares=None)
    assert reasons(check_v1(s.ctx()), "4") == ["unmapped_broker_field"]
    s.state(93, shares=(("AAPL", 100, 0),), shorts=(("unknown-inst", 1),))
    assert reasons(check_v1(s.ctx()), "4") == ["unmapped_broker_field"]
    for bad in (
        working.model_copy(update={"broker_instrument_id": "unknown"}),
        working.model_copy(update={"filled_quantity": None}),
    ):
        s.state(94 + len(s.states) * 0.01, shares=(("AAPL", 100, 0),), working=(bad,))
        assert reasons(check_v1(s.ctx()), "4") == ["unmapped_broker_field"]
    s.state(99, verified=False)
    assert reasons(check_v1(s.ctx()), "4") == ["unverified_account"]


def test_dry_run_two_intents_compete_for_cash() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.state(10, cash="7000", reserved="0")
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-2", occ=PUT))
    f = check_v1(s.ctx())
    assert outcomes(f, "3") == [P, V]
    assert only_obs(f) == ["N=5000 C=7000", "N=5000 C=2000"]


def only_obs(f: tuple[object, ...]) -> list[str | None]:
    return [x.observed_value for x in f if x.sub_item == "3"]  # type: ignore[attr-defined]


def test_dry_run_two_calls_compete_for_shares() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.inst("call-1", CALL)
    s.state(10, shares=(("AAPL", 100, 0),))
    for _ in range(2):
        leg = off_csp(s, iid="call-1", occ=CALL)
        s.decide(DecisionAction.OPEN_CC, leg)
    assert outcomes(check_v1(s.ctx()), "4") == [P, V]


def test_dry_run_close_then_competing_close() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.inst("put-1", PUT)
    s.state(10, shorts=(("put-1", 1),))
    for _ in range(2):
        leg = off_csp(s)
        leg.side = OrderSide.BUY_TO_CLOSE
        s.decide(DecisionAction.CLOSE, leg)
    assert outcomes(check_v1(s.ctx()), "2") == [P, V]


def test_dry_run_conditional_opening_is_unverifiable() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.inst("put-1", PUT)
    s.state(10, shorts=(("put-1", 1),))
    close = off_csp(s)
    close.side = OrderSide.BUY_TO_CLOSE
    close.leg_ref = "leg:roll:close"
    replacement = off_csp(s, iid="put-2", occ=PUT, conditional=True, depends_on="leg:roll:close")
    s.decide(DecisionAction.ROLL, close, replacement)
    f = check_v1(s.ctx())
    assert outcomes(f, "2") == [P]
    assert reasons(f, "3") == ["dry_run_no_execution"]


def test_dry_run_order_and_baseline_problems() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.state(10, cash="7000")
    s.decide(DecisionAction.OPEN_CSP, off_csp(s), priority=None)
    s.decisions[-1].priority = None
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-2", occ=PUT))
    assert reasons(check_v1(s.ctx()), "3") == ["ambiguous_order", "ambiguous_order"]


def test_dry_run_baseline_changed_and_upstream_unknown() -> None:
    s = Scenario(ExecutionMode.OFF)
    base = s.state(10, cash="20000")
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, state=base))
    changed = s.state(20, cash="1000")
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-2", occ=PUT, state=changed))
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, iid="put-3", occ=PUT))
    assert reasons(check_v1(s.ctx()), "3") == ["baseline_changed", "baseline_changed"]
    s2 = Scenario(ExecutionMode.OFF)
    base = s2.state(10, cash="20000")
    same = s2.state(20, cash="20000")
    s2.inst("put-x", PUT2, multiplier=None)
    s2.decide(DecisionAction.OPEN_CSP, off_csp(s2, state=base))
    s2.decide(DecisionAction.OPEN_CSP, off_csp(s2, iid="put-x", occ=PUT2, state=same))
    s2.decide(DecisionAction.OPEN_CSP, off_csp(s2, iid="put-3", occ=PUT))
    f = check_v1(s2.ctx())
    assert outcomes(f, "3")[0] is P
    assert reasons(f, "3") == ["unmapped_broker_field", "upstream_reservation_unknown"]


def test_dry_run_missing_baseline_or_output() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    s.states.clear()
    assert reasons(check_v1(s.ctx()), "3") == ["no_pre_order_state"]
    s2 = Scenario(ExecutionMode.OFF)
    s2.decide(DecisionAction.OPEN_CSP, off_csp(s2))
    s2.states[0] = s2.states[0].model_copy(
        update={"snapshot": s2.states[0].snapshot.model_copy(update={"agentic_verified": False})}
    )
    assert reasons(check_v1(s2.ctx()), "3") == ["unverified_account"]
    s3 = Scenario(ExecutionMode.OFF)
    s3.output_status = DecisionOutputStatus.MISSING
    assert reasons(check_v1(s3.ctx())) == ["missing_final_output"]
    s3.record = False
    assert reasons(check_v1(s3.ctx())) == ["missing_run_record"]
    s4 = Scenario(ExecutionMode.OFF)
    assert outcomes(check_v1(s4.ctx())) == [P]


def test_live_missing_output_still_checks_tool_events() -> None:
    s = Scenario()
    live_csp(s)
    s.output_status = DecisionOutputStatus.MISSING
    f = check_v1(s.ctx())
    assert outcomes(f, "1") == [P]
    assert outcomes(f, "3") == [P]


def test_side_unknown_skips_funding_checks() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    s.place(100, "put-1", PUT, "sell_short", 1, "1.10")
    f = check_v1(s.ctx())
    assert [x.sub_item for x in f] == ["1"]
    assert AttemptStatus.FILLED  # keep import used
