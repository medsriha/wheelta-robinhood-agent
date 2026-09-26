"""V5 No order on an unset rule (incl. BTC per ADR-0014): pass / violation / unverifiable."""

from test_audit_builders import (
    BTC,
    CALL,
    NEW_PUT,
    PUT,
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

from wheelta_robinhood_agent.agent.audit import check_v5
from wheelta_robinhood_agent.config.rules import RuleMarker
from wheelta_robinhood_agent.domain.enums import (
    AuditOutcome,
    DecisionAction,
    ExecutionMode,
    OrderSide,
)
from wheelta_robinhood_agent.domain.run_record import DecisionOutputStatus

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE
TBD = RuleMarker.TBD


def test_all_rules_set_passes() -> None:
    s = Scenario()
    live_csp(s)
    assert outcomes(check_v5(s.ctx())) == [P]
    assert outcomes(check_v5(Scenario().ctx())) == [P]


def test_tbd_filter_or_order_rule_is_violation() -> None:
    s = Scenario()
    live_csp(s)
    s.rules = rules_with(filters={"min_dte": TBD})
    f = only(check_v5(s.ctx()), "1")
    assert f.outcome is V and f.rule_key == "filters.min_dte"
    s.rules = rules_with(orders={"limit_price_rule": "TBD"})
    assert outcomes(check_v5(s.ctx())) == [V]
    s.rules = rules_with(limits={"max_new_positions_per_day": TBD})
    assert outcomes(check_v5(s.ctx())) == [V]
    s.rules = rules_with(filters={"min_volume": RuleMarker.NONE})
    assert outcomes(check_v5(s.ctx())) == [P]


def test_put_only_limit_on_a_call_and_unknown_instrument() -> None:
    s = Scenario()
    s.inst("call-1", CALL)
    p = s.place(100, "call-1", CALL, STO, 1, "1.10")
    s.decide(DecisionAction.OPEN_CC, LegSpec(OrderSide.SELL_TO_OPEN, CALL, "call-1", 1, [p]))
    s.rules = rules_with(limits={"min_cash_reserve_usd": TBD})
    assert outcomes(check_v5(s.ctx())) == [P]
    s.instruments.clear()
    assert reasons(check_v5(s.ctx())) == ["lineage_unknown"]


def test_roll_replacement_does_not_need_new_position_caps() -> None:
    s = Scenario()
    s.inst("put-new", NEW_PUT)
    close = s.place(100, "put-1", PUT, BTC, 1, "0.50")
    repl = s.place(200, "put-new", NEW_PUT, STO, 1, "1.10")
    s.decide(
        DecisionAction.ROLL,
        LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1, [close], leg_ref="leg:c"),
        LegSpec(OrderSide.SELL_TO_OPEN, NEW_PUT, "put-new", 1, [repl], depends_on="leg:c"),
        target_ref="position:1",
    )
    s.rules = rules_with(limits={"max_new_positions_per_run": TBD})
    assert outcomes(check_v5(s.ctx())) == [P, P]
    s.rules = rules_with(management={"max_roll_dte": TBD})
    assert outcomes(check_v5(s.ctx())) == [V, P]


def test_unassociated_sell_to_open_lineage_unknown() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.place(100, "put-1", PUT, STO, 1, "1.10")
    s.rules = rules_with(limits={"max_new_positions_per_run": TBD})
    assert reasons(check_v5(s.ctx())) == ["lineage_unknown"]


def test_buy_to_close_management_dependencies() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    p = s.place(100, "put-1", PUT, BTC, 1, "0.50")
    s.decide(
        DecisionAction.CLOSE,
        LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1, [p]),
        target_ref="position:1",
    )
    assert outcomes(check_v5(s.ctx())) == [P]
    s.rules = rules_with(management={"close_quantity": TBD})
    assert outcomes(check_v5(s.ctx())) == [V]
    s.rules = rules_with(management={"take_profit_capture_ratio": TBD})
    assert reasons(check_v5(s.ctx())) == ["management_rule_attribution_unknown"]
    s.rules = rules_with(filters={"min_dte": TBD})  # entry filters never apply to a close
    assert outcomes(check_v5(s.ctx())) == [P]


def test_unassociated_buy_to_close() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.place(100, "put-1", PUT, BTC, 1, "0.50")
    s.rules = rules_with(management={"close_quantity": TBD})
    assert reasons(check_v5(s.ctx())) == ["management_rule_attribution_unknown"]


def test_unknown_side_is_unverifiable() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.place(100, "put-1", PUT, "buy_to_open", 1, "0.50")
    assert reasons(check_v5(s.ctx())) == ["missing_evidence"]


def test_dry_run_intents() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    assert outcomes(check_v5(s.ctx())) == [P]
    s.rules = rules_with(filters={"max_abs_delta": TBD})
    assert outcomes(check_v5(s.ctx())) == [V]
    s.output_status = DecisionOutputStatus.MISSING
    assert reasons(check_v5(s.ctx())) == ["missing_final_output"]
