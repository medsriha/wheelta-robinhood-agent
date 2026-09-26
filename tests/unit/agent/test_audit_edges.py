"""Edge cases across checks: contradictory evidence, late reads, odd record shapes."""

from dataclasses import replace
from decimal import Decimal

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
    outcomes,
    reasons,
)

from wheelta_robinhood_agent.agent.audit import (
    WorkingOrderObservation,
    check_v1,
    check_v2,
    check_v3,
    check_v4,
    check_v6,
    check_v7,
)
from wheelta_robinhood_agent.agent.audit._common import (
    is_uncertain,
    live_attempts,
    pre_order_state,
)
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditOutcome,
    DecisionAction,
    ExecutionMode,
    OrderSide,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.orders import FillObservationKind

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE


def test_uncertain_requires_dispatch_and_attempt_time_required() -> None:
    s = Scenario()
    call = s.call("place_option_order", 10, status=ToolCallStatus.UNKNOWN)
    assert is_uncertain(call)
    assert not is_uncertain(call.model_copy(update={"dispatched_at": None}))
    live_csp(s)
    attempt = live_attempts(s.ctx())[-1]
    unknown = pre_order_state(s.ctx(), replace(attempt, at=None))
    assert not isinstance(unknown, type(None)) and unknown.reason.value == "missing_evidence"  # type: ignore[union-attr]


def test_contradictory_instrument_results_count_as_missing() -> None:
    s = Scenario()
    live_csp(s)
    s.inst("put-1", PUT, multiplier=10)
    assert reasons(check_v1(s.ctx()), "3-4") == ["missing_evidence"]


def test_quote_source_missing_from_events() -> None:
    s = Scenario()
    live_csp(s)
    s.drop_calls.update(s.quotes[0].source_tool_call_ids)
    f = check_v2(s.ctx())
    assert outcomes(f, "1") == [V]
    assert outcomes(f, "3") == [V]


def test_dry_run_placed_attempt_is_not_an_intent_and_no_time() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.inst("put-1", PUT)
    denied = s.place(100, "put-1", PUT, STO, 1, "1.10", denied=True)
    leg = off_csp(s)
    leg.places = [denied]
    s.decide(DecisionAction.OPEN_CSP, leg)
    assert outcomes(check_v1(s.ctx()), "1") == [P]
    s.fact_sets.clear()
    s.auto_output = False
    assert reasons(check_v3(s.ctx()), "3") == ["missing_evidence"]


def test_dry_run_no_intents_v7() -> None:
    assert outcomes(check_v7(Scenario(ExecutionMode.OFF).ctx())) == [P, P]


def test_dry_run_close_without_instrument() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.state(10, shorts=())
    leg = off_csp(s)
    leg.side = OrderSide.BUY_TO_CLOSE
    leg.iid = None
    s.decide(DecisionAction.CLOSE, leg)
    assert reasons(check_v1(s.ctx()), "2") == ["missing_evidence"]


def test_share_and_exposure_loops_skip_other_contracts() -> None:
    s = Scenario()
    s.inst("call-1", CALL)
    s.inst("put-1", PUT)
    put_order = WorkingOrderObservation(
        broker_order_id="w",
        broker_instrument_id="put-1",
        side_raw=STO,
        quantity=1,
        filled_quantity=0,
        owned=False,
    )
    s.state(90, shares=(("AAPL", 100, 0),), shorts=(("call-1", 0),), working=(put_order,))
    s.place(100, "call-1", CALL, STO, 1, "1.10")
    assert outcomes(check_v1(s.ctx()), "4") == [P]
    s.place(200, "put-1", PUT, STO, 1, "1.10")
    s.state(
        190, value="100000", shorts=(("call-1", 1),), working=(put_order,), reflects=(s.calls[-3],)
    )
    assert P in outcomes(check_v7(s.ctx()), "3")


def test_status_and_fill_reads_after_dispatch_are_ignored() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    first = s.place(100, "put-1", PUT, STO, 2, "1.15")
    s.status(first, 150, AttemptStatus.CANCELLED, filled=1)
    later = s.status(first, 400, AttemptStatus.CANCELLED, filled=1)
    missing = s.status(first, 160, AttemptStatus.CANCELLED)
    s.drop_calls.add(missing.identity.tool_call_id)
    s.fill(first, 450, 1, source=later, kind=FillObservationKind.CUMULATIVE)
    s.state(
        190, reflects=tuple(c for c in s.calls if c.identity.tool.endswith("place_option_order"))
    )
    second = s.place(200, "put-1", PUT, STO, 1, "1.10")
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second])
    )
    f = check_v6(s.ctx())
    assert outcomes(f, "1") == [P, P]


def test_status_read_in_flight_is_overlap() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    first = s.place(100, "put-1", PUT, STO, 2, "1.15")
    s.status(first, 150, AttemptStatus.CANCELLED, filled=0)
    inflight = s.status(first, 180, AttemptStatus.CANCELLED)
    idx = s.calls.index(inflight)
    s.calls[idx] = inflight.model_copy(
        update={"status": ToolCallStatus.UNKNOWN, "completed_at": None, "result_ref": None}
    )
    second = s.place(200, "put-1", PUT, STO, 2, "1.10")
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second])
    )
    assert reasons(check_v6(s.ctx()), "1") == ["uncertain_ordering"]


def test_failed_earlier_place_on_same_contract() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    s.place(100, "put-1", PUT, STO, 1, "1.15", status=ToolCallStatus.FAILED)
    s.place(200, "put-1", PUT, STO, 1, "1.10")
    f = check_v6(s.ctx())
    assert reasons(f, "2")[-1] == "terminal_state_unknown"
    assert outcomes(f, "3")[-1] is V


def test_review_lookup_skips_steps_without_orders_or_other_contracts() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.inst("put-2", NEW_PUT)
    other = s.place(50, "put-2", NEW_PUT, STO, 1, "1.10")
    orphan = s.place(80, "put-1", PUT, STO, 1, "1.10")
    s.place(100, "put-1", PUT, STO, 1, "1.10")
    s.orders.pop(orphan.identity.tool_call_id)
    assert other
    assert outcomes(check_v3(s.ctx()), "1")[-1] is P


def test_record_without_place_arguments() -> None:
    s = Scenario()
    live_csp(s)
    ctx = s.ctx()
    ctx = ctx.model_copy(update={"order_records": ()})
    assert reasons(check_v3(ctx), "2") == ["missing_evidence"]


def test_open_cc_shape_and_book_lookup() -> None:
    s = Scenario()
    s.position("position:a", "put-9", PUT)
    entry = s.position("position:b", "put-1", PUT)
    s.inst("call-1", CALL)
    s.inst("put-1", PUT)
    cc = s.place(100, "call-1", CALL, STO, 1, "1.10")
    s.decide(DecisionAction.OPEN_CC, LegSpec(OrderSide.SELL_TO_OPEN, CALL, "call-1", 1, [cc]))
    close = s.place(200, "put-1", PUT, BTC, 1, "0.50")
    s.decide(
        DecisionAction.CLOSE,
        LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1, [close]),
        target_ref="position:b",
        position_id=entry.position_id,
    )
    s.decide(DecisionAction.HOLD, target_ref="position:a")
    f = check_v4(s.ctx())
    assert outcomes(f, "2") == [P, P, P]
    assert outcomes(f, "1") == [P, P]


def test_live_roll_with_unsubmitted_replacement_leg() -> None:
    s = Scenario()
    entry = s.position("position:1", "put-1", PUT)
    s.inst("put-1", PUT)
    close = s.place(100, "put-1", PUT, BTC, 1, "0.50")
    s.status(close, 110, AttemptStatus.CANCELLED, filled=0)
    s.decide(
        DecisionAction.ROLL,
        LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1, [close], leg_ref="leg:c"),
        LegSpec(OrderSide.SELL_TO_OPEN, NEW_PUT, "put-new", None, depends_on="leg:c"),
        target_ref="position:1",
        position_id=entry.position_id,
    )
    f = check_v4(s.ctx())
    assert outcomes(f, "2") == [P]
    assert outcomes(f, "4") == [P]


def test_v7_call_attempt_and_missing_quantity() -> None:
    s = Scenario()
    s.inst("call-1", CALL)
    s.state(90, shares=(("AAPL", 100, 0),))
    s.place(100, "call-1", CALL, STO, 1, "1.10")
    f = check_v7(s.ctx())
    assert {x.sub_item for x in f} == {"1", "2", "5"}
    ctx = s.ctx()
    order = ctx.order_records[0]
    assert order.intent is not None
    blank = order.model_copy(update={"intent": order.intent.model_copy(update={"quantity": None})})
    f = check_v7(ctx.model_copy(update={"order_records": (blank,)}))
    assert reasons(f, "1") == ["missing_evidence"]


def test_v7_unknown_fill_on_already_counted_leg_and_popped_orders() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90, cash="100000")
    first = s.place(100, "put-1", PUT, STO, 2, "1.15")
    s.status(first, 110, AttemptStatus.CANCELLED, filled=1)
    second = s.place(200, "put-1", PUT, STO, 1, "1.10")
    s.fill(second, 210, 1, kind=FillObservationKind.CUMULATIVE)
    s.fill(second, 220, 0, kind=FillObservationKind.CUMULATIVE)  # decreasing: inconsistent
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second])
    )
    run = [x for x in check_v7(s.ctx()) if x.rule_key == "limits.max_new_positions_per_run"]
    assert run[0].observed_value == "known=1 unknown=0"
    s2 = Scenario()
    s2.inst("put-1", PUT)
    s2.state(90)
    dispatched = s2.place(100, "put-1", PUT, STO, 1, "1.10")
    denied = s2.place(200, "put-1", PUT, STO, 1, "1.10", denied=True)
    ctx = s2.ctx().model_copy(update={"order_records": ()})
    assert dispatched and denied
    run = [x for x in check_v7(ctx) if x.rule_key == "limits.max_new_positions_per_run"]
    assert run == [] or run[0].outcome is P
    assert Decimal("1")
