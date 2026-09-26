"""V6 No retry, stop on unknown: pass / violation / unverifiable, live and dry run."""

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
    outcomes,
    reasons,
)

from wheelta_robinhood_agent.agent.audit import WorkingOrderObservation, check_v6
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditOutcome,
    CancellationStatus,
    DecisionAction,
    ExecutionMode,
    OrderSide,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.orders import BrokerOrder, OrderRecord, StatusObservation

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE


def _walk(s: Scenario, *, target: int = 5, confirm: bool = True, filled: int = 2) -> None:
    s.inst("put-1", PUT)
    s.state(90)
    first = s.place(100, "put-1", PUT, STO, 5, "1.15")
    s.status(first, 110, AttemptStatus.PLACED)
    read = None
    if confirm:
        read = s.status(first, 125, AttemptStatus.CANCELLED, filled=filled)
    s.cancel(first, 120, confirm=read, outcome=CancellationStatus.CONFIRMED)
    s.state(
        190,
        reflects=tuple(
            c
            for c in s.calls
            if c.identity.tool.endswith(("place_option_order", "cancel_option_order"))
        ),
    )
    second = s.place(200, "put-1", PUT, STO, target - filled, "1.10")
    s.status(second, 210, AttemptStatus.FILLED, filled=target - filled)
    s.decide(
        DecisionAction.OPEN_CSP,
        LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", target, [first, second]),
    )


def test_dry_run_is_unverifiable() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    assert reasons(check_v6(s.ctx())) == ["dry_run_no_execution"]


def test_no_placements() -> None:
    assert outcomes(check_v6(Scenario().ctx())) == [P]


def test_single_placement_passes() -> None:
    s = Scenario()
    live_csp(s)
    assert outcomes(check_v6(s.ctx())) == [P, P, P]


def test_two_step_walk_with_fill_during_cancel() -> None:
    s = Scenario()
    _walk(s)
    f = check_v6(s.ctx())
    assert outcomes(f, "1") == [P, P]
    assert outcomes(f, "2") == [P, P]
    assert outcomes(f, "3") == [P, P]


def test_pending_cancel_never_permits_replacement() -> None:
    s = Scenario()
    _walk(s, confirm=False)
    f = check_v6(s.ctx())
    assert outcomes(f, "1") == [P, V]
    assert outcomes(f, "2") == [P, V]


def test_terminal_state_unknown_or_overlapping() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    first = s.place(100, "put-1", PUT, STO, 2, "1.15")
    second = s.place(200, "put-1", PUT, STO, 2, "1.10")
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second])
    )
    f = check_v6(s.ctx())
    assert reasons(f, "1") == ["terminal_state_unknown"]
    assert reasons(f, "2") == ["terminal_state_unknown"]
    s.status(first, 250, AttemptStatus.CANCELLED, filled=0, duration=100)  # read spans dispatch
    f = check_v6(s.ctx())
    assert reasons(f, "1") == ["uncertain_ordering"]


def test_terminal_without_final_fill_observation() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    first = s.place(100, "put-1", PUT, STO, 2, "1.15")
    s.status(first, 150, AttemptStatus.CANCELLED)
    second = s.place(200, "put-1", PUT, STO, 2, "1.10")
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second])
    )
    assert reasons(check_v6(s.ctx()), "1") == ["terminal_state_unknown"]


def test_replacement_after_target_filled_and_unknown_target() -> None:
    s2 = Scenario()
    s2.inst("put-1", PUT)
    s2.state(90)
    first = s2.place(100, "put-1", PUT, STO, 2, "1.15")
    s2.status(first, 150, AttemptStatus.FILLED, filled=2)
    second = s2.place(200, "put-1", PUT, STO, 1, "1.10")
    leg = LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second])
    s2.decide(DecisionAction.OPEN_CSP, leg)
    assert outcomes(check_v6(s2.ctx()), "1") == [P, V]
    leg.target = None
    assert reasons(check_v6(s2.ctx()), "1") == ["target_unknown"]


def test_place_error_later_reconciled_still_forbids_placement() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    failed = s.place(100, "put-1", PUT, STO, 1, "1.15", status=ToolCallStatus.FAILED)
    assert s.orders[failed.identity.tool_call_id].broker is None  # outcome was an error
    s.call("get_option_orders", 150)  # a later read reconciles it; placement is still barred
    s.inst("put-2", NEW_PUT)
    s.place(200, "put-2", NEW_PUT, STO, 1, "1.10")
    f = check_v6(s.ctx())
    assert outcomes(f, "3") == [P, V]


def test_cancel_timeout_then_place_is_violation() -> None:
    s = Scenario()
    place, _ = live_csp(s)
    s.cancel(place, 150, status=ToolCallStatus.TIMED_OUT, outcome=CancellationStatus.UNKNOWN)
    s.inst("put-2", NEW_PUT)
    s.place(200, "put-2", NEW_PUT, STO, 1, "1.10")
    assert outcomes(check_v6(s.ctx()), "3") == [P, V]


def test_unknown_or_overlapping_error_is_unverifiable() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    s.place(100, "put-1", PUT, STO, 1, "1.15", status=ToolCallStatus.UNKNOWN)
    s.inst("put-2", NEW_PUT)
    s.place(200, "put-2", NEW_PUT, STO, 1, "1.10")
    assert reasons(check_v6(s.ctx()), "3") == ["uncertain_ordering"]
    s2 = Scenario()
    s2.inst("put-1", PUT)
    s2.state(90)
    s2.call("cancel_option_order", 100, status=ToolCallStatus.FAILED, duration=200)
    s2.place(150, "put-1", PUT, STO, 1, "1.10")
    assert reasons(check_v6(s2.ctx()), "3") == ["uncertain_ordering"]


def test_partial_roll_close_blocks_replacement() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.inst("put-new", NEW_PUT)
    s.state(90)
    close = s.place(100, "put-1", PUT, BTC, 2, "0.50")
    s.status(close, 150, AttemptStatus.CANCELLED, filled=1)
    close_leg = LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 2, [close], leg_ref="leg:c")
    s.decide(DecisionAction.ROLL, close_leg, target_ref="position:1")
    assert V not in outcomes(check_v6(s.ctx()))
    repl = s.place(200, "put-new", NEW_PUT, STO, 2, "1.10")
    s.decisions[0].legs.append(
        LegSpec(
            OrderSide.SELL_TO_OPEN,
            NEW_PUT,
            "put-new",
            2,
            [repl],
            depends_on="leg:c",
            leg_ref="leg:o",
        )
    )
    f = check_v6(s.ctx())
    assert outcomes(f, "2")[-1] is V


def test_full_roll_close_permits_replacement_and_unknowns() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.inst("put-new", NEW_PUT)
    s.state(90)
    close = s.place(100, "put-1", PUT, BTC, 1, "0.50")
    s.status(close, 150, AttemptStatus.FILLED, filled=1)
    repl = s.place(200, "put-new", NEW_PUT, STO, 1, "1.10")
    close_leg = LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1, [close], leg_ref="leg:c")
    s.decide(
        DecisionAction.ROLL,
        close_leg,
        LegSpec(OrderSide.SELL_TO_OPEN, NEW_PUT, "put-new", 1, [repl], depends_on="leg:c"),
        target_ref="position:1",
    )
    assert outcomes(check_v6(s.ctx()), "2")[-1] is P
    close_leg.target = None
    assert "target_unknown" in reasons(check_v6(s.ctx()), "2")
    close_leg.target = 1
    s.orders[close.identity.tool_call_id].fills.clear()
    assert "terminal_state_unknown" in reasons(check_v6(s.ctx()), "2")


def test_pre_existing_owned_working_order() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    mine = WorkingOrderObservation(
        broker_order_id="old-1",
        broker_instrument_id="put-1",
        side_raw=STO,
        quantity=1,
        filled_quantity=0,
        owned=True,
    )
    theirs = mine.model_copy(update={"broker_order_id": "user-1", "owned": False})
    s.state(90, working=(mine, theirs))
    s.place(100, "put-1", PUT, STO, 1, "1.10")
    assert outcomes(check_v6(s.ctx()), "2") == [V]
    read = s.call("get_option_orders", 50)
    s.extra_orders.append(
        OrderRecord(
            intent=None,
            broker_order=BrokerOrder(
                order_id=read.identity.tool_call_id,
                account_scope_id="acct",
                broker_order_id="old-1",
                intent_id=None,
                first_observed_at=T0,
            ),
            status_history=(
                StatusObservation(
                    status=AttemptStatus.CANCELLED,
                    broker_status_raw="cancelled",
                    observed_at=T0,
                    tool_call_id=read.identity.tool_call_id,
                ),
            ),
        )
    )
    assert outcomes(check_v6(s.ctx()), "2") == [P]
    s.state(95, working=None)
    assert reasons(check_v6(s.ctx()), "2") == ["unmapped_broker_field"]
    s.states.clear()
    assert reasons(check_v6(s.ctx()), "2") == ["no_pre_order_state"]


def test_missing_place_event_and_denied_place() -> None:
    s = Scenario()
    place, _ = live_csp(s)
    s.drop_calls.add(place.identity.tool_call_id)
    s.inst("put-2", NEW_PUT)
    s.place(300, "put-2", NEW_PUT, STO, 1, "1.10", denied=True)
    f = check_v6(s.ctx())
    assert reasons(f, "4") == ["missing_evidence"]
    assert outcomes(f, "3") == [P]


def test_unknown_instrument() -> None:
    s = Scenario()
    s.state(90)
    call = s.place(100, "put-1", PUT, STO, 1, "1.10")
    s.orders.pop(call.identity.tool_call_id)
    f = check_v6(s.ctx())
    assert reasons(f, "1") == ["missing_evidence"]
    assert reasons(f, "2") == ["missing_evidence"]
