"""The unfilled-order next-run cap (ADR-0065): which of a run's orders trigger it."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from wheelta_robinhood_agent.domain.enums import AttemptStatus
from wheelta_robinhood_agent.domain.orders import (
    BrokerOrder,
    FillObservationKind,
    FillRecord,
    OrderIntent,
    OrderRecord,
    StatusObservation,
)
from wheelta_robinhood_agent.orchestrator.schedule import unfilled_orders

RUN = uuid.uuid4()
OTHER_RUN = uuid.uuid4()
SCOPE = "robinhood:agentic:test"
T0 = datetime(2026, 10, 1, 15, 0, tzinfo=UTC)


def _order(
    status: AttemptStatus | None,
    *,
    minute: int = 0,
    instrument: str = "inst-1",
    side: str = "sell_to_open",
    quantity: int | None = 2,
    filled: int | None = None,
    run_id: uuid.UUID = RUN,
    broker: bool = True,
) -> OrderRecord:
    at = T0 + timedelta(minutes=minute)
    intent = OrderIntent(
        intent_id=uuid.uuid4(),
        run_id=run_id,
        place_tool_call_id=uuid.uuid4(),
        account_scope_id=SCOPE,
        occ_symbol=None,
        broker_instrument_id=instrument,
        side_raw=side,
        quantity=quantity,
        order_type_raw="limit",
        time_in_force_raw="gfd",
        limit_price=Decimal("1.25"),
        requested_at=at,
    )
    if not broker:
        return OrderRecord(intent=intent, broker_order=None)
    order_id = uuid.uuid4()
    fills = (
        ()
        if filled is None
        else (
            FillRecord(
                fill_id=uuid.uuid4(),
                order_id=order_id,
                kind=FillObservationKind.CUMULATIVE,
                broker_execution_id=None,
                quantity=filled,
                price=Decimal("1.25"),
                executed_at=at,
                observed_at=at,
                source_tool_call_id=uuid.uuid4(),
            ),
        )
    )
    return OrderRecord(
        intent=intent,
        broker_order=BrokerOrder(
            order_id=order_id,
            account_scope_id=SCOPE,
            broker_order_id=f"ord-{order_id}",
            intent_id=intent.intent_id,
            first_observed_at=at,
        ),
        status_history=()
        if status is None
        else (
            StatusObservation(
                status=status, broker_status_raw="x", observed_at=at, tool_call_id=uuid.uuid4()
            ),
        ),
        fills=fills,
    )


def test_no_orders_and_filled_orders_do_not_cap() -> None:
    assert unfilled_orders((), RUN) == ()
    assert unfilled_orders((_order(AttemptStatus.FILLED, filled=2),), RUN) == ()


def test_cancelled_rejected_expired_and_unknown_orders_cap() -> None:
    for status in (
        AttemptStatus.CANCELLED,
        AttemptStatus.REJECTED,
        AttemptStatus.EXPIRED,
        AttemptStatus.PLACED,
        None,
    ):
        order = _order(status)
        assert unfilled_orders((order,), RUN) == (order,)


def test_a_placement_with_no_broker_order_caps() -> None:
    order = _order(None, broker=False)
    assert unfilled_orders((order,), RUN) == (order,)


def test_a_partial_fill_then_cancel_caps_and_a_full_fill_quantity_does_not() -> None:
    partial = _order(AttemptStatus.CANCELLED, filled=1)
    assert unfilled_orders((partial,), RUN) == (partial,)
    assert unfilled_orders((_order(AttemptStatus.CANCELLED, filled=2),), RUN) == ()


def test_a_price_step_whose_replacement_filled_does_not_cap() -> None:
    first = _order(AttemptStatus.CANCELLED, minute=0)
    replacement = _order(AttemptStatus.FILLED, minute=3, filled=2)
    assert unfilled_orders((replacement, first), RUN) == ()


def test_a_replacement_that_did_not_fill_caps() -> None:
    first = _order(AttemptStatus.CANCELLED, minute=0)
    replacement = _order(AttemptStatus.CANCELLED, minute=3)
    assert unfilled_orders((first, replacement), RUN) == (replacement,)


def test_each_contract_and_side_counts_on_its_own() -> None:
    closed = _order(AttemptStatus.FILLED, side="buy_to_close", filled=2)
    reopen = _order(AttemptStatus.CANCELLED, instrument="inst-2", minute=2)
    assert unfilled_orders((closed, reopen), RUN) == (reopen,)


def test_orders_of_other_runs_are_ignored() -> None:
    observed = _order(AttemptStatus.CANCELLED, run_id=OTHER_RUN)
    assert unfilled_orders((observed,), RUN) == ()
