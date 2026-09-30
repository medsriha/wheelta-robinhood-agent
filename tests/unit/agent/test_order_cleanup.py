"""ADR-0050: classify unresolved owned orders and phrase the cleanup turn."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from wheelta_robinhood_agent.agent.order_cleanup import (
    CLEANUP_TOOLS,
    MAX_ORDER_CLEANUPS,
    CleanupAction,
    cleanup_action,
    cleanup_message,
    needs_cleanup,
    order_scope,
    unresolved_orders,
)
from wheelta_robinhood_agent.agent.simulated_broker import simulated_scope_id
from wheelta_robinhood_agent.domain.enums import AttemptStatus, CancellationStatus, OrderVenue
from wheelta_robinhood_agent.domain.orders import (
    BrokerOrder,
    Cancellation,
    OrderIntent,
    OrderRecord,
    StatusObservation,
)

T0 = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)
RUN = uuid.uuid4()
SCOPE = "acct:1234"


def _broker(order_id: uuid.UUID) -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        account_scope_id=SCOPE,
        broker_order_id="ord-1",
        intent_id=None,
        first_observed_at=T0,
    )


def _record(
    status: AttemptStatus | None = AttemptStatus.PLACED,
    cancel: CancellationStatus | None = None,
) -> OrderRecord:
    order_id = uuid.uuid4()
    return OrderRecord(
        intent=None,
        broker_order=_broker(order_id),
        status_history=(
            ()
            if status is None
            else (
                StatusObservation(
                    status=status,
                    broker_status_raw="queued",
                    observed_at=T0,
                    tool_call_id=uuid.uuid4(),
                ),
            )
        ),
        cancellations=(
            ()
            if cancel is None
            else (
                Cancellation(
                    cancel_tool_call_id=uuid.uuid4(), broker_order_id="ord-1", status=cancel
                ),
            )
        ),
    )


def _intent_only() -> OrderRecord:
    return OrderRecord(
        intent=OrderIntent(
            intent_id=uuid.uuid4(),
            run_id=RUN,
            place_tool_call_id=uuid.uuid4(),
            account_scope_id=SCOPE,
            occ_symbol=None,
            broker_instrument_id="inst-1",
            side_raw="sell",
            quantity=1,
            order_type_raw="limit",
            time_in_force_raw="gfd",
            limit_price=Decimal("1.25"),
            requested_at=T0,
        ),
        broker_order=None,
    )


def test_working_order_without_a_cancel_is_cancelled() -> None:
    assert cleanup_action(_record(AttemptStatus.PLACED)) is CleanupAction.CANCEL
    assert cleanup_action(_record(AttemptStatus.PARTIALLY_FILLED)) is CleanupAction.CANCEL


def test_an_order_whose_cancel_was_sent_is_only_confirmed() -> None:
    """Rule 6: an accepted or uncertain cancellation is never repeated."""
    for status in (CancellationStatus.PENDING, CancellationStatus.UNKNOWN):
        assert cleanup_action(_record(cancel=status)) is CleanupAction.CONFIRM


def test_unknown_state_is_read_first() -> None:
    assert cleanup_action(_record(status=None)) is CleanupAction.READ
    assert cleanup_action(_record(AttemptStatus.UNKNOWN)) is CleanupAction.READ
    assert cleanup_action(_intent_only()) is CleanupAction.READ


def test_order_scope_follows_the_venue() -> None:
    assert order_scope(OrderVenue.BROKER, SCOPE, RUN) == SCOPE
    assert order_scope(OrderVenue.SIMULATED, SCOPE, RUN) == simulated_scope_id(RUN)
    assert order_scope(OrderVenue.NONE, SCOPE, RUN) is None
    assert order_scope(None, SCOPE, RUN) is None


def test_no_scope_reads_nothing() -> None:
    assert unresolved_orders(None, None) == ()  # type: ignore[arg-type]


def test_message_lists_each_order_and_its_action() -> None:
    text = cleanup_message(
        [_record(), _record(cancel=CancellationStatus.PENDING), _intent_only()], 1
    )
    assert f"cleanup 1 of {MAX_ORDER_CLEANUPS}" in text
    assert (
        "broker order ord-1, status placed: last seen working: read get_option_orders; if it "
        "is still working, cancel it" in text
    )
    assert "do not cancel it again" in text
    assert "no broker order known, sell 1 inst-1 @ 1.25, status unknown: state unknown" in text
    assert "placing is not" in text


def test_cleanup_tools_are_order_reads_and_cancel_only() -> None:
    assert CLEANUP_TOOLS == {
        "mcp__robinhood__get_option_orders",
        "mcp__robinhood__get_option_positions",
        "mcp__robinhood__cancel_option_order",
    }


def _prior(tif: str = "gfd", placed: datetime = T0, run: uuid.UUID | None = None) -> OrderRecord:
    record = _intent_only()
    assert record.intent is not None
    intent = record.intent.model_copy(
        update={"run_id": run or uuid.uuid4(), "time_in_force_raw": tif, "requested_at": placed}
    )
    return record.model_copy(update={"intent": intent})


def _needs(record: OrderRecord, calls: frozenset[uuid.UUID] = frozenset()) -> bool:
    return needs_cleanup(record, run_id=RUN, run_tool_call_ids=calls, as_of=T0)


def test_this_runs_orders_always_need_cleanup() -> None:
    assert _needs(_prior(run=RUN, placed=T0 - timedelta(days=3)))


def test_an_older_day_order_has_expired_and_is_skipped() -> None:
    """ADR-0050: a gfd order from an earlier New York date cannot still be working."""
    assert not _needs(_prior(placed=T0 - timedelta(days=1)))
    assert _needs(_prior(placed=T0 - timedelta(hours=2)))  # same New York date
    # 01:00 UTC on T0's date is the previous evening in New York.
    assert not _needs(_prior(placed=T0.replace(hour=1)))


def test_an_older_order_without_a_day_limit_still_needs_cleanup() -> None:
    assert _needs(_prior(tif="gtc", placed=T0 - timedelta(days=5)))
    assert _needs(_prior(tif="GFD", placed=T0))


def test_an_older_order_this_run_observed_or_cancelled_needs_cleanup() -> None:
    old = T0 - timedelta(days=2)
    seen = uuid.uuid4()
    observed = _prior(placed=old).model_copy(
        update={
            "broker_order": _broker(uuid.uuid4()),
            "status_history": (
                StatusObservation(
                    status=AttemptStatus.PLACED,
                    broker_status_raw="queued",
                    observed_at=T0,
                    tool_call_id=seen,
                ),
            ),
        }
    )
    assert _needs(observed, frozenset({seen}))
    assert not _needs(observed)
    cancelled = _prior(placed=old).model_copy(
        update={
            "broker_order": _broker(uuid.uuid4()),
            "cancellations": (
                Cancellation(
                    cancel_tool_call_id=seen,
                    broker_order_id="ord-1",
                    status=CancellationStatus.PENDING,
                ),
            ),
        }
    )
    assert _needs(cancelled, frozenset({seen}))
