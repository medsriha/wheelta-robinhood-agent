"""Order intents, broker orders, order events, and the OrderRecord projection."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import count

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    CancellationStatus,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import FillObservationKind, OrderIntent
from wheelta_robinhood_agent.ledger.errors import DedupConflict, IdentityConflict
from wheelta_robinhood_agent.ledger.events import AppendedEvent
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.ledger.orders import (
    link_decision,
    link_intent,
    observe_cancellation,
    observe_fill,
    observe_status,
    order_record,
    owned_unresolved_orders,
    record_broker_order,
    record_intent_reconciliation,
    record_intent_reviewed,
    record_order_intent,
    record_reconciliation,
    resolve_intent_no_order,
)
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import (
    append_tool_call_outcome,
    record_tool_call_requested,
)

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 16, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=5)
ACCOUNT = "acct-scope-1"
_sdk = count(1)


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


def _call(conn: Conn, run_id: uuid.UUID, tool: str = "get_option_order") -> uuid.UUID:
    return record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id=f"toolu_{next(_sdk)}",
        stage="agent",
        server="robinhood",
        tool=tool,
        tier=ToolTier.X if tool.endswith("_option_order") and "get" not in tool else ToolTier.R,
        arguments_redacted={},
        requested_at=T0,
    ).tool_call_id


def _intent(conn: Conn, run_id: uuid.UUID, account: str = ACCOUNT) -> OrderIntent:
    place = _call(conn, run_id, "place_option_order")
    intent = OrderIntent(
        intent_id=new_id(),
        run_id=run_id,
        place_tool_call_id=place,
        account_scope_id=account,
        occ_symbol=OccSymbol.parse("AAPL  261016P00180000"),
        broker_instrument_id="inst-1",
        side_raw="sell_to_open",
        quantity=2,
        order_type_raw="limit",
        time_in_force_raw="gfd",
        limit_price=Decimal("1.25"),
        requested_at=T0,
    )
    record_order_intent(conn, intent)
    return intent


def _placed_order(
    conn: Conn, run_id: uuid.UUID, broker_id: str = "B-1"
) -> tuple[uuid.UUID, OrderIntent]:
    intent = _intent(conn, run_id)
    order_id = record_broker_order(
        conn, run_id=run_id, account_scope_id=ACCOUNT, broker_order_id=broker_id
    )
    link_intent(
        conn,
        order_id,
        intent.intent_id,
        run_id=run_id,
        observed_at=T0 + timedelta(seconds=1),
        source_tool_call_ids=[intent.place_tool_call_id],
    )
    return order_id, intent


def test_intent_is_idempotent_and_conflicts_on_other_content(conn: Conn, run_id: uuid.UUID) -> None:
    intent = _intent(conn, run_id)
    assert record_order_intent(conn, intent) == intent.intent_id
    with pytest.raises(IdentityConflict):
        record_order_intent(conn, intent.model_copy(update={"quantity": 3}))


def test_order_record_rebuilds_from_events(conn: Conn, run_id: uuid.UUID) -> None:
    order_id, intent = _placed_order(conn, run_id)
    read1, read2 = _call(conn, run_id), _call(conn, run_id)
    observe_status(
        conn,
        order_id,
        run_id=run_id,
        status=AttemptStatus.PLACED,
        observed_at=T0 + timedelta(seconds=2),
        source_tool_call_id=read1,
        broker_status_raw="confirmed",
    )
    observe_fill(
        conn,
        order_id,
        run_id=run_id,
        quantity=2,
        price=Decimal("1.25"),
        broker_execution_id=None,
        observed_at=T0 + timedelta(seconds=3),
        source_tool_call_id=read2,
    )
    observe_status(
        conn,
        order_id,
        run_id=run_id,
        status=AttemptStatus.FILLED,
        observed_at=T0 + timedelta(seconds=3),
        source_tool_call_id=read2,
    )
    link_decision(
        conn, order_id, run_id=run_id, decision_ref="decision:1", leg_ref="leg:1", observed_at=T0
    )

    record = order_record(conn, order_id)
    assert record.intent == intent
    assert record.broker_order is not None
    assert record.broker_order.intent_id == intent.intent_id
    assert record.broker_order.broker_order_id == "B-1"
    assert [s.status for s in record.status_history] == [
        AttemptStatus.PLACED,
        AttemptStatus.FILLED,
    ]
    assert record.status is AttemptStatus.FILLED
    assert record.filled_quantity == 2
    assert record.fills[0].kind is FillObservationKind.CUMULATIVE
    assert (record.decision_ref, record.leg_ref) == ("decision:1", "leg:1")
    # A second rebuild gives the same projection.
    assert order_record(conn, order_id) == record


def test_duplicate_observations_are_deduplicated(conn: Conn, run_id: uuid.UUID) -> None:
    order_id, _ = _placed_order(conn, run_id)
    read = _call(conn, run_id)

    def status(value: AttemptStatus) -> AppendedEvent:
        return observe_status(
            conn,
            order_id,
            run_id=run_id,
            status=value,
            observed_at=T0 + timedelta(seconds=2),
            source_tool_call_id=read,
        )

    first = status(AttemptStatus.PLACED)
    again = status(AttemptStatus.PLACED)
    assert first.created and not again.created
    with pytest.raises(DedupConflict):
        status(AttemptStatus.FILLED)

    for read_id in (_call(conn, run_id), _call(conn, run_id)):
        observe_fill(
            conn,
            order_id,
            run_id=run_id,
            quantity=1,
            price=Decimal("1.20"),
            broker_execution_id="X-1",
            observed_at=T0 + timedelta(seconds=3),
            source_tool_call_id=read_id,
        )
    # The same cumulative total re-read is one observation, never summed.
    for read_id in (_call(conn, run_id), _call(conn, run_id)):
        observe_fill(
            conn,
            order_id,
            run_id=run_id,
            quantity=1,
            price=Decimal("1.20"),
            broker_execution_id=None,
            observed_at=T0 + timedelta(seconds=4),
            source_tool_call_id=read_id,
        )
    with pytest.raises(DedupConflict):
        observe_fill(
            conn,
            order_id,
            run_id=run_id,
            quantity=2,
            price=Decimal("1.20"),
            broker_execution_id="X-1",
            observed_at=T0 + timedelta(seconds=5),
            source_tool_call_id=_call(conn, run_id),
        )
    record = order_record(conn, order_id)
    assert len(record.status_history) == 1
    assert len(record.fills) == 2
    assert record.filled_quantity == 1


def test_cancellation_latest_status_wins_and_evidence_accumulates(
    conn: Conn, run_id: uuid.UUID
) -> None:
    order_id, _ = _placed_order(conn, run_id)
    cancel = _call(conn, run_id, "cancel_option_order")
    confirm = _call(conn, run_id)
    observe_cancellation(
        conn,
        order_id,
        run_id=run_id,
        cancel_tool_call_id=cancel,
        status=CancellationStatus.PENDING,
        observed_at=T0 + timedelta(seconds=5),
    )
    fill = observe_fill(
        conn,
        order_id,
        run_id=run_id,
        quantity=1,
        price=Decimal("1.25"),
        broker_execution_id="X-9",
        observed_at=T0 + timedelta(seconds=6),
        source_tool_call_id=confirm,
    )
    observe_cancellation(
        conn,
        order_id,
        run_id=run_id,
        cancel_tool_call_id=cancel,
        status=CancellationStatus.CONFIRMED,
        observed_at=T0 + timedelta(seconds=6),
        confirmation_tool_call_ids=[confirm],
        fill_ids=[fill.event_id],
    )
    (cancellation,) = order_record(conn, order_id).cancellations
    assert cancellation.status is CancellationStatus.CONFIRMED
    assert cancellation.confirmation_tool_call_ids == (confirm,)
    assert cancellation.fill_ids == (fill.event_id,)
    assert cancellation.broker_order_id == "B-1"


def test_conflicting_decision_links_stay_unlinked_until_corrected(
    conn: Conn, run_id: uuid.UUID
) -> None:
    order_id, _ = _placed_order(conn, run_id)
    wrong = link_decision(
        conn, order_id, run_id=run_id, decision_ref="decision:1", leg_ref=None, observed_at=T0
    )
    link_decision(
        conn, order_id, run_id=run_id, decision_ref="decision:2", leg_ref=None, observed_at=T0
    )
    assert order_record(conn, order_id).decision_ref is None
    fix = record_reconciliation(
        conn,
        order_id,
        run_id=run_id,
        observed_at=T0 + timedelta(minutes=1),
        dedup_key="unlink-decision-1",
        detail="decision:1 link was wrong",
        corrects_event_id=wrong.event_id,
    )
    assert fix.created
    assert order_record(conn, order_id).decision_ref == "decision:2"
    stored = conn.execute(
        "SELECT corrects_event_id FROM order_events WHERE event_id = %s", (fix.event_id,)
    ).fetchone()
    assert stored == (wrong.event_id,)


def test_owned_unresolved_orders_includes_working_and_unknown(
    conn: Conn, run_id: uuid.UUID
) -> None:
    working, _ = _placed_order(conn, run_id, "B-working")
    observe_status(
        conn,
        working,
        run_id=run_id,
        status=AttemptStatus.PARTIALLY_FILLED,
        observed_at=T0 + timedelta(seconds=2),
        source_tool_call_id=_call(conn, run_id),
    )
    never_observed, _ = _placed_order(conn, run_id, "B-unobserved")
    unknown, _ = _placed_order(conn, run_id, "B-unknown")
    observe_status(
        conn,
        unknown,
        run_id=run_id,
        status=AttemptStatus.UNKNOWN,
        observed_at=T0 + timedelta(seconds=2),
        source_tool_call_id=_call(conn, run_id),
    )
    filled, _ = _placed_order(conn, run_id, "B-filled")
    observe_status(
        conn,
        filled,
        run_id=run_id,
        status=AttemptStatus.FILLED,
        observed_at=T0 + timedelta(seconds=2),
        source_tool_call_id=_call(conn, run_id),
    )
    # An unlinked broker order is not ours; an unlinked intent with a timed-out place call is.
    record_broker_order(conn, run_id=run_id, account_scope_id=ACCOUNT, broker_order_id="B-user")
    timed_out = _intent(conn, run_id)
    append_tool_call_outcome(
        conn, timed_out.place_tool_call_id, ToolCallStatus.TIMED_OUT, observed_at=T0
    )
    denied = _intent(conn, run_id)
    append_tool_call_outcome(
        conn, denied.place_tool_call_id, ToolCallStatus.DENIED, observed_at=T0, reason="gate"
    )
    _intent(conn, run_id, account="other-account")

    records = owned_unresolved_orders(conn, ACCOUNT)
    broker_ids = {r.broker_order.order_id for r in records if r.broker_order is not None}
    intent_only = [r.intent.intent_id for r in records if r.broker_order is None and r.intent]
    assert broker_ids == {working, never_observed, unknown}
    assert intent_only == [timed_out.intent_id]


def test_link_intent_rejects_cross_account(conn: Conn, run_id: uuid.UUID) -> None:
    intent = _intent(conn, run_id, account="other-account")
    order_id = record_broker_order(
        conn, run_id=run_id, account_scope_id=ACCOUNT, broker_order_id="B-x"
    )
    with pytest.raises(IdentityConflict):
        link_intent(
            conn,
            order_id,
            intent.intent_id,
            run_id=run_id,
            observed_at=T0,
            source_tool_call_ids=[intent.place_tool_call_id],
        )


def _intent_events(conn: Conn, intent_id: uuid.UUID) -> list[tuple[object, ...]]:
    return conn.execute(
        "SELECT sequence, event_type, source_tool_call_ids FROM order_intent_events "
        "WHERE entity_id = %s ORDER BY sequence",
        (intent_id,),
    ).fetchall()


def test_intent_resolved_no_order_leaves_owned_orders_until_superseded(
    conn: Conn, run_id: uuid.UUID
) -> None:
    intent = _intent(conn, run_id)
    append_tool_call_outcome(
        conn, intent.place_tool_call_id, ToolCallStatus.UNKNOWN, observed_at=T0, reason="timeout"
    )
    review = _call(conn, run_id, "review_option_order")
    first = record_intent_reviewed(
        conn, intent.intent_id, run_id=run_id, review_tool_call_id=review, observed_at=T0
    )
    again = record_intent_reviewed(
        conn, intent.intent_id, run_id=run_id, review_tool_call_id=review, observed_at=T0
    )
    assert first.created and not again.created
    assert [r.intent.intent_id for r in owned_unresolved_orders(conn, ACCOUNT) if r.intent] == [
        intent.intent_id
    ]

    with pytest.raises(ValueError, match="broker reads"):
        resolve_intent_no_order(
            conn,
            intent.intent_id,
            run_id=run_id,
            observed_at=T0,
            source_tool_call_ids=[],
            detail="x",
        )
    read = _call(conn, run_id, "get_option_orders")
    resolved = resolve_intent_no_order(
        conn,
        intent.intent_id,
        run_id=run_id,
        observed_at=T0 + timedelta(minutes=1),
        source_tool_call_ids=[read],
        detail="no order in the broker's order list",
    )
    assert owned_unresolved_orders(conn, ACCOUNT) == ()
    assert [e[1] for e in _intent_events(conn, intent.intent_id)] == [
        "reviewed",
        "resolved_no_order",
    ]

    # A reconciliation superseding the resolution makes the intent unresolved again.
    record_intent_reconciliation(
        conn,
        intent.intent_id,
        run_id=run_id,
        observed_at=T0 + timedelta(minutes=2),
        dedup_key="reopen",
        detail="order list was truncated",
        corrects_event_id=resolved.event_id,
    )
    assert [r.intent.intent_id for r in owned_unresolved_orders(conn, ACCOUNT) if r.intent] == [
        intent.intent_id
    ]


def test_intent_linked_to_an_order_cannot_resolve_as_no_order(
    conn: Conn, run_id: uuid.UUID
) -> None:
    _, intent = _placed_order(conn, run_id, "B-linked")
    with pytest.raises(IdentityConflict, match="linked"):
        resolve_intent_no_order(
            conn,
            intent.intent_id,
            run_id=run_id,
            observed_at=T0,
            source_tool_call_ids=[_call(conn, run_id)],
            detail="x",
        )


def test_intent_evidence_events_need_sources(conn: Conn, run_id: uuid.UUID) -> None:
    intent = _intent(conn, run_id)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO order_intent_events (event_id, entity_id, run_id, sequence, event_type, "
            "observed_at, dedup_key) VALUES (gen_random_uuid(), %s, %s, 1, 'resolved_no_order', "
            "now(), 'k')",
            (intent.intent_id, run_id),
        )
