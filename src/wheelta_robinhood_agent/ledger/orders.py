"""Order intents, broker orders, order events, and the OrderRecord projection.

INTERFACES.md "Orders, fills, and workspace projections" and "Ledger tables and recovery".

- An intent is recorded before dispatch, keyed by its place tool call, even when no broker
  order ID is ever learned. `order_intents.arguments_redacted` holds the normalized place
  arguments of the domain `OrderIntent` (raw strings preserved); the verbatim redacted tool
  input stays on `tool_calls.arguments_redacted`.
- Broker-order identity is unique per (account_scope_id, broker_order_id). Intent and
  decision/leg associations are link events appended only with verified evidence.
- Observations are deduplicated by source key: one status per (order, source read), one fill
  per broker execution ID, one cumulative-total observation per total (never summed), one
  cancellation observation per (cancel call, status).
- Current state is a projection: superseded events (referenced by a later
  `corrects_event_id`) are dropped; conflicting links are left unlinked (None), never guessed.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    CancellationStatus,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.events import OrderEventType, OrderIntentEventType
from wheelta_robinhood_agent.domain.orders import (
    BrokerOrder,
    Cancellation,
    FillObservationKind,
    FillRecord,
    OrderIntent,
    OrderRecord,
    StatusObservation,
)
from wheelta_robinhood_agent.ledger.errors import (
    DedupConflict,
    IdentityConflict,
    LedgerError,
    UnknownEntity,
)
from wheelta_robinhood_agent.ledger.events import AppendedEvent, EventTable, append_event, as_uuid
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.ledger.tool_calls import tool_call_status

Conn = psycopg.Connection[tuple[object, ...]]
Row = dict[str, Any]


# A broker order is resolved once a terminal status is observed; anything else (including no
# observation or an `unknown` one) is unresolved and goes into the prompt's owned_orders.
UNRESOLVED_STATUSES: Final = frozenset(
    {AttemptStatus.PLACED, AttemptStatus.PARTIALLY_FILLED, AttemptStatus.UNKNOWN}
)
_INTENT_FIELDS: Final = frozenset(
    {
        "occ_symbol",
        "broker_instrument_id",
        "side_raw",
        "quantity",
        "order_type_raw",
        "time_in_force_raw",
        "limit_price",
    }
)


def _rows(conn: Conn, query: str, params: Sequence[object]) -> list[Row]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(query.encode(), params).fetchall()


def _one_source(source_tool_call_id: uuid.UUID) -> list[uuid.UUID]:
    return [source_tool_call_id]


# ---------------------------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------------------------
def record_order_intent(conn: Conn, intent: OrderIntent) -> uuid.UUID:
    """Persist a placement intent before dispatch. Idempotent per place tool call.

    The place tool call must exist in the same run with the same `requested_at`. Raises
    IdentityConflict if an intent for that call exists with different content.
    """
    args = intent.model_dump(mode="json", include=set(_INTENT_FIELDS))
    with conn.transaction():
        call = conn.execute(
            "SELECT run_id, requested_at FROM tool_calls WHERE tool_call_id = %s",
            (intent.place_tool_call_id,),
        ).fetchone()
        if call is None:
            raise UnknownEntity(f"tool_calls has no row {intent.place_tool_call_id}")
        if call[0] != intent.run_id:
            raise IdentityConflict("the place tool call belongs to another run")
        if call[1] != intent.requested_at:
            raise IdentityConflict("intent requested_at differs from the place tool call")
        existing = conn.execute(
            "SELECT intent_id, account_scope_id, arguments_redacted FROM order_intents "
            "WHERE place_tool_call_id = %s",
            (intent.place_tool_call_id,),
        ).fetchone()
        if existing is not None:
            if (existing[0], existing[1], existing[2]) != (
                intent.intent_id,
                intent.account_scope_id,
                args,
            ):
                raise IdentityConflict("an intent for this place call records other content")
            return intent.intent_id
        conn.execute(
            "INSERT INTO order_intents (intent_id, run_id, place_tool_call_id, account_scope_id, "
            "arguments_redacted) VALUES (%s, %s, %s, %s, %s)",
            (
                intent.intent_id,
                intent.run_id,
                intent.place_tool_call_id,
                intent.account_scope_id,
                Jsonb(args),
            ),
        )
    return intent.intent_id


def record_broker_order(
    conn: Conn, *, run_id: uuid.UUID, account_scope_id: str, broker_order_id: str
) -> uuid.UUID:
    """Create or get the broker-order identity (unique per account scope + broker ID)."""
    if not account_scope_id or not broker_order_id:
        raise ValueError("account_scope_id and broker_order_id must be non-empty")
    with conn.transaction():
        inserted = conn.execute(
            "INSERT INTO orders (order_id, account_scope_id, broker_order_id, first_run_id) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (account_scope_id, broker_order_id) DO NOTHING RETURNING order_id",
            (new_id(), account_scope_id, broker_order_id, run_id),
        ).fetchone()
        if inserted is not None:
            return as_uuid(inserted[0])
        row = conn.execute(
            "SELECT order_id FROM orders WHERE account_scope_id = %s AND broker_order_id = %s",
            (account_scope_id, broker_order_id),
        ).fetchone()
    if row is None:
        raise LedgerError("order row vanished after a unique conflict")
    return as_uuid(row[0])


# ---------------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------------
def link_intent(
    conn: Conn,
    order_id: uuid.UUID,
    intent_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID],
    review_tool_call_ids: Sequence[uuid.UUID] = (),
) -> AppendedEvent:
    """Link a broker order to its placement intent on verified broker evidence.

    `source_tool_call_ids` must name the evidence (e.g. the place result or an order read).
    `review_tool_call_ids` records the reviews that preceded this placement.
    """
    if not source_tool_call_ids:
        raise ValueError("an intent link needs verified broker evidence")
    with conn.transaction():
        order = conn.execute(
            "SELECT account_scope_id FROM orders WHERE order_id = %s", (order_id,)
        ).fetchone()
        intent = conn.execute(
            "SELECT account_scope_id FROM order_intents WHERE intent_id = %s", (intent_id,)
        ).fetchone()
        if order is None or intent is None:
            raise UnknownEntity("order or intent does not exist")
        if order[0] != intent[0]:
            raise IdentityConflict("intent and order belong to different account scopes")
        return append_event(
            conn,
            EventTable.ORDER,
            entity_id=order_id,
            run_id=run_id,
            event_type=OrderEventType.INTENT_LINKED.value,
            observed_at=observed_at,
            dedup_key=f"intent:{intent_id}",
            source_tool_call_ids=source_tool_call_ids,
            payload={"review_tool_call_ids": [str(i) for i in review_tool_call_ids]},
            extra={"intent_id": intent_id},
        )


def link_decision(
    conn: Conn,
    order_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    decision_ref: str,
    leg_ref: str | None,
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
) -> AppendedEvent:
    """Associate an order with a decision (and leg) once the association is established."""
    if not decision_ref or leg_ref == "":
        raise ValueError("decision_ref must be non-empty; leg_ref non-empty or None")
    return append_event(
        conn,
        EventTable.ORDER,
        entity_id=order_id,
        run_id=run_id,
        event_type=OrderEventType.DECISION_LINKED.value,
        observed_at=observed_at,
        dedup_key=f"decision:{decision_ref}:{leg_ref or ''}",
        source_tool_call_ids=source_tool_call_ids,
        payload={"decision_ref": decision_ref, "leg_ref": leg_ref},
    )


def _check_same(appended: AppendedEvent, conn: Conn, expected: dict[str, object]) -> None:
    """A deduplicated observation must match the stored one; a mismatch is a conflict."""
    if appended.created:
        return
    rows = _rows(conn, "SELECT * FROM order_events WHERE event_id = %s", (appended.event_id,))
    stored = rows[0]
    for column, value in expected.items():
        if stored[column] != value:
            raise DedupConflict(
                f"order event {appended.event_id} already records a different {column}"
            )


def observe_status(
    conn: Conn,
    order_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    status: AttemptStatus,
    observed_at: datetime,
    source_tool_call_id: uuid.UUID,
    broker_status_raw: str | None = None,
    source_as_of: datetime | None = None,
    corrects_event_id: uuid.UUID | None = None,
) -> AppendedEvent:
    """Record one broker status read. Re-recording the same read is a no-op."""
    key = f"status:{source_tool_call_id}"
    if corrects_event_id is not None:
        key += f":corrects:{corrects_event_id}"
    appended = append_event(
        conn,
        EventTable.ORDER,
        entity_id=order_id,
        run_id=run_id,
        event_type=OrderEventType.STATUS_OBSERVED.value,
        observed_at=observed_at,
        dedup_key=key,
        source_as_of=source_as_of,
        source_tool_call_ids=_one_source(source_tool_call_id),
        corrects_event_id=corrects_event_id,
        payload={"broker_status_raw": broker_status_raw},
        extra={"status": status.value},
    )
    _check_same(appended, conn, {"status": status.value})
    return appended


def observe_fill(
    conn: Conn,
    order_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    quantity: int,
    price: Decimal | None,
    broker_execution_id: str | None,
    observed_at: datetime,
    source_tool_call_id: uuid.UUID,
    executed_at: datetime | None = None,
) -> AppendedEvent:
    """Record a fill observation (INTERFACES.md FillRecord).

    With a verified `broker_execution_id` it is one execution (deduplicated by that ID).
    Without one, `quantity` is the broker's cumulative filled total for the order, stored as
    an observation and deduplicated by total; totals are never summed or given a fake ID.
    """
    if isinstance(quantity, bool) or quantity < 0:
        raise ValueError("fill quantity must be a non-negative integer")
    if broker_execution_id == "":
        raise ValueError("broker_execution_id must be non-empty or None")
    if executed_at is not None and executed_at.tzinfo is None:
        raise ValueError("executed_at must be timezone-aware")
    if broker_execution_id is not None:
        if quantity == 0:
            raise ValueError("an execution fill has a positive quantity")
        key = f"fill:execution:{broker_execution_id}"
    else:
        key = f"fill:cumulative:{quantity}"
    appended = append_event(
        conn,
        EventTable.ORDER,
        entity_id=order_id,
        run_id=run_id,
        event_type=OrderEventType.FILL_OBSERVED.value,
        observed_at=observed_at,
        dedup_key=key,
        source_tool_call_ids=_one_source(source_tool_call_id),
        payload={"executed_at": executed_at.isoformat() if executed_at else None},
        extra={
            "broker_execution_id": broker_execution_id,
            "fill_quantity": quantity,
            "fill_price": price,
        },
    )
    _check_same(appended, conn, {"fill_quantity": Decimal(quantity)})
    return appended


def observe_cancellation(
    conn: Conn,
    order_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    cancel_tool_call_id: uuid.UUID,
    status: CancellationStatus,
    observed_at: datetime,
    confirmation_tool_call_ids: Sequence[uuid.UUID] = (),
    fill_ids: Sequence[uuid.UUID] = (),
    evidence_tool_call_ids: Sequence[uuid.UUID] = (),
) -> AppendedEvent:
    """Record a cancel call's observed outcome. `confirmed` needs a confirming order read.

    `fill_ids` are fill events (on this order) that beat the cancellation.
    """
    if status is CancellationStatus.CONFIRMED and not confirmation_tool_call_ids:
        raise ValueError("a confirmed cancellation needs a confirming order read")
    sources = list(
        dict.fromkeys([cancel_tool_call_id, *confirmation_tool_call_ids, *evidence_tool_call_ids])
    )
    return append_event(
        conn,
        EventTable.ORDER,
        entity_id=order_id,
        run_id=run_id,
        event_type=OrderEventType.CANCELLATION_OBSERVED.value,
        observed_at=observed_at,
        dedup_key=f"cancellation:{cancel_tool_call_id}:{status.value}",
        source_tool_call_ids=sources,
        payload={
            "cancel_tool_call_id": str(cancel_tool_call_id),
            "cancellation_status": status.value,
            "confirmation_tool_call_ids": [str(i) for i in confirmation_tool_call_ids],
            "fill_ids": [str(i) for i in fill_ids],
            "evidence_tool_call_ids": [str(i) for i in evidence_tool_call_ids],
        },
    )


def record_reconciliation(
    conn: Conn,
    order_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    observed_at: datetime,
    dedup_key: str,
    detail: str,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
    corrects_event_id: uuid.UUID | None = None,
) -> AppendedEvent:
    """Record an operator/code reconciliation note; with `corrects_event_id` it supersedes
    the referenced event (e.g. a wrong link) in every projection."""
    if not detail:
        raise ValueError("a reconciliation needs a detail")
    return append_event(
        conn,
        EventTable.ORDER,
        entity_id=order_id,
        run_id=run_id,
        event_type=OrderEventType.RECONCILIATION.value,
        observed_at=observed_at,
        dedup_key=f"reconciliation:{dedup_key}",
        source_tool_call_ids=source_tool_call_ids,
        corrects_event_id=corrects_event_id,
        payload={"detail": detail},
    )


# ---------------------------------------------------------------------------------------------
# Intent events (order_intent_events): facts about an intent with no broker order linked
# ---------------------------------------------------------------------------------------------
def _intent_scope(conn: Conn, intent_id: uuid.UUID) -> str:
    row = conn.execute(
        "SELECT account_scope_id FROM order_intents WHERE intent_id = %s", (intent_id,)
    ).fetchone()
    if row is None:
        raise UnknownEntity(f"order_intents has no row {intent_id}")
    return str(row[0])


def record_intent_reviewed(
    conn: Conn,
    intent_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    review_tool_call_id: uuid.UUID,
    observed_at: datetime,
) -> AppendedEvent:
    """Link the review call that preceded the intent's place call (one event per review)."""
    return append_event(
        conn,
        EventTable.ORDER_INTENT,
        entity_id=intent_id,
        run_id=run_id,
        event_type=OrderIntentEventType.REVIEWED.value,
        observed_at=observed_at,
        dedup_key=f"reviewed:{review_tool_call_id}",
        source_tool_call_ids=_one_source(review_tool_call_id),
    )


def resolve_intent_no_order(
    conn: Conn,
    intent_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID],
    detail: str,
) -> AppendedEvent:
    """Record broker reads (`source_tool_call_ids`) proving no order exists for the intent.

    The intent then leaves `owned_unresolved_orders`. Refused if an effective link to a
    broker order exists: a verified order contradicts "no order". A later reconciliation
    with `corrects_event_id` supersedes this resolution (the intent is unresolved again).
    """
    if not source_tool_call_ids:
        raise ValueError("a no-order resolution needs broker reads")
    if not detail:
        raise ValueError("a no-order resolution needs a detail")
    scope = _intent_scope(conn, intent_id)
    linked = _linked_intent_ids(conn, scope)
    if any(intent_id in intents for intents in linked.values()):
        raise IdentityConflict(f"intent {intent_id} is linked to a broker order")
    return append_event(
        conn,
        EventTable.ORDER_INTENT,
        entity_id=intent_id,
        run_id=run_id,
        event_type=OrderIntentEventType.RESOLVED_NO_ORDER.value,
        observed_at=observed_at,
        dedup_key="resolved_no_order",
        source_tool_call_ids=source_tool_call_ids,
        payload={"detail": detail},
    )


def record_intent_reconciliation(
    conn: Conn,
    intent_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    observed_at: datetime,
    dedup_key: str,
    detail: str,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
    corrects_event_id: uuid.UUID | None = None,
) -> AppendedEvent:
    """A reconciliation note on an intent; with `corrects_event_id` it supersedes that event."""
    if not detail:
        raise ValueError("a reconciliation needs a detail")
    return append_event(
        conn,
        EventTable.ORDER_INTENT,
        entity_id=intent_id,
        run_id=run_id,
        event_type=OrderIntentEventType.RECONCILIATION.value,
        observed_at=observed_at,
        dedup_key=f"reconciliation:{dedup_key}",
        source_tool_call_ids=source_tool_call_ids,
        corrects_event_id=corrects_event_id,
        payload={"detail": detail},
    )


def _intents_resolved_no_order(conn: Conn, account_scope_id: str) -> set[uuid.UUID]:
    """Intents in the scope with an effective (not superseded) resolved_no_order event."""
    rows = _rows(
        conn,
        "SELECT e.event_id, e.entity_id, e.event_type, e.corrects_event_id "
        "FROM order_intent_events e JOIN order_intents i ON i.intent_id = e.entity_id "
        "WHERE i.account_scope_id = %s ORDER BY e.entity_id, e.sequence",
        (account_scope_id,),
    )
    return {
        e["entity_id"]
        for e in effective_events(rows)
        if e["event_type"] == OrderIntentEventType.RESOLVED_NO_ORDER
    }


# ---------------------------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _OrderRow:
    order_id: uuid.UUID
    account_scope_id: str
    broker_order_id: str
    recorded_at: datetime


def effective_events(events: Sequence[Row]) -> list[Row]:
    """Events not superseded by a later correction."""
    superseded = {e["corrects_event_id"] for e in events if e["corrects_event_id"] is not None}
    return [e for e in events if e["event_id"] not in superseded]


def _intent(row: Row) -> OrderIntent:
    return OrderIntent.model_validate(
        {
            **row["arguments_redacted"],
            "intent_id": row["intent_id"],
            "run_id": row["run_id"],
            "place_tool_call_id": row["place_tool_call_id"],
            "account_scope_id": row["account_scope_id"],
            "requested_at": row["requested_at"],
        }
    )


_INTENT_SELECT: Final = (
    "SELECT i.intent_id, i.run_id, i.place_tool_call_id, i.account_scope_id, "
    "i.arguments_redacted, t.requested_at FROM order_intents i "
    "JOIN tool_calls t ON t.tool_call_id = i.place_tool_call_id "
)


def _load_intent(conn: Conn, intent_id: uuid.UUID) -> OrderIntent:
    rows = _rows(conn, _INTENT_SELECT + "WHERE i.intent_id = %s", (intent_id,))
    if not rows:
        raise UnknownEntity(f"order_intents has no row {intent_id}")
    return _intent(rows[0])


def _as_count(value: object) -> int:
    if not isinstance(value, Decimal) or value != value.to_integral_value():
        raise LedgerError("stored fill quantity is not an integer")
    return int(value)


def _uuids(values: object) -> tuple[uuid.UUID, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(uuid.UUID(str(v)) for v in values)


def _single_source(event: Row) -> uuid.UUID:
    sources = event["source_tool_call_ids"]
    if len(sources) != 1:
        raise LedgerError(f"order event {event['event_id']} needs exactly one source read")
    return as_uuid(sources[0])


def _build_record(conn: Conn, order: _OrderRow, events: list[Row]) -> OrderRecord:
    live = effective_events(events)
    intent_links = [e for e in live if e["event_type"] == OrderEventType.INTENT_LINKED]
    intent_ids = {e["intent_id"] for e in intent_links}
    intent = _load_intent(conn, intent_ids.pop()) if len(intent_ids) == 1 else None
    reviews: dict[uuid.UUID, None] = {}
    if intent is not None:
        for e in intent_links:
            reviews.update(dict.fromkeys(_uuids(e["payload"].get("review_tool_call_ids"))))

    first_observed = min((e["observed_at"] for e in events), default=order.recorded_at)
    broker_order = BrokerOrder(
        order_id=order.order_id,
        account_scope_id=order.account_scope_id,
        broker_order_id=order.broker_order_id,
        intent_id=intent.intent_id if intent is not None else None,
        first_observed_at=first_observed,
    )

    statuses = tuple(
        StatusObservation(
            status=AttemptStatus(e["status"]),
            broker_status_raw=e["payload"].get("broker_status_raw"),
            observed_at=e["observed_at"],
            tool_call_id=_single_source(e),
        )
        for e in live
        if e["event_type"] == OrderEventType.STATUS_OBSERVED
    )
    fills = tuple(
        FillRecord(
            fill_id=e["event_id"],
            order_id=order.order_id,
            kind=(
                FillObservationKind.EXECUTION
                if e["broker_execution_id"] is not None
                else FillObservationKind.CUMULATIVE
            ),
            broker_execution_id=e["broker_execution_id"],
            quantity=_as_count(e["fill_quantity"]),
            price=e["fill_price"],
            executed_at=(
                datetime.fromisoformat(e["payload"]["executed_at"])
                if e["payload"].get("executed_at")
                else None
            ),
            observed_at=e["observed_at"],
            source_tool_call_id=_single_source(e),
        )
        for e in live
        if e["event_type"] == OrderEventType.FILL_OBSERVED
    )

    # Latest observation per cancel call wins the status; evidence accumulates.
    by_cancel: dict[uuid.UUID, list[Row]] = {}
    for e in live:
        if e["event_type"] == OrderEventType.CANCELLATION_OBSERVED:
            by_cancel.setdefault(uuid.UUID(e["payload"]["cancel_tool_call_id"]), []).append(e)
    cancellations: list[Cancellation] = []
    for cancel_id, observations in by_cancel.items():
        latest = max(observations, key=lambda e: (e["observed_at"], e["sequence"]))
        confirmations: dict[uuid.UUID, None] = {}
        fill_ids: dict[uuid.UUID, None] = {}
        evidence: dict[uuid.UUID, None] = {}
        for e in observations:
            confirmations.update(dict.fromkeys(_uuids(e["payload"]["confirmation_tool_call_ids"])))
            fill_ids.update(dict.fromkeys(_uuids(e["payload"]["fill_ids"])))
            evidence.update(dict.fromkeys(_uuids(e["payload"]["evidence_tool_call_ids"])))
        cancellations.append(
            Cancellation(
                cancel_tool_call_id=cancel_id,
                broker_order_id=order.broker_order_id,
                confirmation_tool_call_ids=tuple(confirmations),
                status=CancellationStatus(latest["payload"]["cancellation_status"]),
                fill_ids=tuple(fill_ids),
                evidence_tool_call_ids=tuple(evidence),
            )
        )

    links = {
        (e["payload"]["decision_ref"], e["payload"].get("leg_ref"))
        for e in live
        if e["event_type"] == OrderEventType.DECISION_LINKED
    }
    decision_ref, leg_ref = links.pop() if len(links) == 1 else (None, None)

    return OrderRecord(
        intent=intent,
        broker_order=broker_order,
        review_tool_call_ids=tuple(reviews),
        status_history=statuses,
        cancellations=tuple(cancellations),
        fills=fills,
        decision_ref=decision_ref,
        leg_ref=leg_ref,
    )


def _order_events(conn: Conn, order_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, list[Row]]:
    grouped: dict[uuid.UUID, list[Row]] = {oid: [] for oid in order_ids}
    if not order_ids:
        return grouped
    for e in _rows(
        conn,
        "SELECT * FROM order_events WHERE entity_id = ANY(%s) ORDER BY entity_id, sequence",
        (list(order_ids),),
    ):
        grouped[e["entity_id"]].append(e)
    return grouped


def order_record(conn: Conn, order_id: uuid.UUID) -> OrderRecord:
    """Rebuild one broker order's OrderRecord from its identity and events."""
    rows = _rows(
        conn,
        "SELECT order_id, account_scope_id, broker_order_id, recorded_at FROM orders "
        "WHERE order_id = %s",
        (order_id,),
    )
    if not rows:
        raise UnknownEntity(f"orders has no row {order_id}")
    order = _OrderRow(**rows[0])
    return _build_record(conn, order, _order_events(conn, [order_id])[order_id])


def _linked_intent_ids(conn: Conn, account_scope_id: str) -> dict[uuid.UUID, set[uuid.UUID]]:
    """order_id -> effective linked intent IDs, for orders in this account scope."""
    rows = _rows(
        conn,
        "SELECT e.* FROM order_events e JOIN orders o ON o.order_id = e.entity_id "
        "WHERE o.account_scope_id = %s AND e.event_type = 'intent_linked' "
        "ORDER BY e.entity_id, e.sequence",
        (account_scope_id,),
    )
    all_corrections = _rows(
        conn,
        "SELECT e.corrects_event_id FROM order_events e JOIN orders o ON o.order_id = e.entity_id "
        "WHERE o.account_scope_id = %s AND e.corrects_event_id IS NOT NULL",
        (account_scope_id,),
    )
    superseded = {r["corrects_event_id"] for r in all_corrections}
    out: dict[uuid.UUID, set[uuid.UUID]] = {}
    for e in rows:
        if e["event_type"] == OrderEventType.INTENT_LINKED and e["event_id"] not in superseded:
            out.setdefault(e["entity_id"], set()).add(e["intent_id"])
    return out


def owned_unresolved_orders(conn: Conn, account_scope_id: str) -> tuple[OrderRecord, ...]:
    """All unresolved owned placements in the account scope (the prompt's owned_orders).

    Owned: a broker order linked to one of our intents, or an intent with no verified broker
    order yet. Included:
    - linked broker orders whose projected status is placed/partially_filled/unknown
      (no observation counts as unknown);
    - intents without a linked order unless their place call was denied (never dispatched)
      or broker reads resolved them as no-order (`resolve_intent_no_order`): the outcome is
      unknown until broker reads link or rule it out (INTERFACES.md recovery).
    Ordered by first observation / request time.
    """
    links = _linked_intent_ids(conn, account_scope_id)
    order_rows = {
        r["order_id"]: _OrderRow(**r)
        for r in _rows(
            conn,
            "SELECT order_id, account_scope_id, broker_order_id, recorded_at FROM orders "
            "WHERE order_id = ANY(%s)",
            (list(links),),
        )
    }
    events = _order_events(conn, list(order_rows))
    records: list[tuple[datetime, OrderRecord]] = []
    for order_id, order in order_rows.items():
        record = _build_record(conn, order, events[order_id])
        if record.status in UNRESOLVED_STATUSES:
            first = record.broker_order.first_observed_at if record.broker_order else None
            records.append((first or order.recorded_at, record))

    linked_intents = set().union(*links.values()) if links else set()
    resolved_no_order = _intents_resolved_no_order(conn, account_scope_id)
    for row in _rows(
        conn,
        _INTENT_SELECT + "WHERE i.account_scope_id = %s ORDER BY t.requested_at, i.intent_id",
        (account_scope_id,),
    ):
        if row["intent_id"] in linked_intents or row["intent_id"] in resolved_no_order:
            continue
        if tool_call_status(conn, row["place_tool_call_id"]) is ToolCallStatus.DENIED:
            continue
        intent = _intent(row)
        records.append((intent.requested_at, OrderRecord(intent=intent, broker_order=None)))
    records.sort(key=lambda item: item[0])
    return tuple(record for _, record in records)
