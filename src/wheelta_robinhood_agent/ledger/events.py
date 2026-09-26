"""Shared append-only event writer (INTERFACES.md "Identity, events, and projections").

Every event table has unique(entity_id, sequence) and unique(entity_id, dedup_key). The
per-entity sequence is allocated inside the caller's transaction after locking the entity's
identity row (`SELECT ... FOR UPDATE`, which fires no UPDATE trigger), so concurrent writers
for one entity serialize and never collide. Re-appending an event with an existing dedup key
returns the stored event instead of writing a second one.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from wheelta_robinhood_agent.ledger.errors import DedupConflict, LedgerError, UnknownEntity
from wheelta_robinhood_agent.ledger.ids import new_id

JsonObject = Mapping[str, object]


class EventTable(StrEnum):
    RUN = "run_events"
    TOOL_CALL = "tool_call_events"
    ORDER_INTENT = "order_intent_events"
    ORDER = "order_events"
    POSITION = "position_events"
    WORKSPACE = "workspace_events"


# Identity table and key column locked for sequence allocation. Closed mapping: table names
# are never taken from callers as strings.
_IDENTITY: Final[dict[EventTable, tuple[str, str]]] = {
    EventTable.RUN: ("runs", "run_id"),
    EventTable.TOOL_CALL: ("tool_calls", "tool_call_id"),
    EventTable.ORDER_INTENT: ("order_intents", "intent_id"),
    EventTable.ORDER: ("orders", "order_id"),
    EventTable.POSITION: ("positions", "position_id"),
    EventTable.WORKSPACE: ("workspace_objects", "object_id"),
}

# Table-specific typed columns a caller may set (see migrations/0001_initial.sql).
_EXTRA_COLUMNS: Final[dict[EventTable, frozenset[str]]] = {
    EventTable.RUN: frozenset({"status"}),
    EventTable.TOOL_CALL: frozenset(
        {
            "reason",
            "effective_arguments_redacted",
            "latency_ms",
            "request_id",
            "result_ref",
            "error_ref",
            "delivered_result_ref",
        }
    ),
    EventTable.ORDER_INTENT: frozenset(),
    EventTable.ORDER: frozenset(
        {"intent_id", "status", "broker_execution_id", "fill_quantity", "fill_price"}
    ),
    EventTable.POSITION: frozenset({"order_id"}),
    EventTable.WORKSPACE: frozenset({"before_state", "after_state"}),
}
_JSON_EXTRAS: Final = frozenset({"effective_arguments_redacted", "before_state", "after_state"})


def as_uuid(value: object) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise LedgerError(f"expected uuid from the database, got {type(value).__name__}")
    return value


def as_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LedgerError(f"expected integer from the database, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class AppendedEvent:
    """An event row as stored. `created` is False when the dedup key already existed."""

    event_id: uuid.UUID
    entity_id: uuid.UUID
    sequence: int
    event_type: str
    created: bool


def append_event(
    conn: psycopg.Connection[tuple[object, ...]],
    table: EventTable,
    *,
    entity_id: uuid.UUID,
    run_id: uuid.UUID,
    event_type: str,
    observed_at: datetime,
    dedup_key: str,
    payload: JsonObject | None = None,
    source_as_of: datetime | None = None,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
    corrects_event_id: uuid.UUID | None = None,
    extra: Mapping[str, object] | None = None,
) -> AppendedEvent:
    """Append one event in its own transaction (a savepoint if the caller has one open)."""
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    if source_as_of is not None and source_as_of.tzinfo is None:
        raise ValueError("source_as_of must be timezone-aware")
    if not dedup_key:
        raise ValueError("dedup_key must be non-empty")
    extra = dict(extra or {})
    unknown = set(extra) - _EXTRA_COLUMNS[table]
    if unknown:
        raise ValueError(f"columns not allowed on {table.value}: {sorted(unknown)}")

    identity_table, identity_key = _IDENTITY[table]
    with conn.transaction():
        locked = conn.execute(
            sql.SQL("SELECT 1 FROM {} WHERE {} = %s FOR UPDATE").format(
                sql.Identifier(identity_table), sql.Identifier(identity_key)
            ),
            (entity_id,),
        ).fetchone()
        if locked is None:
            raise UnknownEntity(f"{identity_table} has no row {entity_id}")

        existing = conn.execute(
            sql.SQL(
                "SELECT event_id, sequence, event_type FROM {} "
                "WHERE entity_id = %s AND dedup_key = %s"
            ).format(sql.Identifier(table.value)),
            (entity_id, dedup_key),
        ).fetchone()
        if existing is not None:
            existing_id, existing_seq, existing_type = existing
            if existing_type != event_type:
                raise DedupConflict(
                    f"{table.value} dedup key {dedup_key!r} already records {existing_type!r}"
                )
            return AppendedEvent(
                as_uuid(existing_id), entity_id, as_int(existing_seq), event_type, created=False
            )

        row = conn.execute(
            sql.SQL("SELECT coalesce(max(sequence), 0) + 1 FROM {} WHERE entity_id = %s").format(
                sql.Identifier(table.value)
            ),
            (entity_id,),
        ).fetchone()
        if row is None:
            raise LedgerError("sequence allocation returned no row")
        sequence = as_int(row[0])

        event_id = new_id()
        values: dict[str, object] = {
            "event_id": event_id,
            "entity_id": entity_id,
            "run_id": run_id,
            "sequence": sequence,
            "event_type": event_type,
            "observed_at": observed_at,
            "source_as_of": source_as_of,
            "source_tool_call_ids": list(source_tool_call_ids),
            "dedup_key": dedup_key,
            "payload": Jsonb(dict(payload or {})),
            "corrects_event_id": corrects_event_id,
        }
        for column, value in extra.items():
            values[column] = Jsonb(value) if column in _JSON_EXTRAS and value is not None else value
        columns = list(values)
        conn.execute(
            sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table.value),
                sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            ),
            [values[c] for c in columns],
        )
    return AppendedEvent(event_id, entity_id, sequence, event_type, created=True)
