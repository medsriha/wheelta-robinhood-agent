"""Tool-call identities and events (INTERFACES.md ToolCallRecord).

`record_tool_call_requested` inserts the identity and its `requested` event in one
transaction, so every tool_calls row has a durable request before any dispatch (including
calls later denied). Outcomes are appended afterwards and deduplicated by source key, so a
result hook and a failure hook reporting the same outcome, or a retried hook, record it once.
An unresolved dispatched financial action is recorded as `unknown`, never inferred as failed.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.events import ToolCallEventType
from wheelta_robinhood_agent.domain.tool_calls import (
    ToolCallDecision,
    ToolCallIdentity,
    ToolCallRecord,
)
from wheelta_robinhood_agent.ledger.errors import IdentityConflict, LedgerError, UnknownEntity
from wheelta_robinhood_agent.ledger.events import AppendedEvent, EventTable, append_event, as_uuid
from wheelta_robinhood_agent.ledger.ids import new_id

REQUESTED_DEDUP_KEY: Final = "requested"
DISPATCHED_DEDUP_KEY: Final = "dispatched"
OUTCOME_STATUSES: Final = frozenset(
    {
        ToolCallStatus.DENIED,
        ToolCallStatus.SUCCEEDED,
        ToolCallStatus.FAILED,
        ToolCallStatus.TIMED_OUT,
        ToolCallStatus.UNKNOWN,
    }
)


@dataclass(frozen=True)
class ToolCallRef:
    """A tool-call identity. `created` is False when (run_id, sdk_tool_use_id) existed."""

    tool_call_id: uuid.UUID
    run_id: uuid.UUID
    sdk_tool_use_id: str
    created: bool


def record_tool_call_requested(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    run_id: uuid.UUID,
    sdk_tool_use_id: str,
    stage: str,
    server: str,
    tool: str,
    tier: ToolTier | None,
    arguments_redacted: Mapping[str, object],
    requested_at: datetime,
) -> ToolCallRef:
    """Persist the identity and `requested` event before dispatch. Idempotent per SDK ID.

    `tier` is None for a tool missing from the registry. Arguments must already be redacted.
    Raises IdentityConflict if the SDK ID exists for a different server/tool.
    """
    if requested_at.tzinfo is None:
        raise ValueError("requested_at must be timezone-aware")
    with conn.transaction():
        existing = conn.execute(
            "SELECT tool_call_id, server, tool FROM tool_calls "
            "WHERE run_id = %s AND sdk_tool_use_id = %s",
            (run_id, sdk_tool_use_id),
        ).fetchone()
        if existing is not None:
            if (existing[1], existing[2]) != (server, tool):
                raise IdentityConflict(
                    f"sdk_tool_use_id {sdk_tool_use_id!r} already records another tool"
                )
            return ToolCallRef(as_uuid(existing[0]), run_id, sdk_tool_use_id, created=False)
        tool_call_id = new_id()
        conn.execute(
            "INSERT INTO tool_calls (tool_call_id, run_id, sdk_tool_use_id, stage, server, tool, "
            "tier, requested_at, arguments_redacted) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                tool_call_id,
                run_id,
                sdk_tool_use_id,
                stage,
                server,
                tool,
                tier.value if tier is not None else None,
                requested_at,
                Jsonb(dict(arguments_redacted)),
            ),
        )
        append_event(
            conn,
            EventTable.TOOL_CALL,
            entity_id=tool_call_id,
            run_id=run_id,
            event_type=ToolCallEventType.REQUESTED.value,
            observed_at=requested_at,
            dedup_key=REQUESTED_DEDUP_KEY,
        )
    return ToolCallRef(tool_call_id, run_id, sdk_tool_use_id, created=True)


def record_tool_call_dispatched(
    conn: psycopg.Connection[tuple[object, ...]],
    tool_call_id: uuid.UUID,
    *,
    dispatched_at: datetime,
    effective_arguments_redacted: Mapping[str, object],
) -> AppendedEvent:
    """Record that the hook allowed the call, with the (possibly rewritten) redacted input."""
    return append_event(
        conn,
        EventTable.TOOL_CALL,
        entity_id=tool_call_id,
        run_id=_run_id_of(conn, tool_call_id),
        event_type=ToolCallEventType.DISPATCHED.value,
        observed_at=dispatched_at,
        dedup_key=DISPATCHED_DEDUP_KEY,
        extra={"effective_arguments_redacted": dict(effective_arguments_redacted)},
    )


def append_tool_call_outcome(
    conn: psycopg.Connection[tuple[object, ...]],
    tool_call_id: uuid.UUID,
    status: ToolCallStatus,
    *,
    observed_at: datetime,
    dedup_key: str | None = None,
    reason: str | None = None,
    latency_ms: int | None = None,
    request_id: str | None = None,
    result_ref: uuid.UUID | None = None,
    error_ref: uuid.UUID | None = None,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
    corrects_event_id: uuid.UUID | None = None,
    payload: Mapping[str, object] | None = None,
) -> AppendedEvent:
    """Append an outcome event (denied/succeeded/failed/timed_out/unknown).

    `dedup_key` defaults to `outcome:<status>`; hooks pass their source key (e.g. the SDK
    event) when one exists. A later reconciliation that resolves an `unknown` appends a new
    outcome with `corrects_event_id`; it never rewrites the earlier one. Denied needs a reason.
    """
    if status not in OUTCOME_STATUSES:
        raise ValueError(f"{status.value} is not an outcome status")
    if status is ToolCallStatus.DENIED and not reason:
        raise ValueError("a denied outcome requires a reason")
    if latency_ms is not None and latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    extra: dict[str, object] = {}
    for column, value in (
        ("reason", reason),
        ("latency_ms", latency_ms),
        ("request_id", request_id),
        ("result_ref", result_ref),
        ("error_ref", error_ref),
    ):
        if value is not None:
            extra[column] = value
    return append_event(
        conn,
        EventTable.TOOL_CALL,
        entity_id=tool_call_id,
        run_id=_run_id_of(conn, tool_call_id),
        event_type=ToolCallEventType(status.value).value,
        observed_at=observed_at,
        dedup_key=dedup_key or f"outcome:{status.value}",
        payload=payload,
        source_tool_call_ids=source_tool_call_ids,
        corrects_event_id=corrects_event_id,
        extra=extra,
    )


def record_tool_call_delivered(
    conn: psycopg.Connection[tuple[object, ...]],
    tool_call_id: uuid.UUID,
    *,
    delivered_result_ref: uuid.UUID,
    observed_at: datetime,
) -> AppendedEvent:
    """Record the exact envelope (a `results` row of kind `delivered`) handed to the model."""
    return append_event(
        conn,
        EventTable.TOOL_CALL,
        entity_id=tool_call_id,
        run_id=_run_id_of(conn, tool_call_id),
        event_type=ToolCallEventType.DELIVERED.value,
        observed_at=observed_at,
        dedup_key="delivered",
        extra={"delivered_result_ref": delivered_result_ref},
    )


def tool_call_status(
    conn: psycopg.Connection[tuple[object, ...]], tool_call_id: uuid.UUID
) -> ToolCallStatus:
    """Projection: the latest status-bearing event (requested or an outcome)."""
    statuses = {s.value for s in ToolCallStatus}
    rows = conn.execute(
        "SELECT event_type FROM tool_call_events WHERE entity_id = %s ORDER BY sequence",
        (tool_call_id,),
    ).fetchall()
    current: ToolCallStatus | None = None
    for (event_type,) in rows:
        if isinstance(event_type, str) and event_type in statuses:
            current = ToolCallStatus(event_type)
    if current is None:
        raise UnknownEntity(f"tool call {tool_call_id} has no recorded request")
    return current


_COMPLETION_EVENTS: Final = frozenset(
    {ToolCallEventType.SUCCEEDED, ToolCallEventType.FAILED, ToolCallEventType.TIMED_OUT}
)


def tool_call_records(
    conn: psycopg.Connection[tuple[object, ...]], run_id: uuid.UUID
) -> tuple[ToolCallRecord, ...]:
    """Projection: every tool call of the run, ordered by requested_at then tool_call_id.

    Built from the identity row and its effective events (an event referenced by a later
    `corrects_event_id` is superseded):
    - `dispatched` sets decision ALLOWED, dispatched_at, and the effective arguments;
      `denied` sets decision DENIED with its reason;
    - status is the latest effective status-bearing event (requested or an outcome), as in
      `tool_call_status`. A dispatched call with no outcome stays REQUESTED; an `unknown`
      outcome stays UNKNOWN. Nothing is inferred from the absence of an event;
    - latency, request ID, result/error refs come from that outcome event. succeeded/
      failed/timed_out set completed_at to its observed_at, unless the outcome corrects an
      earlier `unknown` from broker reads: then it is a reconciliation, completed_at stays
      None and `reconciled_by_tool_call_ids` names the reads;
    - `delivered` sets delivered_result_ref.

    Raises LedgerError if a call's events contradict the ToolCallRecord invariants (e.g. a
    call both denied and dispatched); the record is never repaired by guessing.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        identities = cur.execute(
            "SELECT tool_call_id, sdk_tool_use_id, run_id, stage, server, tool, tier, "
            "requested_at, arguments_redacted FROM tool_calls WHERE run_id = %s "
            "ORDER BY requested_at, tool_call_id",
            (run_id,),
        ).fetchall()
        events = cur.execute(
            "SELECT event_id, entity_id, sequence, event_type, reason, "
            "effective_arguments_redacted, latency_ms, request_id, result_ref, error_ref, "
            "delivered_result_ref, observed_at, source_tool_call_ids, corrects_event_id "
            "FROM tool_call_events WHERE run_id = %s ORDER BY entity_id, sequence",
            (run_id,),
        ).fetchall()
    by_call: dict[uuid.UUID, list[dict[str, Any]]] = {}
    for event in events:
        by_call.setdefault(as_uuid(event["entity_id"]), []).append(event)
    return tuple(_record(row, by_call.get(as_uuid(row["tool_call_id"]), [])) for row in identities)


def _record(identity: dict[str, Any], events: list[dict[str, Any]]) -> ToolCallRecord:
    superseded = {e["corrects_event_id"] for e in events if e["corrects_event_id"] is not None}
    corrected_types = {e["event_id"]: e["event_type"] for e in events}
    live = [e for e in events if e["event_id"] not in superseded]
    fields: dict[str, Any] = {
        "effective_arguments_redacted": None,
        "decision": None,
        "status": None,
        "dispatched_at": None,
        "completed_at": None,
    }
    outcome: dict[str, Any] | None = None
    for event in live:
        kind = ToolCallEventType(event["event_type"])
        if kind is ToolCallEventType.DISPATCHED:
            fields["decision"] = ToolCallDecision.ALLOWED
            fields["dispatched_at"] = event["observed_at"]
            fields["effective_arguments_redacted"] = event["effective_arguments_redacted"]
        elif kind is ToolCallEventType.DELIVERED:
            fields["delivered_result_ref"] = event["delivered_result_ref"]
        elif kind is ToolCallEventType.REQUESTED:
            fields["status"] = ToolCallStatus.REQUESTED
            outcome = None
        else:
            fields["status"] = ToolCallStatus(kind.value)
            outcome = event
        if kind is ToolCallEventType.DENIED:
            fields["decision"] = ToolCallDecision.DENIED
            fields["deny_reason"] = event["reason"]
    if fields["status"] is None:
        raise UnknownEntity(f"tool call {identity['tool_call_id']} has no recorded request")
    if outcome is not None:
        for column in ("latency_ms", "request_id", "result_ref", "error_ref"):
            fields[column] = outcome[column]
        corrected = outcome["corrects_event_id"]
        reconciles_unknown = (
            corrected is not None
            and corrected_types.get(corrected) == ToolCallEventType.UNKNOWN.value
        )
        if reconciles_unknown:
            fields["reconciled_by_tool_call_ids"] = tuple(outcome["source_tool_call_ids"])
        elif ToolCallEventType(outcome["event_type"]) in _COMPLETION_EVENTS:
            fields["completed_at"] = outcome["observed_at"]
    try:
        return ToolCallRecord(
            identity=ToolCallIdentity(
                tool_call_id=identity["tool_call_id"],
                sdk_tool_use_id=identity["sdk_tool_use_id"],
                run_id=identity["run_id"],
                stage=identity["stage"],
                server=identity["server"],
                tool=identity["tool"],
                tier=ToolTier(identity["tier"]) if identity["tier"] is not None else None,
                requested_at=identity["requested_at"],
                arguments_redacted=identity["arguments_redacted"],
            ),
            **fields,
        )
    except ValidationError as exc:
        raise LedgerError(
            f"tool call {identity['tool_call_id']} events are inconsistent: {exc}"
        ) from exc


def _run_id_of(conn: psycopg.Connection[tuple[object, ...]], tool_call_id: uuid.UUID) -> uuid.UUID:
    row = conn.execute(
        "SELECT run_id FROM tool_calls WHERE tool_call_id = %s", (tool_call_id,)
    ).fetchone()
    if row is None:
        raise UnknownEntity(f"tool_calls has no row {tool_call_id}")
    return as_uuid(row[0])
