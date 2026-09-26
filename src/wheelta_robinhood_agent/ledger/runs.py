"""Run identity, run events, and the run-status projection (CLAUDE.md §15, INTERFACES.md).

A run is one row per (environment, slot) with a deterministic run_id. Status is never stored
on that row: it is the latest `status` event. Slot classification:

- NEW: this call inserted the identity row.
- COMPLETED: the latest status event is a finalizing status. The caller no-ops.
- INTERRUPTED: the row exists but no finalizing status was recorded (the process died or was
  killed mid-run). The caller reconciles and finalizes without starting a second trading
  session (INTERFACES.md "Ledger tables and recovery").

Every status except `running` and `skipped_concurrent` finalizes a slot. `skipped_concurrent`
never finalizes: a process that failed to take the lock must not write to the slot at all,
and if such an event exists anyway it cannot mark another process's run as done.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

import psycopg

from wheelta_robinhood_agent.domain.enums import AppEnv, RunStatus
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.run_identity import run_id_for, slot_for
from wheelta_robinhood_agent.ledger.errors import IdentityConflict, LedgerError, UnknownEntity
from wheelta_robinhood_agent.ledger.events import AppendedEvent, EventTable, append_event, as_uuid

NON_FINAL_STATUSES: Final = frozenset({RunStatus.RUNNING, RunStatus.SKIPPED_CONCURRENT})
FINAL_STATUSES: Final = frozenset(set(RunStatus) - NON_FINAL_STATUSES)


class SlotState(StrEnum):
    NEW = "new"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class RunSlot:
    run_id: uuid.UUID
    environment: AppEnv
    slot: datetime
    state: SlotState
    current_status: RunStatus | None


@dataclass(frozen=True)
class RunProjection:
    """Current run state derived from events. Times are UTC observation times from events."""

    run_id: uuid.UUID
    status: RunStatus | None
    started_at: datetime | None
    finished_at: datetime | None


def open_run_slot(
    conn: psycopg.Connection[tuple[object, ...]], environment: AppEnv, slot: datetime
) -> RunSlot:
    """Create or get the run identity for (environment, slot) and classify the slot.

    Call only while holding the single-flight lock (lock.py). Raises ValueError if `slot` is
    not a whole UTC hour and IdentityConflict if a stored row has a different run_id.
    """
    if slot != slot_for(slot):
        raise ValueError(f"slot must be a whole UTC hour, got {slot.isoformat()}")
    run_id = run_id_for(environment, slot)
    with conn.transaction():
        inserted = conn.execute(
            "INSERT INTO runs (run_id, environment, slot) VALUES (%s, %s, %s) "
            "ON CONFLICT (environment, slot) DO NOTHING RETURNING run_id",
            (run_id, environment.value, slot),
        ).fetchone()
        if inserted is not None:
            return RunSlot(run_id, environment, slot, SlotState.NEW, None)
        stored = conn.execute(
            "SELECT run_id FROM runs WHERE environment = %s AND slot = %s",
            (environment.value, slot),
        ).fetchone()
    if stored is None:
        raise LedgerError("run row vanished after a unique conflict")
    if as_uuid(stored[0]) != run_id:
        raise IdentityConflict(f"stored run_id for {environment.value} {slot.isoformat()} differs")
    status = current_run_status(conn, run_id)
    state = SlotState.COMPLETED if status in FINAL_STATUSES else SlotState.INTERRUPTED
    return RunSlot(run_id, environment, slot, state, status)


def append_run_event(
    conn: psycopg.Connection[tuple[object, ...]],
    run_id: uuid.UUID,
    event_type: RunEventType,
    *,
    observed_at: datetime,
    dedup_key: str,
    payload: Mapping[str, object] | None = None,
    status: RunStatus | None = None,
    source_as_of: datetime | None = None,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
    corrects_event_id: uuid.UUID | None = None,
) -> AppendedEvent:
    """Append a run event; `status` is required on STATUS events and forbidden otherwise."""
    if (event_type is RunEventType.STATUS) != (status is not None):
        raise ValueError("status must be given exactly for STATUS events")
    return append_event(
        conn,
        EventTable.RUN,
        entity_id=run_id,
        run_id=run_id,
        event_type=event_type.value,
        observed_at=observed_at,
        dedup_key=dedup_key,
        payload=payload,
        source_as_of=source_as_of,
        source_tool_call_ids=source_tool_call_ids,
        corrects_event_id=corrects_event_id,
        extra={"status": status.value} if status is not None else None,
    )


def current_run_status(
    conn: psycopg.Connection[tuple[object, ...]], run_id: uuid.UUID
) -> RunStatus | None:
    """The latest non-`skipped_concurrent` status event, or None if none was recorded."""
    return run_projection(conn, run_id).status


def run_projection(
    conn: psycopg.Connection[tuple[object, ...]], run_id: uuid.UUID
) -> RunProjection:
    """Rebuild current run state from run_events (no mutable status column exists)."""
    exists = conn.execute("SELECT 1 FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    if exists is None:
        raise UnknownEntity(f"runs has no row {run_id}")
    rows = conn.execute(
        "SELECT event_type, status, observed_at FROM run_events "
        "WHERE entity_id = %s ORDER BY sequence",
        (run_id,),
    ).fetchall()
    status: RunStatus | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    for event_type, raw_status, observed_at in rows:
        if not isinstance(observed_at, datetime):
            raise LedgerError("run event observed_at is not a timestamp")
        if event_type == RunEventType.STARTED.value and started_at is None:
            started_at = observed_at
        if event_type != RunEventType.STATUS.value or not isinstance(raw_status, str):
            continue
        parsed = RunStatus(raw_status)
        if parsed is RunStatus.SKIPPED_CONCURRENT:
            continue
        status = parsed
        finished_at = observed_at if parsed in FINAL_STATUSES else None
    return RunProjection(run_id, status, started_at, finished_at)


def run_event_payloads(
    conn: psycopg.Connection[tuple[object, ...]], run_id: uuid.UUID, event_type: RunEventType
) -> tuple[Mapping[str, object], ...]:
    """Payloads of this run's events of one type, in sequence order (for recovery)."""
    rows = conn.execute(
        "SELECT payload FROM run_events WHERE entity_id = %s AND event_type = %s ORDER BY sequence",
        (run_id, event_type.value),
    ).fetchall()
    return tuple(row[0] for row in rows if isinstance(row[0], dict))
