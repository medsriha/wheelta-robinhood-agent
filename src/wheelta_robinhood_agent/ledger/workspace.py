"""Agentic workspace objects (Tier S): identities, events, ownership and cap projections.

CLAUDE.md §9 "Agentic workspace (Tier S) rules"; INTERFACES.md "Orders, fills, and workspace
projections". An object is owned only when BOTH hold: its identity (account scope, kind,
broker ID) is recorded here, and its current projected name carries the workspace prefix.
State (name, deletion, last modification) is a projection over `workspace_events`; before/
after states are event columns, never overwrites.

A create whose outcome is ambiguous (timeout, unknown result) has no broker ID, so it cannot
have an identity row yet. It is flagged on the create tool call as an `unknown` outcome
carrying the intended kind/name; until resolved it counts against caps and blocks another
create (`unresolved_ambiguous_creates`). Resolution appends a succeeded/failed outcome that
corrects the flag and, when the object was found, records its identity.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row

from wheelta_robinhood_agent.domain.enums import ToolCallStatus
from wheelta_robinhood_agent.domain.events import WorkspaceEventType
from wheelta_robinhood_agent.domain.orders import (
    WorkspaceObject,
    WorkspaceObjectKind,
    WorkspaceObjectState,
    is_owned_name,
)
from wheelta_robinhood_agent.ledger.errors import IdentityConflict, LedgerError, UnknownEntity
from wheelta_robinhood_agent.ledger.events import AppendedEvent, EventTable, append_event, as_uuid
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.ledger.orders import effective_events
from wheelta_robinhood_agent.ledger.tool_calls import append_tool_call_outcome

Conn = psycopg.Connection[tuple[object, ...]]
Row = dict[str, Any]

AMBIGUOUS_CREATE_KEY: Final = "workspace_create:ambiguous"
RESOLVED_CREATE_KEY: Final = "workspace_create:resolved"


_MUTATIONS: Final = frozenset(
    {WorkspaceEventType.CREATED, WorkspaceEventType.UPDATED, WorkspaceEventType.DELETED}
)


def _rows(conn: Conn, query: str, params: Sequence[object]) -> list[Row]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(query.encode(), params).fetchall()


def _check_name(name: str | None) -> None:
    if name == "":
        raise ValueError("name must be non-empty or None")


# ---------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------
def record_workspace_create(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    account_scope_id: str,
    kind: WorkspaceObjectKind,
    broker_id: str,
    name: str,
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID],
    after_state: Mapping[str, object] | None = None,
) -> uuid.UUID:
    """Record a confirmed create: identity + `created` event in one transaction.

    Idempotent per (account scope, kind, broker ID). Ownership is not decided here: the
    projection applies the prefix rule to the recorded name.
    """
    if not account_scope_id or not broker_id or not name:
        raise ValueError("account_scope_id, broker_id and name must be non-empty")
    if not source_tool_call_ids:
        raise ValueError("a create needs its source tool call")
    with conn.transaction():
        inserted = conn.execute(
            "INSERT INTO workspace_objects (object_id, account_scope_id, kind, broker_id, "
            "first_run_id) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (account_scope_id, kind, broker_id) DO NOTHING RETURNING object_id",
            (new_id(), account_scope_id, kind.value, broker_id, run_id),
        ).fetchone()
        if inserted is None:
            row = conn.execute(
                "SELECT object_id FROM workspace_objects "
                "WHERE account_scope_id = %s AND kind = %s AND broker_id = %s",
                (account_scope_id, kind.value, broker_id),
            ).fetchone()
            if row is None:
                raise LedgerError("workspace object vanished after a unique conflict")
            object_id = as_uuid(row[0])
        else:
            object_id = as_uuid(inserted[0])
        append_event(
            conn,
            EventTable.WORKSPACE,
            entity_id=object_id,
            run_id=run_id,
            event_type=WorkspaceEventType.CREATED.value,
            observed_at=observed_at,
            dedup_key="created",
            source_tool_call_ids=source_tool_call_ids,
            payload={"name": name},
            extra={"before_state": None, "after_state": dict(after_state or {})},
        )
    return object_id


def _append(
    conn: Conn,
    object_id: uuid.UUID,
    event_type: WorkspaceEventType,
    *,
    run_id: uuid.UUID,
    observed_at: datetime,
    source_tool_call_id: uuid.UUID,
    name: str | None,
    before_state: Mapping[str, object] | None,
    after_state: Mapping[str, object] | None,
    corrects_event_id: uuid.UUID | None = None,
) -> AppendedEvent:
    _check_name(name)
    key = f"{event_type.value}:{source_tool_call_id}"
    if corrects_event_id is not None:
        key += f":corrects:{corrects_event_id}"
    return append_event(
        conn,
        EventTable.WORKSPACE,
        entity_id=object_id,
        run_id=run_id,
        event_type=event_type.value,
        observed_at=observed_at,
        dedup_key=key,
        source_tool_call_ids=[source_tool_call_id],
        corrects_event_id=corrects_event_id,
        payload={"name": name},
        extra={
            "before_state": dict(before_state) if before_state is not None else None,
            "after_state": dict(after_state) if after_state is not None else None,
        },
    )


def record_workspace_update(
    conn: Conn,
    object_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    name: str | None,
    before_state: Mapping[str, object],
    after_state: Mapping[str, object],
    observed_at: datetime,
    source_tool_call_id: uuid.UUID,
) -> AppendedEvent:
    """Record an update with before/after state. `name` is the new name, if it changed."""
    return _append(
        conn,
        object_id,
        WorkspaceEventType.UPDATED,
        run_id=run_id,
        observed_at=observed_at,
        source_tool_call_id=source_tool_call_id,
        name=name,
        before_state=before_state,
        after_state=after_state,
    )


def record_workspace_delete(
    conn: Conn,
    object_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    before_state: Mapping[str, object],
    observed_at: datetime,
    source_tool_call_id: uuid.UUID,
) -> AppendedEvent:
    return _append(
        conn,
        object_id,
        WorkspaceEventType.DELETED,
        run_id=run_id,
        observed_at=observed_at,
        source_tool_call_id=source_tool_call_id,
        name=None,
        before_state=before_state,
        after_state=None,
    )


def record_workspace_observation(
    conn: Conn,
    object_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    name: str | None,
    state: Mapping[str, object],
    observed_at: datetime,
    source_tool_call_id: uuid.UUID,
    corrects_event_id: uuid.UUID | None = None,
) -> AppendedEvent:
    """Record a read of the object (e.g. a user rename observed at the broker)."""
    return _append(
        conn,
        object_id,
        WorkspaceEventType.OBSERVED,
        run_id=run_id,
        observed_at=observed_at,
        source_tool_call_id=source_tool_call_id,
        name=name,
        before_state=None,
        after_state=state,
        corrects_event_id=corrects_event_id,
    )


# ---------------------------------------------------------------------------------------------
# Ambiguous creates
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AmbiguousCreate:
    """A create whose outcome is unknown; reconcile before another create."""

    create_tool_call_id: uuid.UUID
    flag_event_id: uuid.UUID
    run_id: uuid.UUID
    account_scope_id: str
    kind: WorkspaceObjectKind
    name: str
    observed_at: datetime


def flag_ambiguous_create(
    conn: Conn,
    create_tool_call_id: uuid.UUID,
    *,
    account_scope_id: str,
    kind: WorkspaceObjectKind,
    name: str,
    observed_at: datetime,
    reason: str,
) -> AppendedEvent:
    """Mark a create call's outcome unknown, with the intended object, for reconciliation."""
    if not account_scope_id or not name or not reason:
        raise ValueError("account_scope_id, name and reason must be non-empty")
    return append_tool_call_outcome(
        conn,
        create_tool_call_id,
        ToolCallStatus.UNKNOWN,
        observed_at=observed_at,
        dedup_key=AMBIGUOUS_CREATE_KEY,
        reason=reason,
        payload={
            "workspace_create": {
                "account_scope_id": account_scope_id,
                "kind": kind.value,
                "name": name,
            }
        },
    )


def unresolved_ambiguous_creates(conn: Conn, account_scope_id: str) -> tuple[AmbiguousCreate, ...]:
    rows = _rows(
        conn,
        "SELECT e.event_id, e.entity_id, e.run_id, e.observed_at, e.payload "
        "FROM tool_call_events e "
        "WHERE e.event_type = 'unknown' AND e.dedup_key = %s "
        "AND e.payload->'workspace_create'->>'account_scope_id' = %s "
        "AND NOT EXISTS (SELECT 1 FROM tool_call_events c "
        "                WHERE c.entity_id = e.entity_id AND c.corrects_event_id = e.event_id) "
        "ORDER BY e.observed_at, e.event_id",
        (AMBIGUOUS_CREATE_KEY, account_scope_id),
    )
    return tuple(
        AmbiguousCreate(
            create_tool_call_id=r["entity_id"],
            flag_event_id=r["event_id"],
            run_id=r["run_id"],
            account_scope_id=account_scope_id,
            kind=WorkspaceObjectKind(r["payload"]["workspace_create"]["kind"]),
            name=r["payload"]["workspace_create"]["name"],
            observed_at=r["observed_at"],
        )
        for r in rows
    )


def resolve_ambiguous_create(
    conn: Conn,
    create_tool_call_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    observed_at: datetime,
    broker_id: str | None,
    source_tool_call_ids: Sequence[uuid.UUID],
    after_state: Mapping[str, object] | None = None,
) -> uuid.UUID | None:
    """Resolve a flagged create from broker reads (`source_tool_call_ids`).

    With `broker_id` the object exists: its identity and `created` event are recorded and
    its object ID returned. With None the reads showed no such object. Either way a
    succeeded/failed outcome correcting the flag is appended to the create call.
    """
    if not source_tool_call_ids:
        raise ValueError("resolution needs broker reads")
    with conn.transaction():
        flags = _rows(
            conn,
            "SELECT event_id, payload FROM tool_call_events "
            "WHERE entity_id = %s AND dedup_key = %s",
            (create_tool_call_id, AMBIGUOUS_CREATE_KEY),
        )
        if not flags:
            raise UnknownEntity(f"create call {create_tool_call_id} has no ambiguous flag")
        flag = flags[0]
        intended = flag["payload"]["workspace_create"]
        object_id: uuid.UUID | None = None
        if broker_id is not None:
            object_id = record_workspace_create(
                conn,
                run_id=run_id,
                account_scope_id=intended["account_scope_id"],
                kind=WorkspaceObjectKind(intended["kind"]),
                broker_id=broker_id,
                name=intended["name"],
                observed_at=observed_at,
                source_tool_call_ids=[create_tool_call_id, *source_tool_call_ids],
                after_state=after_state,
            )
        append_tool_call_outcome(
            conn,
            create_tool_call_id,
            ToolCallStatus.SUCCEEDED if broker_id is not None else ToolCallStatus.FAILED,
            observed_at=observed_at,
            dedup_key=RESOLVED_CREATE_KEY,
            source_tool_call_ids=source_tool_call_ids,
            corrects_event_id=flag["event_id"],
            payload={"workspace_create": {**intended, "broker_id": broker_id}},
        )
    return object_id


# ---------------------------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------------------------
def _state(identity_row: Row, events: list[Row]) -> WorkspaceObjectState:
    live = effective_events(events)
    created = next((e for e in live if e["event_type"] == WorkspaceEventType.CREATED), None)
    name: str | None = None
    deleted = False
    last_modified: datetime = identity_row["recorded_at"]
    modified_seen = False
    for e in live:
        kind = WorkspaceEventType(e["event_type"])
        if e["payload"].get("name") is not None:
            name = e["payload"]["name"]
        if kind is WorkspaceEventType.DELETED:
            deleted = True
        if kind in _MUTATIONS:
            if not modified_seen or e["observed_at"] > last_modified:
                last_modified = e["observed_at"]
            modified_seen = True
    identity = WorkspaceObject(
        object_id=identity_row["object_id"],
        account_scope_id=identity_row["account_scope_id"],
        kind=WorkspaceObjectKind(identity_row["kind"]),
        broker_id=identity_row["broker_id"],
        created_run_id=identity_row["first_run_id"],
        created_at=created["observed_at"] if created else identity_row["recorded_at"],
    )
    return WorkspaceObjectState(
        identity=identity,
        current_name=name,
        ledger_recorded=True,
        last_modified_at=last_modified,
        deleted=deleted,
    )


_IDENTITY_SELECT: Final = (
    "SELECT object_id, account_scope_id, kind, broker_id, first_run_id, recorded_at "
    "FROM workspace_objects WHERE "
)
_ORDER: Final = " ORDER BY recorded_at, object_id"
# Closed set of identity filters (no caller-supplied SQL).
_BY_ID: Final = _IDENTITY_SELECT + "object_id = %s" + _ORDER
_BY_BROKER_ID: Final = (
    _IDENTITY_SELECT + "account_scope_id = %s AND kind = %s AND broker_id = %s" + _ORDER
)
_BY_ACCOUNT: Final = _IDENTITY_SELECT + "account_scope_id = %s" + _ORDER


def _states(conn: Conn, query: str, params: Sequence[object]) -> list[WorkspaceObjectState]:
    identities = _rows(conn, query, params)
    ids = [r["object_id"] for r in identities]
    grouped: dict[uuid.UUID, list[Row]] = {i: [] for i in ids}
    if ids:
        for e in _rows(
            conn,
            "SELECT * FROM workspace_events WHERE entity_id = ANY(%s) ORDER BY entity_id, sequence",
            (ids,),
        ):
            grouped[e["entity_id"]].append(e)
    return [_state(r, grouped[r["object_id"]]) for r in identities]


def workspace_object_state(conn: Conn, object_id: uuid.UUID) -> WorkspaceObjectState:
    states = _states(conn, _BY_ID, (object_id,))
    if not states:
        raise UnknownEntity(f"workspace_objects has no row {object_id}")
    return states[0]


def owned_object(
    conn: Conn, account_scope_id: str, kind: WorkspaceObjectKind, broker_id: str
) -> WorkspaceObjectState | None:
    """The recorded object for this broker ID, or None if the ledger never recorded it.

    None means not owned (user-created or unknown). A returned state is owned only if
    `state.owned(prefix)` holds: recorded AND prefixed name AND not deleted.
    """
    states = _states(
        conn,
        _BY_BROKER_ID,
        (account_scope_id, kind.value, broker_id),
    )
    return states[0] if states else None


@dataclass(frozen=True)
class WorkspaceCounts:
    """Inputs for the `[workspace]` caps (limits themselves live in the trading rules).

    `active_owned` counts owned (prefix + recorded), undeleted objects per kind;
    `ambiguous_creates` counts unresolved ambiguous creates per kind, which callers should add
    to `active_owned` when checking caps (fail closed); `mutations_in_run` counts recorded
    create/update/delete events plus ambiguous creates flagged in `run_id`.
    """

    active_owned: Mapping[WorkspaceObjectKind, int]
    ambiguous_creates: Mapping[WorkspaceObjectKind, int]
    mutations_in_run: int


def owned_counts(
    conn: Conn, account_scope_id: str, *, prefix: str, run_id: uuid.UUID
) -> WorkspaceCounts:
    if not prefix:
        raise IdentityConflict("an empty workspace prefix owns nothing")
    active = dict.fromkeys(WorkspaceObjectKind, 0)
    for state in _states(conn, _BY_ACCOUNT, (account_scope_id,)):
        if state.owned(prefix):
            active[state.identity.kind] += 1
    ambiguous_rows = unresolved_ambiguous_creates(conn, account_scope_id)
    ambiguous = dict.fromkeys(WorkspaceObjectKind, 0)
    for a in ambiguous_rows:
        ambiguous[a.kind] += 1
    row = conn.execute(
        "SELECT count(*) FROM workspace_events e "
        "JOIN workspace_objects o ON o.object_id = e.entity_id "
        "WHERE o.account_scope_id = %s AND e.run_id = %s "
        "AND e.event_type IN ('created', 'updated', 'deleted')",
        (account_scope_id, run_id),
    ).fetchone()
    mutations = int(str(row[0])) if row is not None else 0
    mutations += sum(1 for a in ambiguous_rows if a.run_id == run_id)
    return WorkspaceCounts(active, ambiguous, mutations)


__all__ = [
    "AmbiguousCreate",
    "WorkspaceCounts",
    "WorkspaceEventType",
    "flag_ambiguous_create",
    "is_owned_name",
    "owned_counts",
    "owned_object",
    "record_workspace_create",
    "record_workspace_delete",
    "record_workspace_observation",
    "record_workspace_update",
    "resolve_ambiguous_create",
    "unresolved_ambiguous_creates",
    "workspace_object_state",
]
