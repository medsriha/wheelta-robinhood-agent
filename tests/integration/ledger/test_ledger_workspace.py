"""Workspace object identities, events, ownership, caps, and ambiguous creates."""

import uuid
from datetime import UTC, datetime, timedelta
from itertools import count

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv, ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.orders import WorkspaceObjectKind
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import record_tool_call_requested, tool_call_status
from wheelta_robinhood_agent.ledger.workspace import (
    flag_ambiguous_create,
    owned_counts,
    owned_object,
    record_workspace_create,
    record_workspace_delete,
    record_workspace_observation,
    record_workspace_update,
    resolve_ambiguous_create,
    unresolved_ambiguous_creates,
    workspace_object_state,
)

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 15, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=5)
ACCOUNT = "acct-scope-1"
PREFIX = "wra-"
WATCHLIST = WorkspaceObjectKind.WATCHLIST
_sdk = count(1)


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


def _call(conn: Conn, run_id: uuid.UUID, tool: str = "create_watchlist") -> uuid.UUID:
    return record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id=f"toolu_{next(_sdk)}",
        stage="agent",
        server="robinhood",
        tool=tool,
        tier=ToolTier.S,
        arguments_redacted={},
        requested_at=T0,
    ).tool_call_id


def _create(conn: Conn, run_id: uuid.UUID, broker_id: str, name: str) -> uuid.UUID:
    return record_workspace_create(
        conn,
        run_id=run_id,
        account_scope_id=ACCOUNT,
        kind=WATCHLIST,
        broker_id=broker_id,
        name=name,
        observed_at=T0,
        source_tool_call_ids=[_call(conn, run_id)],
        after_state={"name": name, "items": []},
    )


def test_ownership_requires_prefix_and_recorded_id(conn: Conn, run_id: uuid.UUID) -> None:
    _create(conn, run_id, "wl-owned", "wra-held")
    _create(conn, run_id, "wl-unprefixed", "Held names")

    owned = owned_object(conn, ACCOUNT, WATCHLIST, "wl-owned")
    assert owned is not None and owned.owned(PREFIX)
    unprefixed = owned_object(conn, ACCOUNT, WATCHLIST, "wl-unprefixed")
    assert unprefixed is not None and not unprefixed.owned(PREFIX)
    # A prefixed user-created object the ledger never recorded is not ours.
    assert owned_object(conn, ACCOUNT, WATCHLIST, "wl-user-made-wra-name") is None
    # Scope and kind are part of the identity.
    assert owned_object(conn, "other-account", WATCHLIST, "wl-owned") is None
    assert owned_object(conn, ACCOUNT, WorkspaceObjectKind.SCAN, "wl-owned") is None


def test_state_rebuilds_from_events(conn: Conn, run_id: uuid.UUID) -> None:
    object_id = _create(conn, run_id, "wl-1", "wra-held")
    assert _create(conn, run_id, "wl-1", "wra-held") == object_id  # idempotent identity

    rename = _call(conn, run_id, "update_watchlist")
    record_workspace_update(
        conn,
        object_id,
        run_id=run_id,
        name="wra-held-2",
        before_state={"name": "wra-held"},
        after_state={"name": "wra-held-2"},
        observed_at=T0 + timedelta(minutes=1),
        source_tool_call_id=rename,
    )
    state = workspace_object_state(conn, object_id)
    assert state.current_name == "wra-held-2"
    assert state.last_modified_at == T0 + timedelta(minutes=1)
    assert state.owned(PREFIX)

    # A user rename observed at the broker removes ownership (prefix no longer matches).
    record_workspace_observation(
        conn,
        object_id,
        run_id=run_id,
        name="My list",
        state={"name": "My list"},
        observed_at=T0 + timedelta(minutes=2),
        source_tool_call_id=_call(conn, run_id, "get_watchlists"),
    )
    assert not workspace_object_state(conn, object_id).owned(PREFIX)

    record_workspace_delete(
        conn,
        object_id,
        run_id=run_id,
        before_state={"name": "My list"},
        observed_at=T0 + timedelta(minutes=3),
        source_tool_call_id=_call(conn, run_id, "delete_watchlist"),
    )
    state = workspace_object_state(conn, object_id)
    assert state.deleted
    before_after = conn.execute(
        "SELECT event_type, before_state, after_state FROM workspace_events "
        "WHERE entity_id = %s ORDER BY sequence",
        (object_id,),
    ).fetchall()
    assert [r[0] for r in before_after] == ["created", "updated", "observed", "deleted"]
    assert before_after[1][1:] == ({"name": "wra-held"}, {"name": "wra-held-2"})


def test_duplicate_update_observation_is_deduplicated(conn: Conn, run_id: uuid.UUID) -> None:
    object_id = _create(conn, run_id, "wl-1", "wra-held")
    call = _call(conn, run_id, "update_watchlist")
    results = [
        record_workspace_update(
            conn,
            object_id,
            run_id=run_id,
            name=None,
            before_state={"items": []},
            after_state={"items": ["AAPL"]},
            observed_at=T0,
            source_tool_call_id=call,
        )
        for _ in range(2)
    ]
    assert results[0].created and not results[1].created


def test_counts_for_caps(conn: Conn, run_id: uuid.UUID) -> None:
    _create(conn, run_id, "wl-1", "wra-a")
    deleted = _create(conn, run_id, "wl-2", "wra-b")
    record_workspace_delete(
        conn,
        deleted,
        run_id=run_id,
        before_state={},
        observed_at=T0,
        source_tool_call_id=_call(conn, run_id, "delete_watchlist"),
    )
    _create(conn, run_id, "wl-3", "not ours")
    flag_ambiguous_create(
        conn,
        _call(conn, run_id),
        account_scope_id=ACCOUNT,
        kind=WATCHLIST,
        name="wra-c",
        observed_at=T0,
        reason="create timed out",
    )
    counts = owned_counts(conn, ACCOUNT, prefix=PREFIX, run_id=run_id)
    assert counts.active_owned[WATCHLIST] == 1
    assert counts.ambiguous_creates[WATCHLIST] == 1
    # Three creates + one delete recorded, plus one ambiguous create.
    assert counts.mutations_in_run == 5
    other_run = open_run_slot(conn, AppEnv.LOCAL, SLOT + timedelta(hours=1)).run_id
    assert owned_counts(conn, ACCOUNT, prefix=PREFIX, run_id=other_run).mutations_in_run == 0


def test_ambiguous_create_is_flagged_then_reconciled(conn: Conn, run_id: uuid.UUID) -> None:
    create_call = _call(conn, run_id)
    flag = flag_ambiguous_create(
        conn,
        create_call,
        account_scope_id=ACCOUNT,
        kind=WATCHLIST,
        name="wra-held",
        observed_at=T0,
        reason="create timed out",
    )
    (pending,) = unresolved_ambiguous_creates(conn, ACCOUNT)
    assert (pending.create_tool_call_id, pending.name) == (create_call, "wra-held")
    assert tool_call_status(conn, create_call) is ToolCallStatus.UNKNOWN
    assert unresolved_ambiguous_creates(conn, "other-account") == ()

    read = _call(conn, run_id, "get_watchlists")
    object_id = resolve_ambiguous_create(
        conn,
        create_call,
        run_id=run_id,
        observed_at=T0 + timedelta(minutes=1),
        broker_id="wl-found",
        source_tool_call_ids=[read],
    )
    assert object_id is not None
    assert unresolved_ambiguous_creates(conn, ACCOUNT) == ()
    assert tool_call_status(conn, create_call) is ToolCallStatus.SUCCEEDED
    corrected = conn.execute(
        "SELECT corrects_event_id FROM tool_call_events "
        "WHERE entity_id = %s AND event_type = 'succeeded'",
        (create_call,),
    ).fetchone()
    assert corrected == (flag.event_id,)
    state = owned_object(conn, ACCOUNT, WATCHLIST, "wl-found")
    assert state is not None and state.owned(PREFIX)


def test_ambiguous_create_not_found_resolves_as_failed(conn: Conn, run_id: uuid.UUID) -> None:
    create_call = _call(conn, run_id)
    flag_ambiguous_create(
        conn,
        create_call,
        account_scope_id=ACCOUNT,
        kind=WATCHLIST,
        name="wra-held",
        observed_at=T0,
        reason="unknown result",
    )
    result = resolve_ambiguous_create(
        conn,
        create_call,
        run_id=run_id,
        observed_at=T0,
        broker_id=None,
        source_tool_call_ids=[_call(conn, run_id, "get_watchlists")],
    )
    assert result is None
    assert unresolved_ambiguous_creates(conn, ACCOUNT) == ()
    assert tool_call_status(conn, create_call) is ToolCallStatus.FAILED
