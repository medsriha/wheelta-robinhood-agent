"""Tool-call identity, requested-before-outcome ordering, and outcome deduplication."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv, ToolCallStatus, ToolTier
from wheelta_robinhood_agent.ledger.errors import DedupConflict, IdentityConflict, UnknownEntity
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import (
    ToolCallRef,
    append_tool_call_outcome,
    record_tool_call_dispatched,
    record_tool_call_requested,
    tool_call_status,
)

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 16, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=3)


def _request(
    conn: Conn, run_id: uuid.UUID, sdk_id: str = "toolu_1", tool: str = "get_quote"
) -> ToolCallRef:
    return record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id=sdk_id,
        stage="agent",
        server="robinhood",
        tool=tool,
        tier=ToolTier.R,
        arguments_redacted={"symbol": "AAPL"},
        requested_at=T0,
    )


def _events(conn: Conn, tool_call_id: uuid.UUID) -> list[tuple[object, ...]]:
    return conn.execute(
        "SELECT sequence, event_type, dedup_key FROM tool_call_events "
        "WHERE entity_id = %s ORDER BY sequence",
        (tool_call_id,),
    ).fetchall()


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


def test_requested_is_committed_with_the_identity(
    conn: Conn, run_id: uuid.UUID, conn_factory: Callable[[], Conn]
) -> None:
    ref = _request(conn, run_id)
    assert ref.created
    # Visible from another session immediately: durable before dispatch.
    other = conn_factory()
    assert _events(other, ref.tool_call_id) == [(1, "requested", "requested")]
    assert tool_call_status(other, ref.tool_call_id) is ToolCallStatus.REQUESTED


def test_outcome_follows_request_and_is_deduplicated(conn: Conn, run_id: uuid.UUID) -> None:
    ref = _request(conn, run_id)
    record_tool_call_dispatched(
        conn, ref.tool_call_id, dispatched_at=T0, effective_arguments_redacted={"symbol": "AAPL"}
    )
    first = append_tool_call_outcome(
        conn, ref.tool_call_id, ToolCallStatus.SUCCEEDED, observed_at=T0, latency_ms=120
    )
    again = append_tool_call_outcome(
        conn, ref.tool_call_id, ToolCallStatus.SUCCEEDED, observed_at=T0, latency_ms=120
    )
    assert first.created and not again.created and first.event_id == again.event_id
    assert [e[1] for e in _events(conn, ref.tool_call_id)] == [
        "requested",
        "dispatched",
        "succeeded",
    ]
    assert tool_call_status(conn, ref.tool_call_id) is ToolCallStatus.SUCCEEDED


def test_same_source_key_with_a_different_outcome_conflicts(conn: Conn, run_id: uuid.UUID) -> None:
    ref = _request(conn, run_id)
    append_tool_call_outcome(
        conn, ref.tool_call_id, ToolCallStatus.UNKNOWN, observed_at=T0, dedup_key="hook:post"
    )
    with pytest.raises(DedupConflict):
        append_tool_call_outcome(
            conn, ref.tool_call_id, ToolCallStatus.FAILED, observed_at=T0, dedup_key="hook:post"
        )


def test_unknown_is_resolved_by_a_correcting_event(conn: Conn, run_id: uuid.UUID) -> None:
    ref = _request(conn, run_id, tool="place_option_order")
    unknown = append_tool_call_outcome(
        conn, ref.tool_call_id, ToolCallStatus.UNKNOWN, observed_at=T0
    )
    append_tool_call_outcome(
        conn,
        ref.tool_call_id,
        ToolCallStatus.SUCCEEDED,
        observed_at=T0 + timedelta(minutes=1),
        dedup_key="reconcile:order-read",
        corrects_event_id=unknown.event_id,
    )
    assert tool_call_status(conn, ref.tool_call_id) is ToolCallStatus.SUCCEEDED
    assert len(_events(conn, ref.tool_call_id)) == 3


def test_request_is_idempotent_per_sdk_id(conn: Conn, run_id: uuid.UUID) -> None:
    a = _request(conn, run_id)
    b = _request(conn, run_id)
    assert not b.created and a.tool_call_id == b.tool_call_id
    assert len(_events(conn, a.tool_call_id)) == 1
    with pytest.raises(IdentityConflict):
        _request(conn, run_id, tool="get_positions")


def test_denied_requires_reason_and_requested_is_not_an_outcome(
    conn: Conn, run_id: uuid.UUID
) -> None:
    ref = _request(conn, run_id)
    with pytest.raises(ValueError, match="reason"):
        append_tool_call_outcome(conn, ref.tool_call_id, ToolCallStatus.DENIED, observed_at=T0)
    with pytest.raises(ValueError, match="outcome"):
        append_tool_call_outcome(conn, ref.tool_call_id, ToolCallStatus.REQUESTED, observed_at=T0)
    append_tool_call_outcome(
        conn, ref.tool_call_id, ToolCallStatus.DENIED, observed_at=T0, reason="not_allowlisted"
    )
    assert tool_call_status(conn, ref.tool_call_id) is ToolCallStatus.DENIED


def test_outcome_without_a_request_is_rejected(conn: Conn) -> None:
    with pytest.raises(UnknownEntity):
        append_tool_call_outcome(conn, uuid.uuid4(), ToolCallStatus.FAILED, observed_at=T0)


def test_unregistered_tool_has_no_tier(conn: Conn, run_id: uuid.UUID) -> None:
    ref = record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id="toolu_unknown",
        stage="agent",
        server="robinhood",
        tool="mystery_tool",
        tier=None,
        arguments_redacted={},
        requested_at=T0,
    )
    tier = conn.execute(
        "SELECT tier FROM tool_calls WHERE tool_call_id = %s", (ref.tool_call_id,)
    ).fetchone()
    assert tier == (None,)
