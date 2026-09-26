"""ToolCallRecord projection rebuilt from tool_calls + tool_call_events (INTERFACES.md)."""

import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv, ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.orders import WorkspaceObjectKind
from wheelta_robinhood_agent.domain.tool_calls import ToolCallDecision
from wheelta_robinhood_agent.ledger.errors import LedgerError
from wheelta_robinhood_agent.ledger.evidence import ResultKind, insert_result
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import (
    append_tool_call_outcome,
    record_tool_call_delivered,
    record_tool_call_dispatched,
    record_tool_call_requested,
    tool_call_records,
)
from wheelta_robinhood_agent.ledger.workspace import flag_ambiguous_create, resolve_ambiguous_create

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 16, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=3)


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


def _request(
    conn: Conn,
    run_id: uuid.UUID,
    sdk_id: str,
    *,
    at: datetime = T0,
    tool: str = "get_quote",
    tier: ToolTier | None = ToolTier.R,
) -> uuid.UUID:
    return record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id=sdk_id,
        stage="agent",
        server="robinhood",
        tool=tool,
        tier=tier,
        arguments_redacted={"symbol": "AAPL"},
        requested_at=at,
    ).tool_call_id


def _dispatch(conn: Conn, tool_call_id: uuid.UUID, at: datetime = T0) -> None:
    record_tool_call_dispatched(
        conn, tool_call_id, dispatched_at=at, effective_arguments_redacted={"symbol": "MSFT"}
    )


def _result(conn: Conn, run_id: uuid.UUID, tool_call_id: uuid.UUID, kind: str) -> uuid.UUID:
    return insert_result(
        conn, run_id=run_id, tool_call_id=tool_call_id, kind=ResultKind(kind), payload={"ok": 1}
    )


def test_empty_run_has_no_records(conn: Conn, run_id: uuid.UUID) -> None:
    assert tool_call_records(conn, run_id) == ()


def test_succeeded_call_carries_dispatch_completion_and_refs(conn: Conn, run_id: uuid.UUID) -> None:
    call = _request(conn, run_id, "toolu_ok")
    _dispatch(conn, call, T0 + timedelta(seconds=1))
    result = _result(conn, run_id, call, "validated")
    delivered = _result(conn, run_id, call, "delivered")
    append_tool_call_outcome(
        conn,
        call,
        ToolCallStatus.SUCCEEDED,
        observed_at=T0 + timedelta(seconds=2),
        latency_ms=900,
        request_id="req-1",
        result_ref=result,
    )
    record_tool_call_delivered(
        conn, call, delivered_result_ref=delivered, observed_at=T0 + timedelta(seconds=3)
    )
    (record,) = tool_call_records(conn, run_id)
    assert record.identity.tool_call_id == call
    assert record.identity.tier is ToolTier.R
    assert record.identity.arguments_redacted == {"symbol": "AAPL"}
    assert record.effective_arguments_redacted == {"symbol": "MSFT"}
    assert record.decision is ToolCallDecision.ALLOWED
    assert record.status is ToolCallStatus.SUCCEEDED
    assert record.dispatched_at == T0 + timedelta(seconds=1)
    assert record.completed_at == T0 + timedelta(seconds=2)
    assert (record.latency_ms, record.request_id) == (900, "req-1")
    assert record.result_ref == result and record.error_ref is None
    assert record.delivered_result_ref == delivered
    assert record.reconciled_by_tool_call_ids == ()
    assert not record.outcome_unresolved


def test_denied_call_is_never_dispatched(conn: Conn, run_id: uuid.UUID) -> None:
    call = _request(conn, run_id, "toolu_denied", tool="place_equity_order", tier=None)
    append_tool_call_outcome(
        conn, call, ToolCallStatus.DENIED, observed_at=T0, reason="unknown tool"
    )
    (record,) = tool_call_records(conn, run_id)
    assert record.identity.tier is None
    assert record.decision is ToolCallDecision.DENIED
    assert record.status is ToolCallStatus.DENIED
    assert record.deny_reason == "unknown tool"
    assert record.dispatched_at is None and record.completed_at is None


def test_requested_only_call_has_no_decision(conn: Conn, run_id: uuid.UUID) -> None:
    _request(conn, run_id, "toolu_pending")
    (record,) = tool_call_records(conn, run_id)
    assert record.decision is None
    assert record.status is ToolCallStatus.REQUESTED
    assert record.dispatched_at is None and not record.outcome_unresolved


def test_dispatched_without_outcome_stays_requested_not_failed(
    conn: Conn, run_id: uuid.UUID
) -> None:
    call = _request(conn, run_id, "toolu_x", tool="place_option_order", tier=ToolTier.X)
    _dispatch(conn, call)
    (record,) = tool_call_records(conn, run_id)
    assert record.status is ToolCallStatus.REQUESTED
    assert record.completed_at is None
    assert record.outcome_unresolved


def test_unknown_outcome_is_kept_unknown(conn: Conn, run_id: uuid.UUID) -> None:
    call = _request(conn, run_id, "toolu_x", tool="place_option_order", tier=ToolTier.X)
    _dispatch(conn, call)
    append_tool_call_outcome(
        conn, call, ToolCallStatus.UNKNOWN, observed_at=T0, reason="timeout at transport"
    )
    (record,) = tool_call_records(conn, run_id)
    assert record.status is ToolCallStatus.UNKNOWN
    assert record.completed_at is None
    assert record.outcome_unresolved


def test_failed_call_carries_error_ref(conn: Conn, run_id: uuid.UUID) -> None:
    call = _request(conn, run_id, "toolu_fail")
    _dispatch(conn, call)
    error = _result(conn, run_id, call, "error")
    append_tool_call_outcome(
        conn, call, ToolCallStatus.FAILED, observed_at=T0 + timedelta(seconds=1), error_ref=error
    )
    (record,) = tool_call_records(conn, run_id)
    assert record.status is ToolCallStatus.FAILED
    assert record.error_ref == error and record.result_ref is None
    assert record.completed_at == T0 + timedelta(seconds=1)


def test_reconciled_unknown_names_the_broker_reads(conn: Conn, run_id: uuid.UUID) -> None:
    create = _request(conn, run_id, "toolu_create", tool="create_watchlist", tier=ToolTier.S)
    _dispatch(conn, create)
    flag_ambiguous_create(
        conn,
        create,
        account_scope_id="acct",
        kind=WorkspaceObjectKind.WATCHLIST,
        name="agent-wl",
        observed_at=T0,
        reason="timeout",
    )
    read = _request(conn, run_id, "toolu_read", at=T0 + timedelta(minutes=1))
    resolve_ambiguous_create(
        conn,
        create,
        run_id=run_id,
        observed_at=T0 + timedelta(minutes=2),
        broker_id=None,
        source_tool_call_ids=[read],
    )
    records = tool_call_records(conn, run_id)
    assert [r.identity.tool_call_id for r in records] == [create, read]
    resolved = records[0]
    assert resolved.status is ToolCallStatus.FAILED
    assert resolved.reconciled_by_tool_call_ids == (read,)
    assert resolved.completed_at is None


def test_order_is_requested_at_then_tool_call_id(conn: Conn, run_id: uuid.UUID) -> None:
    later = _request(conn, run_id, "toolu_b", at=T0 + timedelta(seconds=5))
    a = _request(conn, run_id, "toolu_a1", at=T0)
    b = _request(conn, run_id, "toolu_a2", at=T0)
    ids = [r.identity.tool_call_id for r in tool_call_records(conn, run_id)]
    assert ids == [*sorted([a, b]), later]


def test_other_runs_are_excluded(conn: Conn, run_id: uuid.UUID) -> None:
    other = open_run_slot(conn, AppEnv.LOCAL, SLOT + timedelta(hours=1)).run_id
    _request(conn, other, "toolu_other")
    mine = _request(conn, run_id, "toolu_mine")
    assert [r.identity.tool_call_id for r in tool_call_records(conn, run_id)] == [mine]


def test_contradictory_events_raise_instead_of_guessing(conn: Conn, run_id: uuid.UUID) -> None:
    call = _request(conn, run_id, "toolu_bad")
    _dispatch(conn, call)
    append_tool_call_outcome(conn, call, ToolCallStatus.DENIED, observed_at=T0, reason="late")
    with pytest.raises(LedgerError, match="inconsistent"):
        tool_call_records(conn, run_id)
