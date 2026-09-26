"""Identity uniqueness and closed-set CHECK constraints."""

import socket
import uuid
from datetime import UTC, datetime

import psycopg
import pytest
from pytest_socket import SocketBlockedError

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.ledger.runs import open_run_slot

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 14, tzinfo=UTC)


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
def test_tcp_sockets_stay_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_run_slot_is_unique_per_environment(conn: Conn) -> None:
    open_run_slot(conn, AppEnv.STAGING, SLOT)
    with pytest.raises(psycopg.errors.UniqueViolation, match="runs_environment_slot_key"):
        conn.execute(
            "INSERT INTO runs (run_id, environment, slot) VALUES (%s, 'staging', %s)",
            (uuid.uuid4(), SLOT),
        )
    open_run_slot(conn, AppEnv.PRODUCTION, SLOT)  # same slot, other environment: allowed


def test_slot_must_be_a_whole_utc_hour(conn: Conn) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO runs (run_id, environment, slot) VALUES (%s, 'local', %s)",
            (uuid.uuid4(), SLOT.replace(minute=30)),
        )


def test_broker_order_id_is_unique_per_account(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    insert = (
        "INSERT INTO orders (order_id, account_scope_id, broker_order_id, first_run_id) "
        "VALUES (%s, %s, %s, %s)"
    )
    conn.execute(insert, (uuid.uuid4(), "acct-a", "B-1", run.run_id))
    conn.execute(insert, (uuid.uuid4(), "acct-b", "B-1", run.run_id))
    with pytest.raises(psycopg.errors.UniqueViolation, match="orders_account_broker_order_key"):
        conn.execute(insert, (uuid.uuid4(), "acct-a", "B-1", run.run_id))


def test_place_tool_call_has_one_intent_and_sdk_id_is_unique(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    tc = uuid.uuid4()
    conn.execute(
        "INSERT INTO tool_calls (tool_call_id, run_id, sdk_tool_use_id, stage, server, tool, "
        "tier, requested_at, arguments_redacted) VALUES (%s, %s, 'toolu_1', 'agent', "
        "'robinhood', 'place_option_order', 'X', now(), '{}')",
        (tc, run.run_id),
    )
    with pytest.raises(psycopg.errors.UniqueViolation, match="tool_calls_run_sdk_tool_use_key"):
        conn.execute(
            "INSERT INTO tool_calls (tool_call_id, run_id, sdk_tool_use_id, stage, server, tool, "
            "tier, requested_at, arguments_redacted) VALUES (%s, %s, 'toolu_1', 'agent', "
            "'robinhood', 'get_quote', 'R', now(), '{}')",
            (uuid.uuid4(), run.run_id),
        )
    intent = (
        "INSERT INTO order_intents (intent_id, run_id, place_tool_call_id, account_scope_id, "
        "arguments_redacted) VALUES (%s, %s, %s, 'acct-a', '{}')"
    )
    conn.execute(intent, (uuid.uuid4(), run.run_id, tc))
    with pytest.raises(psycopg.errors.UniqueViolation, match="order_intents_place_tool_call_key"):
        conn.execute(intent, (uuid.uuid4(), run.run_id, tc))


@pytest.mark.parametrize(
    ("column", "value"),
    [("tier", "Z"), ("stage", "")],
)
def test_tool_call_closed_sets(conn: Conn, column: str, value: str) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    values = {"tier": "R", "stage": "agent", column: value}
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO tool_calls (tool_call_id, run_id, sdk_tool_use_id, stage, server, tool, "
            "tier, requested_at, arguments_redacted) VALUES (%s, %s, 'toolu_x', %s, 's', 't', "
            "%s, now(), '{}')",
            (uuid.uuid4(), run.run_id, values["stage"], values["tier"]),
        )


def test_run_event_status_only_on_status_events(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO run_events (event_id, entity_id, run_id, sequence, event_type, status, "
            "observed_at, dedup_key) VALUES (%s, %s, %s, 1, 'started', 'running', now(), 'k')",
            (uuid.uuid4(), run.run_id, run.run_id),
        )
