"""Every ledger table rejects UPDATE, DELETE, and TRUNCATE (corrections are new rows)."""

import psycopg
import pytest
from psycopg import sql

LEDGER_TABLES = [
    "runs", "tool_calls", "order_intents", "orders", "positions", "workspace_objects",
    "run_events", "tool_call_events", "order_intent_events", "order_events", "position_events",
    "workspace_events",
    "results", "citations", "account_snapshots", "agent_outputs", "decision_facts",
    "agent_decisions", "assembled_run_records", "audit_findings", "alerts_sent",
    "web_cache_entries", "web_cache_entry_tickers",
]  # fmt: skip

Conn = psycopg.Connection[tuple[object, ...]]

# One row in every ledger table, linked through fixed IDs.
SEED_SQL = b"""
INSERT INTO runs (run_id, environment, slot)
VALUES ('00000000-0000-0000-0000-000000000001', 'local', '2026-09-25 14:00+00');
INSERT INTO run_events (event_id, entity_id, run_id, sequence, event_type, observed_at, dedup_key)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001',
        '00000000-0000-0000-0000-000000000001', 1, 'started', now(), 'started');
INSERT INTO tool_calls (tool_call_id, run_id, sdk_tool_use_id, stage, server, tool, tier,
                        requested_at, arguments_redacted)
VALUES ('00000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000001',
        'toolu_1', 'agent', 'robinhood', 'place_option_order', 'X', now(), '{}');
INSERT INTO results (result_id, run_id, tool_call_id, kind, payload)
VALUES ('00000000-0000-0000-0000-000000000003', '00000000-0000-0000-0000-000000000001',
        '00000000-0000-0000-0000-000000000002', 'validated', '{}');
INSERT INTO tool_call_events (event_id, entity_id, run_id, sequence, event_type, observed_at,
                              dedup_key)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000002',
        '00000000-0000-0000-0000-000000000001', 1, 'requested', now(), 'requested');
INSERT INTO order_intents (intent_id, run_id, place_tool_call_id, account_scope_id,
                           arguments_redacted)
VALUES ('00000000-0000-0000-0000-000000000004', '00000000-0000-0000-0000-000000000001',
        '00000000-0000-0000-0000-000000000002', 'acct-1234', '{}');
INSERT INTO order_intent_events (event_id, entity_id, run_id, sequence, event_type,
                                 observed_at, source_tool_call_ids, dedup_key)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000004',
        '00000000-0000-0000-0000-000000000001', 1, 'reviewed', now(),
        '{00000000-0000-0000-0000-000000000002}', 'reviewed');
INSERT INTO orders (order_id, account_scope_id, broker_order_id, first_run_id)
VALUES ('00000000-0000-0000-0000-000000000005', 'acct-1234', 'B-1',
        '00000000-0000-0000-0000-000000000001');
INSERT INTO order_events (event_id, entity_id, run_id, sequence, event_type, intent_id,
                          observed_at, dedup_key)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000005',
        '00000000-0000-0000-0000-000000000001', 1, 'intent_linked',
        '00000000-0000-0000-0000-000000000004', now(), 'link');
INSERT INTO positions (position_id, account_scope_id, first_run_id)
VALUES ('00000000-0000-0000-0000-000000000006', 'acct-1234',
        '00000000-0000-0000-0000-000000000001');
INSERT INTO position_events (event_id, entity_id, run_id, sequence, event_type, observed_at,
                             dedup_key)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000006',
        '00000000-0000-0000-0000-000000000001', 1, 'opened', now(), 'opened');
INSERT INTO workspace_objects (object_id, account_scope_id, kind, broker_id, first_run_id)
VALUES ('00000000-0000-0000-0000-000000000007', 'acct-1234', 'watchlist', 'W-1',
        '00000000-0000-0000-0000-000000000001');
INSERT INTO workspace_events (event_id, entity_id, run_id, sequence, event_type, observed_at,
                              dedup_key)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000007',
        '00000000-0000-0000-0000-000000000001', 1, 'created', now(), 'created');
INSERT INTO citations (citation_id, run_id, tool_call_id, url, retrieved_at, tier, excerpt)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001',
        '00000000-0000-0000-0000-000000000002', 'https://example.com', now(), '1', 'x');
INSERT INTO account_snapshots (snapshot_id, run_id, as_of, retrieved_at, tool_call_ids,
                               account_ref, agentic_verified, quality)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001', now(), now(),
        ARRAY['00000000-0000-0000-0000-000000000002']::uuid[], '...1234', true, 'missing');
INSERT INTO agent_outputs (output_id, run_id, raw_redacted, observed_at)
VALUES ('00000000-0000-0000-0000-000000000008', '00000000-0000-0000-0000-000000000001',
        '{}', now());
INSERT INTO agent_decisions (agent_decision_id, run_id, output_id, schema_version, parse_status)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001',
        '00000000-0000-0000-0000-000000000008', 'v5', 'invalid');
INSERT INTO decision_facts (decision_facts_id, run_id, facts_ref, observed_at,
                            formula_versions, input_hash, facts)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001', 'f1', now(), '{}',
        'h', '{}');
INSERT INTO assembled_run_records (record_id, run_id, schema_version, assembler_version,
                                   input_hash, record, assembled_at)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001', 'v4', 'a1', 'h', '{}',
        now());
INSERT INTO audit_findings (finding_id, run_id, check_id, outcome, effective_execution_mode,
                            detail, audit_version, context_hash)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001', 'V1', 'unverifiable',
        'off', 'd', 'a1', 'h');
INSERT INTO alerts_sent (alert_id, run_id, alert_kind, dedup_key, payload, delivery_status,
                         attempted_at)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001', 'R1', 'k', '{}', 'sent',
        now());
INSERT INTO web_cache_entries (entry_id, run_id, tool_call_id, tool, cache_key, query_raw,
                               result, retrieved_at)
VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001',
        '00000000-0000-0000-0000-000000000002', 'WebSearch', 'aapl', 'AAPL', '{}', now());
INSERT INTO web_cache_entry_tickers (entry_id, ticker)
SELECT entry_id, 'AAPL' FROM web_cache_entries;
"""


@pytest.fixture
def seeded(conn: Conn) -> Conn:
    conn.execute(SEED_SQL)
    for table in LEDGER_TABLES:
        count = conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
        ).fetchone()
        assert count is not None and isinstance(count[0], int) and count[0] >= 1, table
    return conn


@pytest.mark.parametrize("table", LEDGER_TABLES)
def test_update_rejected(seeded: Conn, table: str) -> None:
    with pytest.raises(psycopg.errors.RestrictViolation, match="append-only"):
        seeded.execute(
            sql.SQL("UPDATE {} SET recorded_at = recorded_at").format(sql.Identifier(table))
        )


@pytest.mark.parametrize("table", LEDGER_TABLES)
def test_delete_rejected(seeded: Conn, table: str) -> None:
    with pytest.raises(psycopg.errors.RestrictViolation, match="append-only"):
        seeded.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))


@pytest.mark.parametrize("table", LEDGER_TABLES)
def test_truncate_rejected(seeded: Conn, table: str) -> None:
    with pytest.raises(psycopg.errors.RestrictViolation, match="append-only"):
        seeded.execute(sql.SQL("TRUNCATE {} CASCADE").format(sql.Identifier(table)))


def test_every_public_table_except_migrations_has_guards(conn: Conn) -> None:
    rows = conn.execute(
        "SELECT c.relname, count(t.tgname) FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "LEFT JOIN pg_trigger t ON t.tgrelid = c.oid AND NOT t.tgisinternal "
        "AND t.tgfoid = 'ledger_reject_mutation'::regproc "
        "WHERE n.nspname = 'public' AND c.relkind = 'r' GROUP BY c.relname"
    ).fetchall()
    guards = {str(name): int(str(n)) for name, n in rows}
    assert guards.pop("schema_migrations") == 0
    assert set(guards) == set(LEDGER_TABLES)
    assert all(n == 2 for n in guards.values()), guards
