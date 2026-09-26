"""Forward-only migration runner: apply, idempotency, drift and failure handling."""

import shutil
from pathlib import Path

import pytest
from pydantic import SecretStr

from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.errors import MigrationError
from wheelta_robinhood_agent.ledger.migrate import (
    DEFAULT_MIGRATIONS_DIR,
    apply_migrations,
    discover,
    main,
)

EXPECTED_TABLES = {
    "schema_migrations",
    "runs", "tool_calls", "order_intents", "orders", "positions", "workspace_objects",
    "run_events", "tool_call_events", "order_intent_events", "order_events", "position_events",
    "workspace_events",
    "results", "citations", "account_snapshots", "agent_outputs", "decision_facts",
    "agent_decisions", "assembled_run_records", "audit_findings", "alerts_sent",
    "web_cache_entries", "web_cache_entry_tickers",
}  # fmt: skip


@pytest.fixture
def migrations_copy(tmp_path: Path) -> Path:
    target = tmp_path / "migrations"
    target.mkdir()
    for path in DEFAULT_MIGRATIONS_DIR.glob("*.sql"):
        shutil.copy(path, target / path.name)
    return target


def test_apply_creates_all_tables_and_is_idempotent(empty_db_url: SecretStr) -> None:
    with connect(empty_db_url) as conn:
        first = apply_migrations(conn, DEFAULT_MIGRATIONS_DIR)
        second = apply_migrations(conn, DEFAULT_MIGRATIONS_DIR)
        tables = {
            str(r[0])
            for r in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }
        recorded = conn.execute("SELECT version, filename FROM schema_migrations").fetchall()
    shipped = discover(DEFAULT_MIGRATIONS_DIR)
    assert first == [m.filename for m in shipped]
    assert second == []
    assert tables == EXPECTED_TABLES
    assert sorted(recorded) == [(m.version, m.filename) for m in shipped]


def test_checksum_drift_is_refused(empty_db_url: SecretStr, migrations_copy: Path) -> None:
    with connect(empty_db_url) as conn:
        apply_migrations(conn, migrations_copy)
        target = migrations_copy / "0001_initial.sql"
        target.write_text(target.read_text() + "\n-- edited after apply\n")
        with pytest.raises(MigrationError, match="changed on disk"):
            apply_migrations(conn, migrations_copy)


def test_missing_applied_file_is_refused(empty_db_url: SecretStr, migrations_copy: Path) -> None:
    with connect(empty_db_url) as conn:
        apply_migrations(conn, migrations_copy)
        (migrations_copy / "0001_initial.sql").unlink()
        with pytest.raises(MigrationError, match="missing"):
            apply_migrations(conn, migrations_copy)


def test_bad_file_name_is_refused(tmp_path: Path) -> None:
    (tmp_path / "1_bad.sql").write_text("SELECT 1;")
    with pytest.raises(MigrationError, match="NNNN_name.sql"):
        discover(tmp_path)


def test_failed_migration_rolls_back(empty_db_url: SecretStr, migrations_copy: Path) -> None:
    (migrations_copy / "9999_broken.sql").write_text(
        "CREATE TABLE should_not_exist (id int);\nSELECT * FROM no_such_table;\n"
    )
    with connect(empty_db_url) as conn:
        with pytest.raises(MigrationError, match="9999_broken.sql"):
            apply_migrations(conn, migrations_copy)
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations")]
        leftover = conn.execute("SELECT to_regclass('should_not_exist')").fetchone()
    assert versions == [m.version for m in discover(DEFAULT_MIGRATIONS_DIR)]
    assert leftover == (None,)


def test_cli_reads_database_url_via_config(
    empty_db_url: SecretStr, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", empty_db_url.get_secret_value())
    assert main([]) == 0
    assert main([]) == 0
    monkeypatch.delenv("DATABASE_URL")
    assert main([]) == 1
    assert empty_db_url.get_secret_value() not in capsys.readouterr().err
