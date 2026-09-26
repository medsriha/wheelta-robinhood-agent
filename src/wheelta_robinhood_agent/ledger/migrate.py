"""Forward-only ledger migrations (CLAUDE.md §20, ADR-0002).

Files in `migrations/` named `NNNN_description.sql` are applied in version order, each in its
own transaction together with its `schema_migrations` row. An applied file whose SHA-256 has
changed, or that has disappeared, stops the runner: history is never rewritten. Re-running
applies nothing. A session advisory lock serializes concurrent runners.

CLI (Railway preDeployCommand):
    python -m wheelta_robinhood_agent.ledger.migrate  (reads DATABASE_URL via config)
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import psycopg

from wheelta_robinhood_agent.config.settings import SettingsError, load_database_url
from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.errors import LedgerError, MigrationError
from wheelta_robinhood_agent.ledger.lock import (
    LOCK_CLASS_ID,
    MIGRATION_LOCK_OBJID,
    advisory_unlock,
)

# Repository layout: src/wheelta_robinhood_agent/ledger/migrate.py -> <repo>/migrations.
DEFAULT_MIGRATIONS_DIR: Final = Path(__file__).resolve().parents[3] / "migrations"
# DDL may take longer than the repository default; still bounded.
MIGRATION_STATEMENT_TIMEOUT_MS: Final = 120_000
_FILENAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")

_CREATE_TABLE = b"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    integer     PRIMARY KEY,
    filename   text        NOT NULL,
    sha256     text        NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


@dataclass(frozen=True)
class Migration:
    version: int
    filename: str
    sha256: str
    sql: bytes


def discover(migrations_dir: Path) -> list[Migration]:
    """All migration files, sorted by version. Unrecognized `.sql` names and duplicate
    versions are errors, so a typo cannot silently skip a migration."""
    if not migrations_dir.is_dir():
        raise MigrationError(f"migrations directory not found: {migrations_dir}")
    found: dict[int, Migration] = {}
    for path in sorted(migrations_dir.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if match is None:
            raise MigrationError(f"migration file name must be NNNN_name.sql: {path.name}")
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"duplicate migration version {version:04d}")
        data = path.read_bytes()
        found[version] = Migration(version, path.name, hashlib.sha256(data).hexdigest(), data)
    return [found[v] for v in sorted(found)]


def apply_migrations(
    conn: psycopg.Connection[tuple[object, ...]], migrations_dir: Path
) -> list[str]:
    """Apply pending migrations; return the filenames applied (empty when up to date).

    `conn` must be in autocommit mode (db.connect). Raises MigrationError on checksum drift,
    a missing applied file, or a failed migration (whose transaction is rolled back).
    """
    if not conn.autocommit:
        raise MigrationError("apply_migrations requires an autocommit connection")
    migrations = discover(migrations_dir)
    conn.execute("SELECT pg_advisory_lock(%s, %s)", (LOCK_CLASS_ID, MIGRATION_LOCK_OBJID))
    try:
        conn.execute(_CREATE_TABLE)
        applied = {
            int(str(version)): (str(filename), str(sha))
            for version, filename, sha in conn.execute(
                "SELECT version, filename, sha256 FROM schema_migrations"
            ).fetchall()
        }
        by_version = {m.version: m for m in migrations}
        for version, (filename, sha) in sorted(applied.items()):
            current = by_version.get(version)
            if current is None:
                raise MigrationError(
                    f"applied migration {filename} is missing from {migrations_dir}"
                )
            if current.sha256 != sha or current.filename != filename:
                raise MigrationError(
                    f"applied migration {filename} changed on disk; add a new migration instead"
                )
        done: list[str] = []
        for migration in migrations:
            if migration.version in applied:
                continue
            try:
                with conn.transaction():
                    conn.execute(migration.sql)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, filename, sha256) "
                        "VALUES (%s, %s, %s)",
                        (migration.version, migration.filename, migration.sha256),
                    )
            except psycopg.Error as exc:
                raise MigrationError(
                    f"migration {migration.filename} failed: {type(exc).__name__}: "
                    f"{exc.diag.message_primary or exc}"
                ) from exc
            done.append(migration.filename)
        return done
    finally:
        if not conn.closed:
            advisory_unlock(conn, MIGRATION_LOCK_OBJID)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply forward-only ledger migrations.")
    parser.add_argument("--migrations-dir", type=Path, default=DEFAULT_MIGRATIONS_DIR)
    args = parser.parse_args(argv)
    try:
        url = load_database_url()
        with connect(url, statement_timeout_ms=MIGRATION_STATEMENT_TIMEOUT_MS) as conn:
            applied = apply_migrations(conn, args.migrations_dir)
    except (LedgerError, SettingsError) as exc:
        sys.stderr.write(f"migrate: {exc}\n")
        return 1
    sys.stdout.write(
        f"migrate: applied {', '.join(applied)}\n" if applied else "migrate: up to date\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
