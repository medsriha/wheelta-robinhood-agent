"""Throwaway Postgres for ledger integration tests.

A session fixture initdb's a cluster in a short temp dir and starts it listening only on a
unix socket inside that dir (listen_addresses=''), so no TCP port is opened. Tests skip if
the Postgres server binaries are not installed. pytest runs with --disable-socket; these
tests re-enable unix-domain sockets only (TCP stays blocked).
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from pydantic import SecretStr
from pytest_socket import disable_socket

from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.migrate import DEFAULT_MIGRATIONS_DIR, apply_migrations

_CANDIDATE_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/lib/postgresql/16/bin")
_PORT = 5432
_TEMPLATE_DB = "ledger_template"

Conn = psycopg.Connection[tuple[object, ...]]


def _find_bin_dir() -> Path | None:
    explicit = os.environ.get("PG_BIN")
    candidates = [explicit] if explicit else []
    which = shutil.which("initdb")
    if which:
        candidates.append(str(Path(which).parent))
    candidates.extend(_CANDIDATE_BIN_DIRS)
    for candidate in candidates:
        path = Path(candidate)
        if all((path / name).exists() for name in ("initdb", "pg_ctl", "postgres")):
            return path
    return None


@dataclass
class PgServer:
    socket_dir: Path
    _counter: itertools.count[int]

    def url(self, dbname: str) -> SecretStr:
        return SecretStr(f"postgresql://postgres@/{dbname}?host={self.socket_dir}&port={_PORT}")

    def create_db(self, template: str | None = None) -> SecretStr:
        name = f"t{next(self._counter)}"
        with psycopg.connect(self.url("postgres").get_secret_value(), autocommit=True) as admin:
            tmpl = f" TEMPLATE {template}" if template else ""
            admin.execute(f"CREATE DATABASE {name}{tmpl}".encode())
        return self.url(name)


@pytest.fixture(autouse=True)
def _allow_unix_sockets_only() -> None:
    """Runs after pytest-socket's per-test setup; permits AF_UNIX, keeps TCP/UDP blocked."""
    disable_socket(allow_unix_socket=True)


@pytest.fixture(scope="session")
def pg_server() -> Iterator[PgServer]:
    bin_dir = _find_bin_dir()
    if bin_dir is None:
        pytest.skip("Postgres server binaries (initdb, pg_ctl, postgres) not found")
    disable_socket(allow_unix_socket=True)
    # Unix socket paths are limited to ~104 bytes, so avoid the long pytest tmp paths.
    base = Path(tempfile.mkdtemp(prefix="wra-pg-", dir="/tmp"))
    data = base / "data"
    log = base / "server.log"
    try:
        subprocess.run(  # noqa: S603 - fixed local binaries, no shell
            [
                str(bin_dir / "initdb"),
                "-D",
                str(data),
                "-U",
                "postgres",
                "-A",
                "trust",
                "-E",
                "UTF8",
                "--no-sync",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(  # noqa: S603
            [
                str(bin_dir / "pg_ctl"),
                "-D",
                str(data),
                "-l",
                str(log),
                "-w",
                "-t",
                "30",
                "-o",
                f"-k {base} -c listen_addresses='' -p {_PORT} -c fsync=off",
                "start",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace") if exc.stderr else ""
        if log.exists():
            detail += log.read_text(errors="replace")
        shutil.rmtree(base, ignore_errors=True)
        pytest.fail(f"could not start throwaway Postgres: {detail}")
    server = PgServer(base, itertools.count(1))
    try:
        with psycopg.connect(server.url("postgres").get_secret_value(), autocommit=True) as a:
            a.execute(f"CREATE DATABASE {_TEMPLATE_DB}".encode())
        with connect(server.url(_TEMPLATE_DB)) as conn:
            apply_migrations(conn, DEFAULT_MIGRATIONS_DIR)
        yield server
    finally:
        subprocess.run(  # noqa: S603
            [str(bin_dir / "pg_ctl"), "-D", str(data), "-m", "immediate", "-w", "stop"],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def empty_db_url(pg_server: PgServer) -> SecretStr:
    """A fresh database with no migrations applied."""
    return pg_server.create_db()


@pytest.fixture
def ledger_db_url(pg_server: PgServer) -> SecretStr:
    """A fresh database cloned from the migrated template."""
    return pg_server.create_db(template=_TEMPLATE_DB)


@pytest.fixture
def conn_factory(ledger_db_url: SecretStr) -> Iterator[Callable[[], Conn]]:
    opened: list[Conn] = []

    def make() -> Conn:
        c = connect(ledger_db_url)
        opened.append(c)
        return c

    yield make
    for c in opened:
        c.close()


@pytest.fixture
def conn(conn_factory: Callable[[], Conn]) -> Conn:
    return conn_factory()
