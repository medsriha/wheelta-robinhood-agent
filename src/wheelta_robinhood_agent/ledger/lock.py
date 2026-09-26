"""Single-flight run lock (CLAUDE.md §15): a Postgres session advisory lock per environment.

Key: the two-int form `pg_try_advisory_lock(LOCK_CLASS_ID, objid)`. LOCK_CLASS_ID is the fixed
ASCII tag 'WRA1'; objid is fixed per environment below. Never change these values: a running
old release and a new release must contend for the same key. The lock is session-scoped, so
Postgres releases it if the process dies or the connection drops.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import psycopg

from wheelta_robinhood_agent.domain.enums import AppEnv

LOCK_CLASS_ID: Final = 0x57524131  # 'WRA1'
RUN_LOCK_OBJID: Final[dict[AppEnv, int]] = {
    AppEnv.LOCAL: 1,
    AppEnv.STAGING: 2,
    AppEnv.PRODUCTION: 3,
}
# Serializes concurrent migration runners (e.g. two overlapping preDeployCommands).
MIGRATION_LOCK_OBJID: Final = 100


def try_advisory_lock(conn: psycopg.Connection[tuple[object, ...]], objid: int) -> bool:
    row = conn.execute("SELECT pg_try_advisory_lock(%s, %s)", (LOCK_CLASS_ID, objid)).fetchone()
    return bool(row and row[0])


def advisory_unlock(conn: psycopg.Connection[tuple[object, ...]], objid: int) -> None:
    conn.execute("SELECT pg_advisory_unlock(%s, %s)", (LOCK_CLASS_ID, objid))


@contextmanager
def single_flight(
    conn: psycopg.Connection[tuple[object, ...]], environment: AppEnv
) -> Iterator[bool]:
    """Try to take the environment's run lock without waiting.

    Yields True if acquired (released on exit), False if another session holds it; the
    caller then exits 0 with `skipped_concurrent` and must not write to that slot's run.
    """
    objid = RUN_LOCK_OBJID[environment]
    acquired = try_advisory_lock(conn, objid)
    try:
        yield acquired
    finally:
        if acquired and not conn.closed:
            advisory_unlock(conn, objid)
