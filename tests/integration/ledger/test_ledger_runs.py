"""Run slot classification, run events, sequence allocation, and the status projection."""

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv, RunStatus
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.run_identity import run_id_for
from wheelta_robinhood_agent.ledger.errors import DedupConflict
from wheelta_robinhood_agent.ledger.runs import (
    SlotState,
    append_run_event,
    current_run_status,
    open_run_slot,
    run_projection,
)

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 15, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=2)


def _status(conn: Conn, run_id: uuid.UUID, status: RunStatus, at: datetime) -> None:
    append_run_event(
        conn,
        run_id,
        RunEventType.STATUS,
        observed_at=at,
        dedup_key=f"status:{status.value}",
        status=status,
    )


def test_new_then_interrupted_then_completed(conn: Conn) -> None:
    first = open_run_slot(conn, AppEnv.STAGING, SLOT)
    assert first.state is SlotState.NEW
    assert first.run_id == run_id_for(AppEnv.STAGING, SLOT)

    # Crash before any event: the slot exists but was never finalized.
    assert open_run_slot(conn, AppEnv.STAGING, SLOT).state is SlotState.INTERRUPTED

    append_run_event(conn, first.run_id, RunEventType.STARTED, observed_at=T0, dedup_key="started")
    _status(conn, first.run_id, RunStatus.RUNNING, T0)
    interrupted = open_run_slot(conn, AppEnv.STAGING, SLOT)
    assert interrupted.state is SlotState.INTERRUPTED
    assert interrupted.current_status is RunStatus.RUNNING

    _status(conn, first.run_id, RunStatus.COMPLETED, T0 + timedelta(minutes=5))
    done = open_run_slot(conn, AppEnv.STAGING, SLOT)
    assert done.state is SlotState.COMPLETED
    assert done.current_status is RunStatus.COMPLETED


@pytest.mark.parametrize(
    "status", sorted(set(RunStatus) - {RunStatus.RUNNING, RunStatus.SKIPPED_CONCURRENT})
)
def test_every_finalizing_status_completes_the_slot(conn: Conn, status: RunStatus) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    _status(conn, run.run_id, status, T0)
    assert open_run_slot(conn, AppEnv.LOCAL, SLOT).state is SlotState.COMPLETED


def test_skipped_concurrent_never_finalizes_a_slot(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    _status(conn, run.run_id, RunStatus.RUNNING, T0)
    _status(conn, run.run_id, RunStatus.SKIPPED_CONCURRENT, T0 + timedelta(seconds=1))
    reopened = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    assert reopened.state is SlotState.INTERRUPTED
    assert reopened.current_status is RunStatus.RUNNING


def test_projection_derives_times_from_events(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    append_run_event(conn, run.run_id, RunEventType.STARTED, observed_at=T0, dedup_key="started")
    _status(conn, run.run_id, RunStatus.RUNNING, T0)
    end = T0 + timedelta(minutes=9)
    _status(conn, run.run_id, RunStatus.TIMED_OUT, end)
    projection = run_projection(conn, run.run_id)
    assert projection.status is RunStatus.TIMED_OUT
    assert projection.started_at == T0
    assert projection.finished_at == end
    assert current_run_status(conn, run.run_id) is RunStatus.TIMED_OUT


def test_run_events_deduplicate_by_source_key(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    a = append_run_event(conn, run.run_id, RunEventType.STARTED, observed_at=T0, dedup_key="s")
    b = append_run_event(conn, run.run_id, RunEventType.STARTED, observed_at=T0, dedup_key="s")
    assert a.created and not b.created
    assert (a.event_id, a.sequence) == (b.event_id, b.sequence)
    with pytest.raises(DedupConflict):
        append_run_event(conn, run.run_id, RunEventType.CONTROL, observed_at=T0, dedup_key="s")
    count = conn.execute("SELECT count(*) FROM run_events").fetchone()
    assert count == (1,)


def test_status_argument_matches_event_type(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    with pytest.raises(ValueError, match="STATUS"):
        append_run_event(conn, run.run_id, RunEventType.STATUS, observed_at=T0, dedup_key="x")
    with pytest.raises(ValueError, match="STATUS"):
        append_run_event(
            conn,
            run.run_id,
            RunEventType.CONTROL,
            observed_at=T0,
            dedup_key="x",
            status=RunStatus.STOPPED,
        )


def test_correction_references_an_event_of_the_same_run(conn: Conn) -> None:
    run = open_run_slot(conn, AppEnv.LOCAL, SLOT)
    other = open_run_slot(conn, AppEnv.LOCAL, SLOT + timedelta(hours=1))
    original = append_run_event(
        conn, run.run_id, RunEventType.METADATA, observed_at=T0, dedup_key="m"
    )
    fixed = append_run_event(
        conn,
        run.run_id,
        RunEventType.METADATA,
        observed_at=T0,
        dedup_key="m-fix",
        corrects_event_id=original.event_id,
    )
    assert fixed.sequence == 2
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        append_run_event(
            conn,
            other.run_id,
            RunEventType.METADATA,
            observed_at=T0,
            dedup_key="m-bad",
            corrects_event_id=original.event_id,
        )


def test_non_hour_slot_rejected(conn: Conn) -> None:
    with pytest.raises(ValueError, match="whole UTC hour"):
        open_run_slot(conn, AppEnv.LOCAL, SLOT.replace(minute=1))


def test_sequence_allocation_under_concurrency(conn_factory: Callable[[], Conn]) -> None:
    setup = conn_factory()
    run = open_run_slot(setup, AppEnv.LOCAL, SLOT)
    writers, per_writer = 6, 15
    connections = [conn_factory() for _ in range(writers)]

    def write(index: int) -> None:
        c = connections[index]
        for n in range(per_writer):
            append_run_event(
                c, run.run_id, RunEventType.METADATA, observed_at=T0, dedup_key=f"w{index}-{n}"
            )

    with ThreadPoolExecutor(max_workers=writers) as pool:
        list(pool.map(write, range(writers)))

    sequences = [
        r[0]
        for r in setup.execute(
            "SELECT sequence FROM run_events WHERE entity_id = %s ORDER BY sequence",
            (run.run_id,),
        )
    ]
    assert sequences == list(range(1, writers * per_writer + 1))
