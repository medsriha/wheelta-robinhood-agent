import threading
from datetime import UTC, datetime, timedelta

import pytest

from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason, StopRecord

T0 = datetime(2025, 6, 11, 14, 0, tzinfo=UTC)


def test_initially_not_stopped() -> None:
    control = RunControl()
    assert not control.stop_requested
    assert control.stop_record is None


def test_first_reason_wins_and_irreversible() -> None:
    control = RunControl()
    assert control.request_stop(StopReason.SIGTERM, T0) is True
    assert control.request_stop(StopReason.DEADLINE, T0 + timedelta(seconds=5)) is False
    assert control.stop_requested
    assert control.stop_record == StopRecord(StopReason.SIGTERM, T0)


def test_no_public_way_to_clear() -> None:
    public = {name for name in dir(RunControl) if not name.startswith("_")}
    assert public == {"request_stop", "stop_requested", "stop_record"}


def test_naive_time_rejected_and_latch_untouched() -> None:
    control = RunControl()
    with pytest.raises(ValueError, match="timezone-aware"):
        control.request_stop(StopReason.SIGINT, datetime(2025, 6, 11, 14, 0))  # noqa: DTZ001
    assert not control.stop_requested


def test_thread_safety_exactly_one_winner() -> None:
    control = RunControl()
    reasons = list(StopReason) * 50
    barrier = threading.Barrier(len(reasons))
    wins: list[StopReason] = []
    wins_lock = threading.Lock()

    def worker(reason: StopReason) -> None:
        barrier.wait()
        if control.request_stop(reason, T0):
            with wins_lock:
                wins.append(reason)

    threads = [threading.Thread(target=worker, args=(r,)) for r in reasons]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
    record = control.stop_record
    assert record is not None
    assert record.reason is wins[0]
