import signal
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason
from wheelta_robinhood_agent.orchestrator.signals import (
    RunDeadline,
    install_stop_signal_handlers,
    trip_if_deadline_passed,
)

T0 = datetime(2025, 6, 11, 14, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def original_handlers() -> Iterator[dict[signal.Signals, object]]:
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield saved
    for s, h in saved.items():
        signal.signal(s, h)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sig", "reason"),
    [(signal.SIGTERM, StopReason.SIGTERM), (signal.SIGINT, StopReason.SIGINT)],
)
def test_signal_sets_latch_and_restore(
    original_handlers: dict[signal.Signals, object], sig: signal.Signals, reason: StopReason
) -> None:
    control = RunControl()
    restore = install_stop_signal_handlers(control, FakeClock(T0))
    try:
        signal.raise_signal(sig)
    finally:
        restore()
    record = control.stop_record
    assert record is not None
    assert record.reason is reason
    assert record.requested_at == T0
    for s, h in original_handlers.items():
        assert signal.getsignal(s) == h


def test_second_signal_keeps_first_reason(original_handlers: dict[signal.Signals, object]) -> None:
    control = RunControl()
    restore = install_stop_signal_handlers(control, FakeClock(T0))
    try:
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGTERM)
    finally:
        restore()
    record = control.stop_record
    assert record is not None
    assert record.reason is StopReason.SIGINT


def test_deadline_computation() -> None:
    d = RunDeadline(started_at=T0, budget_seconds=1500)
    assert d.deadline == T0 + timedelta(seconds=1500)
    assert d.remaining_seconds(T0) == 1500
    assert d.remaining_seconds(T0 + timedelta(seconds=2000)) == 0
    assert not d.expired(T0 + timedelta(seconds=1499))
    assert d.expired(T0 + timedelta(seconds=1500))


def test_deadline_trips_latch_with_injected_clock() -> None:
    control = RunControl()
    deadline = RunDeadline(started_at=T0, budget_seconds=60)
    clock = FakeClock(T0 + timedelta(seconds=59))
    assert trip_if_deadline_passed(control, deadline, clock) is False
    assert not control.stop_requested
    clock.now = T0 + timedelta(seconds=60)
    assert trip_if_deadline_passed(control, deadline, clock) is True
    record = control.stop_record
    assert record is not None
    assert record.reason is StopReason.DEADLINE
    assert record.requested_at == clock.now


def test_deadline_does_not_override_earlier_reason() -> None:
    control = RunControl()
    control.request_stop(StopReason.SIGTERM, T0)
    deadline = RunDeadline(started_at=T0, budget_seconds=1)
    assert trip_if_deadline_passed(control, deadline, FakeClock(T0 + timedelta(seconds=5)))
    record = control.stop_record
    assert record is not None
    assert record.reason is StopReason.SIGTERM


def test_deadline_validation() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RunDeadline(started_at=datetime(2025, 6, 11), budget_seconds=10)  # noqa: DTZ001
    with pytest.raises(ValueError, match="positive"):
        RunDeadline(started_at=T0, budget_seconds=0)
