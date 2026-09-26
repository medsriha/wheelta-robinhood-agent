"""Stop signals and the run deadline (ARCHITECTURE.md "Runtime stop"; OPERATIONS.md R7, R9).

SIGTERM/SIGINT and the whole-run budget (`RUN_TIMEOUT_SECONDS`, CLAUDE.md §14) set the
injected RunControl stop latch. Nothing here reads the clock directly: the caller injects it.
"""

import signal
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import FrameType

from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason

Clock = Callable[[], datetime]
Restore = Callable[[], None]

_SIGNAL_REASONS: dict[signal.Signals, StopReason] = {
    signal.SIGTERM: StopReason.SIGTERM,
    signal.SIGINT: StopReason.SIGINT,
}


def install_stop_signal_handlers(control: RunControl, clock: Clock) -> Restore:
    """Install SIGTERM/SIGINT handlers that set the stop latch; return a restore callable.

    The handlers only latch (R9: the orchestrator then interrupts the SDK and finalizes). They
    do not raise, so finalization runs within the remaining budget. Must be called from the
    main thread (a Python `signal` requirement).
    """
    previous: dict[signal.Signals, object] = {}

    def _handler(signum: int, _frame: FrameType | None) -> None:
        control.request_stop(_SIGNAL_REASONS[signal.Signals(signum)], clock())

    for sig in _SIGNAL_REASONS:
        previous[sig] = signal.signal(sig, _handler)

    def _restore() -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]

    return _restore


@dataclass(frozen=True, slots=True)
class RunDeadline:
    """The whole-run budget: `started_at + RUN_TIMEOUT_SECONDS` (CLAUDE.md §14)."""

    started_at: datetime
    budget_seconds: int

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")
        if self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")

    @property
    def deadline(self) -> datetime:
        return self.started_at + timedelta(seconds=self.budget_seconds)

    def remaining_seconds(self, now: datetime) -> float:
        """Seconds left before the deadline; never negative."""
        return max(0.0, (self.deadline - now).total_seconds())

    def expired(self, now: datetime) -> bool:
        """True at or after the deadline."""
        return now >= self.deadline


def trip_if_deadline_passed(control: RunControl, deadline: RunDeadline, clock: Clock) -> bool:
    """Set the latch with reason `deadline` if the deadline has passed (R7).

    Returns True if the deadline has passed (whether or not this call set the latch; an earlier
    reason still wins).
    """
    now = clock()
    if not deadline.expired(now):
        return False
    control.request_stop(StopReason.DEADLINE, now)
    return True
