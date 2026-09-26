"""Runtime stop latch (INTERFACES.md "Run and RunControl"; CLAUDE.md §18 kill switch).

`RunControl` is an explicitly passed runtime object, separate from frozen Settings. SIGTERM,
SIGINT, the run deadline, or an infrastructure failure sets it. Once set it never clears, and
it can only disable activity: there is no operation that enables anything. Agent hooks check
it together with the startup settings before every Tier S/X call.

It lives in `agent/` because the hooks read it and `agent` must not import `orchestrator`.
"""

import threading
from dataclasses import dataclass
from datetime import datetime

from wheelta_robinhood_agent.domain.run import StopReason

__all__ = ["RunControl", "StopReason", "StopRecord"]


@dataclass(frozen=True, slots=True)
class StopRecord:
    """The first stop request: its reason and the caller-supplied UTC-aware time."""

    reason: StopReason
    requested_at: datetime


class RunControl:
    """Thread-safe, irreversible stop latch. The first reason wins; later requests are no-ops.

    `request_stop` may be called from a signal handler, a deadline watcher, or any thread.
    The lock is reentrant because a Python signal handler runs on the main thread and may
    interrupt a `request_stop` already holding it; the single `dict.setdefault` call keeps
    first-reason-wins correct even under that reentry.
    """

    __slots__ = ("_lock", "_slot")

    _KEY = "stop"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slot: dict[str, StopRecord] = {}

    def request_stop(self, reason: StopReason, at: datetime) -> bool:
        """Set the latch. Returns True if this call set it, False if it was already set.

        Raises ValueError for a naive `at` (CLAUDE.md §4: timezone-aware datetimes only).
        """
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("at must be timezone-aware")
        record = StopRecord(reason=reason, requested_at=at)
        with self._lock:
            return self._slot.setdefault(self._KEY, record) is record

    @property
    def stop_requested(self) -> bool:
        return self._KEY in self._slot

    @property
    def stop_record(self) -> StopRecord | None:
        """The first stop request, or None if the latch is not set."""
        return self._slot.get(self._KEY)
