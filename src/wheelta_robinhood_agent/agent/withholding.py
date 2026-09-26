"""Per-run withholding of MCP servers (CLAUDE.md §8 "Discover, then verify"; DATA_QUALITY.md).

A server is withheld when its status is not connected, its discovery diff is not ok, its
registry is unverified, or its result boundary has not passed acceptance. The PreToolUse hook
denies every call to a withheld server. Like `RunControl`, this is a monotonic latch: a server
can be withheld during a run but never released, so nothing here can widen tool access.
"""

import threading
from types import MappingProxyType
from typing import Final

__all__ = ["ServerWithholding"]


class ServerWithholding:
    """Thread-safe, add-only set of withheld server names with the first reason for each."""

    __slots__ = ("_lock", "_reasons")

    _MAX_REASON_LEN: Final = 200

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reasons: dict[str, str] = {}

    def withhold(self, server: str, reason: str) -> bool:
        """Withhold `server`. Returns True if this call withheld it (the first reason wins)."""
        if not server:
            raise ValueError("server must be non-empty")
        with self._lock:
            if server in self._reasons:
                return False
            self._reasons[server] = reason[: self._MAX_REASON_LEN] or "withheld"
            return True

    def reason(self, server: str) -> str | None:
        """Why `server` is withheld, or None if it is not."""
        with self._lock:
            return self._reasons.get(server)

    def snapshot(self) -> MappingProxyType[str, str]:
        """A read-only copy of every withheld server and its reason."""
        with self._lock:
            return MappingProxyType(dict(self._reasons))
