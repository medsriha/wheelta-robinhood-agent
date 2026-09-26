"""Process exit codes per final RunStatus (CLAUDE.md §14 "Exit codes").

`0` for completed (including "nothing to do") and the documented normal skips; distinct
non-zero codes for failures, so Railway and alerting can tell them apart.

`skipped_killed` exits 0: an operator engaged the kill switch deliberately, OPERATIONS.md
lists it among the expected `skipped_*` outcomes, and the kill-switch alert (CLAUDE.md §16)
reports it separately. A non-zero code would page as a failure. (ADR-0014 delegation,
2026-09-26; the owner may revise.)
"""

from types import MappingProxyType
from typing import Final

from wheelta_robinhood_agent.domain.enums import RunStatus

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_TIMED_OUT: Final = 2
EXIT_STOPPED: Final = 3

_EXIT_CODES: Final = MappingProxyType(
    {
        # CLAUDE.md §14: 0 completed (including "nothing to do").
        RunStatus.COMPLETED: EXIT_OK,
        # CLAUDE.md §15: can't acquire the lock -> exit 0 with skipped_concurrent.
        RunStatus.SKIPPED_CONCURRENT: EXIT_OK,
        # ARCHITECTURE.md "Market-session gating": exits 0 with skipped_market_closed.
        RunStatus.SKIPPED_MARKET_CLOSED: EXIT_OK,
        # Deliberate operator stop; alerted separately, not a failure.
        RunStatus.SKIPPED_KILLED: EXIT_OK,
        # CLAUDE.md §14: non-zero for failures; distinct so alerting can tell them apart.
        RunStatus.FAILED: EXIT_FAILED,
        RunStatus.TIMED_OUT: EXIT_TIMED_OUT,
        RunStatus.STOPPED: EXIT_STOPPED,
    }
)


class ExitCodeUndefined(ValueError):
    """The status has no exit code because it is not final (`running`)."""


def exit_code_for(status: RunStatus) -> int:
    """The process exit code for a final run status. Raises ExitCodeUndefined otherwise."""
    code = _EXIT_CODES.get(status)
    if code is None:
        raise ExitCodeUndefined(f"no exit code defined for run status {status.value!r}")
    return code
