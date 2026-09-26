"""Execution-mode resolution. Pure; the inputs come from Settings and code constants."""

from wheelta_robinhood_agent.domain.enums import ExecutionMode


def effective_execution_mode(
    requested: ExecutionMode, *, armed: bool, ceiling: ExecutionMode
) -> ExecutionMode:
    """Return the mode the run actually uses for tools, prompt, and audit.

    Live only if live was requested, the run is armed, and the phase ceiling permits live;
    otherwise off (INTERFACES.md "Run and RunControl"; ADR-0013 caps phase 1 at off).
    """
    if requested is ExecutionMode.LIVE and armed and ceiling is ExecutionMode.LIVE:
        return ExecutionMode.LIVE
    return ExecutionMode.OFF
