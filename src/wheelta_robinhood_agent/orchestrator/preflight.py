"""Preflight gating decision (ARCHITECTURE.md "Run lifecycle" step 4; CLAUDE.md §18).

Pure: decides from explicit inputs only. MCP status, tool discovery, and the single-flight
lock are handled by other stages. Order: kill switch first, then market session.
"""

from dataclasses import dataclass
from enum import StrEnum

from wheelta_robinhood_agent.domain.enums import ExecutionMode, RunStatus
from wheelta_robinhood_agent.domain.gating import effective_execution_mode
from wheelta_robinhood_agent.orchestrator.market_session import MarketSessionResult


class PreflightReason(StrEnum):
    KILL_SWITCH_ENGAGED = "kill_switch_engaged"
    OUTSIDE_REGULAR_SESSION = "outside_regular_session"


@dataclass(frozen=True, slots=True)
class PreflightProceed:
    """The run may continue to MCP status checks, using `effective_mode` for tools and prompt."""

    requested_mode: ExecutionMode
    effective_mode: ExecutionMode


@dataclass(frozen=True, slots=True)
class PreflightSkip:
    """The run ends here with a skip status; the agent session never starts."""

    status: RunStatus
    reason: PreflightReason


PreflightDecision = PreflightProceed | PreflightSkip


def decide_preflight(
    *,
    kill_switch: bool,
    market: MarketSessionResult,
    requested_mode: ExecutionMode,
    armed: bool,
    ceiling: ExecutionMode,
) -> PreflightDecision:
    """Gate the run before any agent activity.

    1. `KILL_SWITCH=true` -> skipped_killed; the run never starts its session (CLAUDE.md §18,
       INTERFACES.md "Run and RunControl").
    2. Outside the NYSE regular session -> skipped_market_closed (ARCHITECTURE.md
       "Market-session gating").
    3. Otherwise proceed with the effective mode: live only if requested live, armed, and the
       phase ceiling permits it (domain.gating; INTERFACES.md "Run and RunControl").
    """
    if kill_switch:
        return PreflightSkip(RunStatus.SKIPPED_KILLED, PreflightReason.KILL_SWITCH_ENGAGED)
    if not market.may_proceed:
        return PreflightSkip(
            RunStatus.SKIPPED_MARKET_CLOSED, PreflightReason.OUTSIDE_REGULAR_SESSION
        )
    return PreflightProceed(
        requested_mode=requested_mode,
        effective_mode=effective_execution_mode(requested_mode, armed=armed, ceiling=ceiling),
    )
