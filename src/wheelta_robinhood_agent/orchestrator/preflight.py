"""Preflight gating decision (ARCHITECTURE.md "Run lifecycle" step 4; CLAUDE.md §18).

Pure: decides from explicit inputs only. MCP status, tool discovery, and the single-flight
lock are handled by other stages. Order: kill switch first; a dry run then proceeds only on
demand, ignoring the calendar and schedule (ADR-0038); live then checks the market session
and whether the tick is due (ADR-0028; orchestrator/schedule.py decides `due`).
"""

from dataclasses import dataclass
from enum import StrEnum

from wheelta_robinhood_agent.domain.enums import ExecutionMode, RunStatus
from wheelta_robinhood_agent.domain.gating import effective_execution_mode
from wheelta_robinhood_agent.orchestrator.market_session import MarketSessionResult


class PreflightReason(StrEnum):
    KILL_SWITCH_ENGAGED = "kill_switch_engaged"
    OUTSIDE_REGULAR_SESSION = "outside_regular_session"
    NOT_DUE = "not_due"
    DRY_RUN_NOT_REQUESTED = "dry_run_not_requested"


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
    due: bool,
    requested_mode: ExecutionMode,
    armed: bool,
    ceiling: ExecutionMode,
    on_demand: bool = False,
) -> PreflightDecision:
    """Gate the run before any agent activity.

    1. `KILL_SWITCH=true` -> skipped_killed; the run never starts its session (CLAUDE.md §18,
       INTERFACES.md "Run and RunControl").
    2. Effective mode off (a dry run, ADR-0038): proceed only on demand (`--run-now`), at any
       time and date; a scheduled tick -> skipped_dry_run_not_requested. Live continues:
    3. Outside the NYSE regular session -> skipped_market_closed (ARCHITECTURE.md
       "Market-session gating").
    4. Before the recorded next-run time -> skipped_not_due (ADR-0028).
    5. Otherwise proceed with the effective mode: live only if requested live, armed, and the
       phase ceiling permits it (domain.gating; INTERFACES.md "Run and RunControl").
    """
    effective = effective_execution_mode(requested_mode, armed=armed, ceiling=ceiling)
    if kill_switch:
        return PreflightSkip(RunStatus.SKIPPED_KILLED, PreflightReason.KILL_SWITCH_ENGAGED)
    if effective is ExecutionMode.OFF:
        if not on_demand:
            return PreflightSkip(
                RunStatus.SKIPPED_DRY_RUN_NOT_REQUESTED, PreflightReason.DRY_RUN_NOT_REQUESTED
            )
        return PreflightProceed(requested_mode=requested_mode, effective_mode=effective)
    if not market.may_proceed:
        return PreflightSkip(
            RunStatus.SKIPPED_MARKET_CLOSED, PreflightReason.OUTSIDE_REGULAR_SESSION
        )
    if not due:
        return PreflightSkip(RunStatus.SKIPPED_NOT_DUE, PreflightReason.NOT_DUE)
    return PreflightProceed(requested_mode=requested_mode, effective_mode=effective)
