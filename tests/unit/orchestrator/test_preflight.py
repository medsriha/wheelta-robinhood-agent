from datetime import UTC, date, datetime

import pytest

from wheelta_robinhood_agent.domain.enums import ExecutionMode, MarketSession, RunStatus
from wheelta_robinhood_agent.orchestrator.market_session import (
    CalendarProvenance,
    MarketSessionResult,
)
from wheelta_robinhood_agent.orchestrator.preflight import (
    PreflightProceed,
    PreflightReason,
    PreflightSkip,
    decide_preflight,
)

PROV = CalendarProvenance("exchange_calendars", "test", "XNYS")


def _market(session: MarketSession) -> MarketSessionResult:
    return MarketSessionResult(
        as_of=datetime(2025, 6, 11, 14, 0, tzinfo=UTC),
        session=session,
        may_proceed=session is MarketSession.REGULAR,
        session_date=date(2025, 6, 11),
        regular_open_utc=None,
        regular_close_utc=None,
        provenance=PROV,
    )


def _decide(kill: bool, session: MarketSession, **kw: object) -> object:
    # Live by default: the calendar and schedule gates apply to live runs only (ADR-0038).
    args: dict[str, object] = {
        "due": True,
        "requested_mode": ExecutionMode.LIVE,
        "armed": True,
        "ceiling": ExecutionMode.LIVE,
    }
    args.update(kw)
    return decide_preflight(kill_switch=kill, market=_market(session), **args)  # type: ignore[arg-type]


@pytest.mark.parametrize("due", [True, False])
@pytest.mark.parametrize("session", list(MarketSession))
def test_kill_switch_checked_first(session: MarketSession, due: bool) -> None:
    assert _decide(True, session, due=due) == PreflightSkip(
        RunStatus.SKIPPED_KILLED, PreflightReason.KILL_SWITCH_ENGAGED
    )


@pytest.mark.parametrize("due", [True, False])
@pytest.mark.parametrize("session", [MarketSession.PRE, MarketSession.POST, MarketSession.CLOSED])
def test_outside_regular_session_skips(session: MarketSession, due: bool) -> None:
    assert _decide(False, session, due=due) == PreflightSkip(
        RunStatus.SKIPPED_MARKET_CLOSED, PreflightReason.OUTSIDE_REGULAR_SESSION
    )


def test_not_due_skips_inside_the_session() -> None:
    assert _decide(False, MarketSession.REGULAR, due=False) == PreflightSkip(
        RunStatus.SKIPPED_NOT_DUE, PreflightReason.NOT_DUE
    )


def test_proceed_live() -> None:
    assert _decide(False, MarketSession.REGULAR) == PreflightProceed(
        ExecutionMode.LIVE, ExecutionMode.LIVE
    )


OFF = {"requested_mode": ExecutionMode.OFF, "armed": False}


@pytest.mark.parametrize("due", [True, False])
@pytest.mark.parametrize("session", list(MarketSession))
def test_dry_run_proceeds_on_demand_at_any_time(session: MarketSession, due: bool) -> None:
    """ADR-0038: an on-demand dry run ignores the calendar and the schedule."""
    assert _decide(False, session, due=due, on_demand=True, **OFF) == PreflightProceed(
        ExecutionMode.OFF, ExecutionMode.OFF
    )


@pytest.mark.parametrize("session", list(MarketSession))
def test_scheduled_tick_in_off_mode_is_not_a_dry_run(session: MarketSession) -> None:
    assert _decide(False, session, **OFF) == PreflightSkip(
        RunStatus.SKIPPED_DRY_RUN_NOT_REQUESTED, PreflightReason.DRY_RUN_NOT_REQUESTED
    )


def test_kill_switch_stops_an_on_demand_dry_run() -> None:
    assert _decide(True, MarketSession.CLOSED, on_demand=True, **OFF) == PreflightSkip(
        RunStatus.SKIPPED_KILLED, PreflightReason.KILL_SWITCH_ENGAGED
    )


def test_unarmed_live_is_an_on_demand_dry_run() -> None:
    kw = {"requested_mode": ExecutionMode.LIVE, "armed": False}
    assert _decide(False, MarketSession.CLOSED, due=False, on_demand=True, **kw) == (
        PreflightProceed(ExecutionMode.LIVE, ExecutionMode.OFF)
    )


@pytest.mark.parametrize(
    ("armed", "ceiling", "effective"),
    [
        (True, ExecutionMode.LIVE, ExecutionMode.LIVE),
        (False, ExecutionMode.LIVE, ExecutionMode.OFF),
        (True, ExecutionMode.OFF, ExecutionMode.OFF),
    ],
)
def test_proceed_effective_mode(
    armed: bool, ceiling: ExecutionMode, effective: ExecutionMode
) -> None:
    result = _decide(
        False,
        MarketSession.REGULAR,
        requested_mode=ExecutionMode.LIVE,
        armed=armed,
        ceiling=ceiling,
        on_demand=effective is ExecutionMode.OFF,
    )
    assert result == PreflightProceed(ExecutionMode.LIVE, effective)
