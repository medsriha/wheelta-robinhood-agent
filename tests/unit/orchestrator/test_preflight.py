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
    args: dict[str, object] = {
        "requested_mode": ExecutionMode.OFF,
        "armed": False,
        "ceiling": ExecutionMode.OFF,
    }
    args.update(kw)
    return decide_preflight(kill_switch=kill, market=_market(session), **args)  # type: ignore[arg-type]


@pytest.mark.parametrize("session", list(MarketSession))
def test_kill_switch_checked_first(session: MarketSession) -> None:
    assert _decide(True, session) == PreflightSkip(
        RunStatus.SKIPPED_KILLED, PreflightReason.KILL_SWITCH_ENGAGED
    )


@pytest.mark.parametrize("session", [MarketSession.PRE, MarketSession.POST, MarketSession.CLOSED])
def test_outside_regular_session_skips(session: MarketSession) -> None:
    assert _decide(False, session) == PreflightSkip(
        RunStatus.SKIPPED_MARKET_CLOSED, PreflightReason.OUTSIDE_REGULAR_SESSION
    )


def test_proceed_off() -> None:
    assert _decide(False, MarketSession.REGULAR) == PreflightProceed(
        ExecutionMode.OFF, ExecutionMode.OFF
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
    )
    assert result == PreflightProceed(ExecutionMode.LIVE, effective)
