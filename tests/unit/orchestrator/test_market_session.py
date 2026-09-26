from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from wheelta_robinhood_agent.domain.enums import MarketSession
from wheelta_robinhood_agent.orchestrator.market_session import (
    CalendarOutOfRange,
    ExchangeCalendarsNyse,
    build_nyse_calendar,
    evaluate_market_session,
)


@pytest.fixture(scope="module")
def cal() -> ExchangeCalendarsNyse:
    return build_nyse_calendar(date(2025, 1, 1), date(2025, 12, 31))


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (_utc(2025, 6, 11, 14, 0), MarketSession.REGULAR),  # Wed 10:00 EDT
        (_utc(2025, 6, 11, 13, 0), MarketSession.PRE),  # 09:00 EDT
        (_utc(2025, 6, 11, 20, 30), MarketSession.POST),  # 16:30 EDT
        (_utc(2025, 6, 14, 15, 0), MarketSession.CLOSED),  # Saturday
        (_utc(2025, 6, 15, 15, 0), MarketSession.CLOSED),  # Sunday
        (_utc(2025, 7, 4, 15, 0), MarketSession.CLOSED),  # Independence Day
        (_utc(2025, 12, 25, 15, 0), MarketSession.CLOSED),  # Christmas
    ],
)
def test_classification(
    cal: ExchangeCalendarsNyse, as_of: datetime, expected: MarketSession
) -> None:
    result = evaluate_market_session(as_of, cal)
    assert result.session is expected
    assert result.may_proceed is (expected is MarketSession.REGULAR)


def test_regular_session_details(cal: ExchangeCalendarsNyse) -> None:
    result = evaluate_market_session(_utc(2025, 6, 11, 14, 0), cal)
    assert result.session_date == date(2025, 6, 11)
    assert result.regular_open_utc == _utc(2025, 6, 11, 13, 30)
    assert result.regular_close_utc == _utc(2025, 6, 11, 20, 0)
    assert result.provenance.library == "exchange_calendars"
    assert result.provenance.calendar_code == "XNYS"
    assert result.provenance.library_version


def test_closed_day_has_no_bounds(cal: ExchangeCalendarsNyse) -> None:
    result = evaluate_market_session(_utc(2025, 7, 4, 15, 0), cal)
    assert result.regular_open_utc is None
    assert result.regular_close_utc is None


def test_open_inclusive_close_exclusive(cal: ExchangeCalendarsNyse) -> None:
    open_, close = _utc(2025, 6, 11, 13, 30), _utc(2025, 6, 11, 20, 0)
    tick = timedelta(microseconds=1)
    assert evaluate_market_session(open_ - tick, cal).session is MarketSession.PRE
    assert evaluate_market_session(open_, cal).may_proceed
    assert evaluate_market_session(close - tick, cal).may_proceed
    assert evaluate_market_session(close, cal).session is MarketSession.POST


def test_early_close_day(cal: ExchangeCalendarsNyse) -> None:
    # Day after Thanksgiving 2025 closes at 13:00 ET (18:00 UTC).
    before = evaluate_market_session(_utc(2025, 11, 28, 17, 30), cal)
    assert before.may_proceed
    assert before.regular_close_utc == _utc(2025, 11, 28, 18, 0)
    after = evaluate_market_session(_utc(2025, 11, 28, 18, 0), cal)
    assert after.session is MarketSession.POST
    assert not after.may_proceed
    assert not evaluate_market_session(_utc(2025, 11, 28, 19, 0), cal).may_proceed


def test_spring_forward_week(cal: ExchangeCalendarsNyse) -> None:
    # DST starts Sun 2025-03-09. Fri 03-07 is EST (open 14:30 UTC); Mon 03-10 is EDT (13:30).
    assert evaluate_market_session(_utc(2025, 3, 7, 14, 0), cal).session is MarketSession.PRE
    assert evaluate_market_session(_utc(2025, 3, 10, 14, 0), cal).may_proceed
    assert evaluate_market_session(_utc(2025, 3, 7, 20, 30), cal).may_proceed
    assert evaluate_market_session(_utc(2025, 3, 10, 20, 30), cal).session is MarketSession.POST


def test_fall_back_week(cal: ExchangeCalendarsNyse) -> None:
    # DST ends Sun 2025-11-02. Fri 10-31 is EDT (close 20:00 UTC); Mon 11-03 is EST (21:00).
    assert evaluate_market_session(_utc(2025, 10, 31, 13, 45), cal).may_proceed
    assert evaluate_market_session(_utc(2025, 11, 3, 13, 45), cal).session is MarketSession.PRE
    assert evaluate_market_session(_utc(2025, 10, 31, 20, 30), cal).session is MarketSession.POST
    assert evaluate_market_session(_utc(2025, 11, 3, 20, 30), cal).may_proceed


def test_session_date_is_new_york_date(cal: ExchangeCalendarsNyse) -> None:
    # 01:00 UTC on 06-12 is 21:00 ET on 06-11.
    result = evaluate_market_session(_utc(2025, 6, 12, 1, 0), cal)
    assert result.session_date == date(2025, 6, 11)
    assert result.session is MarketSession.POST


def test_non_utc_input_normalized(cal: ExchangeCalendarsNyse) -> None:
    as_of = datetime(2025, 6, 11, 10, 0, tzinfo=timezone(timedelta(hours=-4)))
    result = evaluate_market_session(as_of, cal)
    assert result.may_proceed
    assert result.as_of == _utc(2025, 6, 11, 14, 0)
    assert result.as_of.tzinfo is UTC


def test_naive_as_of_rejected(cal: ExchangeCalendarsNyse) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_market_session(datetime(2025, 6, 11, 14, 0), cal)  # noqa: DTZ001


def test_out_of_range_fails_closed(cal: ExchangeCalendarsNyse) -> None:
    with pytest.raises(CalendarOutOfRange):
        evaluate_market_session(_utc(2026, 6, 10, 15, 0), cal)
    with pytest.raises(CalendarOutOfRange):
        evaluate_market_session(_utc(2024, 12, 31, 15, 0), cal)


def test_inverted_range_rejected() -> None:
    with pytest.raises(ValueError, match="precede"):
        build_nyse_calendar(date(2025, 2, 1), date(2025, 1, 1))
