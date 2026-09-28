"""Agent-chosen next runs (ADR-0028): session placement, fallback, and the due check."""

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from wheelta_robinhood_agent.orchestrator.market_session import (
    CalendarOutOfRange,
    build_nyse_calendar,
)
from wheelta_robinhood_agent.orchestrator.schedule import (
    SESSION_SEARCH_DAYS,
    ScheduleSource,
    calendar_window,
    fallback_requested_at,
    into_regular_session,
    is_due,
    latest_next_run,
    next_run,
)

CAL = build_nyse_calendar(date(2026, 9, 1), date(2026, 12, 31))
EDT = timezone(timedelta(hours=-4))


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        # Inside the session (EDT: 13:30-20:00 UTC): unchanged, open inclusive.
        (_utc(2026, 9, 28, 15, 17), _utc(2026, 9, 28, 15, 17)),
        (_utc(2026, 9, 28, 13, 30), _utc(2026, 9, 28, 13, 30)),
        # Before the open: that day's open.
        (_utc(2026, 9, 28, 11, 0), _utc(2026, 9, 28, 13, 30)),
        # At or after the close (exclusive): the next session's open.
        (_utc(2026, 9, 28, 20, 0), _utc(2026, 9, 29, 13, 30)),
        # Friday night and the weekend: Monday's open.
        (_utc(2026, 10, 2, 22, 0), _utc(2026, 10, 5, 13, 30)),
        (_utc(2026, 10, 4, 15, 0), _utc(2026, 10, 5, 13, 30)),
        # After DST ends (2026-11-01) the open is 14:30 UTC.
        (_utc(2026, 11, 2, 14, 0), _utc(2026, 11, 2, 14, 30)),
        # Thanksgiving is closed; the next day closes early at 13:00 ET (18:00 UTC).
        (_utc(2026, 11, 26, 16, 0), _utc(2026, 11, 27, 14, 30)),
        (_utc(2026, 11, 27, 17, 59), _utc(2026, 11, 27, 17, 59)),
        (_utc(2026, 11, 27, 18, 30), _utc(2026, 11, 30, 14, 30)),
    ],
)
def test_into_regular_session(requested: datetime, expected: datetime) -> None:
    assert into_regular_session(requested, CAL) == expected


def test_offsets_are_normalized_to_utc() -> None:
    placed = into_regular_session(datetime(2026, 9, 28, 11, 45, tzinfo=EDT), CAL)
    assert placed == _utc(2026, 9, 28, 15, 45) and placed.tzinfo is UTC


def test_naive_and_uncovered_times_raise() -> None:
    with pytest.raises(ValueError):
        into_regular_session(datetime(2026, 9, 28, 15, 0), CAL)  # noqa: DTZ001
    with pytest.raises(CalendarOutOfRange):
        into_regular_session(_utc(2027, 3, 1, 15, 0), CAL)
    # The search stops at the calendar's end rather than guessing past it.
    with pytest.raises(CalendarOutOfRange):
        into_regular_session(_utc(2026, 12, 31, 22, 0), CAL)


def test_calendar_window_covers_the_search() -> None:
    requested = _utc(2026, 12, 24, 23, 0)
    start, end = calendar_window(requested)
    assert start == date(2026, 12, 24) - timedelta(days=SESSION_SEARCH_DAYS)
    assert end == date(2026, 12, 24) + timedelta(days=SESSION_SEARCH_DAYS)
    assert into_regular_session(requested, build_nyse_calendar(start, end)) == _utc(
        2026, 12, 28, 14, 30
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (_utc(2026, 10, 4, 15, 0), _utc(2026, 10, 5, 13, 30)),  # Sunday
        (_utc(2026, 10, 3, 3, 0), _utc(2026, 10, 5, 13, 30)),  # Friday night ET, Saturday UTC
        (_utc(2026, 9, 7, 15, 0), _utc(2026, 9, 8, 13, 30)),  # Labor Day
        (_utc(2026, 12, 25, 15, 0), _utc(2026, 12, 28, 14, 30)),  # Christmas, then a weekend
        (_utc(2027, 1, 1, 15, 0), _utc(2027, 1, 4, 14, 30)),  # New Year's Day
    ],
)
def test_non_session_days_place_with_the_window_calendar(
    requested: datetime, expected: datetime
) -> None:
    # Regression: a window starting one day back had no session before a weekend or holiday,
    # so the calendar did not "cover" it and the agent's time was rejected.
    start, end = calendar_window(requested)
    assert into_regular_session(requested, build_nyse_calendar(start, end)) == expected


def test_calendar_window_overflows_at_the_end_of_the_date_range() -> None:
    with pytest.raises(OverflowError):
        calendar_window(_utc(9999, 12, 25, 15, 0))


FAR = _utc(2026, 12, 31, 0, 0)  # a cap that does not bind


def test_next_run_records_request_and_placement() -> None:
    moved = next_run(_utc(2026, 10, 3, 12, 0), ScheduleSource.AGENT, CAL, latest_at=FAR)
    assert moved.not_before == _utc(2026, 10, 5, 13, 30)
    assert moved.moved_to_session_open and not moved.capped
    assert moved.event_payload() == {
        "not_before": "2026-10-05T13:30:00+00:00",
        "source": "agent",
        "requested_at": "2026-10-03T12:00:00+00:00",
        "latest_at": "2026-12-31T00:00:00+00:00",
        "capped": False,
        "moved_to_session_open": True,
    }
    kept = next_run(_utc(2026, 9, 28, 16, 0), ScheduleSource.FALLBACK, CAL, latest_at=FAR)
    assert kept.not_before == kept.requested_at and not kept.moved_to_session_open


def test_max_gap_caps_a_later_request() -> None:
    gate = _utc(2026, 9, 28, 15, 0)  # Monday
    latest = latest_next_run(gate, 48)
    assert latest == _utc(2026, 9, 30, 15, 0)
    capped = next_run(_utc(2026, 10, 9, 15, 0), ScheduleSource.AGENT, CAL, latest_at=latest)
    assert capped.capped and capped.capped_at == latest
    assert capped.not_before == latest and not capped.moved_to_session_open
    # Exactly at the limit is not capped.
    at_limit = next_run(latest, ScheduleSource.AGENT, CAL, latest_at=latest)
    assert not at_limit.capped and at_limit.not_before == latest


def test_a_cap_on_a_weekend_moves_to_the_next_open() -> None:
    gate = _utc(2026, 10, 2, 15, 0)  # Friday
    latest = latest_next_run(gate, 48)  # Sunday 15:00 UTC
    run = next_run(_utc(2026, 10, 8, 15, 0), ScheduleSource.AGENT, CAL, latest_at=latest)
    assert run.capped and run.moved_to_session_open
    assert run.not_before == _utc(2026, 10, 5, 13, 30)


def test_next_run_rejects_naive_times() -> None:
    with pytest.raises(ValueError):
        next_run(datetime(2026, 9, 28, 15), ScheduleSource.AGENT, CAL, latest_at=FAR)  # noqa: DTZ001
    with pytest.raises(ValueError):
        next_run(
            _utc(2026, 9, 28, 15),
            ScheduleSource.AGENT,
            CAL,
            latest_at=datetime(2026, 9, 30, 15),  # noqa: DTZ001
        )


def test_fallback_is_minutes_after_the_gate() -> None:
    gate = datetime(2026, 9, 28, 11, 35, 12, tzinfo=EDT)
    assert fallback_requested_at(gate, 60) == _utc(2026, 9, 28, 16, 35, 12)


@pytest.mark.parametrize(
    ("now", "not_before", "due"),
    [
        (_utc(2026, 9, 28, 15, 0), None, True),  # nothing recorded: the initial run
        (_utc(2026, 9, 28, 15, 0), _utc(2026, 9, 28, 15, 0), True),
        (_utc(2026, 9, 28, 15, 5), _utc(2026, 9, 28, 15, 0), True),
        (_utc(2026, 9, 28, 14, 59, 59), _utc(2026, 9, 28, 15, 0), False),
    ],
)
def test_is_due(now: datetime, not_before: datetime | None, due: bool) -> None:
    assert is_due(now, not_before) is due


def test_is_due_rejects_naive_now() -> None:
    with pytest.raises(ValueError):
        is_due(datetime(2026, 9, 28, 15, 0), None)  # noqa: DTZ001
