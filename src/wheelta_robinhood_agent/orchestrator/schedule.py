"""Agent-chosen next runs (ADR-0028).

The Railway cron ticks every 5 minutes. A tick starts a session only when it is due: no
schedule has been recorded yet (the initial run), or the clock has reached the latest
recorded `not_before`. Each run that passes the gate records a fallback first (the rules'
`scheduling.fallback_next_run_minutes` after the gate), so a run that crashes or returns no
valid output is retried on the old hourly cadence. A valid `next_run` in the agent's output
then replaces it.

Two bounds apply to the agent's choice (owner decisions, ADR-0028):

1. A maximum gap: a time later than `scheduling.max_next_run_gap_hours` after the gate is
   capped to that point. There is no minimum gap.
2. The market calendar: a time outside the NYSE regular session (after capping) moves to the
   next session's open. So a cap that lands on a weekend or holiday runs at the next open.

Pure: the caller supplies `now`, the requested time, and the calendar.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Final

from wheelta_robinhood_agent.orchestrator.market_session import (
    NYSE_TZ,
    CalendarOutOfRange,
    TradingCalendar,
)

# How many calendar days after a requested time to look for the next session. The longest
# NYSE closure on record is shorter than this; a longer gap means the calendar is wrong.
SESSION_SEARCH_DAYS: Final = 14


class ScheduleSource(StrEnum):
    """Who chose a recorded next-run time."""

    FALLBACK = "fallback"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class NextRun:
    """A next-run time to record (run event `schedule`).

    `requested_at` is what was asked for (the agent's time, or gate time plus the fallback);
    `latest_at` is the maximum-gap cap; `capped_at` is the earlier of the two; `not_before` is
    `capped_at` moved into the regular session. A tick is due once the clock reaches
    `not_before`.
    """

    not_before: datetime
    source: ScheduleSource
    requested_at: datetime
    latest_at: datetime

    @property
    def capped_at(self) -> datetime:
        return min(self.requested_at, self.latest_at)

    @property
    def capped(self) -> bool:
        return self.requested_at > self.latest_at

    @property
    def moved_to_session_open(self) -> bool:
        return self.not_before != self.capped_at

    def event_payload(self) -> dict[str, object]:
        return {
            "not_before": self.not_before.isoformat(),
            "source": self.source.value,
            "requested_at": self.requested_at.isoformat(),
            "latest_at": self.latest_at.isoformat(),
            "capped": self.capped,
            "moved_to_session_open": self.moved_to_session_open,
        }


def calendar_window(at: datetime) -> tuple[date, date]:
    """The (start, end) dates a calendar must cover to place `at` into a session.

    The start reaches back as far as the search reaches forward: a calendar covers only days
    between its first and last session, so a weekend or holiday `at` needs a session before
    it as well as after. Raises OverflowError near the ends of the date range.
    """
    day = at.astimezone(NYSE_TZ).date()
    span = timedelta(days=SESSION_SEARCH_DAYS)
    return day - span, day + span


def into_regular_session(at: datetime, calendar: TradingCalendar) -> datetime:
    """`at` if it falls inside an NYSE regular session (open inclusive, close exclusive),
    otherwise the open of the next session after it. Returned in UTC.

    Raises ValueError for a naive `at` and CalendarOutOfRange when the calendar does not
    cover the search window or has no session in it.
    """
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("at must be timezone-aware")
    at_utc = at.astimezone(UTC)
    first = at_utc.astimezone(NYSE_TZ).date()
    for offset in range(SESSION_SEARCH_DAYS + 1):
        day = first + timedelta(days=offset)
        if not calendar.covers(day):
            raise CalendarOutOfRange(f"calendar does not cover {day.isoformat()}")
        bounds = calendar.session_bounds(day)
        if bounds is None:
            continue
        if at_utc < bounds.open_utc:
            return bounds.open_utc
        if at_utc < bounds.close_utc:
            return at_utc
    raise CalendarOutOfRange(
        f"no NYSE session within {SESSION_SEARCH_DAYS} days of {first.isoformat()}"
    )


def next_run(
    requested_at: datetime,
    source: ScheduleSource,
    calendar: TradingCalendar,
    *,
    latest_at: datetime,
) -> NextRun:
    """The next-run record for `requested_at`: capped at `latest_at`, then moved into the
    regular session. `calendar` must cover `calendar_window(min(requested_at, latest_at))`."""
    for name, value in (("requested_at", requested_at), ("latest_at", latest_at)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
    requested, latest = requested_at.astimezone(UTC), latest_at.astimezone(UTC)
    return NextRun(
        not_before=into_regular_session(min(requested, latest), calendar),
        source=source,
        requested_at=requested,
        latest_at=latest,
    )


def fallback_requested_at(gate_at: datetime, fallback_minutes: int) -> datetime:
    """The fallback request: `fallback_minutes` after the run passed the gate."""
    return gate_at.astimezone(UTC) + timedelta(minutes=fallback_minutes)


def latest_next_run(gate_at: datetime, max_gap_hours: int) -> datetime:
    """The maximum-gap cap: `max_gap_hours` after the run passed the gate (ADR-0028)."""
    return gate_at.astimezone(UTC) + timedelta(hours=max_gap_hours)


def is_due(now: datetime, not_before: datetime | None) -> bool:
    """A tick is due when nothing is scheduled yet (initial run) or `now >= not_before`."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return not_before is None or now >= not_before
