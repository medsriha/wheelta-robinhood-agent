"""NYSE market-session gating (ARCHITECTURE.md "Market-session gating"; CLAUDE.md §20).

Runs happen only during the NYSE regular session. Holidays, early closes, and DST are taken
from the `exchange_calendars` XNYS calendar, never from hand-rolled lists (CLAUDE.md §4).

The calendar object is built once by the orchestrator (`build_nyse_calendar`) and passed in;
`evaluate_market_session` never reads the clock (the caller supplies `as_of`).
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from typing import Protocol
from zoneinfo import ZoneInfo

import exchange_calendars  # type: ignore[import-untyped]

from wheelta_robinhood_agent.domain.enums import MarketSession

NYSE_CALENDAR_CODE = "XNYS"
CALENDAR_LIBRARY = "exchange_calendars"
NYSE_TZ = ZoneInfo("America/New_York")


class CalendarOutOfRange(ValueError):
    """`as_of` falls outside the dates the calendar was built for. Fail closed; never guess."""


@dataclass(frozen=True, slots=True)
class SessionBounds:
    """Regular-session open and close (UTC) of one trading day; open inclusive, close exclusive."""

    open_utc: datetime
    close_utc: datetime


@dataclass(frozen=True, slots=True)
class CalendarProvenance:
    """Which calendar decided the session (INTERFACES.md `market_session` provenance)."""

    library: str
    library_version: str
    calendar_code: str


class TradingCalendar(Protocol):
    """The minimal typed view of a trading calendar that session gating needs."""

    @property
    def provenance(self) -> CalendarProvenance: ...

    def covers(self, day: date) -> bool:
        """True if `day` is inside the range the calendar was built for."""
        ...

    def session_bounds(self, day: date) -> SessionBounds | None:
        """Regular-session bounds for `day`, or None if `day` is not a trading session."""
        ...


class ExchangeCalendarsNyse:
    """`TradingCalendar` backed by the `exchange_calendars` XNYS calendar.

    Construct once per run via `build_nyse_calendar`; holds no mutable state after init.
    """

    def __init__(self, start: date, end: date) -> None:
        if end < start:
            raise ValueError("calendar end must not precede start")
        self._calendar = exchange_calendars.get_calendar(
            NYSE_CALENDAR_CODE, start=start.isoformat(), end=end.isoformat()
        )
        self._first: date = self._calendar.first_session.date()
        self._last: date = self._calendar.last_session.date()
        self._start = start
        self._end = end
        self._provenance = CalendarProvenance(
            library=CALENDAR_LIBRARY,
            library_version=version("exchange-calendars"),
            calendar_code=NYSE_CALENDAR_CODE,
        )

    @property
    def provenance(self) -> CalendarProvenance:
        return self._provenance

    def covers(self, day: date) -> bool:
        return self._start <= day <= self._end and self._first <= day <= self._last

    def session_bounds(self, day: date) -> SessionBounds | None:
        label = day.isoformat()
        if not self._calendar.is_session(label):
            return None
        open_utc: datetime = self._calendar.session_open(label).to_pydatetime()
        close_utc: datetime = self._calendar.session_close(label).to_pydatetime()
        return SessionBounds(open_utc=open_utc.astimezone(UTC), close_utc=close_utc.astimezone(UTC))


def build_nyse_calendar(start: date, end: date) -> ExchangeCalendarsNyse:
    """Build the XNYS calendar for `[start, end]`. Local computation only; no network."""
    return ExchangeCalendarsNyse(start, end)


@dataclass(frozen=True, slots=True)
class MarketSessionResult:
    """Session state at `as_of` and whether a run may proceed (INTERFACES.md `market_session`)."""

    as_of: datetime
    session: MarketSession
    may_proceed: bool
    session_date: date
    regular_open_utc: datetime | None
    regular_close_utc: datetime | None
    provenance: CalendarProvenance


def evaluate_market_session(as_of: datetime, calendar: TradingCalendar) -> MarketSessionResult:
    """Classify `as_of` against the NYSE regular session.

    Rule (ARCHITECTURE.md "Market-session gating"): the run may proceed only inside the regular
    session, `open <= as_of < close`, where the close is the early close on early-close days.
    Labels: a non-session day is `closed`; on a session day, before the open is `pre` and at or
    after the close is `post`. These labels are relative to the regular session only; the
    calendar does not model extended-hours bounds, and gating never uses them.

    Raises ValueError for a naive `as_of` and CalendarOutOfRange if the calendar does not cover
    the New York date of `as_of`.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of_utc = as_of.astimezone(UTC)
    session_date = as_of_utc.astimezone(NYSE_TZ).date()
    if not calendar.covers(session_date):
        raise CalendarOutOfRange(f"calendar does not cover {session_date.isoformat()}")

    bounds = calendar.session_bounds(session_date)
    if bounds is None:
        session = MarketSession.CLOSED
    elif as_of_utc < bounds.open_utc:
        session = MarketSession.PRE
    elif as_of_utc < bounds.close_utc:
        session = MarketSession.REGULAR
    else:
        session = MarketSession.POST

    return MarketSessionResult(
        as_of=as_of_utc,
        session=session,
        may_proceed=session is MarketSession.REGULAR,
        session_date=session_date,
        regular_open_utc=bounds.open_utc if bounds else None,
        regular_close_utc=bounds.close_utc if bounds else None,
        provenance=calendar.provenance,
    )
