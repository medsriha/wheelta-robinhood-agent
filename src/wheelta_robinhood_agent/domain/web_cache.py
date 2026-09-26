"""Web search/fetch cache rules (ADR-0016). Pure: keys, ticker matching, freshness.

The cache avoids paying twice for the same web research. Entries are recorded results of the
built-in WebSearch/WebFetch tools; they stay untrusted data (CLAUDE.md §11, §24) and carry
their original tool-call provenance. The TTL is `data_quality.freshness.news_max_age_seconds`,
passed in by the caller, never a constant here.
"""

import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import JsonValue, field_validator, model_validator

from wheelta_robinhood_agent.domain.base import DomainModel, NonEmptyStr, UtcDatetime
from wheelta_robinhood_agent.domain.sanity import is_fresh

_TICKER_RE = re.compile(r"[A-Z]{1,5}")
_WHITESPACE_RE = re.compile(r"\s+")


class WebTool(StrEnum):
    WEB_SEARCH = "WebSearch"
    WEB_FETCH = "WebFetch"


def normalize_query(query: str) -> str:
    """Cache key for WebSearch: case-folded, whitespace collapsed, trimmed.

    Two queries share an entry only if they are identical after this normalization; no
    stemming or reordering, so a cache hit never answers a different question.
    """
    return _WHITESPACE_RE.sub(" ", query).strip().casefold()


def normalize_url(url: str) -> str:
    """Cache key for WebFetch: scheme and host lowercased, fragment dropped.

    Path and query string are kept verbatim (they are case-sensitive on most servers).
    Raises ValueError for anything but an absolute http(s) URL.
    """
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ValueError("only absolute http(s) URLs are cacheable")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def validate_ticker(ticker: str) -> str:
    """Tickers are `^[A-Z]{1,5}$` (CLAUDE.md §10). Raises ValueError otherwise."""
    if not _TICKER_RE.fullmatch(ticker):
        raise ValueError(f"invalid ticker: {ticker!r}")
    return ticker


_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,5}(?![A-Za-z0-9])")


def tickers_in(query: str) -> tuple[str, ...]:
    """The tickers a WebSearch query is about: every whole-word, case-sensitive `[A-Z]{1,5}`
    token, deduplicated and sorted. Stored with the entry (`web_cache_entry_tickers`).

    Uppercase words that aren't tickers ("CEO", "ETF") are stored too; they only add rows no
    real lookup asks for. A query that names only the company ("Apple") tags nothing.
    """
    return tuple(sorted(set(_TOKEN_RE.findall(query))))


def ticker_mentioned(ticker: str, text: str) -> bool:
    """An entry belongs to a ticker if the query contains the ticker as a whole word.

    Case-sensitive: `AAPL` matches "AAPL earnings" and "(AAPL)" but not "aapl" or "AAPLX".
    A query that names only the company ("Apple earnings") is not attributed to a ticker.
    """
    validate_ticker(ticker)
    return re.search(rf"(?<![A-Za-z0-9]){ticker}(?![A-Za-z0-9])", text) is not None


class WebCacheEntry(DomainModel):
    """One recorded, successful WebSearch/WebFetch result."""

    entry_id: uuid.UUID
    run_id: uuid.UUID
    tool_call_id: uuid.UUID
    tool: WebTool
    cache_key: NonEmptyStr
    tickers: tuple[str, ...] = ()
    query_raw: str | None
    url: str | None
    result: JsonValue
    retrieved_at: UtcDatetime

    @field_validator("tickers")
    @classmethod
    def _tickers_sorted_unique_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for ticker in value:
            validate_ticker(ticker)
        if list(value) != sorted(set(value)):
            raise ValueError("tickers must be sorted and unique")
        return value

    @model_validator(mode="after")
    def _tickers_match_query(self) -> Self:
        expected = tickers_in(self.query_raw) if self.query_raw is not None else ()
        if self.tickers != expected:
            raise ValueError("tickers must be exactly tickers_in(query_raw)")
        return self

    def is_fresh(self, now: datetime, max_age_seconds: int) -> bool:
        """Fresh while younger than the news max age (CLAUDE.md §13: stale means missing)."""
        return is_fresh(self.retrieved_at, now, max_age_seconds)

    def age_seconds(self, now: datetime) -> int:
        return int((now - self.retrieved_at).total_seconds())
