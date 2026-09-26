"""Web search/fetch cache repository (ADR-0016, migrations/0001_initial.sql).

Entries are appended once per successful WebSearch/WebFetch tool call and never updated.
Freshness is decided at read time from `retrieved_at` and a max age the caller passes in
(`data_quality.freshness.news_max_age_seconds`), so changing the TTL needs no data change.
"""

import uuid
from datetime import datetime, timedelta

import psycopg
from psycopg.types.json import Jsonb
from pydantic import JsonValue

from wheelta_robinhood_agent.domain.web_cache import (
    WebCacheEntry,
    WebTool,
    normalize_query,
    normalize_url,
    tickers_in,
    validate_ticker,
)
from wheelta_robinhood_agent.ledger.ids import new_id

_INSERT_COLUMNS = (
    "entry_id, run_id, tool_call_id, tool, cache_key, query_raw, url, result, retrieved_at"
)
_COLUMNS = (
    "e.entry_id, e.run_id, e.tool_call_id, e.tool, e.cache_key, e.query_raw, e.url, e.result, "
    "e.retrieved_at, ARRAY(SELECT t.ticker FROM web_cache_entry_tickers t "
    "WHERE t.entry_id = e.entry_id ORDER BY t.ticker)"
)


def record_web_result(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    run_id: uuid.UUID,
    tool_call_id: uuid.UUID,
    tool: WebTool,
    tool_input: str,
    result: JsonValue,
    retrieved_at: datetime,
) -> WebCacheEntry:
    """Append the result of one successful call. `tool_input` is the query or the URL.

    Idempotent per tool call: recording the same tool call again returns the stored entry.
    Only record results that passed the PostToolUse validation; errors are never cached.
    """
    if tool is WebTool.WEB_SEARCH:
        cache_key, query_raw, url = normalize_query(tool_input), tool_input, None
    else:
        cache_key, query_raw, url = normalize_url(tool_input), None, tool_input
    entry_id = new_id()
    with conn.transaction():
        inserted = conn.execute(
            f"INSERT INTO web_cache_entries ({_INSERT_COLUMNS}) "  # noqa: S608 - constant list
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (tool_call_id) DO NOTHING "
            "RETURNING entry_id",
            (entry_id, run_id, tool_call_id, tool.value, cache_key, query_raw, url,
             Jsonb(result), retrieved_at),
        ).fetchone()  # fmt: skip
        if inserted is not None and query_raw is not None:
            for ticker in tickers_in(query_raw):
                conn.execute(
                    "INSERT INTO web_cache_entry_tickers (entry_id, ticker) VALUES (%s, %s)",
                    (entry_id, ticker),
                )
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM web_cache_entries e WHERE e.tool_call_id = %s",  # noqa: S608
            (tool_call_id,),
        ).fetchone()
    assert row is not None  # noqa: S101 - inserted or already present in this transaction
    return _entry(row)


def fresh_entry_for_key(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    tool: WebTool,
    tool_input: str,
    now: datetime,
    max_age_seconds: int,
) -> WebCacheEntry | None:
    """The newest fresh entry for exactly this normalized query or URL, if any."""
    key = normalize_query(tool_input) if tool is WebTool.WEB_SEARCH else normalize_url(tool_input)
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM web_cache_entries e "  # noqa: S608 - constant column list
        "WHERE e.tool = %s AND e.cache_key = %s AND e.retrieved_at BETWEEN %s AND %s "
        "ORDER BY e.retrieved_at DESC, e.entry_id LIMIT 1",
        (tool.value, key, now - timedelta(seconds=max_age_seconds), now),
    ).fetchone()
    return None if row is None else _entry(row)


def fresh_entries_for_ticker(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    ticker: str,
    now: datetime,
    max_age_seconds: int,
    limit: int,
) -> tuple[WebCacheEntry, ...]:
    """Fresh WebSearch entries tagged with `ticker` in web_cache_entry_tickers, newest first.

    Tags are written with the entry from domain.web_cache.tickers_in(query).
    """
    validate_ticker(ticker)
    if limit < 1:
        raise ValueError("limit must be positive")
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM web_cache_entries e "  # noqa: S608 - constant column list
        "JOIN web_cache_entry_tickers tag ON tag.entry_id = e.entry_id AND tag.ticker = %s "
        "WHERE e.tool = 'WebSearch' AND e.retrieved_at BETWEEN %s AND %s "
        "ORDER BY e.retrieved_at DESC, e.entry_id LIMIT %s",
        (ticker, now - timedelta(seconds=max_age_seconds), now, limit),
    ).fetchall()
    return tuple(_entry(row) for row in rows)


def _entry(row: tuple[object, ...]) -> WebCacheEntry:
    (entry_id, run_id, tool_call_id, tool, cache_key, query_raw, url, result, retrieved_at,
     tickers) = row  # fmt: skip
    return WebCacheEntry.model_validate(
        {
            "entry_id": entry_id,
            "run_id": run_id,
            "tool_call_id": tool_call_id,
            "tool": tool,
            "cache_key": cache_key,
            "tickers": tuple(tickers) if isinstance(tickers, list) else (),
            "query_raw": query_raw,
            "url": url,
            "result": result,
            "retrieved_at": retrieved_at,
        }
    )
