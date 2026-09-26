"""Web cache entries (ADR-0016): append-only, idempotent per tool call, fresh-by-age reads."""

import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv, ToolTier
from wheelta_robinhood_agent.domain.web_cache import WebTool
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import record_tool_call_requested
from wheelta_robinhood_agent.ledger.web_cache import (
    fresh_entries_for_ticker,
    fresh_entry_for_key,
    record_web_result,
)

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 16, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=3)
TTL = 3600


def _call(conn: Conn, run_id: uuid.UUID, sdk_id: str, tool: WebTool) -> uuid.UUID:
    return record_tool_call_requested(
        conn, run_id=run_id, sdk_tool_use_id=sdk_id, stage="agent", server="builtin",
        tool=tool.value, tier=None, arguments_redacted={}, requested_at=T0,
    ).tool_call_id  # fmt: skip


def _search(conn: Conn, run_id: uuid.UUID, sdk_id: str, query: str, at: datetime) -> uuid.UUID:
    call = _call(conn, run_id, sdk_id, WebTool.WEB_SEARCH)
    record_web_result(conn, run_id=run_id, tool_call_id=call, tool=WebTool.WEB_SEARCH,
                      tool_input=query, result={"links": [query]}, retrieved_at=at)  # fmt: skip
    return call


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


def test_record_is_idempotent_and_append_only(conn: Conn, run_id: uuid.UUID) -> None:
    call = _call(conn, run_id, "toolu_1", WebTool.WEB_SEARCH)
    kwargs = dict(run_id=run_id, tool_call_id=call, tool=WebTool.WEB_SEARCH,
                  tool_input="AAPL earnings", result={"a": 1}, retrieved_at=T0)  # fmt: skip
    first = record_web_result(conn, **kwargs)  # type: ignore[arg-type]
    again = record_web_result(conn, **kwargs)  # type: ignore[arg-type]
    assert first == again
    assert first.cache_key == "aapl earnings"
    assert first.tickers == ("AAPL",)
    (count,) = conn.execute("SELECT count(*) FROM web_cache_entry_tickers").fetchone()  # type: ignore[misc]
    assert count == 1
    with pytest.raises(psycopg.errors.RestrictViolation):
        conn.execute("UPDATE web_cache_entries SET cache_key = 'x'")
    with pytest.raises(psycopg.errors.RestrictViolation):
        conn.execute("DELETE FROM web_cache_entries")


def test_fresh_entry_for_key_respects_ttl_and_normalization(conn: Conn, run_id: uuid.UUID) -> None:
    _search(conn, run_id, "toolu_1", "AAPL  Earnings", T0)
    hit = fresh_entry_for_key(conn, tool=WebTool.WEB_SEARCH, tool_input="aapl earnings",
                              now=T0 + timedelta(seconds=TTL), max_age_seconds=TTL)  # fmt: skip
    assert hit is not None
    late = T0 + timedelta(seconds=TTL + 1)
    stale = fresh_entry_for_key(conn, tool=WebTool.WEB_SEARCH, tool_input="aapl earnings",
                                now=late, max_age_seconds=TTL)  # fmt: skip
    assert stale is None
    future = fresh_entry_for_key(conn, tool=WebTool.WEB_SEARCH, tool_input="aapl earnings",
                                 now=T0 - timedelta(seconds=1), max_age_seconds=TTL)  # fmt: skip
    assert future is None


def test_webfetch_keyed_by_normalized_url(conn: Conn, run_id: uuid.UUID) -> None:
    call = _call(conn, run_id, "toolu_f", WebTool.WEB_FETCH)
    record_web_result(conn, run_id=run_id, tool_call_id=call, tool=WebTool.WEB_FETCH,
                      tool_input="HTTPS://Example.com/IR?q=1#top", result="page",
                      retrieved_at=T0)  # fmt: skip
    hit = fresh_entry_for_key(conn, tool=WebTool.WEB_FETCH,
                              tool_input="https://example.com/IR?q=1", now=T0,
                              max_age_seconds=TTL)  # fmt: skip
    assert hit is not None and hit.url == "HTTPS://Example.com/IR?q=1#top"
    miss = fresh_entry_for_key(conn, tool=WebTool.WEB_FETCH,
                               tool_input="https://example.com/ir?q=1", now=T0,
                               max_age_seconds=TTL)  # fmt: skip
    assert miss is None


def test_fresh_entries_for_ticker(conn: Conn, run_id: uuid.UUID) -> None:
    _search(conn, run_id, "t1", "AAPL earnings date", T0)
    _search(conn, run_id, "t2", "(AAPL) guidance", T0 + timedelta(seconds=10))
    _search(conn, run_id, "t3", "AAPLX fund news", T0)
    _search(conn, run_id, "t4", "aapl lowercase", T0)
    _search(conn, run_id, "t5", "AAPL old", T0 - timedelta(seconds=TTL + 5))
    now = T0 + timedelta(seconds=20)
    entries = fresh_entries_for_ticker(conn, ticker="AAPL", now=now, max_age_seconds=TTL,
                                       limit=5)  # fmt: skip
    assert [e.query_raw for e in entries] == ["(AAPL) guidance", "AAPL earnings date"]
    assert all("AAPL" in e.tickers for e in entries)
    tags = conn.execute(
        "SELECT ticker FROM web_cache_entry_tickers t JOIN web_cache_entries e "
        "USING (entry_id) WHERE e.query_raw = 'AAPLX fund news'"
    ).fetchall()
    assert tags == [("AAPLX",)]
    one = fresh_entries_for_ticker(conn, ticker="AAPL", now=now, max_age_seconds=TTL, limit=1)
    assert len(one) == 1
    with pytest.raises(ValueError):
        fresh_entries_for_ticker(conn, ticker="aapl", now=now, max_age_seconds=TTL, limit=5)
    with pytest.raises(ValueError):
        fresh_entries_for_ticker(conn, ticker="AAPL", now=now, max_age_seconds=TTL, limit=0)


def test_tier_unused_marker() -> None:
    # tool_calls.tier is NULL for built-ins; ToolTier stays importable for readers of this test.
    assert ToolTier.R.value == "R"
