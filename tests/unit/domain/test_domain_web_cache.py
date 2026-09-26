import uuid
from datetime import UTC, datetime, timedelta

import pytest

from wheelta_robinhood_agent.domain.web_cache import (
    WebCacheEntry,
    WebTool,
    normalize_query,
    normalize_url,
    ticker_mentioned,
    tickers_in,
    validate_ticker,
)

T0 = datetime(2026, 9, 25, 16, tzinfo=UTC)


def test_normalize_query() -> None:
    assert normalize_query("  AAPL   Earnings\tDate ") == "aapl earnings date"


def test_normalize_url() -> None:
    assert normalize_url("HTTPS://Example.COM/Path?A=1#frag") == "https://example.com/Path?A=1"
    for bad in ("ftp://x.com/a", "/relative", "example.com", "javascript:alert(1)"):
        with pytest.raises(ValueError):
            normalize_url(bad)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("AAPL earnings", True), ("(AAPL)", True), ("news on AAPL", True), ("AAPL.", True),
     ("aapl earnings", False), ("AAPLX", False), ("XAAPL", False), ("AAPL1", False),
     ("Apple earnings", False)],
)  # fmt: skip
def test_ticker_mentioned(text: str, expected: bool) -> None:
    assert ticker_mentioned("AAPL", text) is expected


@pytest.mark.parametrize("bad", ["aapl", "BRK.B", "TOOLONG", "", "A1"])
def test_validate_ticker_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_ticker(bad)
    with pytest.raises(ValueError):
        ticker_mentioned(bad, "anything")


def test_entry_freshness() -> None:
    entry = WebCacheEntry(
        entry_id=uuid.uuid4(), run_id=uuid.uuid4(), tool_call_id=uuid.uuid4(),
        tool=WebTool.WEB_SEARCH, cache_key="aapl", tickers=("AAPL",), query_raw="AAPL", url=None,
        result={"x": [1, "a"]}, retrieved_at=T0,
    )  # fmt: skip
    assert entry.is_fresh(T0 + timedelta(seconds=3600), 3600)
    assert not entry.is_fresh(T0 + timedelta(seconds=3601), 3600)
    assert not entry.is_fresh(T0 - timedelta(seconds=1), 3600)
    assert entry.age_seconds(T0 + timedelta(seconds=42)) == 42


def test_tickers_in() -> None:
    assert tickers_in("AAPL vs MSFT: AAPL (NVDA) earnings, aapl AAPLXX Q3") == (
        "AAPL",
        "MSFT",
        "NVDA",
    )
    assert tickers_in("Apple earnings") == ()


def test_tickers_in_agrees_with_ticker_mentioned() -> None:
    query = "AAPL vs MSFT, (NVDA) and AAPLX"
    for ticker in ("AAPL", "MSFT", "NVDA", "AAPLX", "TSLA"):
        assert (ticker in tickers_in(query)) is ticker_mentioned(ticker, query)


def test_entry_tickers_must_match_query() -> None:
    base = dict(entry_id=uuid.uuid4(), run_id=uuid.uuid4(), tool_call_id=uuid.uuid4(),
                tool=WebTool.WEB_SEARCH, cache_key="x", query_raw="AAPL MSFT", url=None,
                result=None, retrieved_at=T0)  # fmt: skip
    for bad in ((), ("AAPL",), ("MSFT", "AAPL"), ("AAPL", "MSFT", "NVDA"), ("aapl", "MSFT")):
        with pytest.raises(ValueError):
            WebCacheEntry(tickers=bad, **base)  # type: ignore[arg-type]
    assert WebCacheEntry(tickers=("AAPL", "MSFT"), **base).tickers == ("AAPL", "MSFT")  # type: ignore[arg-type]
