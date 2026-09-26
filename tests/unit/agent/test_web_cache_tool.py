import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import JsonValue

from wheelta_robinhood_agent.agent.tool_access import build_tool_access
from wheelta_robinhood_agent.agent.web_cache import (
    LOCAL_SERVER_NAME,
    UNTRUSTED_NOTE,
    WEB_CACHE_REGISTRY,
    WEB_CACHE_TOOL_NAME,
    build_web_cache_tool,
    cached_search_denial,
    capture_web_result,
    web_tool_input,
)
from wheelta_robinhood_agent.domain.enums import ExecutionMode
from wheelta_robinhood_agent.domain.web_cache import (
    WebCacheEntry,
    WebTool,
    normalize_query,
    normalize_url,
    tickers_in,
)

T0 = datetime(2026, 9, 25, 16, tzinfo=UTC)
TTL = 3600
RUN = uuid.uuid4()


@dataclass
class FakeStore:
    entries: list[WebCacheEntry] = field(default_factory=list)

    def _key(self, tool: WebTool, raw: str) -> str:
        return normalize_query(raw) if tool is WebTool.WEB_SEARCH else normalize_url(raw)

    def fresh_for_ticker(
        self, ticker: str, now: datetime, max_age_seconds: int, limit: int
    ) -> tuple[WebCacheEntry, ...]:
        hits = [
            e
            for e in self.entries
            if e.tool is WebTool.WEB_SEARCH
            and ticker in e.tickers
            and e.is_fresh(now, max_age_seconds)
        ]
        return tuple(sorted(hits, key=lambda e: e.retrieved_at, reverse=True)[:limit])

    def fresh_for_key(
        self, tool: WebTool, tool_input: str, now: datetime, max_age_seconds: int
    ) -> WebCacheEntry | None:
        key = self._key(tool, tool_input)
        hits = [
            e
            for e in self.entries
            if e.tool is tool and e.cache_key == key and e.is_fresh(now, max_age_seconds)
        ]
        return max(hits, key=lambda e: e.retrieved_at, default=None)

    def record(
        self,
        *,
        run_id: uuid.UUID,
        tool_call_id: uuid.UUID,
        tool: WebTool,
        tool_input: str,
        result: JsonValue,
        retrieved_at: datetime,
    ) -> WebCacheEntry:
        entry = WebCacheEntry(
            entry_id=uuid.uuid4(),
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool=tool,
            cache_key=self._key(tool, tool_input),
            tickers=tickers_in(tool_input) if tool is WebTool.WEB_SEARCH else (),
            query_raw=tool_input if tool is WebTool.WEB_SEARCH else None,
            url=tool_input if tool is WebTool.WEB_FETCH else None,
            result=result,
            retrieved_at=retrieved_at,
        )
        self.entries.append(entry)
        return entry


def _capture(store: FakeStore, tool: str, tool_input: dict[str, Any], at: datetime = T0) -> Any:
    return capture_web_result(store, run_id=RUN, tool_call_id=uuid.uuid4(), tool_name=tool,
                              tool_input=tool_input, validated_result={"r": 1},
                              retrieved_at=at)  # fmt: skip


def test_web_tool_input() -> None:
    assert web_tool_input("WebSearch", {"query": "AAPL"}) == (WebTool.WEB_SEARCH, "AAPL")
    assert web_tool_input("WebSearch", {"query": "  "}) is None
    assert web_tool_input("WebSearch", {}) is None
    assert web_tool_input("WebFetch", {"url": "https://x.com/a"}) == (
        WebTool.WEB_FETCH,
        "https://x.com/a",
    )
    assert web_tool_input("WebFetch", {"url": "file:///etc/passwd"}) is None
    assert web_tool_input("WebFetch", {"url": 3}) is None
    assert web_tool_input("mcp__robinhood__get_equity_quotes", {"symbol": "AAPL"}) is None


def test_capture_and_denial() -> None:
    store = FakeStore()
    assert _capture(store, "mcp__wheelta__wheelta_quotes", {"symbols": ["AAPL"]}) is None
    assert cached_search_denial(store, "WebSearch", {"query": "AAPL earnings"}, T0, TTL) is None
    entry = _capture(store, "WebSearch", {"query": "AAPL earnings"})
    assert entry is not None
    reason = cached_search_denial(store, "WebSearch", {"query": "aapl  EARNINGS"},
                                  T0 + timedelta(seconds=30), TTL)  # fmt: skip
    assert reason is not None
    assert "30 s ago" in reason and str(entry.tool_call_id) in reason
    assert f"mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}" in reason
    assert cached_search_denial(store, "WebSearch", {"query": "AAPL earnings"},
                                T0 + timedelta(seconds=TTL + 1), TTL) is None  # fmt: skip
    assert cached_search_denial(store, "WebSearch", {"query": "AAPL guidance"}, T0, TTL) is None
    assert cached_search_denial(store, "Bash", {"command": "ls"}, T0, TTL) is None


def test_lookup_tool_returns_labelled_fresh_entries() -> None:
    store = FakeStore()
    _capture(store, "WebSearch", {"query": "AAPL earnings"})
    _capture(store, "WebSearch", {"query": "MSFT earnings"})
    _capture(store, "WebSearch", {"query": "AAPL old"}, T0 - timedelta(seconds=TTL + 1))
    lookup = build_web_cache_tool(store, lambda: T0 + timedelta(seconds=5), TTL)
    out = asyncio.run(lookup.handler({"ticker": "AAPL"}))
    assert "is_error" not in out
    body = json.loads(out["content"][0]["text"])
    assert body["note"] == UNTRUSTED_NOTE
    assert [e["query"] for e in body["entries"]] == ["AAPL earnings"]
    assert body["entries"][0]["age_seconds"] == 5
    empty = json.loads(asyncio.run(lookup.handler({"ticker": "NVDA"}))["content"][0]["text"])
    assert empty["entries"] == []


def test_lookup_tool_rejects_bad_input() -> None:
    lookup = build_web_cache_tool(FakeStore(), lambda: T0, TTL)
    for args in ({"ticker": "aapl"}, {"ticker": 5}, {}):
        out = asyncio.run(lookup.handler(args))
        assert out["is_error"] is True


def test_lookup_tool_is_allowlisted_in_every_mode() -> None:
    name = f"mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}"
    for mode in ExecutionMode:
        access = build_tool_access(effective_mode=mode, workspace_writes=False,
                                   registries=(WEB_CACHE_REGISTRY,))  # fmt: skip
        assert name in access.allowed_tools
