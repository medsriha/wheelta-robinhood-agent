"""A fake streamable-HTTP MCP server on 127.0.0.1 with controllable tool results.

Built on the `mcp` package's low-level `Server` (mcp 2.x: constructor handlers; FastMCP is
`MCPServer` there and is not used because it would wrap our payloads). It lists every tool
name in a registry (default: the verified Robinhood registry, so discovery diffs are clean) and
answers `tools/call` from a per-tool `Behavior`. Payload shapes are invented for the harness.

A `TransportFailure` behaviour is served by an ASGI wrapper that answers that tool's
`tools/call` POST with HTTP 500 before the MCP layer sees it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server

from wheelta_robinhood_agent.integrations.registry import ToolRegistry
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY


@dataclass(frozen=True)
class Json:
    """A successful result: one text block holding this JSON document."""

    payload: Any


@dataclass(frozen=True)
class RawText:
    """A successful result with arbitrary text (e.g. not JSON at all)."""

    text: str


@dataclass(frozen=True)
class IsError:
    """An MCP `isError: true` result."""

    text: str


@dataclass(frozen=True)
class Slow:
    """Sleep before answering; `started` is set when the call arrives."""

    seconds: float
    then: Json
    started: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class TransportFailure:
    """HTTP 500 for this tool's tools/call request."""


Behavior = Json | RawText | IsError | Slow | TransportFailure


def oversized_payload(sentinel: str, approx_chars: int = 400_000) -> dict[str, Any]:
    """A JSON document far above the CLI's ~25k-token MCP output threshold, sentinel-laced."""
    row = {"note": sentinel, "filler": "x" * 200}
    rows = [row] * (approx_chars // 230 + 1)
    return {"sentinel_head": sentinel, "rows": rows, "sentinel_tail": sentinel}


def _schema(tool: str) -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": True}


class FakeMcpServer:
    """ASGI app: the MCP streamable-HTTP endpoint at /mcp plus the transport-failure wrapper."""

    def __init__(
        self,
        behaviors: Mapping[str, Behavior],
        registry: ToolRegistry = ROBINHOOD_REGISTRY,
    ) -> None:
        self.behaviors = dict(behaviors)
        self.registry = registry
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.http_posts: list[str] = []
        self._lock = threading.Lock()
        server: Server[Any] = Server(
            "harness-" + registry.server,
            version="0.0.0-harness",
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )
        self._app = server.streamable_http_app(
            streamable_http_path="/mcp", json_response=True, stateless_http=True
        )

    def called(self, tool: str) -> int:
        with self._lock:
            return sum(1 for name, _ in self.calls if name == tool)

    async def _list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=t.name, description=f"harness {t.name}", input_schema=_schema(t.name)
                )
                for t in self.registry.tools
            ]
        )

    async def _call_tool(
        self, ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        args = dict(params.arguments or {})
        with self._lock:
            self.calls.append((params.name, args))
        behavior = self.behaviors.get(params.name)
        if isinstance(behavior, Slow):
            behavior.started.set()
            await anyio.sleep(behavior.seconds)
            behavior = behavior.then
        if isinstance(behavior, Json):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(behavior.payload))]
            )
        if isinstance(behavior, RawText):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=behavior.text)]
            )
        if isinstance(behavior, IsError):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=behavior.text)], is_error=True
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"no behaviour for {params.name}")],
            is_error=True,
        )

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self._app(scope, receive, send)
            return
        chunks: list[bytes] = []
        while True:
            message = await receive()
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        raw = b"".join(chunks)
        tool = _called_tool(raw)
        with self._lock:
            self.http_posts.append(tool or "<other>")
        if tool is not None and isinstance(self.behaviors.get(tool), TransportFailure):
            with self._lock:
                self.calls.append((tool, {}))
            await send({"type": "http.response.start", "status": 500, "headers": []})
            await send({"type": "http.response.body", "body": b"upstream exploded"})
            return
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        await self._app(scope, replay, send)


def _called_tool(raw: bytes) -> str | None:
    try:
        message = json.loads(raw)
    except ValueError:
        return None
    if isinstance(message, dict) and message.get("method") == "tools/call":
        name = (message.get("params") or {}).get("name")
        return name if isinstance(name, str) else None
    return None


def wait_for(event: threading.Event, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if event.wait(0.05):
            return True
    return False
