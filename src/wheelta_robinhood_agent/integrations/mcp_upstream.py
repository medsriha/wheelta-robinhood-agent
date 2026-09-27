"""The upstream MCP client behind the validating proxy (ADR-0023; DATA_QUALITY.md).

The agent session never connects to a remote MCP server itself. `agent/proxy.py` serves the
remote tools in-process and forwards each call through an `McpUpstream` opened here, so the
Claude Code CLI only ever receives a result our code produced (CLAUDE.md §8 "Validate before
delivery to the model").

Error texts never carry remote content: every failure maps to a typed `UpstreamError` whose
message is built from fixed strings plus, at most, an HTTP status or MCP error code. A remote
error body, an `isError` result's text, or an exception message from the transport never
reaches `str(error)`.

- `401`/`403` from the server → `UpstreamAuthError` (the source is `needs-auth`).
- A per-call or connect deadline → `UpstreamTimeout`.
- A result whose JSON exceeds `max_result_bytes` → `UpstreamTooLarge` (never delivered).
- Anything else (HTTP status, protocol error, closed stream) → `UpstreamUnavailable`.

No retries (CLAUDE.md §14): a Tier X call must never be repeated, and read retries would
spend Robinhood's undocumented rate limit. HTTP status is observed with an httpx2 response
hook; streamable-HTTP POSTs run on a transport task, so a status is attributed to a call by
counting auth failures seen while that call was in flight (a concurrent call's 401 can mark
this one as an auth failure too, which fails closed).

Verified against mcp 2.2.0 (`mcp.Client`, `streamable_http_client`) with the harness fake
server: HTTP 500 raises `MCPError(-32603)` without the body; a read timeout raises
`MCPError(-32001)`.
"""

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final, Protocol

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from pydantic import JsonValue

from wheelta_robinhood_agent.integrations.status import McpHttpServer

# Operational bounds, not trading values.
MAX_UPSTREAM_RESULT_BYTES: Final = 2_000_000
MAX_TOOL_LIST_PAGES: Final = 20
_AUTH_STATUSES: Final = frozenset({401, 403})
_MCP_REQUEST_TIMEOUT: Final = -32001

__all__ = [
    "MAX_UPSTREAM_RESULT_BYTES",
    "McpUpstream",
    "UpstreamAuthError",
    "UpstreamError",
    "UpstreamResult",
    "UpstreamTimeout",
    "UpstreamTool",
    "UpstreamTooLarge",
    "UpstreamUnavailable",
    "open_http_upstream",
]


class UpstreamError(Exception):
    """A failed upstream exchange. The message is safe to record and deliver (fixed text)."""


class UpstreamAuthError(UpstreamError):
    """The server refused the credential (HTTP 401/403)."""


class UpstreamTimeout(UpstreamError):
    """No answer within the deadline. For a Tier S/X call the outcome is unknown."""


class UpstreamTooLarge(UpstreamError):
    """The result exceeded `max_result_bytes`; it is dropped, never delivered."""


class UpstreamUnavailable(UpstreamError):
    """Transport, HTTP, or protocol failure other than auth or timeout."""


@dataclass(frozen=True, slots=True)
class UpstreamTool:
    """One tool the server listed, with its input schema as the server declared it."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UpstreamResult:
    """A `tools/call` result in the MCP `CallToolResult` JSON shape (camelCase keys), plus the
    size of that JSON. `isError` results are returned, not raised: the validator decides."""

    response: dict[str, JsonValue]
    size_bytes: int


class McpUpstream(Protocol):
    """An open connection to one remote MCP server."""

    @property
    def server(self) -> str: ...

    @property
    def tools(self) -> tuple[UpstreamTool, ...]: ...

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_seconds: float
    ) -> UpstreamResult: ...


class _AuthCounter:
    """Counts 401/403 responses seen by the HTTP client (see module docstring)."""

    def __init__(self) -> None:
        self.count = 0

    async def __call__(self, response: httpx2.Response) -> None:
        if response.status_code in _AUTH_STATUSES:
            self.count += 1


def _classify(exc: BaseException, auth_failed: bool, what: str) -> UpstreamError:
    if auth_failed:
        return UpstreamAuthError(f"{what}: the server refused the credential")
    if isinstance(exc, TimeoutError):
        return UpstreamTimeout(f"{what}: no answer before the deadline")
    if isinstance(exc, MCPError):
        if exc.error.code == _MCP_REQUEST_TIMEOUT:
            return UpstreamTimeout(f"{what}: no answer before the deadline")
        return UpstreamUnavailable(f"{what}: MCP error {exc.error.code}")
    if isinstance(exc, httpx2.HTTPStatusError):
        return UpstreamUnavailable(f"{what}: HTTP {exc.response.status_code}")
    return UpstreamUnavailable(f"{what}: {type(exc).__name__}")


class _HttpUpstream:
    def __init__(
        self,
        server: str,
        client: Client,
        auth: _AuthCounter,
        tools: tuple[UpstreamTool, ...],
        max_result_bytes: int,
    ) -> None:
        self._server = server
        self._client = client
        self._auth = auth
        self._tools = tools
        self._max = max_result_bytes

    @property
    def server(self) -> str:
        return self._server

    @property
    def tools(self) -> tuple[UpstreamTool, ...]:
        return self._tools

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_seconds: float
    ) -> UpstreamResult:
        if timeout_seconds <= 0:
            raise UpstreamTimeout(f"{name}: no time left for the call")
        before = self._auth.count
        try:
            with anyio.fail_after(timeout_seconds):
                result = await self._client.call_tool(
                    name, dict(arguments), read_timeout_seconds=timeout_seconds
                )
        except Exception as exc:  # noqa: BLE001 - mapped to a typed, text-free error
            raise _classify(exc, self._auth.count > before, name) from None
        if self._auth.count > before:
            raise UpstreamAuthError(f"{name}: the server refused the credential")
        response: dict[str, JsonValue] = result.model_dump(
            by_alias=True,
            mode="json",
            exclude_none=True,
            include={"content", "is_error", "structured_content"},
        )
        size = len(json.dumps(response, separators=(",", ":")).encode())
        if size > self._max:
            raise UpstreamTooLarge(f"{name}: result of {size} bytes exceeds {self._max}")
        return UpstreamResult(response=response, size_bytes=size)


async def _list_tools(client: Client) -> tuple[UpstreamTool, ...]:
    tools: list[UpstreamTool] = []
    cursor: str | None = None
    for _ in range(MAX_TOOL_LIST_PAGES):
        page = await client.list_tools(cursor=cursor) if cursor else await client.list_tools()
        tools.extend(UpstreamTool(t.name, t.description, dict(t.input_schema)) for t in page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tuple(tools)
    raise UpstreamUnavailable("tools/list: too many pages")


@asynccontextmanager
async def open_http_upstream(
    server: McpHttpServer,
    *,
    connect_timeout_seconds: float,
    max_result_bytes: int = MAX_UPSTREAM_RESULT_BYTES,
) -> AsyncIterator[McpUpstream]:
    """Connect (initialize handshake + full `tools/list`) within the connect deadline.

    Requires a bearer token: a server that relies on the CLI's stored login (ADR-0018) has no
    credential our code can present, so it cannot be proxied (ValueError). Raises
    `UpstreamError` if the connection or listing fails; the connection closes on exit.

    The deadline is applied through the HTTP client and the session's request timeout, not an
    enclosing cancel scope: the MCP client holds a task group that outlives this function's
    setup, and anyio requires cancel scopes to exit in LIFO order.
    """
    if server.token is None:
        raise ValueError(f"{server.name}: the proxy needs a bearer token")
    auth = _AuthCounter()
    http = httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {server.token.get_secret_value()}"},
        timeout=httpx2.Timeout(connect_timeout_seconds, read=None),
        event_hooks={"response": [auth]},
    )
    client = Client(
        # Robinhood answers the session DELETE with 400 (observed 2026-09-27); the session
        # ends with the connection anyway, so no termination request is sent.
        streamable_http_client(str(server.url), http_client=http, terminate_on_close=False),
        mode="legacy",
        cache=None,
        read_timeout_seconds=connect_timeout_seconds,
    )
    async with http, AsyncExitStack() as stack:
        try:
            await stack.enter_async_context(client)
            tools = await _list_tools(client)
        except UpstreamError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a typed, text-free error
            raise _classify(exc, auth.count > 0, f"{server.name} connect") from None
        yield _HttpUpstream(server.name, client, auth, tools, max_result_bytes)
