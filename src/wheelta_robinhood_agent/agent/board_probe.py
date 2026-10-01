"""Trusted read of the Wheelta board status for the run context (ADR-0062).

The orchestrator agents (Buy-to-Close, Sell Options) may not call `wheelta_board_status`: it
is a market-Mignon tool (ADR-0025). Before the prompt is rendered, trusted code reads it
through the same upstream path the session's validating proxy uses (the run's proxied
Wheelta server and the session's upstream factory, `integrations/mcp_upstream.py`), like
`agent/start_probe.py` reads Robinhood state before the model connects. The model never
sees the payload; only the derived values below reach the prompt and the run's metadata.

Rendered value (compact JSON, `BoardStatusContext.prompt_value`):

- `{"status":"ready","build_id":...,"as_of":...,"next_refresh_at":...}` from a result that
  parses as `integrations.wheelta.schemas.BoardStatus` with `buildState` "ready" (always so
  on a 200, MCP.yaml) and a `buildId`. `next_refresh_at` may be null (absent upstream).
- `{"status":"building"}` for Wheelta's normal `board_building` state (CLAUDE.md §10).
- `{"status":"unavailable","reason":...}` for everything else: no Mignons this run, Wheelta
  withheld or not proxied, the board tool not allowed or not listed, expected Wheelta tools
  missing, a failed connect or call, another Wheelta error, or a result that fails its schema
  (fail closed; nothing is guessed).

The value is context only: a Mignon that queries the board still reads the status and the
rows' build provenance itself. Reasons are built from fixed strings plus a Wheelta error code
or an exception type; no remote text is rendered.
"""

import json
import re
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from enum import StrEnum
from typing import Final

import anyio
from pydantic import BaseModel, ConfigDict, ValidationError

from wheelta_robinhood_agent.agent.mignons import DELEGATION_TOOL, Role, role_tools
from wheelta_robinhood_agent.agent.result_boundary import (
    PayloadError,
    PayloadKind,
    extract_mcp_payload,
)
from wheelta_robinhood_agent.integrations.mcp_upstream import (
    McpUpstream,
    UpstreamError,
    open_http_upstream,
)
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, diff_discovered
from wheelta_robinhood_agent.integrations.status import McpHttpServer
from wheelta_robinhood_agent.integrations.wheelta.registry import SERVER_NAME as WHEELTA
from wheelta_robinhood_agent.integrations.wheelta.schemas import (
    BoardBuilding,
    BoardStatus,
    parse_tool_error,
)

BOARD_STATUS_TOOL: Final = "wheelta_board_status"
QUALIFIED_BOARD_STATUS_TOOL: Final = f"mcp__{WHEELTA}__{BOARD_STATUS_TOOL}"
# Build IDs observed are short hex (`df52f70ac584`); anything else is not rendered.
_BUILD_ID_RE: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_REQUEST_ID_RE: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

UpstreamOpener = Callable[[McpHttpServer, float], AbstractAsyncContextManager[McpUpstream]]


class BoardState(StrEnum):
    READY = "ready"
    BUILDING = "building"
    UNAVAILABLE = "unavailable"


class BoardStatusContext(BaseModel):
    """What the trusted read established (module docstring). Ledger-only fields
    (`retry_after_seconds`, `request_id`, `retrieved_at`) are not rendered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: BoardState
    build_id: str | None = None
    as_of: datetime | None = None
    next_refresh_at: datetime | None = None
    reason: str | None = None
    retry_after_seconds: int | None = None
    request_id: str | None = None
    retrieved_at: datetime | None = None

    def prompt_value(self) -> str:
        """The compact JSON rendered as `{{board_status}}`."""
        value: dict[str, object]
        if self.status is BoardState.READY:
            value = {
                "status": self.status.value,
                "build_id": self.build_id,
                "as_of": self.as_of.isoformat() if self.as_of else None,
                "next_refresh_at": (
                    self.next_refresh_at.isoformat() if self.next_refresh_at else None
                ),
            }
        elif self.status is BoardState.BUILDING:
            value = {"status": self.status.value}
        else:
            value = {"status": self.status.value, "reason": self.reason}
        return json.dumps(value, separators=(",", ":"))

    def event_payload(self) -> dict[str, object]:
        """The run-metadata form (no payload, no remote text)."""
        return {"tool": BOARD_STATUS_TOOL, **self.model_dump(mode="json")}


def unavailable(reason: str, retrieved_at: datetime | None = None) -> BoardStatusContext:
    return BoardStatusContext(
        status=BoardState.UNAVAILABLE, reason=reason, retrieved_at=retrieved_at
    )


def skip_reason(
    *,
    allowed_tools: frozenset[str],
    wheelta_proxied: bool,
    wheelta_withheld: str | None,
) -> str | None:
    """Why no board read is made this run, or None if it is (module docstring)."""
    if DELEGATION_TOOL not in allowed_tools:
        return "no Mignons this run, so no board work"
    if wheelta_withheld is not None:
        return f"Wheelta withheld: {wheelta_withheld}"
    if not wheelta_proxied:
        return "Wheelta is not served through the validating proxy this run"
    if QUALIFIED_BOARD_STATUS_TOOL not in allowed_tools & role_tools(Role.MARKET):
        return f"{BOARD_STATUS_TOOL} is not allowed this run"
    return None


def parse_board_status(response: object, retrieved_at: datetime) -> BoardStatusContext:
    """A `wheelta_board_status` MCP response as context (module docstring). Pure; never
    raises."""
    try:
        kind, payload = extract_mcp_payload(response)
    except PayloadError:
        return unavailable(f"{BOARD_STATUS_TOOL}: unreadable result", retrieved_at)
    if kind is PayloadKind.TOOL_ERROR:
        error = parse_tool_error(payload if isinstance(payload, str) else "")
        request_id = error.request_id
        if request_id is not None and not _REQUEST_ID_RE.fullmatch(request_id):
            request_id = None
        if isinstance(error, BoardBuilding):
            return BoardStatusContext(
                status=BoardState.BUILDING,
                retry_after_seconds=error.retry_after_seconds,
                request_id=request_id,
                retrieved_at=retrieved_at,
            )
        return BoardStatusContext(
            status=BoardState.UNAVAILABLE,
            reason=f"{BOARD_STATUS_TOOL}: Wheelta error {error.code.value}",
            retry_after_seconds=error.retry_after_seconds,
            request_id=request_id,
            retrieved_at=retrieved_at,
        )
    if not isinstance(payload, dict):
        return unavailable(f"{BOARD_STATUS_TOOL}: result is not an object", retrieved_at)
    try:
        status = BoardStatus.model_validate(payload)
    except ValidationError:
        return unavailable(f"{BOARD_STATUS_TOOL}: result failed its schema", retrieved_at)
    if status.build_state != "ready":
        return unavailable(f"{BOARD_STATUS_TOOL}: buildState is not ready", retrieved_at)
    if status.build_id is None or not _BUILD_ID_RE.fullmatch(status.build_id):
        return unavailable(f"{BOARD_STATUS_TOOL}: no usable buildId", retrieved_at)
    return BoardStatusContext(
        status=BoardState.READY,
        build_id=status.build_id,
        as_of=status.as_of,
        next_refresh_at=status.next_refresh_at,
        retrieved_at=retrieved_at,
    )


def _default_opener(
    server: McpHttpServer, connect_timeout_seconds: float
) -> AbstractAsyncContextManager[McpUpstream]:
    return open_http_upstream(server, connect_timeout_seconds=connect_timeout_seconds)


async def probe_board_status(
    server: McpHttpServer,
    registry: ToolRegistry,
    *,
    opener: UpstreamOpener | None,
    connect_timeout_seconds: float,
    tool_timeout_seconds: float,
    clock: Callable[[], datetime],
) -> BoardStatusContext:
    """Connect, verify the tool list, read the board status, close. Never raises for a
    failed read (the context is then UNAVAILABLE with the failure named). No retries."""
    if connect_timeout_seconds <= 0 or tool_timeout_seconds <= 0:
        return unavailable("no time left in the run budget for the board read", clock())
    open_upstream = opener or _default_opener
    try:
        async with open_upstream(server, connect_timeout_seconds) as upstream:
            names = [t.name for t in upstream.tools]
            if BOARD_STATUS_TOOL not in names:
                return unavailable(f"{BOARD_STATUS_TOOL} is not listed by Wheelta", clock())
            if not diff_discovered(registry, names).ok:
                return unavailable("expected Wheelta tools are missing", clock())
            result = await upstream.call_tool(
                BOARD_STATUS_TOOL, {}, timeout_seconds=tool_timeout_seconds
            )
            return parse_board_status(result.response, clock())
    except (UpstreamError, ValueError, TypeError) as exc:
        return unavailable(f"{BOARD_STATUS_TOOL}: {type(exc).__name__}", clock())


def read_board_status(
    server: McpHttpServer,
    registry: ToolRegistry,
    *,
    opener: UpstreamOpener | None,
    connect_timeout_seconds: float,
    tool_timeout_seconds: float,
    clock: Callable[[], datetime],
) -> BoardStatusContext:
    """`probe_board_status` on a fresh event loop (the orchestrator is synchronous)."""

    async def run() -> BoardStatusContext:
        return await probe_board_status(
            server,
            registry,
            opener=opener,
            connect_timeout_seconds=connect_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
            clock=clock,
        )

    return anyio.run(run)


__all__ = [
    "BOARD_STATUS_TOOL",
    "BoardState",
    "BoardStatusContext",
    "QUALIFIED_BOARD_STATUS_TOOL",
    "parse_board_status",
    "probe_board_status",
    "read_board_status",
    "skip_reason",
    "unavailable",
]
