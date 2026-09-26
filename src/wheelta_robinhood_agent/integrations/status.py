"""MCP server-level types shared by the Robinhood and Wheelta integrations (CLAUDE.md §8).

- `map_sdk_status`: Agent SDK server status string → `SourceStatus`, failing closed.
- `SourceObservation`: one server's status at a point in time, plus the discovery diff
  (INTERFACES.md `source_status`).
- `McpHttpServer`: a remote HTTP MCP server config whose bearer token never appears in
  repr, str, or validation errors.

Pure: no network, no clock (callers pass `observed_at`). The Agent SDK is not installed
yet, so everything here is string-based. **Unverified:** the SDK's status field names and
values (assumed: an init/`get_mcp_status` entry per server with `name` and `status`, where
`status` is one of "connected", "pending", "needs-auth", "failed", "disabled"), and the
exact server-config dict shape (taken from `.mcp.json.example`). Confirm against the pinned
`claude-agent-sdk` when it is added.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    AnyHttpUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    SecretStr,
    field_validator,
    model_validator,
)

from wheelta_robinhood_agent.domain.enums import SourceStatus
from wheelta_robinhood_agent.integrations.registry import (
    DiscoveryDiff,
    ToolRegistry,
    diff_discovered,
)

# Exact, case-sensitive SDK strings (unverified, see module docstring).
_SDK_STATUS: dict[str, SourceStatus] = {
    "connected": SourceStatus.CONNECTED,
    "pending": SourceStatus.PENDING,
    "needs-auth": SourceStatus.NEEDS_AUTH,
    "failed": SourceStatus.FAILED,
    "disabled": SourceStatus.DISABLED,
}

_MAX_RAW_STATUS_LEN = 64


def map_sdk_status(raw: object) -> SourceStatus:
    """Map an SDK server status to `SourceStatus`. Anything unrecognised is FAILED.

    CLAUDE.md §2.7 (fail closed): an unknown, non-string, or differently cased value never
    reads as connected.
    """
    if isinstance(raw, str):
        return _SDK_STATUS.get(raw, SourceStatus.FAILED)
    return SourceStatus.FAILED


class SourceObservation(BaseModel):
    """One MCP server's observed state (INTERFACES.md `source_status`, CLAUDE.md §8).

    `discovery` is set only for a CONNECTED server: the init tool list is incomplete while
    pending, and meaningless otherwise. A connected server without a diff is not yet
    verified, so it is not `available`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    server: str
    status: SourceStatus
    observed_at: AwareDatetime
    raw_status: str | None = None  # the SDK string as received, truncated; None if not a str
    discovery: DiscoveryDiff | None = None

    @field_validator("observed_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _discovery_only_when_connected(self) -> "SourceObservation":
        if self.discovery is not None and self.status is not SourceStatus.CONNECTED:
            raise ValueError("a discovery diff is only valid for a connected server")
        return self

    @property
    def available(self) -> bool:
        """Connected and verified: no expected tool missing (unknown tools are denied)."""
        return (
            self.status is SourceStatus.CONNECTED
            and self.discovery is not None
            and self.discovery.ok
        )


def observe_server(
    server: str,
    raw_status: object,
    observed_at: datetime,
    *,
    registry: ToolRegistry | None = None,
    discovered_tools: Iterable[str] | None = None,
) -> SourceObservation:
    """Build an observation from an SDK status and, once connected, the init tool list.

    The discovery diff is computed only when the status maps to CONNECTED and both the
    registry and the discovered (unqualified) tool names are given.
    """
    status = map_sdk_status(raw_status)
    discovery = None
    if status is SourceStatus.CONNECTED and registry is not None and discovered_tools is not None:
        discovery = diff_discovered(registry, discovered_tools)
    return SourceObservation(
        server=server,
        status=status,
        observed_at=observed_at,
        raw_status=raw_status[:_MAX_RAW_STATUS_LEN] if isinstance(raw_status, str) else None,
        discovery=discovery,
    )


class McpHttpServer(BaseModel):
    """A remote MCP server over streamable HTTP with a bearer token (CLAUDE.md §7, §8).

    `repr`/`str` show `SecretStr('**********')`, and validation errors never echo input.
    `to_sdk_config()` returns the only form containing the token: pass it straight to the
    SDK's `mcp_servers` and never log or persist it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str
    url: AnyHttpUrl
    token: SecretStr

    @field_validator("token")
    @classmethod
    def _header_safe(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if not secret or any(c.isspace() or not c.isprintable() for c in secret):
            raise ValueError("bearer token must be non-empty printable text without whitespace")
        return value

    def to_sdk_config(self) -> dict[str, object]:
        """`{"type": "http", "url": ..., "headers": {"Authorization": "Bearer <token>"}}`."""
        transport: Literal["http"] = "http"
        return {
            "type": transport,
            "url": str(self.url),
            "headers": {"Authorization": f"Bearer {self.token.get_secret_value()}"},
        }
