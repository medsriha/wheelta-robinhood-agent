"""Typed parsing of Wheelta MCP results we rely on (CLAUDE.md §10, §12; docs/DATA_QUALITY.md).

Authority: the wheelta-mcp source (read-only, `/Users/deepset/wheelta-mcp`):

- `src/wheelta_mcp/projections.py` `BoardFreshness` and `src/wheelta_mcp/tools/screener.py`
  `BoardStatus` / `BoardQueryResult`: camelCase on the wire (`models.Out`).
- `MCP.yaml` §3: `asOf` is build or 5-minute quote-merge time; `buildId` is NOT advanced by
  the quote merge, so rebuilds are detected by `buildId`; `nextRefreshAt` is the poll target.
- `MCP.yaml` §7 and `src/wheelta_mcp/errors.py`: upstream envelope
  `{"error": {"code", "message", "requestId"}}` with `Retry-After` (bounded 1..3600 s,
  `client._parse_retry_after`). The MCP tool does NOT forward that envelope: it returns
  `isError: true` with prose from `errors.describe()` and a trailing `(requestId: …)`.
  `parse_tool_error` therefore classifies on `describe()`'s fixed opening sentences. Text
  that matches none is `UNRECOGNIZED` (fail closed); a wording change upstream shows up as
  schema drift, never as success.
- `board_building` (503) is a normal daily state (`MCP.yaml` §3). It parses to
  `BoardBuilding`, not an error. Never fall back to an older board (CLAUDE.md §10).

Board option values (strike, bid, Greeks, probOtm, ...) are **build-time screens**, never
quotes: every contract is re-priced from Robinhood before it can be a candidate
(`data_quality.precedence`). Floats in screen rows are converted to `Decimal` via their
shortest repr and are held in fields named `screen_*`. Parsing is pure; models ignore
unknown keys and validate only what we rely on.
"""

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
)
from pydantic.alias_generators import to_camel


class _External(BaseModel):
    """External payload: camelCase aliases, unknown keys ignored, frozen."""

    model_config = ConfigDict(
        extra="ignore", frozen=True, alias_generator=to_camel, populate_by_name=True
    )


def _to_utc(value: datetime | None) -> datetime | None:
    return None if value is None else value.astimezone(UTC)


class BoardFreshness(_External):
    """Build stamps on every board result (projections.py `BoardFreshness`).

    `as_of` is required and must be a timezone-aware ISO-8601 timestamp. `wheelta_board_row`
    can fall back to `asOf=""` when the board is unavailable; that fails validation and the
    result is treated as missing. `build_id` may be absent upstream; then a rebuild can't be
    detected and it stays None (CLAUDE.md §2.3).
    """

    as_of: AwareDatetime
    build_id: str | None = Field(default=None, min_length=1)
    build_state: str | None = None
    next_refresh_at: AwareDatetime | None = None
    scores_updated_at: AwareDatetime | None = None

    _utc = field_validator("as_of", "next_refresh_at", "scores_updated_at")(_to_utc)


class BoardStatus(_External):
    """`wheelta_board_status` result (screener.py `BoardStatus`). Funnel/score range ignored."""

    as_of: AwareDatetime
    build_id: str | None = Field(default=None, min_length=1)
    build_state: str | None = None
    universe_rows: NonNegativeInt
    distinct_symbols: NonNegativeInt
    next_refresh_at: AwareDatetime | None = None
    sectors: tuple[str, ...] = ()

    _utc = field_validator("as_of", "next_refresh_at")(_to_utc)

    @property
    def freshness(self) -> BoardFreshness:
        return BoardFreshness(
            as_of=self.as_of,
            build_id=self.build_id,
            build_state=self.build_state,
            next_refresh_at=self.next_refresh_at,
        )


def _screen_value(value: object) -> object:
    """Floats become Decimal via shortest repr; nested lists/dicts converted recursively."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, list):
        return [_screen_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _screen_value(v) for k, v in value.items()}
    raise ValueError(f"unexpected JSON value type {type(value).__name__}")


class BoardQueryResult(_External):
    """`wheelta_board_query` result (screener.py `BoardQueryResult`).

    `screen_rows` (wire key `rows`) are build-time screen values keyed by dotted column
    path; never a price, premium, or Greek source for a decision (CLAUDE.md §10).
    """

    mode: Literal["rows", "groups"]
    matched: NonNegativeInt
    universe_rows: NonNegativeInt
    freshness: BoardFreshness
    screen_rows: tuple[dict[str, object], ...] | None = Field(default=None, alias="rows")
    returned: NonNegativeInt | None = None

    @field_validator("screen_rows", mode="before")
    @classmethod
    def _decimalize(cls, value: object) -> object:
        if isinstance(value, list):
            return [_screen_value(row) for row in value]
        return value


class BoardBuilding(_External):
    """The board is rebuilding (503 `board_building`): a normal state, not an error.

    Skip Wheelta board work this run; never use an older board (CLAUDE.md §10).
    """

    state: Literal["board_building"] = "board_building"
    retry_after_seconds: int | None = Field(default=None, ge=0, le=3600)
    request_id: str | None = None


class WheeltaErrorCode(StrEnum):
    """Upstream error codes (MCP.yaml §7, errors.py `describe`). UNRECOGNIZED is ours."""

    BOARD_BUILDING = "board_building"
    RATE_LIMITED = "rate_limited"
    SYMBOL_NOT_FOUND = "symbol_not_found"
    SERVICE_SCOPE_FORBIDDEN = "service_scope_forbidden"
    UNAUTHORIZED = "unauthorized"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    HISTORY_UNAVAILABLE = "history_unavailable"
    CONTRACTS_UNPRICED = "contracts_unpriced"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    INVALID_ORIGIN = "invalid_origin"
    SUBSCRIPTION_REQUIRED = "subscription_required"
    REQUEST_TOO_LARGE = "request_too_large"
    UNKNOWN_ERROR = "unknown_error"  # client.py default when the envelope has no code
    UNRECOGNIZED = "unrecognized"  # local: a code or tool-error text we don't know


_CODES = {c.value: c for c in WheeltaErrorCode if c is not WheeltaErrorCode.UNRECOGNIZED}


def _code(raw: str) -> WheeltaErrorCode:
    return _CODES.get(raw, WheeltaErrorCode.UNRECOGNIZED)


class WheeltaError(_External):
    """A Wheelta failure the agent sees. `retry_after_seconds` must be honored exactly (§14)."""

    code: WheeltaErrorCode
    message: str
    request_id: str | None = None
    retry_after_seconds: int | None = Field(default=None, ge=0, le=3600)

    @property
    def retryable(self) -> bool:
        """Transient per MCP.yaml §7 (429, 503). A Tier R read may retry within budget."""
        return self.code in (
            WheeltaErrorCode.RATE_LIMITED,
            WheeltaErrorCode.UPSTREAM_UNAVAILABLE,
            WheeltaErrorCode.CONTRACTS_UNPRICED,
        )


class _EnvelopeBody(_External):
    code: str = Field(min_length=1)
    message: str
    request_id: str | None = None


class _Envelope(_External):
    error: _EnvelopeBody


def parse_error_envelope(
    body: Mapping[str, object], retry_after_seconds: int | None = None
) -> BoardBuilding | WheeltaError:
    """Parse the upstream `{"error": {...}}` envelope (MCP.yaml §7) plus `Retry-After`."""
    err = _Envelope.model_validate(body).error
    code = _code(err.code)
    if code is WheeltaErrorCode.BOARD_BUILDING:
        return BoardBuilding(retry_after_seconds=retry_after_seconds, request_id=err.request_id)
    return WheeltaError(
        code=code,
        message=err.message,
        request_id=err.request_id,
        retry_after_seconds=retry_after_seconds,
    )


# Opening sentences of errors.py `describe()`, one per code.
_TOOL_ERROR_PREFIXES: tuple[tuple[str, WheeltaErrorCode], ...] = (
    ("The screener board is being rebuilt", WheeltaErrorCode.BOARD_BUILDING),
    ("Rate limited by the Wheelta API", WheeltaErrorCode.RATE_LIMITED),
    ("No such symbol upstream", WheeltaErrorCode.SYMBOL_NOT_FOUND),
    ("This API key is not authorized for", WheeltaErrorCode.SERVICE_SCOPE_FORBIDDEN),
    ("The Wheelta API rejected this server's credentials", WheeltaErrorCode.UNAUTHORIZED),
    (
        "A data provider behind the Wheelta API is unavailable",
        WheeltaErrorCode.UPSTREAM_UNAVAILABLE,
    ),  # noqa: E501
    ("Not enough price history resolved", WheeltaErrorCode.HISTORY_UNAVAILABLE),
    ("Neither the board nor the held-contract store", WheeltaErrorCode.CONTRACTS_UNPRICED),
    ("The Wheelta API rejected these inputs", WheeltaErrorCode.VALIDATION_ERROR),
    ("Not found upstream", WheeltaErrorCode.NOT_FOUND),
)
_FALLBACK_RE = re.compile(r"^Wheelta API error \((\d{3}) ([a-z_]+)\):")
_RETRY_RE = re.compile(r"Retry after (\d{1,4})s\.")
_REQUEST_ID_RE = re.compile(r"\(requestId: ([A-Za-z0-9._-]{1,64})\)\s*$")


def parse_tool_error(text: str) -> BoardBuilding | WheeltaError:
    """Classify an `isError` Wheelta tool result's text (errors.py `to_tool_result`).

    Board-building parses to `BoardBuilding`. Unmatched text (including local query errors
    such as an unknown column) is `UNRECOGNIZED`. `Retry-After` and `requestId` are
    extracted from the fixed suffixes `describe()` appends.
    """
    stripped = text.strip()
    code = WheeltaErrorCode.UNRECOGNIZED
    for prefix, candidate in _TOOL_ERROR_PREFIXES:
        if stripped.startswith(prefix):
            code = candidate
            break
    else:
        fallback = _FALLBACK_RE.match(stripped)
        if fallback:
            code = _code(fallback.group(2))
    retry = _RETRY_RE.search(stripped)
    retry_after = int(retry.group(1)) if retry else None
    if retry_after is not None and retry_after > 3600:
        retry_after = None  # outside the documented bound: not trusted
    request = _REQUEST_ID_RE.search(stripped)
    request_id = request.group(1) if request else None
    if code is WheeltaErrorCode.BOARD_BUILDING:
        return BoardBuilding(retry_after_seconds=retry_after, request_id=request_id)
    return WheeltaError(
        code=code, message=stripped, request_id=request_id, retry_after_seconds=retry_after
    )
