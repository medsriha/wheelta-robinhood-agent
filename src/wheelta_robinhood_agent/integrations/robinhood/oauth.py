"""Robinhood OAuth refresh for headless runs (ADR-0021).

Facts verified by `scripts/robinhood_oauth_probe.py` on 2026-09-26:

- Discovery (`DISCOVERY_URL`) names `TOKEN_ENDPOINT`, supports the `authorization_code` and
  `refresh_token` grants, and registers public clients (`token_endpoint_auth_method: none`).
- A refresh is a form POST `{grant_type: refresh_token, refresh_token, client_id}`.
- Refresh tokens rotate: every refresh returns a new refresh token. The old one must be
  treated as dead and never presented again (reuse may revoke the whole token family).

**No retry, ever.** A refresh consumes the refresh token it presents. If the outcome is
ambiguous (timeout, dropped connection, unparseable success body), the server may already
have rotated the token, and a retry would present a dead token and could revoke the family.
So every failure is terminal for the run; the operator re-seeds if needed (OPERATIONS.md).

Errors carry only the HTTP status and a sanitized OAuth `error` code, never the response
body or any token. All I/O goes through an injected `httpx.Client`; no clock (callers pass
`obtained_at`).
"""

import hashlib
from datetime import datetime, timedelta

import httpx
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)

DISCOVERY_URL = "https://agent.robinhood.com/.well-known/oauth-authorization-server/mcp/trading"
# Pinned from discovery (probe, 2026-09-26). `discover_token_endpoint` re-verifies it before
# a refresh; a mismatch fails closed rather than sending a refresh token somewhere new.
TOKEN_ENDPOINT = "https://api.robinhood.com/oauth2/token/"  # noqa: S105 - a URL, not a secret

# RFC 6749 §5.2 and RFC 6750 §3.1 error codes. Anything else is reported as "unrecognized",
# so an error field can never carry arbitrary server text (or an echoed token) into a message.
_OAUTH_ERROR_CODES = frozenset(
    {
        "invalid_request",
        "invalid_client",
        "invalid_grant",
        "unauthorized_client",
        "unsupported_grant_type",
        "invalid_scope",
        "access_denied",
        "server_error",
        "temporarily_unavailable",
        "invalid_token",
        "insufficient_scope",
    }
)


class OAuthError(Exception):
    """Base class. Messages never contain a token, a response body, or a client secret."""


class OAuthDiscoveryFailed(OAuthError):
    """Discovery was unreachable, malformed, or disagrees with the pinned token endpoint."""


class OAuthRefreshFailed(OAuthError):
    """The token endpoint rejected the refresh (HTTP status plus OAuth `error` code only)."""

    def __init__(self, status_code: int, error_code: str) -> None:
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"refresh rejected: HTTP {status_code}, error={error_code}")


class OAuthRefreshOutcomeUnknown(OAuthError):
    """The refresh may or may not have been applied (transport error or unusable success
    body). The presented refresh token may already be rotated: never retry it."""


class TokenPair(BaseModel):
    """An access/refresh token pair from one grant. Tokens never appear in repr or errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    access_token: SecretStr
    refresh_token: SecretStr
    expires_in: int = Field(gt=0)
    obtained_at: AwareDatetime

    @field_validator("access_token", "refresh_token")
    @classmethod
    def _header_safe(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if not secret or any(c.isspace() or not c.isprintable() for c in secret):
            raise ValueError("token must be non-empty printable text without whitespace")
        return value

    @property
    def access_expires_at(self) -> datetime:
        return self.obtained_at + timedelta(seconds=self.expires_in)


class _TokenResponse(BaseModel):
    """The fields of a token response we rely on; everything else is ignored, not stored."""

    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    access_token: SecretStr
    refresh_token: SecretStr
    expires_in: int = Field(gt=0)
    token_type: str | None = None

    @field_validator("token_type")
    @classmethod
    def _bearer(cls, value: str | None) -> str | None:
        if value is not None and value.lower() != "bearer":
            raise ValueError("token_type must be Bearer")
        return value


def fingerprint(secret: SecretStr) -> str:
    """A short, non-reversible label to tell two tokens apart without revealing either."""
    return hashlib.sha256(secret.get_secret_value().encode()).hexdigest()[:8]


def _error_code(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return "unparseable"
    code = body.get("error") if isinstance(body, dict) else None
    if isinstance(code, str) and code in _OAUTH_ERROR_CODES:
        return code
    return "unrecognized"


def discover_token_endpoint(client: httpx.Client, discovery_url: str = DISCOVERY_URL) -> str:
    """Fetch discovery and return the token endpoint if it matches `TOKEN_ENDPOINT` and
    supports the refresh grant. Discovery is an idempotent read, but it is fetched once only:
    it runs just before a refresh, when the run budget matters more than one more attempt."""
    try:
        response = client.get(discovery_url)
    except httpx.HTTPError as exc:
        raise OAuthDiscoveryFailed(f"discovery unreachable ({type(exc).__name__})") from None
    if response.status_code != httpx.codes.OK:
        raise OAuthDiscoveryFailed(f"discovery returned HTTP {response.status_code}")
    try:
        meta = response.json()
    except ValueError:
        raise OAuthDiscoveryFailed("discovery body is not JSON") from None
    if not isinstance(meta, dict):
        raise OAuthDiscoveryFailed("discovery body is not an object")
    if meta.get("token_endpoint") != TOKEN_ENDPOINT:
        raise OAuthDiscoveryFailed("discovery token_endpoint differs from the pinned endpoint")
    grants = meta.get("grant_types_supported")
    if not isinstance(grants, list) or "refresh_token" not in grants:
        raise OAuthDiscoveryFailed("discovery does not list the refresh_token grant")
    return TOKEN_ENDPOINT


def refresh_access_token(
    client: httpx.Client,
    token_endpoint: str,
    client_id: str,
    refresh_token: SecretStr,
    *,
    obtained_at: datetime,
) -> TokenPair:
    """Exchange `refresh_token` for a new pair. Called at most once per credential: the
    presented refresh token is consumed whatever happens (module docstring, no retry).

    A success response without a new refresh token is rejected as OAuthRefreshOutcomeUnknown:
    rotation is the verified behavior, and the presented token must be assumed dead.
    """
    try:
        response = client.post(
            token_endpoint,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token.get_secret_value(),
                "client_id": client_id,
            },
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise OAuthRefreshOutcomeUnknown(
            f"refresh transport error ({type(exc).__name__})"
        ) from None
    if response.status_code >= 400:
        raise OAuthRefreshFailed(response.status_code, _error_code(response))
    if response.status_code != httpx.codes.OK:
        raise OAuthRefreshOutcomeUnknown(f"refresh returned HTTP {response.status_code}")
    try:
        parsed = _TokenResponse.model_validate_json(response.content)
    except ValidationError as exc:
        fields = sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]})
        raise OAuthRefreshOutcomeUnknown(
            f"refresh success body invalid (fields: {', '.join(fields) or 'body'})"
        ) from None
    if parsed.refresh_token.get_secret_value() == refresh_token.get_secret_value():
        raise OAuthRefreshOutcomeUnknown("refresh did not rotate the refresh token")
    try:
        return TokenPair(
            access_token=parsed.access_token,
            refresh_token=parsed.refresh_token,
            expires_in=parsed.expires_in,
            obtained_at=obtained_at,
        )
    except ValidationError:
        raise OAuthRefreshOutcomeUnknown(
            "refresh returned tokens that are not header-safe"
        ) from None
