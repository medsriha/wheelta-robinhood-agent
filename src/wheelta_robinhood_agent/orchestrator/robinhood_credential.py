"""Resolve the Robinhood access token for a headless run (ADR-0021, auth mode refresh_token).

Runs while the single-flight lock is held, after preflight and before the Robinhood server is
built:

1. Load the latest `oauth_credentials` row for (environment, robinhood). None → MISSING.
2. Decrypt it with ROBINHOOD_TOKEN_ENCRYPTION_KEY. Failure → UNDECRYPTABLE.
3. If the access token expires more than `REFRESH_MARGIN` from now → STORED (use it as-is).
4. Otherwise refresh once (never retried; integrations/robinhood/oauth.py) → REFRESH_FAILED
   on any error. The presented refresh token is dead from here on, whatever happened.
5. Persist the rotated pair (source=refresh, superseding the consumed row) BEFORE using it.
   If that fails → PERSIST_FAILED: the only valid refresh token may exist nowhere but this
   process's memory, so the run fails loudly (R17) and the token is not used.

Every non-success means Robinhood is `needs-auth` for this run. The resolution carries only
non-secret facts for the ledger, plus the tokens (hidden from repr) for the server config and
the run's redactor. Tokens are never logged.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol

import httpx
import psycopg
from pydantic import SecretStr

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.integrations.robinhood.oauth import (
    OAuthError,
    TokenPair,
    discover_token_endpoint,
    refresh_access_token,
)
from wheelta_robinhood_agent.integrations.robinhood.token_vault import TokenVault, TokenVaultError
from wheelta_robinhood_agent.ledger import oauth_credentials as ledger_credentials
from wheelta_robinhood_agent.ledger.oauth_credentials import (
    CredentialSource,
    OAuthProvider,
    StoredCredential,
)

Conn = psycopg.Connection[tuple[object, ...]]

# Operational, not a trading rule: refresh when the access token has less than this left, so
# a run never starts with a token that could expire mid-session and a failed refresh leaves
# the operator days (observed lifetimes are ~5.7 and ~9.8 days) rather than minutes to re-seed.
REFRESH_MARGIN = timedelta(hours=48)
OAUTH_HTTP_TIMEOUT_SECONDS = 30.0

SEED_COMMAND = "python -m wheelta_robinhood_agent.orchestrator.seed_robinhood_credential"


class OAuthRefresher(Protocol):
    """Performs exactly one refresh grant; raises on any failure (never retries)."""

    def __call__(
        self, client_id: str, refresh_token: SecretStr, *, obtained_at: datetime
    ) -> TokenPair: ...


def refresh_via_http(
    client_id: str, refresh_token: SecretStr, *, obtained_at: datetime
) -> TokenPair:
    """Production refresher: verify discovery, then one refresh grant (ADR-0021)."""
    with httpx.Client(timeout=OAUTH_HTTP_TIMEOUT_SECONDS) as client:
        endpoint = discover_token_endpoint(client)
        return refresh_access_token(
            client, endpoint, client_id, refresh_token, obtained_at=obtained_at
        )


CredentialInserter = Callable[..., StoredCredential]
CredentialLoader = Callable[[Conn, AppEnv, OAuthProvider], StoredCredential | None]


class CredentialStatus(StrEnum):
    STORED = "stored"
    REFRESHED = "refreshed"
    MISSING = "missing"
    UNDECRYPTABLE = "undecryptable"
    REFRESH_FAILED = "refresh_failed"
    PERSIST_FAILED = "persist_failed"


_OPERATOR_MESSAGES: dict[CredentialStatus, str] = {
    CredentialStatus.MISSING: (
        f"no stored Robinhood credential for this environment; seed one with `{SEED_COMMAND}` "
        "(runbook R2)"
    ),
    CredentialStatus.UNDECRYPTABLE: (
        "the stored Robinhood credential cannot be decrypted with ROBINHOOD_TOKEN_ENCRYPTION_KEY; "
        f"restore the key or re-seed with `{SEED_COMMAND}` (runbook R2)"
    ),
    CredentialStatus.REFRESH_FAILED: (
        "the Robinhood token refresh failed; the stored refresh token must be treated as dead. "
        f"Log in again and re-seed with `{SEED_COMMAND}` (runbook R2)"
    ),
    CredentialStatus.PERSIST_FAILED: (
        "a Robinhood token refresh SUCCEEDED but the rotated credential could NOT be saved; "
        "the stored refresh token is now dead. Follow runbook R17 immediately"
    ),
}


@dataclass(frozen=True)
class CredentialResolution:
    status: CredentialStatus
    access_token: SecretStr | None = field(default=None, repr=False)
    # Every token value seen in this resolution, for the run's redactor.
    secrets: tuple[SecretStr, ...] = field(default=(), repr=False)
    credential_id: uuid.UUID | None = None
    access_expires_at: datetime | None = None
    detail: str | None = None  # a sanitized error description; never a token or a body

    @property
    def usable(self) -> bool:
        return self.access_token is not None

    @property
    def operator_message(self) -> str | None:
        return _OPERATOR_MESSAGES.get(self.status)

    def event_payload(self) -> dict[str, object]:
        """Non-secret facts recorded on the run (no token, no fingerprint)."""
        return {
            "status": self.status.value,
            "credential_id": str(self.credential_id) if self.credential_id else None,
            "access_expires_at": (
                self.access_expires_at.isoformat() if self.access_expires_at else None
            ),
            "refresh_margin_seconds": int(REFRESH_MARGIN.total_seconds()),
            "detail": self.detail,
        }


def resolve_robinhood_credential(
    conn: Conn,
    *,
    environment: AppEnv,
    vault: TokenVault,
    clock: Callable[[], datetime],
    refresher: OAuthRefresher,
    insert: CredentialInserter = ledger_credentials.insert_credential,
    load: CredentialLoader = ledger_credentials.latest_credential,
) -> CredentialResolution:
    """The access token to use this run, or why there is none (module docstring).

    Ledger read errors propagate (the run fails closed); refresh and persist errors are
    captured as statuses because they need specific alerts.
    """
    stored = load(conn, environment, OAuthProvider.ROBINHOOD)
    if stored is None:
        return CredentialResolution(CredentialStatus.MISSING)
    try:
        tokens = vault.decrypt(stored.ciphertext)
    except TokenVaultError as exc:
        return CredentialResolution(
            CredentialStatus.UNDECRYPTABLE, credential_id=stored.credential_id, detail=str(exc)
        )
    held = (tokens.access_token, tokens.refresh_token)
    if stored.access_expires_at - clock() > REFRESH_MARGIN:
        return CredentialResolution(
            CredentialStatus.STORED,
            access_token=tokens.access_token,
            secrets=held,
            credential_id=stored.credential_id,
            access_expires_at=stored.access_expires_at,
        )

    obtained_at = clock()
    try:
        pair = refresher(stored.client_id, tokens.refresh_token, obtained_at=obtained_at)
    except Exception as exc:  # noqa: BLE001 - any failure consumed the token; never retry
        return CredentialResolution(
            CredentialStatus.REFRESH_FAILED,
            secrets=held,
            credential_id=stored.credential_id,
            access_expires_at=stored.access_expires_at,
            detail=_safe_detail(exc, held),
        )
    seen = (*held, pair.access_token, pair.refresh_token)
    try:
        new = insert(
            conn,
            environment=environment,
            provider=OAuthProvider.ROBINHOOD,
            client_id=stored.client_id,
            ciphertext=vault.encrypt(pair.access_token, pair.refresh_token),
            access_expires_at=pair.access_expires_at,
            obtained_at=pair.obtained_at,
            source=CredentialSource.REFRESH,
            supersedes_credential_id=stored.credential_id,
        )
    except Exception as exc:  # noqa: BLE001 - the new pair is unsaved; fail loudly, don't use it
        return CredentialResolution(
            CredentialStatus.PERSIST_FAILED,
            secrets=seen,
            credential_id=stored.credential_id,
            access_expires_at=pair.access_expires_at,
            detail=type(exc).__name__,
        )
    return CredentialResolution(
        CredentialStatus.REFRESHED,
        access_token=pair.access_token,
        secrets=seen,
        credential_id=new.credential_id,
        access_expires_at=new.access_expires_at,
    )


def _safe_detail(exc: Exception, held: tuple[SecretStr, ...]) -> str:
    """OAuthError messages are sanitized by construction; anything else is reduced to its
    type name. Belt and braces: a message that contains a held token is dropped."""
    if not isinstance(exc, OAuthError):
        return type(exc).__name__
    message = str(exc)
    if any(s.get_secret_value() in message for s in held):
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"
