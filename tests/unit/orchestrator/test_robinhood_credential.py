"""Robinhood credential resolution (ADR-0021) with fake ledger and refresher. No network."""

import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.integrations.robinhood.oauth import (
    OAuthRefreshFailed,
    OAuthRefreshOutcomeUnknown,
    TokenPair,
)
from wheelta_robinhood_agent.integrations.robinhood.token_vault import TokenVault
from wheelta_robinhood_agent.ledger.errors import CredentialConflict
from wheelta_robinhood_agent.ledger.oauth_credentials import (
    CredentialSource,
    OAuthProvider,
    StoredCredential,
)
from wheelta_robinhood_agent.orchestrator.robinhood_credential import (
    REFRESH_MARGIN,
    Conn,
    CredentialResolution,
    CredentialStatus,
    resolve_robinhood_credential,
)

NOW = datetime(2026, 9, 28, 15, tzinfo=UTC)
OLD_ID = uuid.UUID("00000000-0000-7000-8000-000000000001")
FAKE_CONN = cast(Conn, object())


class World:
    """Fake ledger + refresher that record the order of side effects."""

    def __init__(self, *, expires_in: timedelta | None, key: SecretStr | None = None) -> None:
        self.key = key or SecretStr(Fernet.generate_key().decode())
        self.vault = TokenVault(self.key)
        self.log: list[str] = []
        self.old_access = "old-" + secrets.token_urlsafe(24)
        self.old_refresh = "oldr-" + secrets.token_urlsafe(24)
        self.new_access = "new-" + secrets.token_urlsafe(24)
        self.new_refresh = "newr-" + secrets.token_urlsafe(24)
        self.stored = (
            None
            if expires_in is None
            else StoredCredential(
                credential_id=OLD_ID,
                environment=AppEnv.STAGING,
                provider=OAuthProvider.ROBINHOOD,
                client_id="client-1",
                ciphertext=self.vault.encrypt(
                    SecretStr(self.old_access), SecretStr(self.old_refresh)
                ),
                access_expires_at=NOW + expires_in,
                obtained_at=NOW - timedelta(days=5),
                source=CredentialSource.SEED,
                supersedes_credential_id=None,
                recorded_at=NOW - timedelta(days=5),
            )
        )
        self.inserted: list[dict[str, Any]] = []
        self.refresh_error: Exception | None = None
        self.insert_error: Exception | None = None

    def load(self, conn: Conn, env: AppEnv, provider: OAuthProvider) -> StoredCredential | None:
        self.log.append("load")
        assert env is AppEnv.STAGING and provider is OAuthProvider.ROBINHOOD
        return self.stored

    def refresher(
        self, client_id: str, refresh_token: SecretStr, *, obtained_at: datetime
    ) -> TokenPair:
        self.log.append("refresh")
        assert client_id == "client-1"
        assert refresh_token.get_secret_value() == self.old_refresh
        if self.refresh_error is not None:
            raise self.refresh_error
        return TokenPair(
            access_token=SecretStr(self.new_access),
            refresh_token=SecretStr(self.new_refresh),
            expires_in=842736,
            obtained_at=obtained_at,
        )

    def insert(self, conn: Conn, **kwargs: Any) -> StoredCredential:
        self.log.append("insert")
        if self.insert_error is not None:
            raise self.insert_error
        self.inserted.append(kwargs)
        return StoredCredential(
            credential_id=uuid.UUID("00000000-0000-7000-8000-000000000002"),
            recorded_at=NOW,
            **kwargs,
        )

    def resolve(self) -> CredentialResolution:
        return resolve_robinhood_credential(
            FAKE_CONN,
            environment=AppEnv.STAGING,
            vault=self.vault,
            clock=lambda: NOW,
            refresher=self.refresher,
            insert=self.insert,
            load=self.load,
        )

    def tokens(self) -> list[str]:
        return [self.old_access, self.old_refresh, self.new_access, self.new_refresh]


def _assert_no_tokens(world: World, resolution: CredentialResolution) -> None:
    text = f"{resolution!r} {resolution.event_payload()!r} {resolution.operator_message}"
    for token in world.tokens():
        assert token not in text


def test_no_credential_is_missing_and_never_refreshes() -> None:
    w = World(expires_in=None)
    r = w.resolve()
    assert r.status is CredentialStatus.MISSING and not r.usable
    assert "seed_robinhood_credential" in (r.operator_message or "")
    assert w.log == ["load"]


def test_fresh_token_is_used_without_refresh() -> None:
    w = World(expires_in=REFRESH_MARGIN + timedelta(hours=1))
    r = w.resolve()
    assert r.status is CredentialStatus.STORED
    assert r.access_token is not None and r.access_token.get_secret_value() == w.old_access
    assert r.credential_id == OLD_ID
    assert w.log == ["load"]
    _assert_no_tokens(w, r)


@pytest.mark.parametrize("left", [REFRESH_MARGIN, timedelta(hours=1), -timedelta(days=1)])
def test_near_expiry_refreshes_then_persists_before_use(left: timedelta) -> None:
    w = World(expires_in=left)
    r = w.resolve()
    assert w.log == ["load", "refresh", "insert"]
    assert r.status is CredentialStatus.REFRESHED
    assert r.access_token is not None and r.access_token.get_secret_value() == w.new_access
    [row] = w.inserted
    assert row["source"] is CredentialSource.REFRESH
    assert row["supersedes_credential_id"] == OLD_ID
    assert row["client_id"] == "client-1"
    assert row["access_expires_at"] == NOW + timedelta(seconds=842736)
    stored = w.vault.decrypt(row["ciphertext"])
    assert stored.access_token.get_secret_value() == w.new_access
    assert stored.refresh_token.get_secret_value() == w.new_refresh
    assert {s.get_secret_value() for s in r.secrets} == set(w.tokens())
    _assert_no_tokens(w, r)


@pytest.mark.parametrize(
    "error",
    [CredentialConflict("already superseded"), RuntimeError("db down"), OSError("socket")],
)
def test_persist_failure_after_refresh_is_not_used(error: Exception) -> None:
    w = World(expires_in=timedelta(hours=1))
    w.insert_error = error
    r = w.resolve()
    assert w.log == ["load", "refresh", "insert"]
    assert r.status is CredentialStatus.PERSIST_FAILED
    assert r.access_token is None and not r.usable
    assert r.detail == type(error).__name__
    assert "R17" in (r.operator_message or "")
    assert w.new_refresh in {s.get_secret_value() for s in r.secrets}  # still redacted
    _assert_no_tokens(w, r)


@pytest.mark.parametrize(
    "error",
    [
        OAuthRefreshFailed(400, "invalid_grant"),
        OAuthRefreshOutcomeUnknown("refresh transport error (ReadTimeout)"),
    ],
)
def test_refresh_failure_is_needs_auth_without_persist(error: Exception) -> None:
    w = World(expires_in=timedelta(hours=1))
    w.refresh_error = error
    r = w.resolve()
    assert w.log == ["load", "refresh"]
    assert r.status is CredentialStatus.REFRESH_FAILED and not r.usable
    assert r.detail is not None and type(error).__name__ in r.detail
    _assert_no_tokens(w, r)


def test_unexpected_refresh_error_detail_is_reduced_to_its_type() -> None:
    w = World(expires_in=timedelta(hours=1))
    w.refresh_error = ValueError(f"boom {w.old_refresh}")
    r = w.resolve()
    assert r.status is CredentialStatus.REFRESH_FAILED
    assert r.detail == "ValueError"
    _assert_no_tokens(w, r)


def test_wrong_key_is_undecryptable() -> None:
    w = World(expires_in=timedelta(days=5))
    w.vault = TokenVault(SecretStr(Fernet.generate_key().decode()))
    r = w.resolve()
    assert r.status is CredentialStatus.UNDECRYPTABLE and not r.usable
    assert w.log == ["load"]
    _assert_no_tokens(w, r)
