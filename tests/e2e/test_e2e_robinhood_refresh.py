"""Full runs in ROBINHOOD_MCP_AUTH=refresh_token mode (ADR-0021) with a fake OAuth refresher.

Real ledger (oauth_credentials), vault, orchestrator, and session; the refresher and, where
stated, the credential insert are fakes. No network.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
from cryptography.fernet import Fernet
from e2e_support import SESSION_TIME, FakeClock, RecordingNotifier
from pydantic import SecretStr
from test_e2e_orchestrator import Harness

from wheelta_robinhood_agent.domain.enums import AppEnv, RunStatus
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.run_identity import run_id_for, slot_for
from wheelta_robinhood_agent.integrations.robinhood.oauth import OAuthRefreshFailed, TokenPair
from wheelta_robinhood_agent.integrations.robinhood.token_vault import TokenVault
from wheelta_robinhood_agent.ledger.oauth_credentials import (
    CredentialSource,
    OAuthProvider,
    StoredCredential,
    insert_credential,
    latest_credential,
)
from wheelta_robinhood_agent.orchestrator.robinhood_credential import REFRESH_MARGIN


class Tokens:
    def __init__(self) -> None:
        self.old_access = "oa-" + secrets.token_urlsafe(32)
        self.old_refresh = "or-" + secrets.token_urlsafe(32)
        self.new_access = "na-" + secrets.token_urlsafe(32)
        self.new_refresh = "nr-" + secrets.token_urlsafe(32)

    def all(self) -> list[str]:
        return [self.old_access, self.old_refresh, self.new_access, self.new_refresh]


class Refresher:
    def __init__(self, tokens: Tokens, error: Exception | None = None) -> None:
        self.tokens = tokens
        self.error = error
        self.calls: list[str] = []

    def __call__(
        self, client_id: str, refresh_token: SecretStr, *, obtained_at: datetime
    ) -> TokenPair:
        self.calls.append(refresh_token.get_secret_value())
        if self.error is not None:
            raise self.error
        return TokenPair(
            access_token=SecretStr(self.tokens.new_access),
            refresh_token=SecretStr(self.tokens.new_refresh),
            expires_in=842736,
            obtained_at=obtained_at,
        )


@pytest.fixture
def key() -> SecretStr:
    return SecretStr(Fernet.generate_key().decode())


@pytest.fixture
def refresh_harness(
    make_settings: Any, notifier: RecordingNotifier, key: SecretStr
) -> Callable[..., Harness]:
    def make() -> Harness:
        settings = make_settings(
            ROBINHOOD_MCP_AUTH="refresh_token",
            ROBINHOOD_MCP_ACCESS_TOKEN=None,
            ROBINHOOD_TOKEN_ENCRYPTION_KEY=key.get_secret_value(),
        )
        return Harness(settings, notifier, FakeClock(SESSION_TIME))

    return make


def _seed(h: Harness, key: SecretStr, tokens: Tokens, expires_in: timedelta) -> StoredCredential:
    with h.conn() as c:
        return insert_credential(
            c,
            environment=AppEnv.LOCAL,
            provider=OAuthProvider.ROBINHOOD,
            client_id="client-1",
            ciphertext=TokenVault(key).encrypt(
                SecretStr(tokens.old_access), SecretStr(tokens.old_refresh)
            ),
            access_expires_at=SESSION_TIME + expires_in,
            obtained_at=SESSION_TIME - timedelta(days=5),
            source=CredentialSource.SEED,
        )


def _credential_event(h: Harness) -> dict[str, Any]:
    events = [e for e in h.events(RunEventType.METADATA) if "robinhood_credential" in e]
    assert len(events) == 1
    return dict(events[0]["robinhood_credential"])


def _bearer(h: Harness) -> str:
    """The Authorization header the proxy's upstream presents (ADR-0023). The CLI itself is
    handed no credential at all."""
    servers = h.clis[0].options.mcp_servers
    assert isinstance(servers, dict)
    assert all("headers" not in cfg for cfg in servers.values())
    (upstream,) = [s for s in h.world.upstream_servers if s.name == "robinhood"]
    assert upstream.token is not None
    return f"Bearer {upstream.token.get_secret_value()}"


def _assert_no_token_leak(h: Harness, tokens: Tokens, caplog: pytest.LogCaptureFixture) -> None:
    logged = " ".join(f"{r.getMessage()} {r.__dict__!r}" for r in caplog.records)
    with h.conn() as c:
        events = c.execute("SELECT payload::text FROM run_events").fetchall()
        alerts = c.execute("SELECT payload::text FROM alerts_sent").fetchall()
    stored = f"{events!r} {alerts!r} {[a.model_dump_json() for a in h.notifier.alerts]!r}"
    for token in tokens.all():
        assert token not in logged
        assert token not in stored


def test_no_credential_is_needs_auth_with_a_seed_message(
    refresh_harness: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    h = refresh_harness()
    refresher = Refresher(Tokens())
    assert h.run(oauth_refresher=refresher) == 1
    assert h.clis == [] and refresher.calls == []
    assert h.status() is RunStatus.FAILED
    [alert] = [a for a in h.notifier.alerts if a.kind.value == "robinhood_needs_auth"]
    assert "seed_robinhood_credential" in alert.message
    assert _credential_event(h)["status"] == "missing"
    statuses = {e["server"]: e["status"] for e in h.events(RunEventType.SOURCE_STATUS)}
    assert statuses["robinhood"] == "needs-auth"


def test_fresh_token_is_used_without_refresh(
    refresh_harness: Callable[..., Harness], key: SecretStr, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    h = refresh_harness()
    tokens = Tokens()
    seeded = _seed(h, key, tokens, REFRESH_MARGIN + timedelta(days=1))
    refresher = Refresher(tokens)
    assert h.run(oauth_refresher=refresher) == 0
    assert refresher.calls == []
    assert _bearer(h) == f"Bearer {tokens.old_access}"
    event = _credential_event(h)
    assert event["status"] == "stored" and event["credential_id"] == str(seeded.credential_id)
    _assert_no_token_leak(h, tokens, caplog)


def test_near_expiry_refreshes_and_persists_before_use(
    refresh_harness: Callable[..., Harness], key: SecretStr, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    h = refresh_harness()
    tokens = Tokens()
    seeded = _seed(h, key, tokens, timedelta(hours=2))
    order: list[str] = []
    refresher = Refresher(tokens)

    def recording_refresher(*args: Any, **kwargs: Any) -> TokenPair:
        order.append("refresh")
        return refresher(*args, **kwargs)

    def recording_insert(conn: Any, **kwargs: Any) -> StoredCredential:
        order.append("insert")
        return insert_credential(conn, **kwargs)

    code = h.run(oauth_refresher=recording_refresher, insert_credential=recording_insert)
    assert code == 0
    assert refresher.calls == [tokens.old_refresh]
    assert order == ["refresh", "insert"]  # the session (and the token's use) came after
    assert _bearer(h) == f"Bearer {tokens.new_access}"
    with h.conn() as c:
        current = latest_credential(c, AppEnv.LOCAL, OAuthProvider.ROBINHOOD)
    assert current is not None and current.source is CredentialSource.REFRESH
    assert current.supersedes_credential_id == seeded.credential_id
    stored = TokenVault(key).decrypt(current.ciphertext)
    assert stored.refresh_token.get_secret_value() == tokens.new_refresh
    assert _credential_event(h)["status"] == "refreshed"
    _assert_no_token_leak(h, tokens, caplog)


def test_persist_failure_after_refresh_fails_the_run_loudly(
    refresh_harness: Callable[..., Harness], key: SecretStr, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    h = refresh_harness()
    tokens = Tokens()
    seeded = _seed(h, key, tokens, timedelta(hours=2))

    def failing_insert(conn: Any, **kwargs: Any) -> StoredCredential:
        raise RuntimeError(f"db write failed for {kwargs['client_id']}")

    code = h.run(oauth_refresher=Refresher(tokens), insert_credential=failing_insert)
    assert code == 1
    assert h.clis == []  # the unsaved token is never used
    assert h.status() is RunStatus.FAILED
    kinds = h.notifier.alert_kinds()
    assert "robinhood_credential_unsaved" in kinds and "robinhood_needs_auth" not in kinds
    [alert] = [a for a in h.notifier.alerts if a.kind.value == "robinhood_credential_unsaved"]
    assert alert.runbook == "R17" and alert.severity.value == "error"
    statuses = [e for e in h.events(RunEventType.STATUS) if e and "reason" in e]
    assert statuses[-1]["reason"] == "robinhood_credential_unsaved"
    with h.conn() as c:
        current = latest_credential(c, AppEnv.LOCAL, OAuthProvider.ROBINHOOD)
    assert current is not None and current.credential_id == seeded.credential_id
    _assert_no_token_leak(h, tokens, caplog)


def test_refresh_failure_is_needs_auth(
    refresh_harness: Callable[..., Harness], key: SecretStr, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    h = refresh_harness()
    tokens = Tokens()
    _seed(h, key, tokens, timedelta(hours=2))
    refresher = Refresher(tokens, error=OAuthRefreshFailed(400, "invalid_grant"))
    assert h.run(oauth_refresher=refresher) == 1
    assert len(refresher.calls) == 1  # never retried
    assert h.clis == []
    assert "robinhood_needs_auth" in h.notifier.alert_kinds()
    event = _credential_event(h)
    assert event["status"] == "refresh_failed" and "invalid_grant" in event["detail"]
    _assert_no_token_leak(h, tokens, caplog)


# -- ADR-0024: dry runs are local only; production keeps its credential fresh --------------------


def test_production_off_mode_refreshes_the_credential_but_starts_no_session(
    make_settings: Any, notifier: RecordingNotifier, key: SecretStr
) -> None:
    settings = make_settings(
        APP_ENV="production",
        ROBINHOOD_MCP_AUTH="refresh_token",
        ROBINHOOD_MCP_ACCESS_TOKEN=None,
        ROBINHOOD_TOKEN_ENCRYPTION_KEY=key.get_secret_value(),
    )
    h = Harness(settings, notifier, FakeClock(SESSION_TIME))
    h.run_id = run_id_for(AppEnv.PRODUCTION, slot_for(SESSION_TIME))
    tokens = Tokens()
    with h.conn() as c:
        insert_credential(
            c,
            environment=AppEnv.PRODUCTION,
            provider=OAuthProvider.ROBINHOOD,
            client_id="client-1",
            ciphertext=TokenVault(key).encrypt(
                SecretStr(tokens.old_access), SecretStr(tokens.old_refresh)
            ),
            access_expires_at=SESSION_TIME + timedelta(hours=2),
            obtained_at=SESSION_TIME - timedelta(days=5),
            source=CredentialSource.SEED,
        )
    refresher = Refresher(tokens)
    assert h.run(oauth_refresher=refresher) == 0
    assert refresher.calls == [tokens.old_refresh]  # the rotating token stays alive
    assert h.clis == [] and h.world.upstream_servers == []  # no session, no MCP connection
    assert h.status() is RunStatus.SKIPPED_DRY_RUN_NOT_LOCAL
    assert _credential_event(h)["status"] == "refreshed"
    assert [hb.status.value for hb in notifier.heartbeats] == ["success"]
