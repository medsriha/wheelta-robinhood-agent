"""oauth_credentials (ADR-0021): insert/latest, append-only, supersede-once, seed command."""

import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.integrations.robinhood.token_vault import TokenVault
from wheelta_robinhood_agent.ledger.errors import CredentialConflict
from wheelta_robinhood_agent.ledger.oauth_credentials import (
    CredentialSource,
    OAuthProvider,
    StoredCredential,
    insert_credential,
    latest_credential,
)
from wheelta_robinhood_agent.orchestrator import seed_robinhood_credential as seed_cmd

Conn = psycopg.Connection[tuple[object, ...]]
T0 = datetime(2026, 9, 26, 15, tzinfo=UTC)


def _insert(
    conn: Conn,
    *,
    env: AppEnv = AppEnv.STAGING,
    source: CredentialSource = CredentialSource.SEED,
    supersedes: StoredCredential | None = None,
    blob: bytes = b"ciphertext",
) -> StoredCredential:
    return insert_credential(
        conn,
        environment=env,
        provider=OAuthProvider.ROBINHOOD,
        client_id="client-1",
        ciphertext=blob,
        access_expires_at=T0 + timedelta(days=5),
        obtained_at=T0,
        source=source,
        supersedes_credential_id=supersedes.credential_id if supersedes else None,
    )


def test_latest_is_none_until_seeded_and_scoped_by_environment(conn: Conn) -> None:
    assert latest_credential(conn, AppEnv.STAGING, OAuthProvider.ROBINHOOD) is None
    seeded = _insert(conn, blob=b"\x00\x01binary")
    got = latest_credential(conn, AppEnv.STAGING, OAuthProvider.ROBINHOOD)
    assert got == seeded
    assert got is not None and got.ciphertext == b"\x00\x01binary"
    assert "binary" not in repr(got)
    assert latest_credential(conn, AppEnv.PRODUCTION, OAuthProvider.ROBINHOOD) is None


def test_refresh_row_becomes_current(conn: Conn) -> None:
    seeded = _insert(conn)
    refreshed = _insert(conn, source=CredentialSource.REFRESH, supersedes=seeded, blob=b"new")
    current = latest_credential(conn, AppEnv.STAGING, OAuthProvider.ROBINHOOD)
    assert current is not None and current.credential_id == refreshed.credential_id
    assert current.supersedes_credential_id == seeded.credential_id


def test_a_row_can_be_superseded_only_once(conn: Conn) -> None:
    seeded = _insert(conn)
    _insert(conn, source=CredentialSource.REFRESH, supersedes=seeded)
    with pytest.raises(CredentialConflict):
        _insert(conn, source=CredentialSource.REFRESH, supersedes=seeded)


def test_source_and_supersedes_must_agree(conn: Conn) -> None:
    seeded = _insert(conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(conn, source=CredentialSource.REFRESH)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(conn, source=CredentialSource.SEED, supersedes=seeded)


def test_rows_are_append_only(conn: Conn) -> None:
    _insert(conn)
    for statement in (
        b"UPDATE oauth_credentials SET client_id = 'x'",
        b"DELETE FROM oauth_credentials",
        b"TRUNCATE oauth_credentials",
    ):
        with pytest.raises(psycopg.errors.RestrictViolation, match="append-only"):
            conn.execute(statement)


# -- seed command (fake values only) -------------------------------------------------------------

SEED_VARS = (
    "APP_ENV", "DATABASE_URL", "ROBINHOOD_TOKEN_ENCRYPTION_KEY", "ROBINHOOD_OAUTH_CLIENT_ID",
    "ROBINHOOD_OAUTH_ACCESS_TOKEN", "ROBINHOOD_OAUTH_REFRESH_TOKEN",
    "ROBINHOOD_OAUTH_OBTAINED_AT", "ROBINHOOD_OAUTH_EXPIRES_IN",
)  # fmt: skip


def _oauth_values() -> dict[str, str]:
    return {
        "ROBINHOOD_OAUTH_CLIENT_ID": "client-seed",
        "ROBINHOOD_OAUTH_ACCESS_TOKEN": "acc-" + secrets.token_urlsafe(24),
        "ROBINHOOD_OAUTH_REFRESH_TOKEN": "ref-" + secrets.token_urlsafe(24),
        "ROBINHOOD_OAUTH_OBTAINED_AT": T0.isoformat(),
        "ROBINHOOD_OAUTH_EXPIRES_IN": "496235",
    }


@pytest.fixture
def seed_env(monkeypatch: pytest.MonkeyPatch, ledger_db_url: SecretStr) -> SecretStr:
    for name in SEED_VARS:
        monkeypatch.delenv(name, raising=False)
    key = SecretStr(Fernet.generate_key().decode())
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("DATABASE_URL", ledger_db_url.get_secret_value())
    monkeypatch.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", key.get_secret_value())
    return key


def _assert_seeded(conn: Conn, key: SecretStr, values: dict[str, str], out: str) -> None:
    stored = latest_credential(conn, AppEnv.STAGING, OAuthProvider.ROBINHOOD)
    assert stored is not None and stored.source is CredentialSource.SEED
    assert stored.client_id == "client-seed"
    assert stored.access_expires_at == T0 + timedelta(seconds=496235)
    assert values["ROBINHOOD_OAUTH_ACCESS_TOKEN"].encode() not in stored.ciphertext
    tokens = TokenVault(key).decrypt(stored.ciphertext)
    assert tokens.access_token.get_secret_value() == values["ROBINHOOD_OAUTH_ACCESS_TOKEN"]
    assert tokens.refresh_token.get_secret_value() == values["ROBINHOOD_OAUTH_REFRESH_TOKEN"]
    assert "staging" in out and str(stored.credential_id) in out


def test_seed_from_environment_variables(
    seed_env: SecretStr, monkeypatch: pytest.MonkeyPatch, conn: Conn,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    values = _oauth_values()
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    assert seed_cmd.main([]) == 0
    out = capsys.readouterr()
    for token in (values["ROBINHOOD_OAUTH_ACCESS_TOKEN"], values["ROBINHOOD_OAUTH_REFRESH_TOKEN"]):
        assert token not in out.out + out.err
    _assert_seeded(conn, seed_env, values, out.out)


def test_seed_from_an_env_file_with_process_env_precedence(
    seed_env: SecretStr, conn: Conn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = _oauth_values()
    env_file = tmp_path / ".env"
    lines = [f"{k}={v}" for k, v in values.items()]
    env_file.write_text("\n".join(["APP_ENV=local", "UNRELATED=1", *lines]) + "\n")
    os.chmod(env_file, 0o600)
    assert seed_cmd.main(["--env-file", str(env_file)]) == 0
    out = capsys.readouterr()
    for token in (values["ROBINHOOD_OAUTH_ACCESS_TOKEN"], values["ROBINHOOD_OAUTH_REFRESH_TOKEN"]):
        assert token not in out.out + out.err
    _assert_seeded(conn, seed_env, values, out.out)  # APP_ENV=staging from the shell wins
    assert latest_credential(conn, AppEnv.LOCAL, OAuthProvider.ROBINHOOD) is None


def test_seed_rejects_missing_or_malformed_values_without_echo(
    seed_env: SecretStr, monkeypatch: pytest.MonkeyPatch, conn: Conn,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    values = _oauth_values()
    values["ROBINHOOD_OAUTH_OBTAINED_AT"] = "2026-09-26T15:00:00"  # naive: rejected
    del values["ROBINHOOD_OAUTH_REFRESH_TOKEN"]
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    assert seed_cmd.main([]) == 1
    err = capsys.readouterr().err
    assert "ROBINHOOD_OAUTH_REFRESH_TOKEN" in err and "ROBINHOOD_OAUTH_OBTAINED_AT" in err
    assert values["ROBINHOOD_OAUTH_ACCESS_TOKEN"] not in err
    assert latest_credential(conn, AppEnv.STAGING, OAuthProvider.ROBINHOOD) is None


def test_seed_requires_the_encryption_key(
    seed_env: SecretStr, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY")
    for name, value in _oauth_values().items():
        monkeypatch.setenv(name, value)
    assert seed_cmd.main([]) == 1
    assert "ROBINHOOD_TOKEN_ENCRYPTION_KEY" in capsys.readouterr().err
