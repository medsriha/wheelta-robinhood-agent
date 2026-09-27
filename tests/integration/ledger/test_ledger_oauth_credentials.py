"""oauth_credentials (ADR-0021): insert/latest, append-only, supersede-once, seed command."""

import json
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


# -- seed command --------------------------------------------------------------------------------


def _token_file(tmp_path: Path, mode: int = 0o600) -> tuple[Path, dict[str, object]]:
    record: dict[str, object] = {
        "client_id": "client-seed",
        "obtained_at": T0.isoformat(),
        "access_token": "acc-" + secrets.token_urlsafe(24),
        "refresh_token": "ref-" + secrets.token_urlsafe(24),
        "expires_in": 496235,
        "token_type": "Bearer",
        "scope": "internal",
    }
    path = tmp_path / "robinhood_oauth.json"
    path.write_text(json.dumps(record))
    os.chmod(path, mode)
    return path, record


@pytest.fixture
def seed_env(monkeypatch: pytest.MonkeyPatch, ledger_db_url: SecretStr) -> SecretStr:
    key = SecretStr(Fernet.generate_key().decode())
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("DATABASE_URL", ledger_db_url.get_secret_value())
    monkeypatch.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", key.get_secret_value())
    return key


def test_seed_inserts_an_encrypted_seed_row(
    seed_env: SecretStr, conn: Conn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, record = _token_file(tmp_path)
    assert seed_cmd.main([str(path)]) == 0
    out = capsys.readouterr()
    for token in (record["access_token"], record["refresh_token"]):
        assert str(token) not in out.out + out.err
    stored = latest_credential(conn, AppEnv.STAGING, OAuthProvider.ROBINHOOD)
    assert stored is not None and stored.source is CredentialSource.SEED
    assert stored.client_id == "client-seed"
    assert stored.access_expires_at == T0 + timedelta(seconds=496235)
    assert str(record["access_token"]).encode() not in stored.ciphertext
    tokens = TokenVault(seed_env).decrypt(stored.ciphertext)
    assert tokens.refresh_token.get_secret_value() == record["refresh_token"]
    assert "staging" in out.out and str(stored.credential_id) in out.out


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o700])
def test_seed_refuses_a_file_broader_than_0600(
    seed_env: SecretStr, conn: Conn, tmp_path: Path, mode: int, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = _token_file(tmp_path, mode)
    assert seed_cmd.main([str(path)]) == 1
    assert "chmod 600" in capsys.readouterr().err
    assert latest_credential(conn, AppEnv.STAGING, OAuthProvider.ROBINHOOD) is None


def test_seed_rejects_a_malformed_file_without_echo(
    seed_env: SecretStr, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"access_token": "leaky-token-value", "client_id": "c"}))
    os.chmod(path, 0o600)
    assert seed_cmd.main([str(path)]) == 1
    err = capsys.readouterr().err
    assert "not a valid token file" in err and "leaky-token-value" not in err


def test_seed_requires_the_encryption_key(
    seed_env: SecretStr, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    monkeypatch.delenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY")
    path, _ = _token_file(tmp_path)
    assert seed_cmd.main([str(path)]) == 1
    assert "ROBINHOOD_TOKEN_ENCRYPTION_KEY" in capsys.readouterr().err
