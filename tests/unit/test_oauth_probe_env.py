"""scripts/robinhood_oauth_probe.py writes the seed-only ROBINHOOD_OAUTH_* keys into `.env`
(ADR-0021): replaced in place or appended, other lines untouched, mode 0600. Fake values only."""

import importlib.util
import stat
from pathlib import Path
from types import ModuleType

import pytest
from dotenv import dotenv_values

from wheelta_robinhood_agent.config.settings import load_credential_seed_settings

PROBE = Path(__file__).resolve().parents[2] / "scripts" / "robinhood_oauth_probe.py"
RECORD = {
    "client_id": "client-1",
    "obtained_at": "2026-09-26T15:00:00+00:00",
    "access_token": "fake-access-token#1",
    "refresh_token": "fake-refresh-token=2",
    "expires_in": 496235,
}


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("robinhood_oauth_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ENV_FILE", tmp_path / ".env")
    return module


def test_appends_keys_and_keeps_other_lines(probe: ModuleType, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nAPP_ENV=local\nWHEELTA_MCP_TOKEN=keep-me")  # no final newline
    env.chmod(0o644)
    probe.save_tokens(RECORD)
    text = env.read_text()
    assert text.startswith("# comment\nAPP_ENV=local\nWHEELTA_MCP_TOKEN=keep-me\n")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    values = dotenv_values(env)
    assert values["ROBINHOOD_OAUTH_ACCESS_TOKEN"] == RECORD["access_token"]
    assert values["ROBINHOOD_OAUTH_REFRESH_TOKEN"] == RECORD["refresh_token"]
    assert values.get("WHEELTA_MCP_TOKEN") == "keep-me"
    assert not (tmp_path / ".env.probe.tmp").exists()


def test_replaces_existing_keys_in_place(probe: ModuleType, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "A=1\nROBINHOOD_OAUTH_REFRESH_TOKEN=old\nB=2\nexport ROBINHOOD_OAUTH_REFRESH_TOKEN=dup\n"
    )
    probe.save_tokens(RECORD)
    lines = env.read_text().splitlines()
    assert lines[0] == "A=1" and lines[2] == "B=2"
    assert lines[1] == "ROBINHOOD_OAUTH_REFRESH_TOKEN='fake-refresh-token=2'"
    assert sum("ROBINHOOD_OAUTH_REFRESH_TOKEN" in line for line in lines) == 1
    rotated = {**RECORD, "refresh_token": "rotated"}
    probe.save_tokens(rotated)
    assert dotenv_values(env)["ROBINHOOD_OAUTH_REFRESH_TOKEN"] == rotated["refresh_token"]


def test_written_file_loads_through_seed_settings(
    probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("APP_ENV", *probe.ENV_KEYS.values()):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    monkeypatch.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", "a" * 43 + "=")
    probe.save_tokens(RECORD)
    seed = load_credential_seed_settings(tmp_path / ".env")
    assert seed.ROBINHOOD_OAUTH_CLIENT_ID == "client-1"
    assert seed.ROBINHOOD_OAUTH_ACCESS_TOKEN.get_secret_value() == "fake-access-token#1"
    assert seed.ROBINHOOD_OAUTH_EXPIRES_IN == 496235


def test_refuses_unexpected_values_and_saves_nothing(probe: ModuleType, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    with pytest.raises(SystemExit, match="ROBINHOOD_OAUTH_ACCESS_TOKEN"):
        probe.save_tokens({**RECORD, "access_token": "has space"})
    assert env.read_text() == "A=1\n"
