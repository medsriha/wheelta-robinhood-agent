import pytest

from wheelta_robinhood_agent.config.settings import (
    PHASE_EXECUTION_CEILING,
    Settings,
    SettingsError,
    load_credential_seed_settings,
    load_database_url,
    load_settings,
)
from wheelta_robinhood_agent.domain.enums import ExecutionMode

REQUIRED = {
    "ANTHROPIC_API_KEY": "sk-test-secret",
    "AGENT_MODEL": "claude-test-model",
    "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER": "5RA123456789",
    "WHEELTA_MCP_TOKEN": "wheelta-secret",
    "DATABASE_URL": "postgresql://u:p@localhost/db",
}

ALL_VARS = [
    *REQUIRED,
    "APP_ENV",
    "LOG_LEVEL",
    "RUN_TIMEOUT_SECONDS",
    "EXECUTION_MODE",
    "EXECUTION_ARMED",
    "KILL_SWITCH",
    "MCP_TIMEOUT",
    "MCP_TOOL_TIMEOUT",
    "ROBINHOOD_MCP_URL",
    "ROBINHOOD_MCP_ACCESS_TOKEN",
    "ROBINHOOD_MCP_AUTH",
    "ROBINHOOD_TOKEN_ENCRYPTION_KEY",
    "ROBINHOOD_OAUTH_CLIENT_ID",
    "ROBINHOOD_OAUTH_ACCESS_TOKEN",
    "ROBINHOOD_OAUTH_REFRESH_TOKEN",
    "ROBINHOOD_OAUTH_OBTAINED_AT",
    "ROBINHOOD_OAUTH_EXPIRES_IN",
    "ROBINHOOD_WORKSPACE_WRITES",
    "ROBINHOOD_WORKSPACE_PREFIX",
    "WHEELTA_MCP_URL",
    "HEARTBEAT_URL",
    "ALERT_WEBHOOK_URL",
]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_safe_defaults(env: pytest.MonkeyPatch) -> None:
    s = load_settings()
    assert s.requested_execution_mode is ExecutionMode.OFF
    assert s.effective_execution_mode is ExecutionMode.OFF
    assert s.EXECUTION_ARMED is False
    assert s.KILL_SWITCH is False
    assert s.ROBINHOOD_MCP_ACCESS_TOKEN is None


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_missing_required_fails_fast(env: pytest.MonkeyPatch, name: str) -> None:
    env.delenv(name)
    with pytest.raises(SettingsError, match=name):
        load_settings()


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "WHEELTA_MCP_TOKEN", "DATABASE_URL"])
def test_blank_required_secret_fails(env: pytest.MonkeyPatch, name: str) -> None:
    env.setenv(name, "   ")
    with pytest.raises(SettingsError, match=name):
        load_settings()


def test_error_never_echoes_secret_values(env: pytest.MonkeyPatch) -> None:
    env.setenv("RUN_TIMEOUT_SECONDS", "sk-live-should-not-appear")
    with pytest.raises(SettingsError) as info:
        load_settings()
    assert "sk-live-should-not-appear" not in str(info.value)
    assert "sk-test-secret" not in str(info.value)


@pytest.mark.parametrize("raw", ["", "LIVE", "Live", "propose", "review", "on", "true"])
def test_unknown_execution_mode_is_off(env: pytest.MonkeyPatch, raw: str) -> None:
    env.setenv("EXECUTION_MODE", raw)
    env.setenv("EXECUTION_ARMED", "true")
    s = load_settings()
    assert s.requested_execution_mode is ExecutionMode.OFF
    assert s.effective_execution_mode is ExecutionMode.OFF
    assert s.config_snapshot()["execution_mode_raw"] == raw


@pytest.mark.parametrize("armed", ["true", "false"])
def test_phase_1_caps_live_at_off(env: pytest.MonkeyPatch, armed: str) -> None:
    assert PHASE_EXECUTION_CEILING is ExecutionMode.OFF
    env.setenv("EXECUTION_MODE", "live")
    env.setenv("EXECUTION_ARMED", armed)
    s = load_settings()
    assert s.requested_execution_mode is ExecutionMode.LIVE
    assert s.effective_execution_mode is ExecutionMode.OFF


@pytest.mark.parametrize("value", ["3600", "7200", "0", "-5", "abc"])
def test_run_timeout_must_be_below_cron_interval(env: pytest.MonkeyPatch, value: str) -> None:
    env.setenv("RUN_TIMEOUT_SECONDS", value)
    with pytest.raises(SettingsError, match="RUN_TIMEOUT_SECONDS"):
        load_settings()


@pytest.mark.parametrize("name", ["EXECUTION_ARMED", "KILL_SWITCH"])
def test_malformed_safety_bool_fails(env: pytest.MonkeyPatch, name: str) -> None:
    env.setenv(name, "maybe")
    with pytest.raises(SettingsError, match=name):
        load_settings()


def test_blank_optional_values_are_absent(env: pytest.MonkeyPatch) -> None:
    for name in ("ROBINHOOD_MCP_ACCESS_TOKEN", "HEARTBEAT_URL", "ALERT_WEBHOOK_URL"):
        env.setenv(name, "")
    s = load_settings()
    assert s.ROBINHOOD_MCP_ACCESS_TOKEN is None
    assert s.HEARTBEAT_URL is None
    assert s.ALERT_WEBHOOK_URL is None


def test_config_snapshot_has_no_secrets(env: pytest.MonkeyPatch) -> None:
    env.setenv("ROBINHOOD_MCP_ACCESS_TOKEN", "rh-token-secret")
    env.setenv("HEARTBEAT_URL", "https://hc.example/secret-path")
    s = load_settings()
    snapshot = repr(s.config_snapshot())
    for secret in [*REQUIRED.values(), "rh-token-secret", "secret-path"]:
        if secret in (REQUIRED["AGENT_MODEL"],):
            continue
        assert secret not in snapshot
    assert s.config_snapshot()["robinhood_account_last4"] == "6789"
    assert s.config_snapshot()["robinhood_token_present"] is True


def test_settings_repr_redacts_secrets(env: pytest.MonkeyPatch) -> None:
    text = repr(load_settings())
    assert "sk-test-secret" not in text
    assert "5RA123456789" not in text


def test_settings_are_frozen(env: pytest.MonkeyPatch) -> None:
    s = load_settings()
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        s.KILL_SWITCH = True  # type: ignore[misc]
    assert isinstance(s, Settings)


def test_env_file_loads_when_given(
    env: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    env.delenv("AGENT_MODEL")
    path = tmp_path / ".env"  # type: ignore[operator]
    path.write_text("AGENT_MODEL=from-file\n")
    assert load_settings(path).AGENT_MODEL == "from-file"


def test_load_database_url_alone(env: pytest.MonkeyPatch) -> None:
    env.delenv("ANTHROPIC_API_KEY")
    assert load_database_url().get_secret_value() == REQUIRED["DATABASE_URL"]
    env.setenv("DATABASE_URL", " ")
    with pytest.raises(SettingsError, match="DATABASE_URL"):
        load_database_url()
    env.delenv("DATABASE_URL")
    with pytest.raises(SettingsError, match="DATABASE_URL"):
        load_database_url()


def test_claude_code_login_mode_is_local_only(env: pytest.MonkeyPatch) -> None:
    env.setenv("ROBINHOOD_MCP_AUTH", "claude_code_login")
    s = load_settings()
    assert s.config_snapshot()["robinhood_mcp_auth"] == "claude_code_login"
    env.setenv("APP_ENV", "staging")
    with pytest.raises(SettingsError, match="requires APP_ENV=local"):
        load_settings()


def test_claude_code_login_mode_excludes_a_token(env: pytest.MonkeyPatch) -> None:
    env.setenv("ROBINHOOD_MCP_AUTH", "claude_code_login")
    env.setenv("ROBINHOOD_MCP_ACCESS_TOKEN", "rh-token")
    with pytest.raises(SettingsError, match="excludes ROBINHOOD_MCP_ACCESS_TOKEN"):
        load_settings()


def test_unknown_robinhood_auth_mode_fails(env: pytest.MonkeyPatch) -> None:
    env.setenv("ROBINHOOD_MCP_AUTH", "browser_cookies")
    with pytest.raises(SettingsError, match="ROBINHOOD_MCP_AUTH"):
        load_settings()


def test_local_remote_risk_acceptance(env: pytest.MonkeyPatch) -> None:
    assert load_settings().remote_result_risk_accepted is False
    env.setenv("LOCAL_ACCEPT_REMOTE_RESULT_RISK", "true")
    assert load_settings().remote_result_risk_accepted is True
    env.setenv("APP_ENV", "production")
    with pytest.raises(SettingsError, match="requires APP_ENV=local"):
        load_settings()


# -- ROBINHOOD_MCP_AUTH=refresh_token (ADR-0021) ---------------------------------------------------

FERNET_KEY = "a" * 43 + "="  # urlsafe base64 of 32 bytes; a test key, not a real one


@pytest.mark.parametrize("app_env", ["local", "staging", "production"])
def test_refresh_token_mode_is_allowed_in_every_env(env: pytest.MonkeyPatch, app_env: str) -> None:
    env.setenv("APP_ENV", app_env)
    env.setenv("ROBINHOOD_MCP_AUTH", "refresh_token")
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", FERNET_KEY)
    s = load_settings()
    snapshot = s.config_snapshot()
    assert snapshot["robinhood_mcp_auth"] == "refresh_token"
    assert snapshot["robinhood_token_encryption_key_present"] is True
    assert FERNET_KEY not in repr(snapshot)
    assert FERNET_KEY not in repr(s)


def test_refresh_token_mode_requires_the_key(env: pytest.MonkeyPatch) -> None:
    env.setenv("ROBINHOOD_MCP_AUTH", "refresh_token")
    with pytest.raises(SettingsError, match="requires ROBINHOOD_TOKEN_ENCRYPTION_KEY"):
        load_settings()
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", "")
    with pytest.raises(SettingsError, match="requires ROBINHOOD_TOKEN_ENCRYPTION_KEY"):
        load_settings()


def test_refresh_token_mode_excludes_a_static_token(env: pytest.MonkeyPatch) -> None:
    env.setenv("ROBINHOOD_MCP_AUTH", "refresh_token")
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", FERNET_KEY)
    env.setenv("ROBINHOOD_MCP_ACCESS_TOKEN", "rh-token")
    with pytest.raises(SettingsError, match="excludes ROBINHOOD_MCP_ACCESS_TOKEN"):
        load_settings()


@pytest.mark.parametrize("mode", ["token", "claude_code_login"])
def test_key_is_rejected_in_other_modes(env: pytest.MonkeyPatch, mode: str) -> None:
    env.setenv("ROBINHOOD_MCP_AUTH", mode)
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", FERNET_KEY)
    with pytest.raises(SettingsError, match="only valid with ROBINHOOD_MCP_AUTH=refresh_token"):
        load_settings()


@pytest.mark.parametrize("bad", ["not-a-key", "a" * 44, "YWJj"])
def test_malformed_key_fails_without_echo(env: pytest.MonkeyPatch, bad: str) -> None:
    env.setenv("ROBINHOOD_MCP_AUTH", "refresh_token")
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", bad)
    with pytest.raises(SettingsError, match="Fernet key") as info:
        load_settings()
    assert bad not in str(info.value)


def test_key_absent_by_default(env: pytest.MonkeyPatch) -> None:
    snapshot = load_settings().config_snapshot()
    assert snapshot["robinhood_token_encryption_key_present"] is False


SEED_OAUTH = {
    "ROBINHOOD_OAUTH_CLIENT_ID": "client-1",
    "ROBINHOOD_OAUTH_ACCESS_TOKEN": "fake-access-token-value",
    "ROBINHOOD_OAUTH_REFRESH_TOKEN": "fake-refresh-token-value",
    "ROBINHOOD_OAUTH_OBTAINED_AT": "2026-09-26T15:00:00+00:00",
    "ROBINHOOD_OAUTH_EXPIRES_IN": "496235",
}


def test_credential_seed_settings_need_no_agent_secrets(env: pytest.MonkeyPatch) -> None:
    env.delenv("ANTHROPIC_API_KEY")
    env.delenv("WHEELTA_MCP_TOKEN")
    for name, value in SEED_OAUTH.items():
        env.setenv(name, value)
    with pytest.raises(SettingsError, match="ROBINHOOD_TOKEN_ENCRYPTION_KEY"):
        load_credential_seed_settings()
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", FERNET_KEY)
    env.setenv("APP_ENV", "staging")
    seed = load_credential_seed_settings()
    assert seed.APP_ENV.value == "staging"
    assert seed.ROBINHOOD_OAUTH_CLIENT_ID == "client-1"
    assert seed.access_expires_at.isoformat() == "2026-10-02T08:50:35+00:00"
    text = repr(seed)
    for secret in ("fake-access-token-value", "fake-refresh-token-value", FERNET_KEY):
        assert secret not in text


@pytest.mark.parametrize("name", sorted(SEED_OAUTH))
def test_credential_seed_settings_require_every_oauth_value(
    env: pytest.MonkeyPatch, name: str
) -> None:
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", FERNET_KEY)
    for key, value in SEED_OAUTH.items():
        if key != name:
            env.setenv(key, value)
    with pytest.raises(SettingsError, match=name) as info:
        load_credential_seed_settings()
    assert "fake-access-token-value" not in str(info.value)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("ROBINHOOD_OAUTH_OBTAINED_AT", "2026-09-26T15:00:00"),
        ("ROBINHOOD_OAUTH_EXPIRES_IN", "0"),
        ("ROBINHOOD_OAUTH_ACCESS_TOKEN", " "),
        ("ROBINHOOD_OAUTH_CLIENT_ID", ""),
    ],
)
def test_credential_seed_settings_reject_malformed_values(
    env: pytest.MonkeyPatch, name: str, bad: str
) -> None:
    env.setenv("ROBINHOOD_TOKEN_ENCRYPTION_KEY", FERNET_KEY)
    for key, value in SEED_OAUTH.items():
        env.setenv(key, value)
    env.setenv(name, bad)
    with pytest.raises(SettingsError, match=name):
        load_credential_seed_settings()


def test_mignon_models_default_to_agent_model_and_parse_an_allowlist(
    env: pytest.MonkeyPatch,
) -> None:
    """ADR-0025: MIGNON_AGENT_MODELS is optional; blank or unset means AGENT_MODEL only."""
    s = load_settings()
    assert s.MIGNON_AGENT_MODELS is None and s.mignon_models == (s.AGENT_MODEL,)
    env.setenv("MIGNON_AGENT_MODELS", "")
    assert load_settings().mignon_models == (s.AGENT_MODEL,)
    env.setenv("MIGNON_AGENT_MODELS", " claude-haiku-4-5, claude-sonnet-5,claude-opus-4-8 ")
    pinned = load_settings()
    assert pinned.mignon_models == ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8")
    assert pinned.config_snapshot()["mignon_agent_models"] == list(pinned.mignon_models)


@pytest.mark.parametrize(
    "value",
    [
        "opus",
        "claude-opus-4-8,Sonnet",
        "claude-opus-4-8,inherit",
        "claude-x,claude-x",
        "claude-x,",
        "Claude Opus",
        "claude--x",
    ],
)
def test_bad_mignon_model_allowlists_are_rejected(env: pytest.MonkeyPatch, value: str) -> None:
    env.setenv("MIGNON_AGENT_MODELS", value)
    with pytest.raises(SettingsError, match="MIGNON_AGENT_MODELS"):
        load_settings()


@pytest.mark.parametrize("alias", ["opus", "Sonnet", "inherit"])
def test_agent_model_aliases_are_rejected(env: pytest.MonkeyPatch, alias: str) -> None:
    env.setenv("AGENT_MODEL", alias)
    with pytest.raises(SettingsError, match="AGENT_MODEL"):
        load_settings()
