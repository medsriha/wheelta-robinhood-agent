import pytest

from wheelta_robinhood_agent.config.settings import (
    PHASE_EXECUTION_CEILING,
    Settings,
    SettingsError,
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
