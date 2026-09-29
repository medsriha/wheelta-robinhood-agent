"""The production entrypoint fails closed with exit 1 instead of crash-dumping."""

from typing import Any

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv, ExecutionMode
from wheelta_robinhood_agent.orchestrator import main as orchestrator_main


def _patch_startup(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(orchestrator_main, "load_settings", lambda: _Settings())
    monkeypatch.setattr(orchestrator_main, "load_rules", lambda: object())
    monkeypatch.setattr(orchestrator_main, "load_prompt", lambda: object())
    monkeypatch.setattr(orchestrator_main, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator_main, "settings_secrets", lambda s: ())

    def boom(*args: Any, **kwargs: Any) -> int:
        raise error

    monkeypatch.setattr(orchestrator_main, "run_once", boom)


class _Settings:
    APP_ENV = AppEnv.LOCAL
    LOG_LEVEL = "INFO"
    ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = None
    ALERT_WEBHOOK_URL = None
    HEARTBEAT_URL = None
    RUN_SUMMARY_EMAIL_ENABLED = False
    effective_execution_mode = ExecutionMode.OFF


@pytest.mark.parametrize(
    "error", [psycopg.errors.UndefinedTable("relation runs does not exist"), RuntimeError("x")]
)
def test_unhandled_errors_exit_1(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    _patch_startup(monkeypatch, error)
    assert orchestrator_main.main() == 1


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, env: AppEnv, mode: ExecutionMode = ExecutionMode.OFF
) -> list[dict[str, Any]]:
    _patch_startup(monkeypatch, RuntimeError("unused"))
    settings = _Settings()
    settings.APP_ENV = env
    settings.effective_execution_mode = mode
    monkeypatch.setattr(orchestrator_main, "load_settings", lambda: settings)
    monkeypatch.setattr(orchestrator_main, "load_mignon_prompts", lambda: {})
    calls: list[dict[str, Any]] = []

    def fake_run_once(*args: Any, **kwargs: Any) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(orchestrator_main, "run_once", fake_run_once)
    return calls


@pytest.mark.parametrize("argv", [None, []])
def test_no_arguments_is_a_scheduled_tick(monkeypatch: pytest.MonkeyPatch, argv: Any) -> None:
    calls = _patch_run(monkeypatch, AppEnv.PRODUCTION)
    assert orchestrator_main.main(argv) == 0
    assert calls == [{"run_now": False}]


def test_run_now_starts_a_local_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR-0039: dry runs are local and on demand."""
    calls = _patch_run(monkeypatch, AppEnv.LOCAL)
    assert orchestrator_main.main(["--run-now"]) == 0
    assert calls == [{"run_now": True}]


@pytest.mark.parametrize("env", [AppEnv.PRODUCTION, AppEnv.STAGING])
@pytest.mark.parametrize("mode", list(ExecutionMode))
def test_run_now_outside_local_fails_before_any_run(
    monkeypatch: pytest.MonkeyPatch, env: AppEnv, mode: ExecutionMode
) -> None:
    """ADR-0039: production runs live on its schedule only; no hand-started run there."""
    calls = _patch_run(monkeypatch, env, mode)
    assert orchestrator_main.main(["--run-now"]) == 1
    assert calls == []


@pytest.mark.parametrize("argv", [["--now"], ["--run-now", "extra"], ["run-now"]])
def test_unknown_arguments_fail_before_loading_anything(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    def never() -> Any:
        raise AssertionError("settings must not load")

    monkeypatch.setattr(orchestrator_main, "load_settings", never)
    assert orchestrator_main.main(argv) == 1
