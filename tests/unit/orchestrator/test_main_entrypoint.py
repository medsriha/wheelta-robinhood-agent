"""The production entrypoint fails closed with exit 1 instead of crash-dumping."""

from typing import Any

import psycopg
import pytest

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
    LOG_LEVEL = "INFO"
    ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = None
    ALERT_WEBHOOK_URL = None
    HEARTBEAT_URL = None


@pytest.mark.parametrize(
    "error", [psycopg.errors.UndefinedTable("relation runs does not exist"), RuntimeError("x")]
)
def test_unhandled_errors_exit_1(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    _patch_startup(monkeypatch, error)
    assert orchestrator_main.main() == 1
