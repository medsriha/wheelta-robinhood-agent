"""E2E fixtures: the throwaway Postgres from the ledger tests, settings, clock, notifier.

The Postgres fixtures are loaded from tests/integration/ledger/conftest.py by path (test
directories have no __init__.py), so both suites share one implementation: unix socket only,
no TCP port, and a skip when the server binaries are missing.

Tests marked `requires_cli` need the real Claude Code CLI and are skipped unless
`WRA_RUN_REQUIRES_CLI=1` (docs/TESTING.md "Result boundary").
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from e2e_support import RecordingNotifier
from pydantic import SecretStr

from wheelta_robinhood_agent.config.settings import Settings

_LEDGER_CONFTEST = Path(__file__).resolve().parents[1] / "integration" / "ledger" / "conftest.py"
_spec = importlib.util.spec_from_file_location("wra_e2e_ledger_conftest", _LEDGER_CONFTEST)
assert _spec is not None and _spec.loader is not None
_ledger = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ledger  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(_ledger)

pg_server = _ledger.pg_server
ledger_db_url = _ledger.ledger_db_url
_allow_unix_sockets_only = _ledger._allow_unix_sockets_only


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("WRA_RUN_REQUIRES_CLI") == "1":
        return
    skip = pytest.mark.skip(reason="needs the real Claude Code CLI (WRA_RUN_REQUIRES_CLI=1)")
    for item in items:
        if "requires_cli" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
def make_settings(ledger_db_url: SecretStr) -> Any:
    from e2e_fakes import ACCOUNT_NUMBER

    def make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "APP_ENV": "local",
            "ANTHROPIC_API_KEY": "sk-test-not-a-real-key",
            "AGENT_MODEL": "claude-test-model",
            "ROBINHOOD_MCP_ACCESS_TOKEN": "rh-test-token",
            "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER": ACCOUNT_NUMBER,
            "WHEELTA_MCP_TOKEN": "wt-test-token",
            "DATABASE_URL": ledger_db_url.get_secret_value(),
            "RUN_TIMEOUT_SECONDS": 1500,
        }
        values.update(overrides)
        return Settings(**values)

    return make
