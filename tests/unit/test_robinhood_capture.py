"""The registry and account-scope table match our own capture of robinhood-trading (ADR-0017)."""

import json
from pathlib import Path

from wheelta_robinhood_agent.agent.account_scope import (
    ROBINHOOD_ACCOUNT_SCOPE,
    AccountScope,
    AccountScopeSpec,
)
from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.registry import diff_discovered
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY

CAPTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "robinhood" / "tools_2026-09-26.json").read_text()
)
TOOLS = {t["name"]: t for t in CAPTURE["tools"]}


def test_capture_provenance() -> None:
    assert CAPTURE["provenance"]["server"] == {"name": "robinhood-trading", "version": "1.6.0"}
    assert len(TOOLS) == 76


def test_registry_covers_the_capture_and_nothing_expected_is_missing() -> None:
    diff = diff_discovered(ROBINHOOD_REGISTRY, TOOLS)
    assert diff.unknown == frozenset()
    assert diff.missing == frozenset()


def test_write_tools_are_never_tier_r() -> None:
    for name, entry in TOOLS.items():
        spec = ROBINHOOD_REGISTRY.get(name)
        assert spec is not None
        if not entry["read_only_hint"]:
            assert spec.tier is not ToolTier.R, name


def test_scope_table_matches_captured_account_arguments() -> None:
    for name, entry in TOOLS.items():
        if not entry.get("schema_captured") or ROBINHOOD_REGISTRY.get(name).tier is not ToolTier.R:  # type: ignore[union-attr]
            continue
        spec = ROBINHOOD_ACCOUNT_SCOPE[name]
        if entry["account_arg"] is not None:
            assert spec == AccountScopeSpec.verified(entry["account_arg"]), name
        elif entry["login_scoped"] or entry.get("lists_all_accounts"):
            assert spec.scope is AccountScope.UNVERIFIED, name
        else:
            assert spec.scope is AccountScope.NOT_ACCOUNT_SCOPED, name
