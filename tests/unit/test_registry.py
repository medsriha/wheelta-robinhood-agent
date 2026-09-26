import pytest

from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.registry import diff_discovered, make_registry
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    LIVE_ORDER_TOOLS,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY


def test_robinhood_registry_shape() -> None:
    assert ROBINHOOD_REGISTRY.verified is False  # third-party snapshot until captured
    live = {t.name for t in ROBINHOOD_REGISTRY.tools if t.live_order_tool}
    assert live == set(LIVE_ORDER_TOOLS)
    assert all(ROBINHOOD_REGISTRY.get(n).tier is ToolTier.X for n in live)  # type: ignore[union-attr]
    for t in ROBINHOOD_REGISTRY.tools:
        if t.name.startswith(("place_", "cancel_", "review_", "preview_", "exercise_")):
            assert t.tier in (ToolTier.X, ToolTier.EXCLUDED), t.name
        if "crypto" in t.name:
            assert t.tier is ToolTier.EXCLUDED, t.name
    assert ROBINHOOD_REGISTRY.get("nope") is None
    assert ROBINHOOD_REGISTRY.qualified("get_accounts") == "mcp__robinhood__get_accounts"


def test_wheelta_registry_is_read_only() -> None:
    assert len(WHEELTA_REGISTRY.tools) == 12
    assert all(t.tier is ToolTier.R for t in WHEELTA_REGISTRY.tools)


def test_discovery_diff() -> None:
    names = {t.name for t in ROBINHOOD_REGISTRY.tools}
    diff = diff_discovered(ROBINHOOD_REGISTRY, names | {"surprise_tool"})
    assert diff.unknown == {"surprise_tool"}
    assert diff.missing == frozenset()
    assert diff.ok
    # A missing denied tool is harmless; a missing read tool fails the stage.
    diff = diff_discovered(ROBINHOOD_REGISTRY, names - {"exercise_option", "get_option_quotes"})
    assert diff.missing == {"get_option_quotes"}
    assert not diff.ok


def test_make_registry_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        make_registry("s", {ToolTier.R: ("a", "a")}, verified=False)
    with pytest.raises(ValueError, match="Tier X"):
        make_registry("s", {ToolTier.R: ("a",)}, verified=False, live_order_tools=("a",))
