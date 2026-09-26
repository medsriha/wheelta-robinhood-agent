import itertools

import pytest

from wheelta_robinhood_agent.agent.tool_access import (
    ALLOWED_BUILTINS,
    DISALLOWED_BUILTINS,
    build_tool_access,
)
from wheelta_robinhood_agent.config.settings import PHASE_EXECUTION_CEILING
from wheelta_robinhood_agent.domain.enums import ExecutionMode, ToolTier
from wheelta_robinhood_agent.domain.gating import effective_execution_mode
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    LIVE_ORDER_TOOLS,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

REGISTRIES = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY)
ORDER_TOOLS = {f"mcp__robinhood__{n}" for n in LIVE_ORDER_TOOLS}
DENIED_X = {
    ROBINHOOD_REGISTRY.qualified(t.name)
    for t in ROBINHOOD_REGISTRY.tools
    if (t.tier is ToolTier.X and not t.live_order_tool) or t.tier is ToolTier.EXCLUDED
}
TIER_S = {ROBINHOOD_REGISTRY.qualified(t.name) for t in ROBINHOOD_REGISTRY.by_tier(ToolTier.S)}
TIER_R = {r.qualified(t.name) for r in REGISTRIES for t in r.by_tier(ToolTier.R)}


def _access(mode: ExecutionMode, writes: bool = True):  # type: ignore[no-untyped-def]
    return build_tool_access(effective_mode=mode, workspace_writes=writes, registries=REGISTRIES)


@pytest.mark.parametrize("writes", [True, False])
def test_off_mode_never_exposes_order_tools(writes: bool) -> None:
    access = _access(ExecutionMode.OFF, writes)
    assert not ORDER_TOOLS & set(access.allowed_tools)
    assert ORDER_TOOLS <= set(access.disallowed_tools)


def test_live_mode_exposes_exactly_the_three_order_tools() -> None:
    access = _access(ExecutionMode.LIVE)
    x_allowed = (
        {n for n in access.allowed_tools if n.startswith("mcp__robinhood__")} - TIER_R - TIER_S
    )
    assert x_allowed == ORDER_TOOLS


@pytest.mark.parametrize(("mode", "writes"), list(itertools.product(ExecutionMode, [True, False])))
def test_invariants_in_every_mode(mode: ExecutionMode, writes: bool) -> None:
    access = _access(mode, writes)
    allowed, disallowed = set(access.allowed_tools), set(access.disallowed_tools)
    assert not allowed & disallowed
    assert DENIED_X <= disallowed
    assert TIER_R <= allowed
    assert set(DISALLOWED_BUILTINS) <= disallowed
    assert set(ALLOWED_BUILTINS) <= allowed
    assert list(access.allowed_tools) == sorted(access.allowed_tools)
    if writes:
        assert TIER_S <= allowed
    else:
        assert TIER_S <= disallowed


@pytest.mark.parametrize(
    ("requested", "armed"), list(itertools.product(ExecutionMode, [True, False]))
)
def test_phase_1_settings_path_never_exposes_order_tools(
    requested: ExecutionMode, armed: bool
) -> None:
    mode = effective_execution_mode(requested, armed=armed, ceiling=PHASE_EXECUTION_CEILING)
    assert not ORDER_TOOLS & set(_access(mode).allowed_tools)


def test_unregistered_tool_is_in_neither_list() -> None:
    access = _access(ExecutionMode.LIVE)
    name = "mcp__robinhood__brand_new_tool"
    assert name not in access.allowed_tools
    assert name not in access.disallowed_tools
