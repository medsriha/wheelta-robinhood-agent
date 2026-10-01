import itertools

import pytest

from wheelta_robinhood_agent.agent.mignons import AGENT_ORCHESTRATOR_TOOLS, ROLE_TOOLS, Role
from wheelta_robinhood_agent.agent.order_walk import (
    ORDER_WORK_REGISTRY,
    QUALIFIED_AWAIT_TOOL,
    QUALIFIED_WORK_TOOL,
)
from wheelta_robinhood_agent.agent.tool_access import (
    DISALLOWED_BUILTINS,
    SESSION_BUILTINS,
    build_tool_access,
)
from wheelta_robinhood_agent.config.settings import PHASE_EXECUTION_CEILING
from wheelta_robinhood_agent.domain.enums import AgentRole, ExecutionMode, ToolTier
from wheelta_robinhood_agent.domain.gating import effective_execution_mode
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    LIVE_ORDER_TOOLS,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.websearch.registry import TAVILY_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

REGISTRIES = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY)
ORDER_TOOLS = {f"mcp__robinhood__{n}" for n in LIVE_ORDER_TOOLS}
# ADR-0066: code reviews and places; the model keeps cancel and starts order-work jobs.
MODEL_ORDER_TOOLS = {"mcp__robinhood__cancel_option_order", QUALIFIED_WORK_TOOL}
EXECUTOR_ONLY = {"mcp__robinhood__review_option_order", "mcp__robinhood__place_option_order"}
DENIED_X = {
    ROBINHOOD_REGISTRY.qualified(t.name)
    for t in ROBINHOOD_REGISTRY.tools
    if (t.tier is ToolTier.X and not t.live_order_tool) or t.tier is ToolTier.EXCLUDED
}
TIER_S = {ROBINHOOD_REGISTRY.qualified(t.name) for t in ROBINHOOD_REGISTRY.by_tier(ToolTier.S)}
TIER_R = {r.qualified(t.name) for r in REGISTRIES for t in r.by_tier(ToolTier.R)}
ALL_ROLES = frozenset().union(*ROLE_TOOLS.values())


def _access(mode: ExecutionMode, writes: bool = True):  # type: ignore[no-untyped-def]
    return build_tool_access(effective_mode=mode, workspace_writes=writes, registries=REGISTRIES)


@pytest.mark.parametrize("writes", [True, False])
def test_off_mode_never_exposes_order_tools(writes: bool) -> None:
    access = _access(ExecutionMode.OFF, writes)
    assert not ORDER_TOOLS & set(access.allowed_tools)
    assert ORDER_TOOLS <= set(access.disallowed_tools)


def test_live_mode_exposes_exactly_cancel_and_work_order() -> None:
    """ADR-0066: with an order venue the model's Tier X tools are cancel_option_order and
    work_option_order; review and place stay disallowed (only the executor calls them)."""
    access = build_tool_access(
        effective_mode=ExecutionMode.LIVE,
        workspace_writes=True,
        registries=(*REGISTRIES, ORDER_WORK_REGISTRY),
    )
    x_allowed = set(access.allowed_tools) - TIER_R - TIER_S - {QUALIFIED_AWAIT_TOOL}
    x_allowed = {n for n in x_allowed if n.startswith("mcp__")}
    assert x_allowed == MODEL_ORDER_TOOLS
    assert EXECUTOR_ONLY <= set(access.disallowed_tools)
    assert QUALIFIED_AWAIT_TOOL in access.allowed_tools


def test_without_a_venue_no_order_tool_is_allowed() -> None:
    access = build_tool_access(
        effective_mode=ExecutionMode.OFF,
        workspace_writes=True,
        registries=(*REGISTRIES, ORDER_WORK_REGISTRY),
    )
    assert not (ORDER_TOOLS | {QUALIFIED_WORK_TOOL}) & set(access.allowed_tools)
    assert ORDER_TOOLS | {QUALIFIED_WORK_TOOL} <= set(access.disallowed_tools)


@pytest.mark.parametrize(("mode", "writes"), list(itertools.product(ExecutionMode, [True, False])))
def test_invariants_in_every_mode(mode: ExecutionMode, writes: bool) -> None:
    access = _access(mode, writes)
    allowed, disallowed = set(access.allowed_tools), set(access.disallowed_tools)
    assert not allowed & disallowed
    assert DENIED_X <= disallowed
    assert TIER_R & ALL_ROLES <= allowed
    assert TIER_R - ALL_ROLES <= disallowed  # no role uses them (e.g. trusted-only get_accounts)
    assert set(DISALLOWED_BUILTINS) <= disallowed
    assert set(SESSION_BUILTINS) <= allowed
    assert list(access.allowed_tools) == sorted(access.allowed_tools)
    assert "Task" not in disallowed  # the CLI alias of Agent: disallowing it disables Agent
    if writes:
        assert TIER_S <= allowed
    else:
        assert TIER_S <= disallowed


@pytest.mark.parametrize(
    ("requested", "armed"), list(itertools.product(ExecutionMode, [True, False]))
)
def test_settings_path_exposes_order_tools_only_when_armed_live(
    requested: ExecutionMode, armed: bool
) -> None:
    mode = effective_execution_mode(requested, armed=armed, ceiling=PHASE_EXECUTION_CEILING)
    exposed = bool(ORDER_TOOLS & set(_access(mode).allowed_tools))
    assert exposed is (requested is ExecutionMode.LIVE and armed)


def test_unregistered_tool_is_in_neither_list() -> None:
    access = _access(ExecutionMode.LIVE)
    name = "mcp__robinhood__brand_new_tool"
    assert name not in access.allowed_tools
    assert name not in access.disallowed_tools


@pytest.mark.parametrize("mode", list(ExecutionMode))
def test_without_mignons_only_the_orchestrator_tools_remain(mode: ExecutionMode) -> None:
    access = build_tool_access(
        effective_mode=mode, workspace_writes=True, registries=REGISTRIES, mignons=False
    )
    assert set(access.allowed_tools) <= ROLE_TOOLS[Role.ORCHESTRATOR]
    assert {"Agent", "WebSearch", "WebFetch"} <= set(access.disallowed_tools)


@pytest.mark.parametrize("mode", list(ExecutionMode))
def test_web_research_is_tavily_for_mignons_only(mode: ExecutionMode) -> None:
    """ADR-0058: the built-in web tools are disabled; Tavily search/extract are allowed for
    the company and macro Mignons, and its credit-heavy tools never are."""
    access = build_tool_access(
        effective_mode=mode, workspace_writes=True, registries=(*REGISTRIES, TAVILY_REGISTRY)
    )
    assert {"WebSearch", "WebFetch"} <= set(access.disallowed_tools)
    web = {"mcp__tavily__tavily_search", "mcp__tavily__tavily_extract"}
    assert web <= set(access.allowed_tools)
    excluded = {f"mcp__tavily__tavily_{t}" for t in ("crawl", "map", "research")}
    assert excluded <= set(access.disallowed_tools)
    assert not web & ROLE_TOOLS[Role.ORCHESTRATOR]
    assert not web & ROLE_TOOLS[Role.MARKET]
    assert web <= ROLE_TOOLS[Role.COMPANY] and web <= ROLE_TOOLS[Role.MACRO]


SCAN_TOOLS = {
    f"mcp__robinhood__{t}"
    for t in ("get_scans", "create_scan", "update_scan_filters", "update_scan_config")
}


def test_close_agent_sees_no_scan_tools_but_keeps_orders() -> None:
    """ADR-0059: the Buy-to-Close session does not list scan tools; the model never sees them."""
    access = build_tool_access(
        effective_mode=ExecutionMode.LIVE,
        workspace_writes=True,
        registries=REGISTRIES,
        agent=AgentRole.CLOSE,
    )
    assert not SCAN_TOOLS & set(access.allowed_tools)
    assert SCAN_TOOLS <= set(access.disallowed_tools)
    assert "mcp__robinhood__cancel_option_order" in access.allowed_tools
    assert not EXECUTOR_ONLY & set(access.allowed_tools)  # ADR-0066: code reviews and places
    # A Mignon's scanner stays: run_scan is the market Mignon's, not the orchestrator's.
    assert "mcp__robinhood__run_scan" in access.allowed_tools


@pytest.mark.parametrize("agent", [AgentRole.SELL, AgentRole.WHEEL])
def test_sell_and_legacy_agents_keep_scan_tools(agent: AgentRole) -> None:
    access = build_tool_access(
        effective_mode=ExecutionMode.LIVE, workspace_writes=True, registries=REGISTRIES, agent=agent
    )
    assert SCAN_TOOLS <= set(access.allowed_tools)


def test_agent_orchestrator_sets_are_within_the_orchestrator_role() -> None:
    for agent in AgentRole:
        assert AGENT_ORCHESTRATOR_TOOLS[agent] <= ROLE_TOOLS[Role.ORCHESTRATOR]
    assert ROLE_TOOLS[Role.ORCHESTRATOR] - AGENT_ORCHESTRATOR_TOOLS[AgentRole.CLOSE] == SCAN_TOOLS
