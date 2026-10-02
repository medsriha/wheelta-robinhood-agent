"""Orchestrator/Mignon role table and limits (ADR-0025, agent/mignons.py)."""

import pytest

from wheelta_robinhood_agent.agent.account_scope import (
    ROBINHOOD_ACCOUNT_SCOPE,
    AccountScope,
)
from wheelta_robinhood_agent.agent.facts_tool import FACTS_TOOL_NAME
from wheelta_robinhood_agent.agent.local_server import LOCAL_REGISTRY
from wheelta_robinhood_agent.agent.mignons import (
    DELEGATION_TOOL,
    ROLE_TOOLS,
    WEB_TOOLS,
    MignonLimits,
    Role,
    agent_name,
    cli_env,
    mignon_limits,
    parse_agent_name,
    role_of,
)
from wheelta_robinhood_agent.agent.order_walk import (
    ORDER_WORK_REGISTRY,
    ORDER_WORK_SERVER,
    QUALIFIED_AWAIT_TOOL,
    QUALIFIED_WORK_TOOL,
)
from wheelta_robinhood_agent.config.rules import RuleMarker, load_rules
from wheelta_robinhood_agent.domain.enums import MignonType, ToolTier
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

REGISTRIES = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY, LOCAL_REGISTRY, ORDER_WORK_REGISTRY)
KNOWN = {r.qualified(t.name): (r, t) for r in REGISTRIES for t in r.tools} | {
    name: None for name in (DELEGATION_TOOL, *WEB_TOOLS)
}
MIGNON_ROLES = [Role(m.value) for m in MignonType]
RULES = load_rules().rules


@pytest.mark.parametrize("role", list(Role))
def test_every_role_tool_exists(role: Role) -> None:
    assert ROLE_TOOLS[role] <= set(KNOWN), ROLE_TOOLS[role] - set(KNOWN)


@pytest.mark.parametrize("role", MIGNON_ROLES)
def test_mignons_hold_no_account_state_actions_or_delegation(role: Role) -> None:
    for name in ROLE_TOOLS[role]:
        entry = KNOWN[name]
        assert name != DELEGATION_TOOL
        if entry is None:
            continue
        registry, spec = entry
        assert spec.tier is ToolTier.R, name
        if registry is ROBINHOOD_REGISTRY:
            scope = ROBINHOOD_ACCOUNT_SCOPE[spec.name].scope
            assert scope in (AccountScope.NOT_ACCOUNT_SCOPED, AccountScope.LOGIN_SCOPED), name
    assert f"mcp__wra_local__{FACTS_TOOL_NAME}" not in ROLE_TOOLS[role]


@pytest.mark.parametrize("role", MIGNON_ROLES)
def test_every_mignon_holds_every_research_read(role: Role) -> None:
    """ADR-0068: each Mignon may call any non-account Robinhood read and any Wheelta tool."""
    for name, spec in ROBINHOOD_ACCOUNT_SCOPE.items():
        if spec.scope is AccountScope.NOT_ACCOUNT_SCOPED:
            assert ROBINHOOD_REGISTRY.qualified(name) in ROLE_TOOLS[role], name
    for spec in WHEELTA_REGISTRY.tools:
        assert WHEELTA_REGISTRY.qualified(spec.name) in ROLE_TOOLS[role], spec.name


def test_orchestrator_reads_no_web_and_delegates() -> None:
    tools = ROLE_TOOLS[Role.ORCHESTRATOR]
    assert DELEGATION_TOOL in tools
    assert not set(WEB_TOOLS) & tools
    assert "mcp__wra_local__web_cache_lookup" not in tools
    assert not any(t.startswith("mcp__wheelta__") for t in tools)
    assert f"mcp__wra_local__{FACTS_TOOL_NAME}" in tools  # the spelled-out name matches


# ADR-0066: only the order-walk executor reviews and places; no role holds them.
EXECUTOR_ONLY = frozenset({"review_option_order", "place_option_order"})


def test_every_account_scoped_read_belongs_to_the_orchestrator_only() -> None:
    for name, spec in ROBINHOOD_ACCOUNT_SCOPE.items():
        if spec.scope is AccountScope.VERIFIED and name not in EXECUTOR_ONLY:
            qualified = ROBINHOOD_REGISTRY.qualified(name)
            assert qualified in ROLE_TOOLS[Role.ORCHESTRATOR], name


def test_review_and_place_belong_to_no_role() -> None:
    for role in Role:
        for name in EXECUTOR_ONLY:
            assert ROBINHOOD_REGISTRY.qualified(name) not in ROLE_TOOLS[role], (role, name)


def test_orchestrator_holds_the_order_work_tools_and_mignons_do_not() -> None:
    """The spelled-out names in agent/mignons.py match agent/order_walk.py."""
    assert {QUALIFIED_WORK_TOOL, QUALIFIED_AWAIT_TOOL} <= ROLE_TOOLS[Role.ORCHESTRATOR]
    for role in MIGNON_ROLES:
        assert not any(f"__{ORDER_WORK_SERVER}__" in t for t in ROLE_TOOLS[role])


def test_role_of_accepts_only_mignon_types_on_allowed_models() -> None:
    models = ("claude-opus-4-8",)
    assert role_of("mignon-macro--claude-opus-4-8", models) is Role.MACRO
    for other in (
        "mignon-macro",
        "mignon-macro--claude-haiku-4-5",
        "mignon-macro--opus",
        "orchestrator--claude-opus-4-8",
        "general-purpose",
        None,
        3,
    ):
        assert role_of(other, models) is None, other


def test_agent_names_round_trip() -> None:
    for mignon in MignonType:
        assert parse_agent_name(agent_name(mignon, "claude-sonnet-5")) == (
            mignon,
            "claude-sonnet-5",
        )
    assert parse_agent_name("mignon-market--Claude Opus") is None


def test_limits_come_from_the_rules() -> None:
    assert mignon_limits(RULES) == MignonLimits(12, 4, 40)


@pytest.mark.parametrize("value", [RuleMarker.TBD, RuleMarker.NONE, RuleMarker.AGENT_DISCRETION, 0])
@pytest.mark.parametrize("key", ["max_per_run", "max_concurrent", "max_turns_per_mignon"])
def test_any_non_positive_or_marker_limit_disables_mignons(key: str, value: object) -> None:
    rules = RULES.model_copy(update={"mignons": RULES.mignons.model_copy(update={key: value})})
    assert mignon_limits(rules) is None


def test_cli_env_always_disables_background_and_builtin_agents() -> None:
    assert cli_env(None) == {
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
    }
    assert cli_env(MignonLimits(8, 3, 40))["CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS"] == "3"
