"""build_agent_options (CLAUDE.md §8).

Named test_agent_options.py because tests/unit/test_options.py already exists and
pytest's default import mode rejects duplicate test module basenames."""

import dataclasses
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import HookMatcher
from pydantic import SecretStr

from wheelta_robinhood_agent.agent.mignons import MignonLimits, Role, role_allowed
from wheelta_robinhood_agent.agent.options import (
    AgentOptionsError,
    assert_safe_options,
    build_agent_options,
)
from wheelta_robinhood_agent.agent.tool_access import ToolAccess, build_tool_access
from wheelta_robinhood_agent.domain.enums import ExecutionMode, MignonType
from wheelta_robinhood_agent.domain.gating import effective_execution_mode
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    LIVE_ORDER_TOOLS,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.status import McpHttpServer
from wheelta_robinhood_agent.integrations.websearch.registry import TAVILY_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

REGISTRIES = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY, TAVILY_REGISTRY)
ORDER_TOOLS = {ROBINHOOD_REGISTRY.qualified(n) for n in LIVE_ORDER_TOOLS}
SCRATCH = Path("/private/tmp/wra-scratch")
PROMPTS = {m: f"rendered {m.value} prompt" for m in MignonType}
LIMITS = MignonLimits(max_per_run=8, max_concurrent=4, max_turns_per_mignon=40)
MK = "mignon-market--claude-test-model"  # the default: every Mignon on the session model


async def _noop(*_: Any) -> Any:
    return {}


HOOKS: Any = {
    "PreToolUse": [HookMatcher(hooks=[_noop])],
    "PostToolUse": [HookMatcher(hooks=[_noop])],
    "PostToolUseFailure": [HookMatcher(hooks=[_noop])],
}
SERVERS = [
    McpHttpServer(name="robinhood", url="https://rh.example/mcp", token=SecretStr("rh-token")),  # type: ignore[arg-type]
    McpHttpServer(name="wheelta", url="https://wt.example/mcp", token=SecretStr("wt-token")),  # type: ignore[arg-type]
]


def _access(mode: ExecutionMode) -> ToolAccess:
    return build_tool_access(effective_mode=mode, workspace_writes=True, registries=REGISTRIES)


def _build(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "tool_access": _access(ExecutionMode.OFF),
        "mcp_servers": SERVERS,
        "hooks": HOOKS,
        "system_prompt": "rendered wheel_agent prompt",
        "model": "claude-test-model",
        "scratch_dir": SCRATCH,
        "max_turns": 40,
        "max_budget_usd": Decimal("2.50"),
        "mcp_timeout_ms": 30000,
        "mcp_tool_timeout_ms": 60000,
        "mignon_prompts": PROMPTS,
        "mignon_limits": LIMITS,
    }
    kwargs.update(overrides)
    return build_agent_options(**kwargs)


def test_options_contract() -> None:
    o = _build()
    assert o.permission_mode == "dontAsk"
    assert o.setting_sources == []
    assert o.strict_mcp_config is True
    assert o.tools == ["Agent"]
    assert o.model == "claude-test-model"
    assert o.cwd == SCRATCH
    assert o.max_turns == 40 and o.max_budget_usd == 2.5
    assert o.hooks is HOOKS
    assert o.system_prompt == "rendered wheel_agent prompt"
    assert o.env == {
        "MCP_TIMEOUT": "30000",
        "MCP_TOOL_TIMEOUT": "60000",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
        "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": "4",
    }
    assert o.can_use_tool is None and o.skills is None
    assert set(o.agents) == {f"{m.value}--claude-test-model" for m in MignonType}
    assert o.mcp_servers == {
        "robinhood": {
            "type": "http",
            "url": "https://rh.example/mcp",
            "headers": {"Authorization": "Bearer rh-token"},
        },
        "wheelta": {
            "type": "http",
            "url": "https://wt.example/mcp",
            "headers": {"Authorization": "Bearer wt-token"},
        },
    }


def test_no_budget_is_allowed() -> None:
    assert _build(max_budget_usd=None).max_budget_usd is None


@pytest.mark.parametrize(
    "mode",
    [
        ExecutionMode.OFF,
        effective_execution_mode(ExecutionMode.LIVE, armed=False, ceiling=ExecutionMode.LIVE),
    ],
)
def test_order_tools_absent_in_off_and_unarmed(mode: ExecutionMode) -> None:
    assert mode is ExecutionMode.OFF
    o = _build(tool_access=_access(mode))
    assert not ORDER_TOOLS & set(o.allowed_tools)
    assert ORDER_TOOLS <= set(o.disallowed_tools)


def test_order_tools_allowed_only_in_armed_live() -> None:
    o = _build(tool_access=_access(ExecutionMode.LIVE))
    assert ORDER_TOOLS <= set(o.allowed_tools)
    assert "mcp__robinhood__place_equity_order" in o.disallowed_tools


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"model": " "}, "model"),
        ({"system_prompt": ""}, "prompt"),
        ({"scratch_dir": Path("relative")}, "absolute"),
        ({"max_turns": 0}, "max_turns"),
        ({"max_budget_usd": Decimal("0")}, "budget"),
        ({"max_budget_usd": Decimal("NaN")}, "budget"),
        ({"mcp_timeout_ms": 0}, "timeouts"),
        ({"mcp_tool_timeout_ms": 0}, "timeouts"),
        ({"hooks": {"PreToolUse": HOOKS["PreToolUse"]}}, "hooks missing"),
        ({"mcp_servers": [SERVERS[0], SERVERS[0]]}, "duplicate"),
    ],
)
def test_invalid_inputs_rejected(override: dict[str, Any], fragment: str) -> None:
    with pytest.raises(AgentOptionsError, match=fragment):
        _build(**override)


def test_unsafe_tool_access_rejected() -> None:
    base = _access(ExecutionMode.OFF)
    both = base.model_copy(update={"allowed_tools": (*base.allowed_tools, "Bash")})
    with pytest.raises(AgentOptionsError, match="both allowed and disallowed"):
        _build(tool_access=both)
    extra = base.model_copy(update={"allowed_tools": (*base.allowed_tools, "Glob2")})
    with pytest.raises(AgentOptionsError, match="built-ins beyond"):
        _build(tool_access=extra)
    missing = base.model_copy(
        update={"disallowed_tools": tuple(t for t in base.disallowed_tools if t != "Bash")}
    )
    with pytest.raises(AgentOptionsError, match="unneeded built-in"):
        _build(tool_access=missing)


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"permission_mode": "bypassPermissions"}, "bypassPermissions"),
        ({"permission_mode": "default"}, "dontAsk"),
        ({"setting_sources": None}, "settings"),
        ({"tools": ["Agent", "WebSearch", "WebFetch"]}, "built-in"),
        ({"skills": "all"}, "skills"),
    ],
)
def test_assert_safe_options_rejects_unsafe(change: dict[str, Any], fragment: str) -> None:
    unsafe = dataclasses.replace(_build(), **change)
    with pytest.raises(AgentOptionsError, match=fragment):
        assert_safe_options(unsafe)


def test_mignon_definitions_follow_their_roles() -> None:
    o = _build()
    for mignon in MignonType:
        d = o.agents[f"{mignon.value}--claude-test-model"]
        assert d.prompt == PROMPTS[mignon]
        assert tuple(d.tools) == role_allowed(Role(mignon.value), o.allowed_tools)
        assert d.model == o.model and d.maxTurns == 40 and d.background is False
        assert not {"Agent", *ORDER_TOOLS} & set(d.tools)
        assert not any("get_option_positions" in t or "get_portfolio" in t for t in d.tools)
    assert "mcp__tavily__tavily_search" not in o.agents[MK].tools
    assert "mcp__tavily__tavily_extract" in o.agents["mignon-company--claude-test-model"].tools
    assert not {"WebSearch", "WebFetch"} & {t for d in o.agents.values() for t in d.tools}


def test_disabled_mignons_leave_no_agents_and_no_web() -> None:
    access = build_tool_access(
        effective_mode=ExecutionMode.OFF,
        workspace_writes=True,
        registries=REGISTRIES,
        mignons=False,
    )
    o = _build(tool_access=access, mignon_prompts=None, mignon_limits=None)
    assert not o.agents
    assert {"Agent", "WebSearch", "WebFetch"} <= set(o.disallowed_tools)
    assert o.env["CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS"] == "1"
    assert "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS" not in o.env


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"mignon_prompts": None}, "no Mignons are configured"),
        ({"mignon_limits": None}, "no Mignons are configured"),
        ({"mignon_prompts": {MignonType.MARKET: "p"}}, "every Mignon type"),
        ({"mignon_prompts": {**PROMPTS, MignonType.MACRO: " "}}, "every Mignon type"),
    ],
)
def test_mignon_configuration_rejected(override: dict[str, Any], fragment: str) -> None:
    with pytest.raises(AgentOptionsError, match=fragment):
        _build(**override)


def _with_agent(o: Any, name: str, **changes: Any) -> Any:
    agents = dict(o.agents)
    agents[name] = dataclasses.replace(agents.get(name) or agents[MK], **changes)
    return dataclasses.replace(o, agents=agents)


@pytest.mark.parametrize(
    ("name", "changes", "fragment"),
    [
        ("general-purpose", {}, "not a Mignon type"),
        ("mignon-market", {}, "not a Mignon type"),
        (MK, {"tools": ["mcp__tavily__tavily_search"]}, "tools differ"),
        (MK, {"tools": None}, "tools differ"),
        (MK, {"model": "inherit"}, "pinned model its name names"),
        (MK, {"model": "opus"}, "pinned model its name names"),
        (MK, {"model": None}, "pinned model its name names"),
        (MK, {"model": "claude-other"}, "pinned model its name names"),
        (MK, {"background": True}, "background"),
        (MK, {"background": None}, "background"),
        (MK, {"maxTurns": None}, "turn cap"),
        (MK, {"permissionMode": "bypassPermissions"}, "overrides"),
        (MK, {"mcpServers": ["other"]}, "inline loopback servers"),
        (MK, {"skills": ["x"]}, "overrides"),
    ],
)
def test_assert_safe_options_rejects_unsafe_mignons(
    name: str, changes: dict[str, Any], fragment: str
) -> None:
    with pytest.raises(AgentOptionsError, match=fragment):
        assert_safe_options(_with_agent(_build(), name, **changes))


@pytest.mark.parametrize(
    "key", ["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS"]
)
def test_assert_safe_options_requires_the_cli_variables(key: str) -> None:
    o = _build()
    env = {k: v for k, v in o.env.items() if k != key}
    with pytest.raises(AgentOptionsError, match=key):
        assert_safe_options(dataclasses.replace(o, env=env))


def test_agents_without_the_agent_tool_rejected() -> None:
    o = _build()
    allowed = [t for t in o.allowed_tools if t != "Agent"]
    with pytest.raises(AgentOptionsError, match="Agent is not allowed"):
        assert_safe_options(dataclasses.replace(o, allowed_tools=allowed))


def test_each_mignon_type_is_offered_once_per_allowed_model() -> None:
    o = _build(mignon_models=("claude-haiku-4-5", "claude-opus-4-8"))
    assert o.model == "claude-test-model"
    assert set(o.agents) == {
        f"{m.value}--{model}"
        for m in MignonType
        for model in ("claude-haiku-4-5", "claude-opus-4-8")
    }
    for name, d in o.agents.items():
        assert d.model == name.split("--", 1)[1]
    haiku = o.agents["mignon-company--claude-haiku-4-5"]
    assert haiku.tools == o.agents["mignon-company--claude-opus-4-8"].tools
    assert haiku.description.startswith("mignon-company on claude-haiku-4-5")


# ---- ADR-0063: Mignon-only tools on inline loopback servers -----------------------------------

BEARER = {"Authorization": "Bearer " + "t" * 43}


def _inline(role: Role, source: str, **changes: Any) -> Any:
    config: dict[str, Any] = {
        "type": "http",
        "url": f"http://127.0.0.1:41234/{role.value}/{source}/mcp",
        "headers": BEARER,
    }
    config.update(changes)
    return config


def _mignon_servers() -> Any:
    return {
        Role.MARKET: {"robinhood": _inline(Role.MARKET, "robinhood")},
        Role.COMPANY: {
            "tavily": _inline(Role.COMPANY, "tavily"),
            "robinhood": _inline(Role.COMPANY, "robinhood"),
        },
    }


def test_each_mignon_carries_its_roles_inline_servers_sorted() -> None:
    o = _build(mignon_mcp_servers=_mignon_servers())
    company = o.agents["mignon-company--claude-test-model"].mcpServers
    assert company == [
        {"robinhood": _inline(Role.COMPANY, "robinhood")},
        {"tavily": _inline(Role.COMPANY, "tavily")},
    ]
    assert o.agents[MK].mcpServers == [{"robinhood": _inline(Role.MARKET, "robinhood")}]
    assert o.agents["mignon-macro--claude-test-model"].mcpServers is None
    # Inline servers never enter the session's own server list.
    assert set(o.mcp_servers) == {"robinhood", "wheelta"}


def test_mignon_servers_without_mignons_are_rejected() -> None:
    with pytest.raises(AgentOptionsError, match="no Mignons are configured"):
        _build(mignon_mcp_servers=_mignon_servers(), mignon_limits=None, mignon_prompts=None)


@pytest.mark.parametrize(
    ("servers", "fragment"),
    [
        ([], "empty MCP server list"),
        (["robinhood"], "inline loopback servers"),
        ([{"robinhood": _inline(Role.MARKET, "robinhood"), "x": {}}], "inline loopback"),
        ([{"wra_local": _inline(Role.MARKET, "wra_local")}], "not a proxied source"),
        ([{"other": _inline(Role.MARKET, "other")}], "not a proxied source"),
        (
            [
                {"robinhood": _inline(Role.MARKET, "robinhood")},
                {"robinhood": _inline(Role.MARKET, "robinhood")},
            ],
            "not a proxied source",
        ),
        ([{"robinhood": {"type": "sdk", "name": "robinhood"}}], "malformed"),
        ([{"robinhood": {"type": "stdio", "command": "x", "url": "", "headers": {}}}], "mal"),
        ([{"robinhood": _inline(Role.MARKET, "robinhood", type="sse")}], "loopback URL"),
        (
            [{"robinhood": _inline(Role.MARKET, "robinhood", url="https://rh.example/mcp")}],
            "loopback URL",
        ),
        (
            [
                {
                    "robinhood": _inline(
                        Role.MARKET, "robinhood", url="http://localhost:1/market/robinhood/mcp"
                    )
                }
            ],
            "loopback URL",
        ),
        (
            [
                {
                    "robinhood": _inline(
                        Role.MARKET, "robinhood", url="http://127.0.0.1/market/robinhood/mcp"
                    )
                }
            ],
            "loopback URL",
        ),
        ([{"robinhood": _inline(Role.COMPANY, "robinhood")}], "loopback URL"),  # another role
        ([{"robinhood": _inline(Role.MARKET, "wheelta")}], "loopback URL"),  # another source
        (
            [
                {
                    "robinhood": _inline(
                        Role.MARKET,
                        "robinhood",
                        url="http://127.0.0.1:1/market/robinhood/mcp?x=1",
                    )
                }
            ],
            "loopback URL",
        ),
        (
            [
                {
                    "robinhood": _inline(
                        Role.MARKET,
                        "robinhood",
                        url="http://u@127.0.0.1:1/market/robinhood/mcp",
                    )
                }
            ],
            "loopback URL",
        ),
        ([{"robinhood": _inline(Role.MARKET, "robinhood", headers={})}], "bearer header"),
        (
            [{"robinhood": _inline(Role.MARKET, "robinhood", headers={"Authorization": "x"})}],
            "bearer header",
        ),
        (
            [{"robinhood": _inline(Role.MARKET, "robinhood", headers={**BEARER, "X-A": "1"})}],
            "bearer header",
        ),
    ],
)
def test_assert_safe_options_accepts_only_the_roles_loopback_servers(
    servers: Any, fragment: str
) -> None:
    with pytest.raises(AgentOptionsError, match=fragment):
        assert_safe_options(_with_agent(_build(), MK, mcpServers=servers))


def test_the_roles_loopback_server_passes_the_check() -> None:
    assert_safe_options(
        _with_agent(_build(), MK, mcpServers=[{"robinhood": _inline(Role.MARKET, "robinhood")}])
    )
