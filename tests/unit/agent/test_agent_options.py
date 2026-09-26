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

from wheelta_robinhood_agent.agent.options import (
    AgentOptionsError,
    assert_safe_options,
    build_agent_options,
)
from wheelta_robinhood_agent.agent.tool_access import ToolAccess, build_tool_access
from wheelta_robinhood_agent.domain.enums import ExecutionMode
from wheelta_robinhood_agent.domain.gating import effective_execution_mode
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    LIVE_ORDER_TOOLS,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.status import McpHttpServer
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

REGISTRIES = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY)
ORDER_TOOLS = {ROBINHOOD_REGISTRY.qualified(n) for n in LIVE_ORDER_TOOLS}
SCRATCH = Path("/private/tmp/wra-scratch")


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
    }
    kwargs.update(overrides)
    return build_agent_options(**kwargs)


def test_options_contract() -> None:
    o = _build()
    assert o.permission_mode == "dontAsk"
    assert o.setting_sources == []
    assert o.strict_mcp_config is True
    assert o.tools == ["WebSearch", "WebFetch"]
    assert o.model == "claude-test-model"
    assert o.cwd == SCRATCH
    assert o.max_turns == 40 and o.max_budget_usd == 2.5
    assert o.hooks is HOOKS
    assert o.system_prompt == "rendered wheel_agent prompt"
    assert o.env == {"MCP_TIMEOUT": "30000", "MCP_TOOL_TIMEOUT": "60000"}
    assert o.can_use_tool is None and o.skills is None and not o.agents
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
        ({"tools": ["WebSearch", "WebFetch", "Bash"]}, "built-in"),
        ({"skills": "all"}, "skills"),
    ],
)
def test_assert_safe_options_rejects_unsafe(change: dict[str, Any], fragment: str) -> None:
    unsafe = dataclasses.replace(_build(), **change)
    with pytest.raises(AgentOptionsError, match=fragment):
        assert_safe_options(unsafe)
