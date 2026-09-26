"""Build the `ClaudeAgentOptions` for the single agent session (CLAUDE.md §8, ADR-0001, ADR-0006).

Pure: builds configuration only, no I/O. The session:

- sees only `tool_access` (layers 1 and 2), with `permission_mode="dontAsk"`, never
  `bypassPermissions` (it ignores `allowed_tools`);
- has built-ins restricted to WebSearch/WebFetch through `tools`;
- loads no filesystem settings or CLAUDE.md (`setting_sources=[]`) and only the MCP servers
  given here (`strict_mcp_config=True`): remote HTTP servers plus, optionally, in-process SDK
  servers named in `LOCAL_SDK_SERVER_NAMES` (`wra_local`: web_cache_lookup and
  get_decision_facts). No other server type (stdio, SSE) is ever configured;
- runs in an explicit scratch directory;
- uses the pinned model (`Settings.AGENT_MODEL`) and the given hooks (layer 3).

Field names verified against claude-agent-sdk 0.2.160 `types.py` `ClaudeAgentOptions`.
"""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal, cast

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from claude_agent_sdk.types import (
    HookEvent,
    McpHttpServerConfig,
    McpSdkServerConfig,
    McpServerConfig,
)

from wheelta_robinhood_agent.agent.tool_access import (
    ALLOWED_BUILTINS,
    DISALLOWED_BUILTINS,
    ToolAccess,
)
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.integrations.status import McpHttpServer

PERMISSION_MODE: Final[Literal["dontAsk"]] = "dontAsk"
# The only in-process SDK MCP servers a session may carry. Their tools are local Tier R code.
LOCAL_SDK_SERVER_NAMES: Final = frozenset({LOCAL_SERVER_NAME})
_REQUIRED_HOOK_EVENTS: tuple[HookEvent, ...] = ("PreToolUse", "PostToolUse", "PostToolUseFailure")


class AgentOptionsError(ValueError):
    """The options would violate the session's safety contract."""


def _sdk_server(server: McpHttpServer) -> McpHttpServerConfig:
    config = server.to_sdk_config()
    return McpHttpServerConfig(
        type="http",
        url=str(config["url"]),
        headers=cast(dict[str, str], config["headers"]),
    )


def build_agent_options(
    *,
    tool_access: ToolAccess,
    mcp_servers: Sequence[McpHttpServer],
    hooks: dict[HookEvent, list[HookMatcher]],
    system_prompt: str,
    model: str,
    scratch_dir: Path,
    max_turns: int,
    max_budget_usd: Decimal | None,
    mcp_timeout_ms: int,
    mcp_tool_timeout_ms: int,
    sdk_servers: Mapping[str, McpSdkServerConfig] | None = None,
) -> ClaudeAgentOptions:
    """Assemble the session options. Raises `AgentOptionsError` on any unsafe input.

    `max_budget_usd` is a Decimal in our code and converted to the SDK's float only here.
    `sdk_servers` maps a name in `LOCAL_SDK_SERVER_NAMES` to its `create_sdk_mcp_server`
    config; the name must match the config's own name and no HTTP server may share it.
    """
    if not model.strip():
        raise AgentOptionsError("model must be pinned")
    if not system_prompt.strip():
        raise AgentOptionsError("system prompt is empty")
    if not scratch_dir.is_absolute():
        raise AgentOptionsError("scratch_dir must be absolute")
    if max_turns < 1:
        raise AgentOptionsError("max_turns must be positive")
    if max_budget_usd is not None and not (max_budget_usd.is_finite() and max_budget_usd > 0):
        raise AgentOptionsError("max_budget_usd must be positive")
    if mcp_timeout_ms < 1 or mcp_tool_timeout_ms < 1:
        raise AgentOptionsError("MCP timeouts must be positive")
    missing_hooks = [e for e in _REQUIRED_HOOK_EVENTS if not hooks.get(e)]
    if missing_hooks:
        raise AgentOptionsError(f"hooks missing for {missing_hooks}")
    allowed = set(tool_access.allowed_tools)
    if allowed & set(tool_access.disallowed_tools):
        raise AgentOptionsError("a tool is both allowed and disallowed")
    builtins_allowed = {t for t in allowed if not t.startswith("mcp__")}
    if not builtins_allowed <= set(ALLOWED_BUILTINS):
        raise AgentOptionsError(f"built-ins beyond {ALLOWED_BUILTINS} are allowed")
    if not set(DISALLOWED_BUILTINS) <= set(tool_access.disallowed_tools):
        raise AgentOptionsError("not every unneeded built-in is disallowed")
    names = [s.name for s in mcp_servers]
    if len(names) != len(set(names)):
        raise AgentOptionsError("duplicate MCP server names")
    local = dict(sdk_servers or {})
    if set(local) & set(names):
        raise AgentOptionsError("an SDK server shares a name with an HTTP server")

    servers: dict[str, McpServerConfig] = {s.name: _sdk_server(s) for s in mcp_servers}
    servers.update(local)
    options = ClaudeAgentOptions(
        tools=list(ALLOWED_BUILTINS),
        allowed_tools=list(tool_access.allowed_tools),
        disallowed_tools=list(tool_access.disallowed_tools),
        permission_mode=PERMISSION_MODE,
        setting_sources=[],
        strict_mcp_config=True,
        mcp_servers=servers,
        hooks=hooks,
        system_prompt=system_prompt,
        model=model,
        cwd=scratch_dir,
        max_turns=max_turns,
        max_budget_usd=float(max_budget_usd) if max_budget_usd is not None else None,
        env={"MCP_TIMEOUT": str(mcp_timeout_ms), "MCP_TOOL_TIMEOUT": str(mcp_tool_timeout_ms)},
    )
    assert_safe_options(options)
    return options


def assert_safe_options(options: ClaudeAgentOptions) -> None:
    """Re-check the built options: dontAsk, no bypass, isolated settings, restricted tools."""
    if options.permission_mode == "bypassPermissions":
        raise AgentOptionsError("bypassPermissions is never allowed")
    if options.permission_mode != PERMISSION_MODE:
        raise AgentOptionsError(f"permission_mode must be {PERMISSION_MODE}")
    if options.setting_sources != []:
        raise AgentOptionsError("filesystem settings must not load")
    if options.tools != list(ALLOWED_BUILTINS):
        raise AgentOptionsError("built-in tools must be limited to WebSearch/WebFetch")
    if options.can_use_tool is not None or options.skills is not None or options.agents:
        raise AgentOptionsError("no permission callback, skills, or sub-agents")
    if not isinstance(options.mcp_servers, dict):
        raise AgentOptionsError("MCP servers must be given inline, never as a config file path")
    for name, config in options.mcp_servers.items():
        _check_server(name, config)


def _check_server(name: str, config: McpServerConfig) -> None:
    """Only remote HTTP servers and the named local SDK servers; nothing that spawns a process."""
    kind = config.get("type")
    if kind == "http":
        if not isinstance(config.get("url"), str):
            raise AgentOptionsError(f"HTTP server {name!r} has no URL")
        return
    if kind == "sdk":
        if name not in LOCAL_SDK_SERVER_NAMES or config.get("name") != name:
            raise AgentOptionsError(f"SDK server {name!r} is not an allowed local server")
        if config.get("instance") is None:
            raise AgentOptionsError(f"SDK server {name!r} has no instance")
        return
    raise AgentOptionsError(f"MCP server {name!r} has a disallowed type {kind!r}")
