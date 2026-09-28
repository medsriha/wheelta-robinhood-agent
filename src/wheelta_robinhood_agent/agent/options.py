"""Build the `ClaudeAgentOptions` for the single agent session (CLAUDE.md §8, ADR-0001, ADR-0006).

Pure: builds configuration only, no I/O. The session:

- sees only `tool_access` (layers 1 and 2), with `permission_mode="dontAsk"`, never
  `bypassPermissions` (it ignores `allowed_tools`);
- has built-ins restricted to `Agent` (spawns Mignons) and WebSearch/WebFetch (Mignons only)
  through `tools`, and carries exactly the Mignon definitions built from `agent/mignons.py`
  (ADR-0025) plus the CLI variables that design depends on (`mignons.cli_env`);
- loads no filesystem settings or CLAUDE.md (`setting_sources=[]`) and only the MCP servers
  given here (`strict_mcp_config=True`): remote HTTP servers (direct delivery, ADR-0019 local
  dry runs only) plus in-process SDK servers named in `LOCAL_SDK_SERVER_NAMES` (`wra_local`:
  web_cache_lookup and get_decision_facts) or `PROXY_SDK_SERVER_NAMES` (the validating proxy
  for `robinhood` and `wheelta`, ADR-0023). No other server type (stdio, SSE) is ever
  configured;
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
    AgentDefinition,
    HookEvent,
    McpHttpServerConfig,
    McpSdkServerConfig,
    McpServerConfig,
)

from wheelta_robinhood_agent.agent.mignons import (
    DELEGATION_TOOL,
    MignonLimits,
    Role,
    build_agent_definitions,
    cli_env,
    role_allowed,
)
from wheelta_robinhood_agent.agent.tool_access import (
    DISALLOWED_BUILTINS,
    SESSION_BUILTINS,
    ToolAccess,
)
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.config.settings import MODEL_ALIASES
from wheelta_robinhood_agent.domain.enums import MignonType
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME as ROBINHOOD
from wheelta_robinhood_agent.integrations.status import McpHttpServer
from wheelta_robinhood_agent.integrations.wheelta.registry import SERVER_NAME as WHEELTA

PERMISSION_MODE: Final[Literal["dontAsk"]] = "dontAsk"
# The only in-process SDK MCP servers a session may carry. Their tools are local Tier R code.
LOCAL_SDK_SERVER_NAMES: Final = frozenset({LOCAL_SERVER_NAME})
# In-process validating proxies (agent/proxy.py) for the remote sources, under their own names.
PROXY_SDK_SERVER_NAMES: Final = frozenset({ROBINHOOD, WHEELTA})
_REQUIRED_HOOK_EVENTS: tuple[HookEvent, ...] = ("PreToolUse", "PostToolUse", "PostToolUseFailure")


class AgentOptionsError(ValueError):
    """The options would violate the session's safety contract."""


def _sdk_server(server: McpHttpServer) -> McpHttpServerConfig:
    config = server.to_sdk_config()
    if "headers" not in config:  # the CLI's stored login (ADR-0018)
        return McpHttpServerConfig(type="http", url=str(config["url"]))
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
    mignon_prompts: Mapping[MignonType, str] | None = None,
    mignon_limits: MignonLimits | None = None,
    mignon_model: str | None = None,
) -> ClaudeAgentOptions:
    """Assemble the session options. Raises `AgentOptionsError` on any unsafe input.

    `max_budget_usd` is a Decimal in our code and converted to the SDK's float only here.
    `sdk_servers` maps a name in `LOCAL_SDK_SERVER_NAMES` or `PROXY_SDK_SERVER_NAMES` to its
    in-process server config; the name must match the config's own name and no HTTP server
    may share it. Mignons are configured only with both `mignon_prompts` (one rendered prompt
    per type) and `mignon_limits`, and only when `Agent` is allowed. `mignon_model` is their
    pinned model (`Settings.mignon_model`); None means the session's `model`.
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
    if not builtins_allowed <= set(SESSION_BUILTINS):
        raise AgentOptionsError(f"built-ins beyond {SESSION_BUILTINS} are allowed")
    delegating = DELEGATION_TOOL in allowed
    if delegating and (mignon_prompts is None or mignon_limits is None):
        raise AgentOptionsError("Agent is allowed but no Mignons are configured")
    agents: dict[str, AgentDefinition] | None = None
    if delegating and mignon_prompts is not None and mignon_limits is not None:
        if set(mignon_prompts) != set(MignonType) or not all(
            p.strip() for p in mignon_prompts.values()
        ):
            raise AgentOptionsError("every Mignon type needs a non-empty rendered prompt")
        agents = build_agent_definitions(
            prompts=mignon_prompts,
            allowed_tools=tool_access.allowed_tools,
            model=mignon_model if mignon_model is not None else model,
            limits=mignon_limits,
        )
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
        tools=list(SESSION_BUILTINS),
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
        agents=agents,
        env={
            "MCP_TIMEOUT": str(mcp_timeout_ms),
            "MCP_TOOL_TIMEOUT": str(mcp_tool_timeout_ms),
            **cli_env(mignon_limits if agents else None),
        },
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
    if options.tools != list(SESSION_BUILTINS):
        raise AgentOptionsError(f"built-in tools must be exactly {SESSION_BUILTINS}")
    if options.can_use_tool is not None or options.skills is not None:
        raise AgentOptionsError("no permission callback or skills")
    for key, value in cli_env(None).items():
        if options.env.get(key) != value:
            raise AgentOptionsError(f"CLI variable {key} must be {value}")
    _check_agents(options)
    if not isinstance(options.mcp_servers, dict):
        raise AgentOptionsError("MCP servers must be given inline, never as a config file path")
    for name, config in options.mcp_servers.items():
        _check_server(name, config)


def _check_agents(options: ClaudeAgentOptions) -> None:
    """Sub-agents are only the Mignons: known types, their role's allowed tools, one shared
    pinned model ID (never an alias such as `inherit`), no permission/MCP/skill/memory
    overrides, never in background."""
    agents = options.agents or {}
    if agents and DELEGATION_TOOL not in options.allowed_tools:
        raise AgentOptionsError("sub-agents are defined but Agent is not allowed")
    for name, definition in agents.items():
        try:
            role = Role(MignonType(name).value)
        except ValueError:
            raise AgentOptionsError(f"sub-agent {name!r} is not a Mignon type") from None
        if definition.tools is None or tuple(definition.tools) != role_allowed(
            role, options.allowed_tools
        ):
            raise AgentOptionsError(f"Mignon {name} tools differ from its role")
        model = definition.model
        if not isinstance(model, str) or not model.strip() or model.lower() in MODEL_ALIASES:
            raise AgentOptionsError(f"Mignon {name} must use a pinned model ID")
        if definition.background is not False:
            raise AgentOptionsError(f"Mignon {name} must not run in the background")
        if not isinstance(definition.maxTurns, int) or definition.maxTurns < 1:
            raise AgentOptionsError(f"Mignon {name} needs a positive turn cap")
        overrides = (
            definition.disallowedTools,
            definition.mcpServers,
            definition.skills,
            definition.memory,
            definition.permissionMode,
            definition.initialPrompt,
        )
        if any(o is not None for o in overrides):
            raise AgentOptionsError(f"Mignon {name} overrides a session setting")
    if len({d.model for d in agents.values()}) > 1:
        raise AgentOptionsError("every Mignon must use the same pinned model")


def _check_server(name: str, config: McpServerConfig) -> None:
    """Only remote HTTP servers and the named in-process SDK servers; nothing that spawns a
    process."""
    kind = config.get("type")
    if kind == "http":
        if not isinstance(config.get("url"), str):
            raise AgentOptionsError(f"HTTP server {name!r} has no URL")
        return
    if kind == "sdk":
        allowed = LOCAL_SDK_SERVER_NAMES | PROXY_SDK_SERVER_NAMES
        if name not in allowed or config.get("name") != name:
            raise AgentOptionsError(f"SDK server {name!r} is not an allowed in-process server")
        if config.get("instance") is None:
            raise AgentOptionsError(f"SDK server {name!r} has no instance")
        return
    raise AgentOptionsError(f"MCP server {name!r} has a disallowed type {kind!r}")
