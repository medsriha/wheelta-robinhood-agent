"""Which tools the agent session may see, per effective execution mode (CLAUDE.md §8, §18).

Layers 1 and 2 of tool access: `disallowed_tools` removes denied tools from the model's
context, and `allowed_tools` is the explicit allowlist used with `permission_mode="dontAsk"`.
Layer 3 (the PreToolUse hook) re-checks every call. The prompt never decides any of this.
"""

from pydantic import BaseModel, ConfigDict

from wheelta_robinhood_agent.agent.mignons import DELEGATION_TOOL, ROLE_TOOLS, Role
from wheelta_robinhood_agent.domain.enums import ExecutionMode, ToolTier
from wheelta_robinhood_agent.integrations.registry import ToolRegistry

# Built-in Agent SDK tools the session never needs (CLAUDE.md §8). It needs MCP tools, web
# search/fetch (Mignons only), and `Agent` to spawn Mignons (ADR-0025). `Task` is the CLI's
# alias of `Agent`: listing it here would disable `Agent` as well (real CLI 2.1.283), so it
# is left out; it has no tier and the hook denies it. `SendMessage` (resume a subagent) is
# denied: follow-ups spawn a new Mignon.
DISALLOWED_BUILTINS = (
    "Bash",
    "BashOutput",
    "Edit",
    "Glob",
    "Grep",
    "KillShell",
    "MultiEdit",
    "NotebookEdit",
    "Read",
    "SendMessage",
    "TodoWrite",
    "Write",
)
# Research built-ins (Tier R): used by Mignons only.
ALLOWED_BUILTINS = ("WebSearch", "WebFetch")
# Every built-in the session loads (`ClaudeAgentOptions.tools`).
SESSION_BUILTINS = (DELEGATION_TOOL, *ALLOWED_BUILTINS)


class ToolAccess(BaseModel):
    """The SDK tool lists for one run. Both are sorted, so they are deterministic to record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_mode: ExecutionMode
    allowed_tools: tuple[str, ...]
    disallowed_tools: tuple[str, ...]


def build_tool_access(
    *,
    effective_mode: ExecutionMode,
    workspace_writes: bool,
    registries: tuple[ToolRegistry, ...],
    mignons: bool = True,
) -> ToolAccess:
    """Compute allowed and disallowed tools.

    By tier:
    - Tier R: allowed.
    - Tier S: allowed only if `workspace_writes` (ROBINHOOD_WORKSPACE_WRITES); else disallowed.
    - Tier X live order tools: allowed only when `effective_mode` is live (already requires
      armed and the phase ceiling); otherwise disallowed so the model never sees them.
    - Every other Tier X tool and every EXCLUDED tool: disallowed in every mode.
    By role (ADR-0025, agent/mignons.py): a tool is allowed only if some role may use it.
    With `mignons` False (their limits are not integers) only the orchestrator's tools count,
    and `Agent` and the web built-ins are disallowed.
    A tool absent from every registry is in neither list; `dontAsk` and the hook deny it.
    """
    roles = tuple(Role) if mignons else (Role.ORCHESTRATOR,)
    usable = frozenset().union(*(ROLE_TOOLS[r] for r in roles))
    if not mignons:
        usable -= {DELEGATION_TOOL}
    allowed: set[str] = {b for b in SESSION_BUILTINS if b in usable}
    disallowed: set[str] = set(DISALLOWED_BUILTINS) | (set(SESSION_BUILTINS) - allowed)
    live = effective_mode is ExecutionMode.LIVE
    for registry in registries:
        for tool in registry.tools:
            name = registry.qualified(tool.name)
            permitted = name in usable and (
                tool.tier is ToolTier.R
                or (tool.tier is ToolTier.S and workspace_writes)
                or (tool.tier is ToolTier.X and tool.live_order_tool and live)
            )
            (allowed if permitted else disallowed).add(name)
    return ToolAccess(
        effective_mode=effective_mode,
        allowed_tools=tuple(sorted(allowed)),
        disallowed_tools=tuple(sorted(disallowed)),
    )
