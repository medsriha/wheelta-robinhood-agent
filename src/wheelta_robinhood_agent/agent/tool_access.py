"""Which tools the agent session may see, per effective execution mode (CLAUDE.md §8, §18).

Layers 1 and 2 of tool access: `disallowed_tools` removes denied tools from the model's
context, and `allowed_tools` is the explicit allowlist used with `permission_mode="dontAsk"`.
Layer 3 (the PreToolUse hook) re-checks every call. The prompt never decides any of this.
"""

from pydantic import BaseModel, ConfigDict

from wheelta_robinhood_agent.domain.enums import ExecutionMode, ToolTier
from wheelta_robinhood_agent.integrations.registry import ToolRegistry

# Built-in Agent SDK tools the agent never needs (CLAUDE.md §8). The agent needs MCP tools
# and web search/fetch only.
DISALLOWED_BUILTINS = (
    "Agent",
    "Bash",
    "BashOutput",
    "Edit",
    "Glob",
    "Grep",
    "KillShell",
    "MultiEdit",
    "NotebookEdit",
    "Read",
    "Task",
    "TodoWrite",
    "Write",
)
ALLOWED_BUILTINS = ("WebSearch", "WebFetch")


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
) -> ToolAccess:
    """Compute allowed and disallowed tools.

    - Tier R: always allowed.
    - Tier S: allowed only if `workspace_writes` (ROBINHOOD_WORKSPACE_WRITES); else disallowed.
    - Tier X live order tools: allowed only when `effective_mode` is live (already requires
      armed and the phase ceiling); otherwise disallowed so the model never sees them.
    - Every other Tier X tool and every EXCLUDED tool: disallowed in every mode.
    A tool absent from every registry is in neither list; `dontAsk` and the hook deny it.
    """
    allowed: set[str] = set(ALLOWED_BUILTINS)
    disallowed: set[str] = set(DISALLOWED_BUILTINS)
    live = effective_mode is ExecutionMode.LIVE
    for registry in registries:
        for tool in registry.tools:
            name = registry.qualified(tool.name)
            permitted = (
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
