"""The in-process `wra_local` MCP server and its registry (ADR-0011, ADR-0016).

Both local tools are Tier R and verified (they are our own code), so `tool_access` allows them
in every mode and the PreToolUse hook resolves them like any registered tool:

- `mcp__wra_local__web_cache_lookup` (agent/web_cache.py)
- `mcp__wra_local__get_decision_facts` (agent/facts_tool.py)

One registry per server name: the hook resolves a server to the first registry with that
name, so the local tools must share this one (it supersedes `WEB_CACHE_REGISTRY` for sessions).
"""

from collections.abc import Sequence
from typing import Any

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server
from claude_agent_sdk.types import McpSdkServerConfig

from wheelta_robinhood_agent.agent.facts_tool import FACTS_TOOL_NAME
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME, WEB_CACHE_TOOL_NAME
from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, make_registry

LOCAL_SERVER_VERSION = "1"

LOCAL_REGISTRY: ToolRegistry = make_registry(
    LOCAL_SERVER_NAME, {ToolTier.R: (WEB_CACHE_TOOL_NAME, FACTS_TOOL_NAME)}, verified=True
)


class LocalServerError(ValueError):
    """The local tools do not match the local registry."""


def build_local_server(tools: Sequence[SdkMcpTool[Any]]) -> McpSdkServerConfig:
    """`create_sdk_mcp_server` for exactly the registered local tools (no more, no fewer)."""
    names = sorted(t.name for t in tools)
    expected = sorted(t.name for t in LOCAL_REGISTRY.tools)
    if names != expected:
        raise LocalServerError(f"local tools {names} do not match the registry {expected}")
    return create_sdk_mcp_server(LOCAL_SERVER_NAME, version=LOCAL_SERVER_VERSION, tools=list(tools))
