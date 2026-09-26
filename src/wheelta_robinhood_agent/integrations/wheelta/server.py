"""Agent SDK MCP server config for Wheelta (CLAUDE.md §8, §10; docs/INTEGRATIONS.md).

Registered as `wheelta`, so tools appear as `mcp__wheelta__<tool>`. The bearer is the
agent's own `WHEELTA_MCP_TOKEN` (its dedicated entry in `WHEELTA_HTTP_AUTH_TOKENS` on the
wheelta-mcp service, CLAUDE.md §7). Pure: builds config, performs no I/O.
"""

from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.integrations.status import McpHttpServer
from wheelta_robinhood_agent.integrations.wheelta.registry import SERVER_NAME


def build_wheelta_server(settings: Settings) -> McpHttpServer:
    """The Wheelta server config. The token is required by Settings, so this always exists."""
    return McpHttpServer(
        name=SERVER_NAME, url=settings.WHEELTA_MCP_URL, token=settings.WHEELTA_MCP_TOKEN
    )
