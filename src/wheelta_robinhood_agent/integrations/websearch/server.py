"""MCP server config for Tavily web search (ADR-0058; docs/INTEGRATIONS.md).

Registered as `tavily`, so tools appear as `mcp__tavily__<tool>`. The bearer is
`TAVILY_API_KEY` (Tavily accepts `Authorization: Bearer <key>` on the remote server). Without a
key the source is `disabled`: research Mignons then have no web tools and the run continues
(web context is never required to trade). Pure: builds config, performs no I/O.
"""

from datetime import datetime

from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.enums import SourceStatus
from wheelta_robinhood_agent.integrations.status import McpHttpServer, SourceObservation
from wheelta_robinhood_agent.integrations.websearch.registry import SERVER_NAME


def build_tavily_server(
    settings: Settings, observed_at: datetime
) -> McpHttpServer | SourceObservation:
    """The Tavily server config, or a DISABLED observation when no API key is configured."""
    if settings.TAVILY_API_KEY is None:
        return SourceObservation(
            server=SERVER_NAME, status=SourceStatus.DISABLED, observed_at=observed_at
        )
    return McpHttpServer(
        name=SERVER_NAME, url=settings.TAVILY_MCP_URL, token=settings.TAVILY_API_KEY
    )
