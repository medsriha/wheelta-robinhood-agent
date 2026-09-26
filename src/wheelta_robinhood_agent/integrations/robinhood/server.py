"""Agent SDK MCP server config for Robinhood (CLAUDE.md §8, §9; docs/INTEGRATIONS.md).

Registered as `robinhood`, so tools appear as `mcp__robinhood__<tool>`. The SDK does not
run interactive OAuth; it needs an access token in headers (docs/INTEGRATIONS.md). How a
headless run obtains that token is unresolved (ADR-0004, Open). Without
`ROBINHOOD_MCP_ACCESS_TOKEN` there is no server config: the source is recorded as
`needs-auth` and the run cannot trade or manage positions. No workaround (scraping, stored
credentials, replayed browser sessions) is attempted. Pure: builds config, no I/O.
"""

from datetime import datetime

from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.enums import SourceStatus
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME
from wheelta_robinhood_agent.integrations.status import McpHttpServer, SourceObservation


def build_robinhood_server(
    settings: Settings, observed_at: datetime
) -> McpHttpServer | SourceObservation:
    """The Robinhood server config, or a NEEDS_AUTH observation when no token is configured.

    `observed_at` is passed in (no clock in pure code) and stamps the observation.
    """
    token = settings.ROBINHOOD_MCP_ACCESS_TOKEN
    if token is None:
        return SourceObservation(
            server=SERVER_NAME, status=SourceStatus.NEEDS_AUTH, observed_at=observed_at
        )
    return McpHttpServer(name=SERVER_NAME, url=settings.ROBINHOOD_MCP_URL, token=token)
