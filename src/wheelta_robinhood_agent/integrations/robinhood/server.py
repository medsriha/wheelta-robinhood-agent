"""Agent SDK MCP server config for Robinhood (CLAUDE.md §8, §9; docs/INTEGRATIONS.md).

Registered as `robinhood`, so tools appear as `mcp__robinhood__<tool>`. Two auth modes
(`ROBINHOOD_MCP_AUTH`):

- `token` (default): a bearer token from `ROBINHOOD_MCP_ACCESS_TOKEN`. How a headless run
  obtains one is unresolved (ADR-0004, Open); without it the source is `needs-auth` and the
  run cannot trade or manage positions.
- `claude_code_login` (local only, ADR-0018): no header; the Claude Code CLI that hosts the
  session reuses the OAuth login the owner completed with `/mcp` for the `robinhood` server.
  Our code never sees or stores that credential.

No other workaround (scraping, stored credentials, replayed browser sessions) is attempted.
Pure: builds config, no I/O.
"""

from datetime import datetime

from wheelta_robinhood_agent.config.settings import RobinhoodMcpAuth, Settings
from wheelta_robinhood_agent.domain.enums import SourceStatus
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME
from wheelta_robinhood_agent.integrations.status import McpHttpServer, SourceObservation


def build_robinhood_server(
    settings: Settings, observed_at: datetime
) -> McpHttpServer | SourceObservation:
    """The Robinhood server config, or a NEEDS_AUTH observation when no token is configured.

    `observed_at` is passed in (no clock in pure code) and stamps the observation.
    """
    if settings.ROBINHOOD_MCP_AUTH is RobinhoodMcpAuth.CLAUDE_CODE_LOGIN:
        return McpHttpServer(
            name=SERVER_NAME, url=settings.ROBINHOOD_MCP_URL, uses_stored_cli_login=True
        )
    token = settings.ROBINHOOD_MCP_ACCESS_TOKEN
    if token is None:
        return SourceObservation(
            server=SERVER_NAME, status=SourceStatus.NEEDS_AUTH, observed_at=observed_at
        )
    return McpHttpServer(name=SERVER_NAME, url=settings.ROBINHOOD_MCP_URL, token=token)
