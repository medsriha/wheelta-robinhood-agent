"""Agent SDK MCP server config for Robinhood (CLAUDE.md §8, §9; docs/INTEGRATIONS.md).

Registered as `robinhood`, so tools appear as `mcp__robinhood__<tool>`. Two auth modes
(`ROBINHOOD_MCP_AUTH`):

- `token` (default): a static bearer token from `ROBINHOOD_MCP_ACCESS_TOKEN`; without it the
  source is `needs-auth` and the run cannot trade or manage positions.
- `refresh_token` (ADR-0021): the orchestrator resolves an access token from the ledger's
  encrypted, rotating OAuth credential (refreshing it when near expiry) and passes it in as
  `access_token`. None means no usable credential: `needs-auth`.
- `claude_code_login` (local only, ADR-0018): no header; the Claude Code CLI that hosts the
  session reuses the OAuth login the owner completed with `/mcp` for the `robinhood` server.
  Our code never sees or stores that credential.

No other workaround (scraping, stored passwords, replayed browser sessions) is attempted.
Pure: builds config, no I/O.
"""

from datetime import datetime

from pydantic import SecretStr

from wheelta_robinhood_agent.config.settings import RobinhoodMcpAuth, Settings
from wheelta_robinhood_agent.domain.enums import SourceStatus
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME
from wheelta_robinhood_agent.integrations.status import McpHttpServer, SourceObservation


def build_robinhood_server(
    settings: Settings, observed_at: datetime, access_token: SecretStr | None = None
) -> McpHttpServer | SourceObservation:
    """The Robinhood server config, or a NEEDS_AUTH observation when no token is available.

    `observed_at` is passed in (no clock in pure code) and stamps the observation.
    `access_token` is the orchestrator-resolved token and is valid only in refresh_token mode.
    """
    refresh_mode = settings.ROBINHOOD_MCP_AUTH is RobinhoodMcpAuth.REFRESH_TOKEN
    if access_token is not None and not refresh_mode:
        raise ValueError("access_token is only accepted with ROBINHOOD_MCP_AUTH=refresh_token")
    if settings.ROBINHOOD_MCP_AUTH is RobinhoodMcpAuth.CLAUDE_CODE_LOGIN:
        return McpHttpServer(
            name=SERVER_NAME, url=settings.ROBINHOOD_MCP_URL, uses_stored_cli_login=True
        )
    token = access_token if refresh_mode else settings.ROBINHOOD_MCP_ACCESS_TOKEN
    if token is None:
        return SourceObservation(
            server=SERVER_NAME, status=SourceStatus.NEEDS_AUTH, observed_at=observed_at
        )
    return McpHttpServer(name=SERVER_NAME, url=settings.ROBINHOOD_MCP_URL, token=token)
