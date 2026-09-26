"""Wheelta MCP tool registry. Every Wheelta tool is read-only (CLAUDE.md §10).

Names are from the wheelta-mcp contract (`/Users/deepset/wheelta-mcp/MCP.yaml`), which this
repository consumes but does not own.
"""

from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, make_registry

SERVER_NAME = "wheelta"

WHEELTA_REGISTRY: ToolRegistry = make_registry(
    SERVER_NAME,
    {
        ToolTier.R: (
            "wheelta_board_query",
            "wheelta_board_fields",
            "wheelta_board_row",
            "wheelta_board_status",
            "wheelta_assignment_rates",
            "wheelta_macro_snapshot",
            "wheelta_macro_series",
            "wheelta_candles",
            "wheelta_quotes",
            "wheelta_correlations",
            "wheelta_calendar_events",
            "wheelta_company_research",
        ),
    },  # fmt: skip
    verified=False,
)
