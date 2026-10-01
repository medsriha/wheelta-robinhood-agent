"""Tavily MCP tool registry (ADR-0058; CLAUDE.md §8, §11).

Names and input schemas are from our own `tools/list` capture of the hosted server
`https://mcp.tavily.com/mcp/` on 2026-09-30 (`tests/fixtures/tavily/tools_2026-09-30.json`).
Search and extract are research reads (Tier R, research Mignons only). Crawl, map, and
research are excluded: crawl and map spend credits per page, and research returns an
LLM-written report, which is model text and can never be a source.
"""

from typing import Final

from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, make_registry

SERVER_NAME: Final = "tavily"
SEARCH_TOOL: Final = "tavily_search"
EXTRACT_TOOL: Final = "tavily_extract"

TAVILY_REGISTRY: ToolRegistry = make_registry(
    SERVER_NAME,
    {
        ToolTier.R: (SEARCH_TOOL, EXTRACT_TOOL),
        ToolTier.EXCLUDED: ("tavily_crawl", "tavily_map", "tavily_research"),
    },
    verified=True,
)
