-- ADR-0058: web research moves from the built-in WebSearch/WebFetch to Tavily's MCP tools, and
-- the web cache records `tavily_search` results. Forward-only: 0001 is applied in production
-- and never edited. Backwards-compatible: existing WebSearch/WebFetch rows stay valid, and the
-- previous release writes only values these constraints still accept.
--
-- Replaces 0001's inline tool CHECK (default name web_cache_entries_tool_check) and its
-- table CHECK tying a query to the search tool (default name web_cache_entries_check),
-- mirroring domain/web_cache.py WebTool and SEARCH_TOOLS.

ALTER TABLE web_cache_entries DROP CONSTRAINT web_cache_entries_tool_check;
ALTER TABLE web_cache_entries ADD CONSTRAINT web_cache_entries_tool_check
    CHECK (tool IN ('WebSearch', 'WebFetch', 'tavily_search'));

ALTER TABLE web_cache_entries DROP CONSTRAINT web_cache_entries_check;
ALTER TABLE web_cache_entries ADD CONSTRAINT web_cache_entries_query_check
    CHECK ((tool IN ('WebSearch', 'tavily_search')) = (query_raw IS NOT NULL));
