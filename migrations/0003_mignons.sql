-- ADR-0025: the orchestrator spawns research Mignons through the built-in `Agent` tool
-- (tier D, delegation), and every tool call records which Mignon made it. Forward-only:
-- 0001 is applied in production and never edited.
--
-- `agent_id`/`agent_type` are the SDK hook input's sub-agent identity; both NULL for the
-- orchestrator's own calls (and for every row written before this migration).
-- Replaces the inline tier CHECK of tool_calls (default name tool_calls_tier_check),
-- mirroring domain/enums.py ToolTier.

ALTER TABLE tool_calls ADD COLUMN agent_id text CHECK (length(agent_id) > 0);
ALTER TABLE tool_calls ADD COLUMN agent_type text CHECK (length(agent_type) > 0);
ALTER TABLE tool_calls ADD CONSTRAINT tool_calls_agent_attribution_check
    CHECK ((agent_id IS NULL) = (agent_type IS NULL));

ALTER TABLE tool_calls DROP CONSTRAINT tool_calls_tier_check;
ALTER TABLE tool_calls ADD CONSTRAINT tool_calls_tier_check
    CHECK (tier IN ('R', 'S', 'X', 'D', 'excluded'));
