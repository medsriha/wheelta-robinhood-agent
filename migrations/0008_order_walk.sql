-- ADR-0066: code works each option order (the order walk). Every call the executor makes
-- for a `work_option_order` job is recorded like a model call, linked to that job's own
-- tool call. Forward-only: 0001 is applied in production and never edited.
-- Backwards-compatible: the previous release's INSERT omits the column (NULL: a model call).
--
-- `parent_tool_call_id` is the `work_option_order` call that started the job; NULL for every
-- call the model (orchestrator or Mignon) made, and for every row written before this
-- migration. An executor call is never a Mignon's.

ALTER TABLE tool_calls ADD COLUMN parent_tool_call_id uuid REFERENCES tool_calls (tool_call_id);
ALTER TABLE tool_calls ADD CONSTRAINT tool_calls_executor_attribution_check
    CHECK (parent_tool_call_id IS NULL OR agent_id IS NULL);
CREATE INDEX tool_calls_parent_idx ON tool_calls (parent_tool_call_id)
    WHERE parent_tool_call_id IS NOT NULL;
