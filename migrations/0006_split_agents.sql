-- ADR-0057: a due tick runs two agents in order, Buy-to-Close then Sell Options, each as its
-- own run of the same slot. Forward-only: 0001 is applied in production and never edited.
--
-- 1. runs.agent names the agent (domain/enums.py AgentRole). Rows of earlier releases are
--    the single agent, 'wheel'; their run_ids are unchanged (domain/run_identity.py).
-- 2. Run identity is unique per (environment, slot, agent). Replaces
--    runs_environment_slot_key from 0001.
-- 3. New run statuses `skipped_no_open_shorts` and `skipped_insufficient_balance` (the two
--    agents' start conditions), mirroring domain/enums.py RunStatus. Replaces
--    run_events_status_check (last set by 0005).

ALTER TABLE runs ADD COLUMN agent text NOT NULL DEFAULT 'wheel'
    CHECK (agent IN ('wheel', 'close', 'sell'));

ALTER TABLE runs DROP CONSTRAINT runs_environment_slot_key;
ALTER TABLE runs ADD CONSTRAINT runs_environment_slot_agent_key UNIQUE (environment, slot, agent);

ALTER TABLE run_events DROP CONSTRAINT run_events_status_check;
ALTER TABLE run_events ADD CONSTRAINT run_events_status_check CHECK (status IN (
    'running', 'completed', 'skipped_concurrent', 'skipped_killed',
    'skipped_market_closed', 'skipped_dry_run_not_local', 'skipped_not_due',
    'skipped_dry_run_not_requested', 'skipped_no_open_shorts',
    'skipped_insufficient_balance', 'stopped', 'timed_out', 'failed'));
