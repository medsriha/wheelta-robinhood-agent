-- ADR-0028: the cron ticks every 5 minutes and the agent chooses when the next session runs.
-- Forward-only: 0001 is applied in production and never edited.
--
-- 1. Slots are whole 5-minute UTC boundaries (whole hours, the earlier slots, still qualify).
--    Replaces the inline slot CHECK of runs (default name runs_slot_check).
-- 2. New run status `skipped_not_due` (a tick before the recorded next-run time), mirroring
--    domain/enums.py RunStatus. Replaces run_events_status_check (last set by 0002).
-- 3. New run event type `schedule` (the next-run time), mirroring domain/events.py
--    RunEventType. Replaces the inline event_type CHECK (default name
--    run_events_event_type_check).

ALTER TABLE runs DROP CONSTRAINT runs_slot_check;
ALTER TABLE runs ADD CONSTRAINT runs_slot_check CHECK (
    date_trunc('minute', slot AT TIME ZONE 'UTC') = slot AT TIME ZONE 'UTC'
    AND extract(minute FROM slot AT TIME ZONE 'UTC')::integer % 5 = 0);

ALTER TABLE run_events DROP CONSTRAINT run_events_status_check;
ALTER TABLE run_events ADD CONSTRAINT run_events_status_check CHECK (status IN (
    'running', 'completed', 'skipped_concurrent', 'skipped_killed',
    'skipped_market_closed', 'skipped_dry_run_not_local', 'skipped_not_due', 'stopped',
    'timed_out', 'failed'));

ALTER TABLE run_events DROP CONSTRAINT run_events_event_type_check;
ALTER TABLE run_events ADD CONSTRAINT run_events_event_type_check CHECK (event_type IN (
    'started', 'recovery_started', 'status', 'control',
    'source_status', 'market_session', 'metadata', 'audit_status', 'schedule'));
