-- ADR-0024: dry runs run locally only; an off-mode run outside APP_ENV=local finalizes as
-- `skipped_dry_run_not_local`. Forward-only: 0001 is applied in production and never edited.
-- Replaces the inline status CHECK of run_events (default name run_events_status_check),
-- mirroring domain/enums.py RunStatus.

ALTER TABLE run_events DROP CONSTRAINT run_events_status_check;
ALTER TABLE run_events ADD CONSTRAINT run_events_status_check CHECK (status IN (
    'running', 'completed', 'skipped_concurrent', 'skipped_killed',
    'skipped_market_closed', 'skipped_dry_run_not_local', 'stopped', 'timed_out', 'failed'));
