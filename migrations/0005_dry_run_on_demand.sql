-- ADR-0038: dry runs start only on demand; a scheduled tick while the effective mode is off
-- finalizes as `skipped_dry_run_not_requested`. Forward-only: 0001 is applied in production
-- and never edited. Replaces run_events_status_check (last set by 0004), mirroring
-- domain/enums.py RunStatus.

ALTER TABLE run_events DROP CONSTRAINT run_events_status_check;
ALTER TABLE run_events ADD CONSTRAINT run_events_status_check CHECK (status IN (
    'running', 'completed', 'skipped_concurrent', 'skipped_killed',
    'skipped_market_closed', 'skipped_dry_run_not_local', 'skipped_not_due',
    'skipped_dry_run_not_requested', 'stopped', 'timed_out', 'failed'));
