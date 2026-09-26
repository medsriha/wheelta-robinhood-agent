"""Run identity (CLAUDE.md §15): one run per (environment, scheduled UTC hour)."""

import uuid
from datetime import UTC, datetime

from wheelta_robinhood_agent.domain.enums import AppEnv

# Fixed namespace so run_id is reproducible from (environment, slot) forever. Never change it.
RUN_ID_NAMESPACE = uuid.UUID("6f1d3c52-8a41-4e0b-9d57-2b7a4c9e1f30")


def slot_for(fired_at: datetime) -> datetime:
    """The scheduled slot: the fire time truncated to the UTC hour.

    Cron may drift by minutes but stays inside its hour, so the truncation is stable.
    Raises ValueError for a naive datetime.
    """
    if fired_at.tzinfo is None or fired_at.utcoffset() is None:
        raise ValueError("fired_at must be timezone-aware")
    return fired_at.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def run_id_for(environment: AppEnv, slot: datetime) -> uuid.UUID:
    """Deterministic run_id: UUIDv5 of `<environment>:<slot ISO-8601 UTC>`."""
    if slot != slot_for(slot):
        raise ValueError(f"slot must be a whole UTC hour, got {slot.isoformat()}")
    return uuid.uuid5(RUN_ID_NAMESPACE, f"{environment.value}:{slot.astimezone(UTC).isoformat()}")
