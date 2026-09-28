"""Run identity (CLAUDE.md §15): one run per (environment, scheduled 5-minute UTC slot).

The cron ticks every `SLOT_MINUTES` (ADR-0028); the agent chooses which tick runs a session.
Whole UTC hours, the slots of earlier releases, are still valid slots with the same run_id.
"""

import uuid
from datetime import UTC, datetime
from typing import Final

from wheelta_robinhood_agent.domain.enums import AppEnv

# Fixed namespace so run_id is reproducible from (environment, slot) forever. Never change it.
RUN_ID_NAMESPACE = uuid.UUID("6f1d3c52-8a41-4e0b-9d57-2b7a4c9e1f30")
# The Railway cron interval (.railway/railway.py) and the slot size. Railway's minimum is 5.
SLOT_MINUTES: Final = 5


def slot_for(fired_at: datetime) -> datetime:
    """The scheduled slot: the fire time truncated to its 5-minute UTC boundary.

    Cron may drift, but by less than the interval, so the truncation is stable.
    Raises ValueError for a naive datetime.
    """
    if fired_at.tzinfo is None or fired_at.utcoffset() is None:
        raise ValueError("fired_at must be timezone-aware")
    utc = fired_at.astimezone(UTC)
    return utc.replace(minute=utc.minute - utc.minute % SLOT_MINUTES, second=0, microsecond=0)


def run_id_for(environment: AppEnv, slot: datetime) -> uuid.UUID:
    """Deterministic run_id: UUIDv5 of `<environment>:<slot ISO-8601 UTC>`."""
    if slot != slot_for(slot):
        raise ValueError(
            f"slot must be a whole {SLOT_MINUTES}-minute UTC slot, got {slot.isoformat()}"
        )
    return uuid.uuid5(RUN_ID_NAMESPACE, f"{environment.value}:{slot.astimezone(UTC).isoformat()}")
