import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.domain.run_identity import SLOT_MINUTES, run_id_for, slot_for


def test_slot_truncates_to_five_minutes_utc() -> None:
    est = timezone(timedelta(hours=-5))
    fired = datetime(2026, 9, 25, 10, 3, 17, tzinfo=est)
    assert slot_for(fired) == datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
    assert slot_for(datetime(2026, 9, 25, 15, 44, 59, 999, tzinfo=UTC)) == datetime(
        2026, 9, 25, 15, 40, tzinfo=UTC
    )
    assert SLOT_MINUTES == 5


def test_naive_rejected() -> None:
    with pytest.raises(ValueError):
        slot_for(datetime(2026, 9, 25, 10, 3))  # noqa: DTZ001


def test_run_id_deterministic_and_distinct() -> None:
    slot = datetime(2026, 9, 25, 15, tzinfo=UTC)
    a = run_id_for(AppEnv.STAGING, slot)
    assert a == run_id_for(AppEnv.STAGING, slot_for(slot + timedelta(minutes=4)))
    assert a != run_id_for(AppEnv.PRODUCTION, slot)
    assert a != run_id_for(AppEnv.STAGING, slot + timedelta(minutes=5))


def test_hour_slots_keep_their_run_id() -> None:
    # ADR-0028: slots recorded by the hourly release stay valid with the same run_id.
    slot = datetime(2026, 9, 25, 15, tzinfo=UTC)
    assert run_id_for(AppEnv.PRODUCTION, slot) == run_id_for(AppEnv.PRODUCTION, slot_for(slot))


def test_run_id_requires_whole_slot() -> None:
    with pytest.raises(ValueError):
        run_id_for(AppEnv.LOCAL, datetime(2026, 9, 25, 15, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        run_id_for(AppEnv.LOCAL, datetime(2026, 9, 25, 15, 5, 30, tzinfo=UTC))


def test_railway_cron_ticks_once_per_slot() -> None:
    # ADR-0028: one tick per slot. A faster cron would collide on slots; a slower one would
    # delay the agent's chosen time by more than a slot.
    config = (Path(__file__).resolve().parents[2] / ".railway" / "railway.py").read_text()
    match = re.search(r'^CRON_SCHEDULE = "\*/(\d+) ', config, re.M)
    assert match is not None and int(match.group(1)) == SLOT_MINUTES
