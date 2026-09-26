from datetime import UTC, datetime, timedelta, timezone

import pytest

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.domain.run_identity import run_id_for, slot_for


def test_slot_truncates_to_utc_hour() -> None:
    est = timezone(timedelta(hours=-5))
    fired = datetime(2026, 9, 25, 10, 3, 17, tzinfo=est)
    assert slot_for(fired) == datetime(2026, 9, 25, 15, 0, tzinfo=UTC)


def test_naive_rejected() -> None:
    with pytest.raises(ValueError):
        slot_for(datetime(2026, 9, 25, 10, 3))  # noqa: DTZ001


def test_run_id_deterministic_and_distinct() -> None:
    slot = datetime(2026, 9, 25, 15, tzinfo=UTC)
    a = run_id_for(AppEnv.STAGING, slot)
    assert a == run_id_for(AppEnv.STAGING, slot_for(slot + timedelta(minutes=59)))
    assert a != run_id_for(AppEnv.PRODUCTION, slot)
    assert a != run_id_for(AppEnv.STAGING, slot + timedelta(hours=1))


def test_run_id_requires_whole_hour() -> None:
    with pytest.raises(ValueError):
        run_id_for(AppEnv.LOCAL, datetime(2026, 9, 25, 15, 1, tzinfo=UTC))
