"""Alert and heartbeat payloads (CLAUDE.md §16)."""

import json
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import SecretStr, ValidationError

from wheelta_robinhood_agent.domain.enums import AppEnv, RunStatus
from wheelta_robinhood_agent.observability.alerts import (
    ALERT_SPECS,
    AlertKind,
    AlertPayload,
    AlertSeverity,
    HeartbeatPayload,
    HeartbeatStatus,
    build_alert,
    build_heartbeat,
    heartbeat_status_for,
)
from wheelta_robinhood_agent.observability.redaction import REDACTED, Redactor

T0 = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
ACCOUNT = "5QR12345678"


def test_every_kind_has_a_spec() -> None:
    assert set(ALERT_SPECS) == set(AlertKind)


def test_runbook_ids_match_operations_md() -> None:
    assert ALERT_SPECS[AlertKind.TIER_X_ERROR_OR_UNKNOWN_OUTCOME].runbook == "R8"
    assert ALERT_SPECS[AlertKind.KILL_SWITCH_ENGAGED].runbook == "R9"
    assert ALERT_SPECS[AlertKind.AUDIT_VIOLATION].runbook == "R12"
    assert ALERT_SPECS[AlertKind.ORDER_ACTIVITY].severity is AlertSeverity.INFO


def test_build_alert_redacts_details_and_message() -> None:
    redactor = Redactor(account_number=SecretStr(ACCOUNT), secrets=[SecretStr("tok-abcdefgh")])
    alert = build_alert(
        AlertKind.ROBINHOOD_NEEDS_AUTH,
        run_id="run-1",
        environment=AppEnv.STAGING,
        occurred_at=T0,
        message=f"auth failed for {ACCOUNT} with tok-abcdefgh",
        details={"authorization": "Bearer xyzxyzxyzxyz", "account_number": ACCOUNT, "n": 1},
        redactor=redactor,
    )
    dumped = alert.model_dump_json()
    assert ACCOUNT not in dumped and "tok-abcdefgh" not in dumped and "xyzxyz" not in dumped
    assert alert.severity is AlertSeverity.ERROR
    assert alert.runbook == "R2"
    assert alert.details == {"authorization": REDACTED, "account_number": "****5678", "n": 1}


def test_direct_construction_still_redacts_and_enforces_spec() -> None:
    alert = AlertPayload(
        kind=AlertKind.SCHEMA_DRIFT,
        severity=AlertSeverity.WARNING,
        runbook="R6",
        run_id=None,
        environment=AppEnv.LOCAL,
        occurred_at=T0,
        message="Bearer abcdefghijklmn",
        details={"token": "t", "s": SecretStr("hidden-value")},
    )
    assert "abcdefghijklmn" not in alert.message
    assert alert.details == {"token": REDACTED, "s": REDACTED}
    with pytest.raises(ValidationError):
        AlertPayload(
            kind=AlertKind.SCHEMA_DRIFT,
            severity=AlertSeverity.INFO,
            runbook="R6",
            run_id=None,
            environment=AppEnv.LOCAL,
            occurred_at=T0,
            message="m",
            details={},
        )


@pytest.mark.parametrize(
    ("run_status", "expected"),
    [
        (RunStatus.COMPLETED, HeartbeatStatus.SUCCESS),
        (RunStatus.SKIPPED_MARKET_CLOSED, HeartbeatStatus.SUCCESS),
        (RunStatus.SKIPPED_CONCURRENT, HeartbeatStatus.FAILURE),
        (RunStatus.SKIPPED_KILLED, HeartbeatStatus.FAILURE),
        (RunStatus.STOPPED, HeartbeatStatus.FAILURE),
        (RunStatus.TIMED_OUT, HeartbeatStatus.FAILURE),
        (RunStatus.FAILED, HeartbeatStatus.FAILURE),
    ],
)
def test_heartbeat_status_mapping(run_status: RunStatus, expected: HeartbeatStatus) -> None:
    hb = build_heartbeat(
        run_status, run_id="r", environment=AppEnv.PRODUCTION, slot=T0, occurred_at=T0
    )
    assert hb.status is expected
    assert json.loads(hb.model_dump_json())["status"] == expected.value


def test_running_is_not_a_heartbeat() -> None:
    with pytest.raises(ValueError):
        heartbeat_status_for(RunStatus.RUNNING)


_non_success = st.sampled_from(
    [s for s in RunStatus if s not in (RunStatus.COMPLETED, RunStatus.SKIPPED_MARKET_CLOSED)]
)


@given(run_status=_non_success)
def test_failure_heartbeat_can_never_report_success(run_status: RunStatus) -> None:
    with pytest.raises(ValidationError):
        HeartbeatPayload(
            status=HeartbeatStatus.SUCCESS,
            run_status=run_status,
            run_id="r",
            environment=AppEnv.PRODUCTION,
            slot=None,
            occurred_at=T0,
        )


def test_heartbeat_reason_redacted_and_aware_times_required() -> None:
    hb = build_heartbeat(
        RunStatus.FAILED,
        run_id=None,
        environment=AppEnv.LOCAL,
        slot=None,
        occurred_at=T0,
        reason="db postgres://u:pw1234@h/db",
    )
    assert hb.reason is not None and "pw1234" not in hb.reason
    with pytest.raises(ValidationError):
        build_heartbeat(
            RunStatus.COMPLETED,
            run_id=None,
            environment=AppEnv.LOCAL,
            slot=None,
            occurred_at=datetime(2026, 1, 1),  # noqa: DTZ001 - naive on purpose
        )
