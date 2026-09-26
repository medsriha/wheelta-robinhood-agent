"""Alert and heartbeat payload models (CLAUDE.md §16). No HTTP: delivery is in
``integrations/notifications``.

Every alert kind in CLAUDE.md §16 maps to a fixed severity and, where docs/OPERATIONS.md has
one, a runbook id. Severity is derived from the kind, never chosen at the call site. Details
are redacted at construction, so a payload can never carry a secret or a full account number.

A heartbeat reports the status of one scheduled cron invocation. ``success`` covers only a
completed run or the normal market-closed skip; every other outcome is ``failure``. A failed
invocation can never produce a success heartbeat.
"""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    JsonValue,
    field_validator,
    model_validator,
)

from wheelta_robinhood_agent.domain.enums import AppEnv, RunStatus
from wheelta_robinhood_agent.observability.redaction import Redactor

_DEFAULT_REDACTOR = Redactor()


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AlertKind(StrEnum):
    """Every alert in CLAUDE.md §16."""

    ORDER_ACTIVITY = "order_activity"
    ROBINHOOD_NEEDS_AUTH = "robinhood_needs_auth"
    REPEATED_SOURCE_FAILURE = "repeated_source_failure"
    SCHEMA_DRIFT = "schema_drift"
    TOOL_DRIFT = "tool_drift"
    KILL_SWITCH_ENGAGED = "kill_switch_engaged"
    TIER_X_ERROR_OR_UNKNOWN_OUTCOME = "tier_x_error_or_unknown_outcome"
    INVALID_AGENT_OUTPUT = "invalid_agent_output"
    RUN_TIMEOUT = "run_timeout"
    CONSECUTIVE_RUN_FAILURES = "consecutive_run_failures"
    AUDIT_VIOLATION = "audit_violation"
    AUDIT_FAILURE = "audit_failure"


class AlertSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: AlertSeverity
    runbook: str | None


# OPERATIONS.md severities: info -> info, medium -> warning, high/critical -> error.
# Runbook ids refer to docs/OPERATIONS.md; every kind has one.
ALERT_SPECS: Mapping[AlertKind, AlertSpec] = {
    AlertKind.ORDER_ACTIVITY: AlertSpec(severity=AlertSeverity.INFO, runbook="R11"),
    AlertKind.ROBINHOOD_NEEDS_AUTH: AlertSpec(severity=AlertSeverity.ERROR, runbook="R2"),
    AlertKind.REPEATED_SOURCE_FAILURE: AlertSpec(severity=AlertSeverity.WARNING, runbook="R15"),
    AlertKind.SCHEMA_DRIFT: AlertSpec(severity=AlertSeverity.WARNING, runbook="R6"),
    AlertKind.TOOL_DRIFT: AlertSpec(severity=AlertSeverity.ERROR, runbook="R5"),
    AlertKind.KILL_SWITCH_ENGAGED: AlertSpec(severity=AlertSeverity.INFO, runbook="R9"),
    AlertKind.TIER_X_ERROR_OR_UNKNOWN_OUTCOME: AlertSpec(
        severity=AlertSeverity.ERROR, runbook="R8"
    ),
    AlertKind.INVALID_AGENT_OUTPUT: AlertSpec(severity=AlertSeverity.ERROR, runbook="R13"),
    AlertKind.RUN_TIMEOUT: AlertSpec(severity=AlertSeverity.WARNING, runbook="R7"),
    AlertKind.CONSECUTIVE_RUN_FAILURES: AlertSpec(severity=AlertSeverity.ERROR, runbook="R14"),
    AlertKind.AUDIT_VIOLATION: AlertSpec(severity=AlertSeverity.ERROR, runbook="R12"),
    AlertKind.AUDIT_FAILURE: AlertSpec(severity=AlertSeverity.ERROR, runbook="R16"),
}


class AlertPayload(BaseModel):
    """One operational alert. Severity and runbook must match ``ALERT_SPECS[kind]``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: AlertKind
    severity: AlertSeverity
    runbook: str | None
    run_id: str | None
    environment: AppEnv
    occurred_at: AwareDatetime
    message: str
    details: dict[str, JsonValue]

    @field_validator("message")
    @classmethod
    def _redact_message(cls, value: str) -> str:
        return _DEFAULT_REDACTOR.redact_text(value)

    @field_validator("details", mode="before")
    @classmethod
    def _redact_details(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return _DEFAULT_REDACTOR.redact_mapping({str(k): v for k, v in value.items()})
        return value

    @model_validator(mode="after")
    def _matches_spec(self) -> Self:
        spec = ALERT_SPECS[self.kind]
        if self.severity is not spec.severity or self.runbook != spec.runbook:
            raise ValueError(f"severity/runbook for {self.kind.value} must match ALERT_SPECS")
        return self


def build_alert(
    kind: AlertKind,
    *,
    run_id: str | None,
    environment: AppEnv,
    occurred_at: datetime,
    message: str,
    details: Mapping[str, object] | None = None,
    redactor: Redactor | None = None,
) -> AlertPayload:
    """Build an alert with severity/runbook from ``ALERT_SPECS``.

    Pass the run's configured ``redactor`` so known secret values (the account number, tokens)
    are scrubbed too; without it only key-, type- and pattern-based redaction applies.
    """
    spec = ALERT_SPECS[kind]
    scrub = redactor if redactor is not None else _DEFAULT_REDACTOR
    return AlertPayload(
        kind=kind,
        severity=spec.severity,
        runbook=spec.runbook,
        run_id=run_id,
        environment=environment,
        occurred_at=occurred_at,
        message=scrub.redact_text(message),
        details=scrub.redact_mapping(details or {}),
    )


class HeartbeatStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


# Only these finalized statuses are healthy invocations (OPERATIONS.md "healthy"). A concurrent
# skip means the previous run overran its sub-hour budget; a kill-switch skip needs attention.
_SUCCESS_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.SKIPPED_MARKET_CLOSED})


def heartbeat_status_for(run_status: RunStatus) -> HeartbeatStatus:
    """Map a finalized run status to the heartbeat status. ``running`` is not finalized."""
    if run_status is RunStatus.RUNNING:
        raise ValueError("a heartbeat is sent only for a finalized invocation")
    return HeartbeatStatus.SUCCESS if run_status in _SUCCESS_STATUSES else HeartbeatStatus.FAILURE


class HeartbeatPayload(BaseModel):
    """Status of one scheduled invocation. ``status`` must agree with ``run_status``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["heartbeat"] = "heartbeat"
    status: HeartbeatStatus
    run_status: RunStatus
    run_id: str | None
    environment: AppEnv
    slot: AwareDatetime | None
    occurred_at: AwareDatetime
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def _redact_reason(cls, value: str | None) -> str | None:
        return None if value is None else _DEFAULT_REDACTOR.redact_text(value)

    @model_validator(mode="after")
    def _status_matches_run(self) -> Self:
        if self.status is not heartbeat_status_for(self.run_status):
            raise ValueError(f"run_status {self.run_status.value} cannot report {self.status}")
        return self


def build_heartbeat(
    run_status: RunStatus,
    *,
    run_id: str | None,
    environment: AppEnv,
    slot: datetime | None,
    occurred_at: datetime,
    reason: str | None = None,
) -> HeartbeatPayload:
    """Build the heartbeat for a finalized invocation; status is derived, never passed in."""
    return HeartbeatPayload(
        status=heartbeat_status_for(run_status),
        run_status=run_status,
        run_id=run_id,
        environment=environment,
        slot=slot,
        occurred_at=occurred_at,
        reason=reason,
    )
