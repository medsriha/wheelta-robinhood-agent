"""Run identity, run events, config snapshot, the Run projection, and audit findings.

Contracts: INTERFACES.md "Run and RunControl" and "AuditFinding"; CLAUDE.md §15, §17.
`RunControl` (the mutable stop latch) is a runtime object owned by the orchestrator and is
not modeled here; its changes are recorded as `RunEvent`s with a `StopReason`.
"""

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import JsonValue, model_validator

from wheelta_robinhood_agent.domain.base import (
    Count,
    DomainModel,
    NonEmptyStr,
    Ref,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AuditCheck,
    AuditOutcome,
    ExecutionMode,
    MarketSession,
    RunStatus,
    SourceStatus,
)
from wheelta_robinhood_agent.domain.run_identity import run_id_for, slot_for


class RunIdentity(DomainModel):
    """`run_id` is deterministic from `(environment, slot)`; slot is a whole UTC hour."""

    run_id: UUID
    environment: AppEnv
    slot: UtcDatetime

    @model_validator(mode="after")
    def _check_identity(self) -> Self:
        if self.slot != slot_for(self.slot):
            raise ValueError("slot must be a whole UTC hour")
        if self.run_id != run_id_for(self.environment, self.slot):
            raise ValueError("run_id does not match (environment, slot)")
        return self


class ConfigSnapshot(DomainModel):
    """Non-secret settings recorded on every run (INTERFACES.md `config_snapshot`).

    `values` holds the full non-secret snapshot from `Settings.config_snapshot()`. The typed
    fields repeat the safety-relevant ones; where `values` also carries them they must agree.
    Effective mode is live only if requested live and armed (the phase ceiling may still
    force off).
    """

    requested_execution_mode: ExecutionMode
    effective_execution_mode: ExecutionMode
    execution_armed: bool
    kill_switch: bool
    rules_version: NonEmptyStr
    rules_hash: NonEmptyStr
    values: dict[str, JsonValue]

    @model_validator(mode="after")
    def _check_config(self) -> Self:
        if self.effective_execution_mode is ExecutionMode.LIVE and not (
            self.requested_execution_mode is ExecutionMode.LIVE and self.execution_armed
        ):
            raise ValueError("effective live requires requested live and armed")
        typed: dict[str, JsonValue] = {
            "requested_execution_mode": self.requested_execution_mode.value,
            "effective_execution_mode": self.effective_execution_mode.value,
            "execution_armed": self.execution_armed,
            "kill_switch": self.kill_switch,
        }
        for key, expected in typed.items():
            if key in self.values and self.values[key] != expected:
                raise ValueError(f"config snapshot disagrees on {key}")
        return self


class SourceStatusObservation(DomainModel):
    """Per-server MCP status and discovery diff (INTERFACES.md `source_status`)."""

    server: NonEmptyStr
    status: SourceStatus
    observed_at: UtcDatetime
    missing_tools: tuple[NonEmptyStr, ...] = ()
    unknown_tools: tuple[NonEmptyStr, ...] = ()


class MarketSessionInfo(DomainModel):
    """Session state with the calendar provenance that determined it."""

    session: MarketSession
    calendar_provenance: NonEmptyStr


class StopReason(StrEnum):
    """Why the RunControl stop latch was set (INTERFACES.md "Run and RunControl")."""

    SIGTERM = "sigterm"
    SIGINT = "sigint"
    DEADLINE = "deadline"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class AuditStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class Run(DomainModel):
    """Run projection, derived from identity + run events (INTERFACES.md "Run").

    A killed run never starts its session; a skipped/killed run has no model metadata
    requirement. `finished_at` is present iff the status is terminal.
    """

    identity: RunIdentity
    started_at: UtcDatetime
    finished_at: UtcDatetime | None
    status: RunStatus
    git_sha: NonEmptyStr | None
    image_digest: NonEmptyStr | None
    config_snapshot: ConfigSnapshot
    market_session: MarketSessionInfo | None
    source_status: tuple[SourceStatusObservation, ...] = ()
    prompt_id: NonEmptyStr | None = None
    prompt_hash: NonEmptyStr | None = None
    rendered_prompt_hash: NonEmptyStr | None = None
    model_id: NonEmptyStr | None = None
    stop_reason: StopReason | None = None
    audit_status: AuditStatus | None = None

    @model_validator(mode="after")
    def _check_run(self) -> Self:
        terminal = self.status is not RunStatus.RUNNING
        if terminal != (self.finished_at is not None):
            raise ValueError("finished_at is present iff the run is terminal")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished before started")
        if self.status is RunStatus.SKIPPED_KILLED and self.model_id is not None:
            raise ValueError("a killed run never starts its session")
        require_unique(tuple(s.server for s in self.source_status), "source server")
        return self

    @property
    def run_id(self) -> UUID:
        return self.identity.run_id

    @property
    def slot(self) -> datetime:
        return self.identity.slot


class AuditFinding(DomainModel):
    """An append-only post-run audit finding (INTERFACES.md "AuditFinding").

    `rule_value` and `observed_value` are canonical strings so Decimals keep exact form.
    Corrections are new findings referencing `corrects_finding_id`.
    """

    finding_id: UUID
    run_id: UUID
    check_id: AuditCheck
    sub_item: NonEmptyStr | None = None
    outcome: AuditOutcome
    effective_execution_mode: ExecutionMode
    decision_ref: Ref | None = None
    leg_ref: Ref | None = None
    attempt_index: Count | None = None
    rule_key: NonEmptyStr | None = None
    rule_value: NonEmptyStr | None = None
    observed_value: NonEmptyStr | None = None
    tool_call_ids: tuple[UUID, ...] = ()
    detail: NonEmptyStr
    audit_version: NonEmptyStr
    context_hash: NonEmptyStr
    corrects_finding_id: UUID | None = None

    @model_validator(mode="after")
    def _check_finding(self) -> Self:
        if self.corrects_finding_id == self.finding_id:
            raise ValueError("a finding cannot correct itself")
        if self.attempt_index is not None and self.leg_ref is None:
            raise ValueError("attempt_index requires leg_ref")
        require_unique(self.tool_call_ids, "tool call id")
        return self
