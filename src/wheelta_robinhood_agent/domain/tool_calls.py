"""ToolCallRecord identity and projection (INTERFACES.md "ToolCallRecord").

Events are stored with the `ToolCallEventType` vocabulary (domain/events.py); the ledger
rebuilds `ToolCallRecord` from them (ledger/tool_calls.py `tool_call_records`).

One identity per SDK tool-use ID within a run. A durable `requested` event is committed
before dispatch, including for calls later denied. An unresolved dispatched financial action
(Tier X) is UNKNOWN, never inferred to have failed.
"""

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import JsonValue, model_validator

from wheelta_robinhood_agent.domain.base import (
    Count,
    DomainModel,
    NonEmptyStr,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier

RedactedArguments = dict[str, JsonValue]
"""Tool arguments after redaction (secrets, account numbers). JSON data only."""


class ToolCallDecision(StrEnum):
    """Hook decision on a requested tool call."""

    ALLOWED = "allowed"
    DENIED = "denied"


class ToolCallIdentity(DomainModel):
    """Inserted once, before dispatch. Unique per (run_id, sdk_tool_use_id).

    `tier` is None for a tool absent from the registry; such a call is always denied.
    """

    tool_call_id: UUID
    sdk_tool_use_id: NonEmptyStr
    run_id: UUID
    stage: NonEmptyStr
    server: NonEmptyStr
    tool: NonEmptyStr
    tier: ToolTier | None
    requested_at: UtcDatetime
    arguments_redacted: RedactedArguments


_TERMINAL_WITH_COMPLETION = frozenset(
    {ToolCallStatus.SUCCEEDED, ToolCallStatus.FAILED, ToolCallStatus.TIMED_OUT}
)


class ToolCallRecord(DomainModel):
    """Projection of one tool call, rebuilt from identity + events.

    Invariants (INTERFACES.md "ToolCallRecord"):
    - denied ⇔ status DENIED, never dispatched, with a reason;
    - succeeded/failed/timed_out require dispatch and completion times;
    - requested ≤ dispatched ≤ completed;
    - a dispatched Tier X call without a completion stays REQUESTED (in flight) or UNKNOWN;
    - an outcome resolved later from broker reads (a correction of an UNKNOWN outcome)
      names those reads in `reconciled_by_tool_call_ids`. It has no completion time and may
      have no result_ref: the reads, not this call, are its evidence.
    """

    identity: ToolCallIdentity
    effective_arguments_redacted: RedactedArguments | None
    decision: ToolCallDecision | None
    deny_reason: NonEmptyStr | None = None
    status: ToolCallStatus
    dispatched_at: UtcDatetime | None
    completed_at: UtcDatetime | None
    latency_ms: Count | None = None
    request_id: NonEmptyStr | None = None
    result_ref: UUID | None = None
    error_ref: UUID | None = None
    delivered_result_ref: UUID | None = None
    reconciled_by_tool_call_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def _check_tool_record(self) -> Self:
        require_unique(self.reconciled_by_tool_call_ids, "reconciling tool call id")
        reconciled = bool(self.reconciled_by_tool_call_ids)
        denied = self.decision is ToolCallDecision.DENIED
        if denied != (self.status is ToolCallStatus.DENIED):
            raise ValueError("status DENIED iff the decision is denied")
        if denied and (self.dispatched_at is not None or self.deny_reason is None):
            raise ValueError("a denied call is never dispatched and records its reason")
        if self.dispatched_at is not None and self.decision is not ToolCallDecision.ALLOWED:
            raise ValueError("only an allowed call can be dispatched")
        if reconciled and self.dispatched_at is None:
            raise ValueError("only a dispatched call's outcome is reconciled")
        if self.status in _TERMINAL_WITH_COMPLETION and (
            self.dispatched_at is None or (self.completed_at is None and not reconciled)
        ):
            raise ValueError(f"status {self.status.value} requires dispatch and completion")
        if self.completed_at is not None and self.dispatched_at is None:
            raise ValueError("completion without dispatch")
        requested = self.identity.requested_at
        if self.dispatched_at is not None and self.dispatched_at < requested:
            raise ValueError("dispatched before requested")
        if (
            self.completed_at is not None
            and self.dispatched_at is not None
            and self.completed_at < self.dispatched_at
        ):
            raise ValueError("completed before dispatched")
        if self.status is ToolCallStatus.SUCCEEDED and self.result_ref is None and not reconciled:
            raise ValueError("a succeeded call needs a result_ref")
        return self

    @property
    def outcome_unresolved(self) -> bool:
        """A dispatched call whose outcome is not known (Tier X: treat as unknown, alert)."""
        return self.dispatched_at is not None and self.status in (
            ToolCallStatus.REQUESTED,
            ToolCallStatus.UNKNOWN,
        )
