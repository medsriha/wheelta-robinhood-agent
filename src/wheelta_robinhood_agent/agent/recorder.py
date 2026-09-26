"""Tool-call event recording for the agent hooks (INTERFACES.md "ToolCallRecord"; CLAUDE.md §17).

`ToolEventRecorder` is what the hooks depend on: they never talk to the database directly.
Order of calls for one tool use:

1. `requested` - before any decision, so denied calls also have a durable request.
2. `dispatched` - the hook allowed the call; records the (possibly rewritten) arguments.
3. `outcome` - denied / succeeded / failed / timed_out / unknown.
4. `store_result` + `delivered` - the exact envelope handed to the model.

Every method raises on failure. The hooks treat any exception as a recording failure and
fail closed (deny, stop latch, interrupt). Arguments and payloads must already be redacted.

`LedgerToolEventRecorder` adapts `ledger/tool_calls.py`. Writing `results` rows needs a ledger
function that does not exist yet, so it takes an injected `ResultWriter` (the ledger owns the
SQL; `agent/` never touches the database, CLAUDE.md §3).
"""

import uuid
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Protocol

import psycopg
from pydantic import JsonValue

from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier
from wheelta_robinhood_agent.ledger import tool_calls as ledger_tool_calls

AGENT_STAGE = "agent"


class ResultKind(StrEnum):
    """`results.kind` values (migrations/0001_initial.sql)."""

    VALIDATED = "validated"
    ERROR = "error"
    RAW_INVALID = "raw_invalid"
    DELIVERED = "delivered"


class ToolEventRecorder(Protocol):
    """Durable tool-call recording used by the PreToolUse/PostToolUse/PostToolUseFailure hooks."""

    def requested(
        self,
        *,
        sdk_tool_use_id: str,
        server: str,
        tool: str,
        tier: ToolTier | None,
        arguments_redacted: Mapping[str, JsonValue],
        requested_at: datetime,
    ) -> uuid.UUID:
        """Persist the identity and `requested` event; return the tool_call_id."""
        ...

    def dispatched(
        self,
        tool_call_id: uuid.UUID,
        *,
        effective_arguments_redacted: Mapping[str, JsonValue],
        dispatched_at: datetime,
    ) -> None: ...

    def outcome(
        self,
        tool_call_id: uuid.UUID,
        status: ToolCallStatus,
        *,
        observed_at: datetime,
        dedup_key: str | None = None,
        reason: str | None = None,
        result_ref: uuid.UUID | None = None,
        error_ref: uuid.UUID | None = None,
    ) -> None: ...

    def store_result(
        self,
        tool_call_id: uuid.UUID,
        kind: ResultKind,
        payload: JsonValue,
    ) -> uuid.UUID:
        """Persist a `results` row and return its result_id."""
        ...

    def delivered(
        self,
        tool_call_id: uuid.UUID,
        *,
        delivered_result_ref: uuid.UUID,
        observed_at: datetime,
    ) -> None: ...


class ResultWriter(Protocol):
    """Inserts one `results` row. Provided by `ledger/` (planned: no function exists yet)."""

    def __call__(
        self,
        conn: psycopg.Connection[tuple[object, ...]],
        *,
        run_id: uuid.UUID,
        tool_call_id: uuid.UUID,
        kind: str,
        payload: JsonValue,
    ) -> uuid.UUID: ...


class LedgerToolEventRecorder:
    """`ToolEventRecorder` over `ledger/tool_calls.py`, for one run and one connection."""

    def __init__(
        self,
        conn: psycopg.Connection[tuple[object, ...]],
        *,
        run_id: uuid.UUID,
        result_writer: ResultWriter,
        stage: str = AGENT_STAGE,
    ) -> None:
        self._conn = conn
        self._run_id = run_id
        self._result_writer = result_writer
        self._stage = stage

    def requested(
        self,
        *,
        sdk_tool_use_id: str,
        server: str,
        tool: str,
        tier: ToolTier | None,
        arguments_redacted: Mapping[str, JsonValue],
        requested_at: datetime,
    ) -> uuid.UUID:
        ref = ledger_tool_calls.record_tool_call_requested(
            self._conn,
            run_id=self._run_id,
            sdk_tool_use_id=sdk_tool_use_id,
            stage=self._stage,
            server=server,
            tool=tool,
            tier=tier,
            arguments_redacted=arguments_redacted,
            requested_at=requested_at,
        )
        return ref.tool_call_id

    def dispatched(
        self,
        tool_call_id: uuid.UUID,
        *,
        effective_arguments_redacted: Mapping[str, JsonValue],
        dispatched_at: datetime,
    ) -> None:
        ledger_tool_calls.record_tool_call_dispatched(
            self._conn,
            tool_call_id,
            dispatched_at=dispatched_at,
            effective_arguments_redacted=effective_arguments_redacted,
        )

    def outcome(
        self,
        tool_call_id: uuid.UUID,
        status: ToolCallStatus,
        *,
        observed_at: datetime,
        dedup_key: str | None = None,
        reason: str | None = None,
        result_ref: uuid.UUID | None = None,
        error_ref: uuid.UUID | None = None,
    ) -> None:
        ledger_tool_calls.append_tool_call_outcome(
            self._conn,
            tool_call_id,
            status,
            observed_at=observed_at,
            dedup_key=dedup_key,
            reason=reason,
            result_ref=result_ref,
            error_ref=error_ref,
        )

    def store_result(
        self,
        tool_call_id: uuid.UUID,
        kind: ResultKind,
        payload: JsonValue,
    ) -> uuid.UUID:
        return self._result_writer(
            self._conn,
            run_id=self._run_id,
            tool_call_id=tool_call_id,
            kind=kind.value,
            payload=payload,
        )

    def delivered(
        self,
        tool_call_id: uuid.UUID,
        *,
        delivered_result_ref: uuid.UUID,
        observed_at: datetime,
    ) -> None:
        ledger_tool_calls.record_tool_call_delivered(
            self._conn,
            tool_call_id,
            delivered_result_ref=delivered_result_ref,
            observed_at=observed_at,
        )
