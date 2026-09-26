"""Canonical ledger event vocabularies (INTERFACES.md "Event vocabularies").

One enum per event table. Values are exactly the `event_type` CHECK sets in
`migrations/0001_initial.sql`; `tests/unit/test_event_vocabulary.py` asserts the parity.
The ledger writers import these; there is no second vocabulary for the same table.
"""

from enum import StrEnum


class RunEventType(StrEnum):
    """run_events.event_type. `status` is the only event that carries a RunStatus."""

    STARTED = "started"
    RECOVERY_STARTED = "recovery_started"
    STATUS = "status"
    CONTROL = "control"
    SOURCE_STATUS = "source_status"
    MARKET_SESSION = "market_session"
    METADATA = "metadata"
    AUDIT_STATUS = "audit_status"


class ToolCallEventType(StrEnum):
    """tool_call_events.event_type.

    `requested` is committed before dispatch; `dispatched` records the allowed (possibly
    rewritten) arguments; denied/succeeded/failed/timed_out/unknown are outcomes (the
    ToolCallStatus values); `delivered` references the exact envelope handed to the model.
    """

    REQUESTED = "requested"
    DISPATCHED = "dispatched"
    DENIED = "denied"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"
    DELIVERED = "delivered"


class OrderIntentEventType(StrEnum):
    """order_intent_events.event_type: facts about a placement intent with no broker order.

    `reviewed` links the review call that preceded the place call; `resolved_no_order`
    records broker reads proving no order exists for the intent; `reconciliation` records a
    reconciliation note that does not resolve it.
    """

    REVIEWED = "reviewed"
    RESOLVED_NO_ORDER = "resolved_no_order"
    RECONCILIATION = "reconciliation"


class OrderEventType(StrEnum):
    """order_events.event_type. Fills are stored inline (fill_quantity/fill_price)."""

    INTENT_LINKED = "intent_linked"
    DECISION_LINKED = "decision_linked"
    STATUS_OBSERVED = "status_observed"
    FILL_OBSERVED = "fill_observed"
    CANCELLATION_OBSERVED = "cancellation_observed"
    RECONCILIATION = "reconciliation"


class PositionEventType(StrEnum):
    """position_events.event_type."""

    OPENED = "opened"
    FILL_LINKED = "fill_linked"
    ROLL = "roll"
    ASSIGNMENT = "assignment"
    CLOSED = "closed"
    GAP = "gap"
    RECONCILIATION = "reconciliation"
    NOTE = "note"


class WorkspaceEventType(StrEnum):
    """workspace_events.event_type. Before/after state is stored inline on the event."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    OBSERVED = "observed"
    RECONCILIATION = "reconciliation"


__all__ = [
    "OrderEventType",
    "OrderIntentEventType",
    "PositionEventType",
    "RunEventType",
    "ToolCallEventType",
    "WorkspaceEventType",
]
