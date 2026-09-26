"""Step 1 of assembly: rebuild every place/cancel action from recorded events.

OUTPUT_ASSEMBLY.md "Final record assembly" step 1: denied calls are not dispatched;
dispatched actions without a confirmed outcome stay unknown; fills are deduplicated with
`orders.filled_quantity` (never summing repeated cumulative totals); a confirmed cancellation
needs a successful terminal order read after the cancel was dispatched, and a pending
cancellation releases nothing. Nothing here depends on model output.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    CancelDispatchStatus,
    CancellationStatus,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.orders import (
    Attempt,
    Cancellation,
    OrderIntent,
    OrderRecord,
    ReasonCode,
)
from wheelta_robinhood_agent.domain.run_record import AssemblyFinding
from wheelta_robinhood_agent.domain.tool_calls import ToolCallRecord

PLACE_TOOL: Final = "place_option_order"
CANCEL_TOOL: Final = "cancel_option_order"
REVIEW_TOOL: Final = "review_option_order"
ORDERS_READ_TOOL: Final = "get_option_orders"

TERMINAL_STATUSES: Final = frozenset(
    {
        AttemptStatus.FILLED,
        AttemptStatus.CANCELLED,
        AttemptStatus.REJECTED,
        AttemptStatus.EXPIRED,
    }
)


def tool_name(call: ToolCallRecord) -> str:
    """The bare tool name (`mcp__robinhood__place_option_order` → `place_option_order`)."""
    return call.identity.tool.rsplit("__", 1)[-1]


def call_sort_key(call: ToolCallRecord) -> tuple[str, str, str]:
    """Chronological order of tool calls: request time, then SDK id, then identity."""
    ident = call.identity
    return (ident.requested_at.isoformat(), ident.sdk_tool_use_id, str(ident.tool_call_id))


@dataclass(frozen=True, slots=True)
class PlaceAction:
    """A recorded place call rebuilt as an attempt (index assigned later per leg)."""

    call: ToolCallRecord
    intent: OrderIntent | None
    attempt: Attempt


@dataclass(slots=True)
class EventIndex:
    """Everything step 1 derived from events, in deterministic chronological order."""

    calls_by_id: dict[UUID, ToolCallRecord]
    places: list[PlaceAction]
    cancellations: list[Cancellation]
    order_place_call_of_cancel: dict[UUID, UUID | None]
    findings: list[AssemblyFinding] = field(default_factory=list)

    def place(self, call_id: UUID) -> PlaceAction | None:
        return next((p for p in self.places if p.call.identity.tool_call_id == call_id), None)


def _finding(code: str, detail: str, *call_ids: UUID) -> AssemblyFinding:
    return AssemblyFinding(code=code, detail=detail, tool_call_ids=tuple(call_ids))


def _orders_by_place_call(
    orders: Iterable[OrderRecord], findings: list[AssemblyFinding]
) -> dict[UUID, OrderRecord | None]:
    """Map place call → its order record; conflicting duplicates map to None (ambiguous)."""
    result: dict[UUID, OrderRecord | None] = {}
    for order in orders:
        if order.intent is None:
            continue
        key = order.intent.place_tool_call_id
        if key in result and result[key] != order:
            findings.append(
                _finding(
                    "ambiguous_order_record",
                    "more than one distinct order record claims this place call",
                    key,
                )
            )
            result[key] = None
        elif key not in result:
            result[key] = order
    return result


def _status_and_reasons(
    call: ToolCallRecord, order: OrderRecord | None, findings: list[AssemblyFinding]
) -> tuple[AttemptStatus, list[ReasonCode]]:
    call_id = call.identity.tool_call_id
    has_broker = order is not None and order.broker_order is not None
    reasons: list[ReasonCode] = []
    if call.status is ToolCallStatus.DENIED:
        if not has_broker:
            return AttemptStatus.NOT_PLACED, [ReasonCode.NOT_DISPATCHED]
        findings.append(
            _finding(
                "denied_call_has_broker_order",
                "a denied place call has a broker order; broker observations are kept",
                call_id,
            )
        )
    if order is not None and order.status_history:
        status = order.status
    elif call.status is ToolCallStatus.SUCCEEDED and has_broker:
        status = AttemptStatus.PLACED
    else:
        status = AttemptStatus.UNKNOWN
    if status is AttemptStatus.NOT_PLACED and (has_broker or (order is not None and order.fills)):
        findings.append(
            _finding(
                "contradictory_order_status",
                "order observed as not_placed but has a broker order or fills",
                call_id,
            )
        )
        status = AttemptStatus.UNKNOWN
    if call.status is ToolCallStatus.FAILED:
        reasons.append(ReasonCode.TOOL_ERROR)
    if status is AttemptStatus.REJECTED:
        reasons.append(ReasonCode.BROKER_REJECTED)
    if status is AttemptStatus.UNKNOWN:
        reasons.append(ReasonCode.OUTCOME_UNKNOWN)
    return status, reasons


def build_place_attempt(
    call: ToolCallRecord,
    order: OrderRecord | None,
    snapshot_ref: UUID | None,
    quote_refs: tuple[UUID, ...],
    findings: list[AssemblyFinding],
) -> Attempt:
    """One attempt from a recorded place call; requested fields are the recorded arguments."""
    call_id = call.identity.tool_call_id
    status, reasons = _status_and_reasons(call, order, findings)
    intent = order.intent if order is not None else None
    if intent is None:
        findings.append(
            _finding(
                "place_intent_missing",
                "no recorded order intent for this place call; requested fields unknown",
                call_id,
            )
        )
    not_placed = status is AttemptStatus.NOT_PLACED
    fills = () if order is None or not_placed else order.fills
    fill_ids = tuple(f.fill_id for f in sorted(fills, key=lambda f: (f.observed_at, f.fill_id)))
    filled: int | None
    if not_placed:
        filled = 0
    elif order is None:
        filled = None
    else:
        filled = order.filled_quantity
        if filled is None:
            findings.append(
                _finding(
                    "inconsistent_fills",
                    "fill observations for this order are inconsistent; filled quantity unknown",
                    call_id,
                )
            )
    cancellations = () if order is None else order.cancellations
    broker = None if order is None or order.broker_order is None or not_placed else order
    return Attempt(
        index=0,
        place_tool_call_id=call_id,
        proposal_ref=None,
        requested_quantity=intent.quantity if intent else None,
        order_type_raw=intent.order_type_raw if intent else None,
        time_in_force_raw=intent.time_in_force_raw if intent else None,
        limit_price=intent.limit_price if intent else None,
        snapshot_ref=snapshot_ref,
        quote_refs=quote_refs,
        status=status,
        broker_order_id=broker.broker_order.broker_order_id
        if broker is not None and broker.broker_order is not None
        else None,
        review_tool_call_ids=() if order is None else order.review_tool_call_ids,
        cancel_tool_call_ids=tuple(c.cancel_tool_call_id for c in cancellations),
        confirmation_tool_call_ids=tuple(
            dict.fromkeys(i for c in cancellations for i in c.confirmation_tool_call_ids)
        ),
        fill_ids=fill_ids,
        filled_quantity=filled,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _confirmation_verified(
    cancellation: Cancellation,
    cancel_call: ToolCallRecord,
    order: OrderRecord | None,
    calls_by_id: dict[UUID, ToolCallRecord],
) -> bool:
    """Confirmed needs a terminal order and successful order reads after cancel dispatch."""
    if order is None or order.status not in TERMINAL_STATUSES:
        return False
    dispatched = cancel_call.dispatched_at
    if dispatched is None:
        return False
    for read_id in cancellation.confirmation_tool_call_ids:
        read = calls_by_id.get(read_id)
        if (
            read is None
            or tool_name(read) != ORDERS_READ_TOOL
            or read.status is not ToolCallStatus.SUCCEEDED
            or read.completed_at is None
            or read.completed_at < dispatched
        ):
            return False
    return True


def _unconfirmed_status(cancel_call: ToolCallRecord) -> CancellationStatus:
    return (
        CancellationStatus.PENDING
        if cancel_call.status is ToolCallStatus.SUCCEEDED
        else CancellationStatus.UNKNOWN
    )


def _dispatch_status(cancel_call: ToolCallRecord) -> CancelDispatchStatus:
    if cancel_call.status is ToolCallStatus.DENIED:
        return CancelDispatchStatus.DENIED
    if cancel_call.dispatched_at is not None:
        return CancelDispatchStatus.DISPATCHED
    return CancelDispatchStatus.NO_DISPATCH_RECORDED


def _with_dispatch(cancellation: Cancellation, cancel_call: ToolCallRecord) -> Cancellation:
    """Stamp the cancel's dispatch status/reason; an undispatched cancel is never pending."""
    dispatch = _dispatch_status(cancel_call)
    update: dict[str, object] = {
        "dispatch_status": dispatch,
        "dispatch_reason": cancel_call.deny_reason
        if dispatch is CancelDispatchStatus.DENIED
        else None,
    }
    if dispatch is not CancelDispatchStatus.DISPATCHED:
        update["status"] = CancellationStatus.UNKNOWN
    return Cancellation.model_validate({**cancellation.model_dump(), **update})


def build_event_index(
    run_id: UUID,
    tool_calls: Iterable[ToolCallRecord],
    orders: tuple[OrderRecord, ...],
    evidence: dict[UUID, tuple[UUID | None, tuple[UUID, ...]]],
) -> EventIndex:
    """Rebuild all place attempts and cancellations for this run from recorded events."""
    findings: list[AssemblyFinding] = []
    calls: list[ToolCallRecord] = []
    for call in sorted(tool_calls, key=call_sort_key):
        if call.identity.run_id != run_id:
            findings.append(
                _finding(
                    "foreign_tool_call",
                    "tool call belongs to another run and is ignored",
                    call.identity.tool_call_id,
                )
            )
            continue
        calls.append(call)
    calls_by_id = {c.identity.tool_call_id: c for c in calls}
    by_place = _orders_by_place_call(orders, findings)

    places: list[PlaceAction] = []
    for call in calls:
        if tool_name(call) != PLACE_TOOL:
            continue
        order = by_place.get(call.identity.tool_call_id)
        snapshot_ref, quote_refs = evidence.get(call.identity.tool_call_id, (None, ()))
        attempt = build_place_attempt(call, order, snapshot_ref, quote_refs, findings)
        places.append(PlaceAction(call, order.intent if order else None, attempt))

    # Cancellation records per cancel call, from order projections (deduplicated).
    recorded: dict[UUID, tuple[Cancellation, OrderRecord] | None] = {}
    for order in orders:
        for cancellation in order.cancellations:
            key = cancellation.cancel_tool_call_id
            if key in recorded:
                prior = recorded[key]
                if prior is None or prior[0] != cancellation:
                    recorded[key] = None
            else:
                recorded[key] = (cancellation, order)
    cancel_calls = [c for c in calls if tool_name(c) == CANCEL_TOOL]
    cancel_ids = {c.identity.tool_call_id for c in cancel_calls}
    for key in sorted(k for k in recorded if k not in cancel_ids):
        findings.append(
            _finding(
                "cancellation_without_call",
                "an order projection lists a cancellation with no recorded cancel call",
                key,
            )
        )

    cancellations: list[Cancellation] = []
    place_of_cancel: dict[UUID, UUID | None] = {}
    for call in cancel_calls:
        call_id = call.identity.tool_call_id
        entry = recorded.get(call_id)
        if call_id in recorded and entry is None:
            findings.append(
                _finding(
                    "ambiguous_cancellation_record",
                    "conflicting cancellation records for this cancel call",
                    call_id,
                )
            )
        if call.status is ToolCallStatus.DENIED:
            findings.append(
                _finding("cancel_not_dispatched", "the cancel call was denied", call_id)
            )
        if entry is None:
            cancellations.append(
                _with_dispatch(
                    Cancellation(
                        cancel_tool_call_id=call_id,
                        broker_order_id=None,
                        status=_unconfirmed_status(call),
                    ),
                    call,
                )
            )
            place_of_cancel[call_id] = None
            continue
        cancellation, order = entry
        if cancellation.status is CancellationStatus.CONFIRMED and not _confirmation_verified(
            cancellation, call, order, calls_by_id
        ):
            findings.append(
                _finding(
                    "cancel_confirmation_unverified",
                    "confirmed cancellation lacks a successful terminal order read after "
                    "the cancel; recorded as not confirmed",
                    call_id,
                )
            )
            cancellation = cancellation.model_copy(update={"status": _unconfirmed_status(call)})
        cancellations.append(_with_dispatch(cancellation, call))
        place_of_cancel[call_id] = order.intent.place_tool_call_id if order.intent else None

    return EventIndex(
        calls_by_id=calls_by_id,
        places=places,
        cancellations=cancellations,
        order_place_call_of_cancel=place_of_cancel,
        findings=findings,
    )


__all__ = [
    "CANCEL_TOOL",
    "ORDERS_READ_TOOL",
    "PLACE_TOOL",
    "REVIEW_TOOL",
    "TERMINAL_STATUSES",
    "EventIndex",
    "PlaceAction",
    "build_event_index",
    "build_place_attempt",
    "call_sort_key",
    "tool_name",
]
