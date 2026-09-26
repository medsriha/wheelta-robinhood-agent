"""Order intents, broker orders, fills, attempts, cancellations, and workspace objects.

Contracts: INTERFACES.md "Orders, fills, and workspace projections" and the Attempt /
Cancellation rows of "DecisionFacts, AgentDecisionOutput v5, and RunRecord v4".

Recorded tool arguments are preserved exactly as submitted (side, type, TIF are raw
strings), so an order that violated the rules is still representable and auditable (V1).
Broker facts override model claims. Unknown outcomes stay unknown; no broker execution ID is
ever manufactured and repeated cumulative fill totals are never summed.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import StrictInt, model_validator

from wheelta_robinhood_agent.domain.base import (
    Count,
    DomainModel,
    NonEmptyStr,
    PosDec,
    Ref,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    CancelDispatchStatus,
    CancellationStatus,
    OrderSide,
)
from wheelta_robinhood_agent.domain.options import OccSymbol


class ReasonCode(StrEnum):
    """Factual reason codes stamped by code on attempts/legs (OUTPUT_ASSEMBLY.md).

    Only codes named in the docs are listed; the full closed set is **TBD** (not defined in
    INTERFACES.md). Add members with the doc change that defines them.
    """

    DRY_RUN = "DRY_RUN"
    NO_CAPACITY = "no_capacity"
    QUANTITY_UNAVAILABLE = "quantity_unavailable"
    NOT_DISPATCHED = "not_dispatched"
    OUTCOME_UNKNOWN = "outcome_unknown"
    BROKER_REJECTED = "broker_rejected"
    TOOL_ERROR = "tool_error"


class OrderIntent(DomainModel):
    """Placement intent, identified by its place tool call even if no broker ID is known.

    Fields are the recorded place arguments, verbatim (INTERFACES.md "Orders").
    """

    intent_id: UUID
    run_id: UUID
    place_tool_call_id: UUID
    account_scope_id: NonEmptyStr
    occ_symbol: OccSymbol | None
    broker_instrument_id: NonEmptyStr | None
    side_raw: NonEmptyStr | None
    quantity: StrictInt | None
    order_type_raw: NonEmptyStr | None
    time_in_force_raw: NonEmptyStr | None
    limit_price: PosDec | None
    requested_at: UtcDatetime

    @property
    def side(self) -> OrderSide | None:
        """The permitted side the raw argument names, or None (not STO/BTC, or absent)."""
        if self.side_raw is None:
            return None
        try:
            return OrderSide(self.side_raw)
        except ValueError:
            return None


class BrokerOrder(DomainModel):
    """Broker-order identity: unique per (account_scope_id, broker_order_id)."""

    order_id: UUID
    account_scope_id: NonEmptyStr
    broker_order_id: NonEmptyStr
    intent_id: UUID | None
    first_observed_at: UtcDatetime


class FillObservationKind(StrEnum):
    """How the broker reported a fill (INTERFACES.md "FillRecord")."""

    EXECUTION = "execution"
    """A single execution with a verified broker execution ID."""
    CUMULATIVE = "cumulative"
    """A cumulative filled total for the order; deltas are derived once, never summed."""


class FillRecord(DomainModel):
    """An immutable fill observation.

    EXECUTION requires `broker_execution_id` and a positive quantity; CUMULATIVE forbids an
    execution ID (never manufactured) and records the running total. `price` is the reported
    execution or average price, when the broker gave one.
    """

    fill_id: UUID
    order_id: UUID
    kind: FillObservationKind
    broker_execution_id: NonEmptyStr | None
    quantity: Count
    price: PosDec | None
    executed_at: UtcDatetime | None
    observed_at: UtcDatetime
    source_tool_call_id: UUID

    @model_validator(mode="after")
    def _check_fill(self) -> Self:
        if self.kind is FillObservationKind.EXECUTION:
            if self.broker_execution_id is None:
                raise ValueError("an execution fill needs a broker execution id")
            if self.quantity == 0:
                raise ValueError("an execution fill has a positive quantity")
        elif self.broker_execution_id is not None:
            raise ValueError("a cumulative observation has no broker execution id")
        return self


def filled_quantity(fills: Iterable[FillRecord]) -> int | None:
    """Cumulative filled contracts for one broker order, or None if inconsistent.

    Rules (INTERFACES.md "FillRecord"): executions are deduplicated by broker execution ID
    and summed; cumulative totals are observations, so the filled quantity is the latest
    total (by `observed_at`) and totals must never decrease. If both kinds exist they must
    agree. Conflicting duplicates, mixed orders, or decreasing totals return None.
    """
    items = list(fills)
    if not items:
        return 0
    if len({f.order_id for f in items}) != 1:
        return None
    executions: dict[str, int] = {}
    cumulative: list[FillRecord] = []
    for fill in items:
        if fill.kind is FillObservationKind.EXECUTION:
            if fill.broker_execution_id is None:  # unreachable: enforced by FillRecord
                return None
            prior = executions.get(fill.broker_execution_id)
            if prior is not None and prior != fill.quantity:
                return None
            executions[fill.broker_execution_id] = fill.quantity
        else:
            cumulative.append(fill)
    cumulative.sort(key=lambda f: f.observed_at)
    totals = [f.quantity for f in cumulative]
    if any(later < earlier for earlier, later in zip(totals, totals[1:], strict=False)):
        return None
    execution_total = sum(executions.values()) if executions else None
    cumulative_total = totals[-1] if totals else None
    if execution_total is not None and cumulative_total is not None:
        return execution_total if execution_total == cumulative_total else None
    return execution_total if execution_total is not None else cumulative_total


class StatusObservation(DomainModel):
    """One observed broker status, with its source tool call."""

    status: AttemptStatus
    broker_status_raw: NonEmptyStr | None
    observed_at: UtcDatetime
    tool_call_id: UUID


class Cancellation(DomainModel):
    """A cancel call and its observed outcome (INTERFACES.md "Cancellation").

    `confirmed` requires an order read proving terminal state; pending releases nothing.
    Fills that beat the cancellation are kept in `fill_ids`. `dispatch_status` and
    `dispatch_reason` are set by run assembly from the cancel's ToolCallRecord (None in the
    ledger's order projection); a denied or undispatched cancel stays `unknown`, never
    pending or confirmed.
    """

    cancel_tool_call_id: UUID
    broker_order_id: NonEmptyStr | None
    confirmation_tool_call_ids: tuple[UUID, ...] = ()
    status: CancellationStatus
    fill_ids: tuple[UUID, ...] = ()
    evidence_tool_call_ids: tuple[UUID, ...] = ()
    dispatch_status: CancelDispatchStatus | None = None
    dispatch_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check_cancellation(self) -> Self:
        if self.status is CancellationStatus.CONFIRMED and not self.confirmation_tool_call_ids:
            raise ValueError("a confirmed cancellation needs a confirming order read")
        undispatched = self.dispatch_status in (
            CancelDispatchStatus.DENIED,
            CancelDispatchStatus.NO_DISPATCH_RECORDED,
        )
        if undispatched and self.status is not CancellationStatus.UNKNOWN:
            raise ValueError("an undispatched cancel is neither confirmed nor pending")
        if (self.dispatch_status is CancelDispatchStatus.DENIED) != (
            self.dispatch_reason is not None
        ):
            raise ValueError("dispatch_reason is required exactly for a denied cancel")
        require_unique(self.fill_ids, "fill id")
        return self


class Attempt(DomainModel):
    """One execution attempt within a leg (INTERFACES.md "Attempt").

    Identity is either the recorded place call (`place_tool_call_id`) or, for an
    unsubmitted dry-run proposal, a code-derived `proposal_ref`; exactly one is set.
    Requested parameters are the recorded arguments, never recopied from model JSON.
    A proposal is `not_placed` with no broker, review, cancel, or fill references and a
    positive computed quantity (OUTPUT_ASSEMBLY.md: DRY_RUN only with known positive size).
    A denied place call is `not_placed` with no broker order ID.
    """

    index: Count
    place_tool_call_id: UUID | None
    proposal_ref: Ref | None
    requested_quantity: StrictInt | None
    order_type_raw: NonEmptyStr | None
    time_in_force_raw: NonEmptyStr | None
    limit_price: PosDec | None
    snapshot_ref: UUID | None
    quote_refs: tuple[UUID, ...] = ()
    status: AttemptStatus
    broker_order_id: NonEmptyStr | None
    review_tool_call_ids: tuple[UUID, ...] = ()
    cancel_tool_call_ids: tuple[UUID, ...] = ()
    confirmation_tool_call_ids: tuple[UUID, ...] = ()
    fill_ids: tuple[UUID, ...] = ()
    filled_quantity: Count | None
    reason_codes: tuple[ReasonCode, ...] = ()

    @model_validator(mode="after")
    def _check_attempt(self) -> Self:
        if (self.place_tool_call_id is None) == (self.proposal_ref is None):
            raise ValueError("exactly one of place_tool_call_id / proposal_ref is required")
        require_unique(self.reason_codes, "reason code")
        require_unique(self.fill_ids, "fill id")
        if self.proposal_ref is not None:
            if self.status is not AttemptStatus.NOT_PLACED:
                raise ValueError("an unsubmitted proposal is not_placed")
            if (
                self.broker_order_id is not None
                or self.review_tool_call_ids
                or self.cancel_tool_call_ids
                or self.confirmation_tool_call_ids
                or self.fill_ids
            ):
                raise ValueError("an unsubmitted proposal has no broker-action references")
            if self.requested_quantity is None or self.requested_quantity <= 0:
                raise ValueError("a proposal attempt needs a known positive quantity")
        if self.status is AttemptStatus.NOT_PLACED:
            if self.broker_order_id is not None:
                raise ValueError("a not_placed attempt has no broker order id")
            if self.fill_ids or self.filled_quantity:
                raise ValueError("a not_placed attempt has no fills")
        return self


class OrderRecord(DomainModel):
    """Projection of one placement: intent, broker identity, events, and fills.

    Rebuilt from identities and `order_events`; never the only copy of evidence. The status
    is the latest observation (by `observed_at`), or UNKNOWN when nothing was observed.
    """

    intent: OrderIntent | None
    broker_order: BrokerOrder | None
    review_tool_call_ids: tuple[UUID, ...] = ()
    status_history: tuple[StatusObservation, ...] = ()
    cancellations: tuple[Cancellation, ...] = ()
    fills: tuple[FillRecord, ...] = ()
    decision_ref: Ref | None = None
    leg_ref: Ref | None = None

    @model_validator(mode="after")
    def _check_order_record(self) -> Self:
        if self.intent is None and self.broker_order is None:
            raise ValueError("an order record needs an intent or a broker order")
        if (
            self.intent is not None
            and self.broker_order is not None
            and self.broker_order.intent_id not in (None, self.intent.intent_id)
        ):
            raise ValueError("broker order is linked to a different intent")
        if self.fills:
            if self.broker_order is None:
                raise ValueError("fills require a broker order identity")
            if any(f.order_id != self.broker_order.order_id for f in self.fills):
                raise ValueError("fills must belong to this order")
        require_unique(tuple(f.fill_id for f in self.fills), "fill id")
        return self

    @property
    def status(self) -> AttemptStatus:
        if not self.status_history:
            return AttemptStatus.UNKNOWN
        return max(self.status_history, key=lambda s: s.observed_at).status

    @property
    def filled_quantity(self) -> int | None:
        return filled_quantity(self.fills)


class WorkspaceObjectKind(StrEnum):
    """Agentic workspace object kinds (CLAUDE.md §9 Tier S)."""

    SCAN = "scan"
    WATCHLIST = "watchlist"
    OPTION_WATCHLIST = "option_watchlist"
    ALERT = "alert"


class WorkspaceObject(DomainModel):
    """Immutable workspace object identity: unique per (account_scope_id, kind, broker_id)."""

    object_id: UUID
    account_scope_id: NonEmptyStr
    kind: WorkspaceObjectKind
    broker_id: NonEmptyStr
    created_run_id: UUID
    created_at: UtcDatetime


def is_owned_name(name: str, prefix: str) -> bool:
    """The ownership name rule: the object's name starts with the workspace prefix.

    Necessary but not sufficient: ownership also requires the ledger to have recorded the
    object's ID (CLAUDE.md §9 "Ownership namespace"). An empty prefix owns nothing.
    """
    return bool(prefix) and name.startswith(prefix)


class WorkspaceObjectState(DomainModel):
    """Projection of a workspace object's current name, ownership, and deletion state."""

    identity: WorkspaceObject
    current_name: NonEmptyStr | None
    ledger_recorded: bool
    last_modified_at: UtcDatetime
    deleted: bool

    def owned(self, prefix: str) -> bool:
        """Mutable by the agent only if prefix AND ledger record match and not deleted."""
        return (
            self.ledger_recorded
            and not self.deleted
            and self.current_name is not None
            and is_owned_name(self.current_name, prefix)
        )


__all__ = [
    "Attempt",
    "BrokerOrder",
    "Cancellation",
    "FillObservationKind",
    "FillRecord",
    "OrderIntent",
    "OrderRecord",
    "ReasonCode",
    "StatusObservation",
    "WorkspaceObject",
    "WorkspaceObjectKind",
    "WorkspaceObjectState",
    "filled_quantity",
    "is_owned_name",
]
