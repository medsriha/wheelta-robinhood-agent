"""AuditContext: the frozen, recorded input of the post-run audit (docs/VALIDATION.md).

The orchestrator/agent layer builds this context from ledger data after the session ends.
It holds both the assembled RunRecord and the independent evidence it was built from (tool
events, broker order projections, fills, quotes, instrument results, broker state reads),
so the checks can compare the record against original evidence instead of against itself.

Evidence models here cover broker facts the domain package does not model yet (instrument
multiplier/tick, review results, positions/open-order reads). Every broker field mapping is
**unverified** until captured fixtures exist (CLAUDE.md §9); unknown values are `None`, and
the checks treat `None` as unverifiable, never as zero.
"""

import hashlib
import json
from datetime import date
from typing import Self
from uuid import UUID

from pydantic import PrivateAttr, StrictInt, model_validator

from wheelta_robinhood_agent.config.rules import TradingRules
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import (
    Count,
    DomainModel,
    NonEmptyStr,
    PosCount,
    PosDec,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.decision_output import AgentDecisionOutput
from wheelta_robinhood_agent.domain.enums import ExecutionMode
from wheelta_robinhood_agent.domain.facts import DecisionFacts
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import OrderRecord
from wheelta_robinhood_agent.domain.positions import PositionBook
from wheelta_robinhood_agent.domain.run_record import Quote, RunRecord
from wheelta_robinhood_agent.domain.tool_calls import ToolCallRecord

AUDIT_VERSION = "audit-v1"


class InstrumentFact(DomainModel):
    """A validated option-instrument result: identity, multiplier, and tick increment.

    `multiplier`/`tick_increment` are None when the broker field is unmapped or absent.
    """

    broker_instrument_id: NonEmptyStr
    occ_symbol: OccSymbol
    multiplier: PosCount | None
    tick_increment: PosDec | None
    source_tool_call_id: UUID


class ReviewObservation(DomainModel):
    """A recorded `review_option_order` result, normalized.

    `succeeded` is True only when the broker accepted the review; any warning is recorded in
    `warnings` and prevents the review from counting as a matching successful review (V3.1).
    Reviewed parameters are None when the broker field is unmapped.
    """

    review_tool_call_id: UUID
    completed_at: UtcDatetime
    succeeded: bool
    warnings: tuple[NonEmptyStr, ...] = ()
    broker_instrument_id: NonEmptyStr | None
    side_raw: NonEmptyStr | None
    quantity: StrictInt | None
    order_type_raw: NonEmptyStr | None
    time_in_force_raw: NonEmptyStr | None
    limit_price: PosDec | None


class ShortOptionHolding(DomainModel):
    """A short option position on the Agentic account, as the broker reported it."""

    broker_instrument_id: NonEmptyStr
    short_quantity: Count


class ShareHolding(DomainModel):
    """Owned shares of one underlying.

    `other_reserved_shares` holds broker-confirmed share reservations other than short calls
    and working sell-to-open calls; None means the broker field is unmapped/unknown.
    """

    underlying: NonEmptyStr
    quantity: Count
    other_reserved_shares: Count | None


class WorkingOrderObservation(DomainModel):
    """An outstanding option order on the account (any actor), from an open-orders read.

    `filled_quantity` None means the filled portion is unknown; `owned` marks orders the
    agent placed (ledger-recorded).
    """

    broker_order_id: NonEmptyStr
    broker_instrument_id: NonEmptyStr
    side_raw: NonEmptyStr | None
    quantity: PosCount
    filled_quantity: Count | None
    owned: bool

    @model_validator(mode="after")
    def _check_working(self) -> Self:
        if self.filled_quantity is not None and self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed quantity")
        return self


class BrokerState(DomainModel):
    """An AccountSnapshot plus the position/open-order reads taken with it.

    `completed_at` is the completion time of the latest read composing this state; a state
    completed after a place dispatch is not pre-order evidence (VALIDATION.md). Collections
    are None when the read is missing or unmapped. `reflects_mutation_tool_call_ids` lists
    the place/cancel calls this state is verified to include (reservation visibility).
    """

    snapshot: AccountSnapshot
    completed_at: UtcDatetime
    short_options: tuple[ShortOptionHolding, ...] | None
    share_holdings: tuple[ShareHolding, ...] | None
    working_orders: tuple[WorkingOrderObservation, ...] | None
    reflects_mutation_tool_call_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def _check_state(self) -> Self:
        if self.completed_at < self.snapshot.retrieved_at:
            raise ValueError("a broker state completes no earlier than its snapshot retrieval")
        require_unique(self.reflects_mutation_tool_call_ids, "mutation tool call id")
        return self


class DayHistory(DomainModel):
    """New position lineages first filled earlier on this NYSE trading day (V7.2)."""

    trading_date: date
    prior_new_lineage_ids: tuple[UUID, ...]

    @model_validator(mode="after")
    def _check_day(self) -> Self:
        require_unique(self.prior_new_lineage_ids, "lineage id")
        return self


class AuditContext(DomainModel):
    """Everything the V1–V7 checks read. Built outside the pure functions; no I/O here.

    - `tool_calls`: every recorded tool call of the run (ToolCallRecord projections), in
      recorded order; tool events are the authority for live chronology.
    - `order_records`: broker-order projections (intent = verbatim place arguments, status
      history, cancellations, fills) for this run's placements and any order they touched.
    - `run_record` / `decision_output`: None when assembly or model output is unavailable.
    - `position_book`: the initial PositionBook delivered to the model (None if unknown).
    - `day_history`: None when earlier same-day lineages cannot be established.
    """

    run_id: UUID
    effective_execution_mode: ExecutionMode
    rules: TradingRules
    rules_version: NonEmptyStr
    rules_hash: NonEmptyStr
    audit_version: NonEmptyStr = AUDIT_VERSION
    assembler_version: NonEmptyStr | None = None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    order_records: tuple[OrderRecord, ...] = ()
    quotes: tuple[Quote, ...] = ()
    instruments: tuple[InstrumentFact, ...] = ()
    reviews: tuple[ReviewObservation, ...] = ()
    broker_states: tuple[BrokerState, ...] = ()
    decision_facts: tuple[DecisionFacts, ...] = ()
    decision_output: AgentDecisionOutput | None = None
    run_record: RunRecord | None = None
    position_book: PositionBook | None = None
    day_history: DayHistory | None = None

    _hash: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_context(self) -> Self:
        require_unique(tuple(t.identity.tool_call_id for t in self.tool_calls), "tool call id")
        require_unique(tuple(q.quote_id for q in self.quotes), "quote id")
        require_unique(tuple(r.review_tool_call_id for r in self.reviews), "review call id")
        require_unique(tuple(f.facts_ref for f in self.decision_facts), "facts ref")
        if any(t.identity.run_id != self.run_id for t in self.tool_calls):
            raise ValueError("every tool call must belong to the audited run")
        if self.run_record is not None:
            if self.run_record.run_id != self.run_id:
                raise ValueError("the run record belongs to another run")
            if self.run_record.effective_execution_mode is not self.effective_execution_mode:
                raise ValueError("the run record disagrees on the effective execution mode")
        return self

    @property
    def context_hash(self) -> str:
        """SHA-256 of the canonical JSON of this context (sorted keys, exact Decimals)."""
        if self._hash is None:
            payload = json.dumps(
                self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            )
            self._hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self._hash


__all__ = [
    "AUDIT_VERSION",
    "AuditContext",
    "BrokerState",
    "DayHistory",
    "InstrumentFact",
    "ReviewObservation",
    "ShareHolding",
    "ShortOptionHolding",
    "WorkingOrderObservation",
]
