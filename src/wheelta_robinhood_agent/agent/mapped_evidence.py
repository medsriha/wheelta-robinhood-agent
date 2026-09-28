"""Typed evidence contract between remote-tool result mappers and the result boundary.

Split out of `result_boundary.py` so verified mappers (`robinhood_mappers.py`) can build
`MappedEvidence` while `result_boundary.VERIFIED_MAPPERS` registers them, without an import
cycle. `result_boundary` re-exports every name here.
"""

import uuid
from collections.abc import Callable
from decimal import Decimal
from typing import Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import NonEmptyStr, Ref
from wheelta_robinhood_agent.domain.enums import AttemptStatus, CandidateOrigin
from wheelta_robinhood_agent.domain.facts_compute import (
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
    UnderlyingQuote,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.run_record import Quote


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateEvidence(_Model):
    """A code-issued candidate reference and the instrument it names (OUTPUT_ASSEMBLY.md).

    The model may select `candidate_ref`; it can never mint one from a ticker string.
    """

    candidate_ref: Ref
    origin: CandidateOrigin
    underlying: NonEmptyStr
    instrument_evidence_id: uuid.UUID
    broker_instrument_id: NonEmptyStr
    occ_symbol: OccSymbol


class _Observed(_Model):
    evidence_id: uuid.UUID
    as_of: AwareDatetime
    source_tool_call_ids: tuple[uuid.UUID, ...]


class OrderLeg(_Model):
    """One leg of a broker order or review, identified by the broker instrument ID.

    `side_raw` joins the broker's `side` and `position_effect` (`sell` + `open` ->
    `sell_to_open`), so it compares directly with `OrderSide` values; other combinations are
    kept verbatim and simply match no permitted side.
    """

    broker_instrument_id: NonEmptyStr
    side_raw: NonEmptyStr
    occ_symbol: OccSymbol | None = None


class Execution(_Model):
    """One broker execution (fill) of an order leg."""

    broker_execution_id: NonEmptyStr
    quantity: int
    price: Decimal
    executed_at: AwareDatetime


class BrokerOrderObservation(_Observed):
    """One option order as the broker reported it (place result or order read, ADR-0034).

    `status` is the broker `state` mapped onto `AttemptStatus`; `state_raw` keeps the
    original. Quantities are whole contracts. `placed_agent` says who placed it.
    """

    broker_order_id: NonEmptyStr
    state_raw: NonEmptyStr
    status: AttemptStatus
    underlying: NonEmptyStr
    order_type_raw: NonEmptyStr
    trigger_raw: NonEmptyStr
    time_in_force_raw: NonEmptyStr
    quantity: int
    processed_quantity: int
    pending_quantity: int
    canceled_quantity: int
    limit_price: Decimal | None
    multiplier: int
    placed_agent: str | None
    created_at: AwareDatetime
    legs: tuple[OrderLeg, ...]
    executions: tuple[Execution, ...] = ()


class OrderReviewObservation(_Observed):
    """A `review_option_order` result: the echoed order and the broker's pre-trade check.

    `clean` is True only when `order_checks` is the empty object, which the tool defines as
    "clean and safe to place". Otherwise `alert_type` names the issue.
    """

    legs: tuple[OrderLeg, ...]
    quantity: int
    order_type_raw: NonEmptyStr
    time_in_force_raw: str | None
    limit_price: Decimal | None
    clean: bool
    alert_type: str | None


class CancelRequestObservation(_Observed):
    """A `cancel_option_order` result. `accepted` means the broker accepted the request, not
    that the order is cancelled; only a later order read confirms that."""

    broker_order_id: NonEmptyStr
    accepted: bool


class HeldOptionRow(_Model):
    """A short option position row before its contract (strike, right) is resolved."""

    broker_instrument_id: NonEmptyStr
    underlying: NonEmptyStr
    short_quantity: int
    multiplier: int


class PendingOptionPositions(_Observed):
    """A complete options-positions read whose short rows still need their contracts.

    `get_option_positions` reports no strike or call/put, so the facts service resolves each
    row against this run's instrument evidence (`get_option_instruments ids=...`). Until every
    row resolves, no options half exists (ADR-0034)."""

    rows: tuple[HeldOptionRow, ...]


class MappedEvidence(_Model):
    """Normalized, typed evidence produced from one validated tool result."""

    instruments: tuple[OptionInstrument, ...] = ()
    option_quotes: tuple[Quote, ...] = ()
    underlying_quotes: tuple[UnderlyingQuote, ...] = ()
    account_snapshots: tuple[AccountSnapshot, ...] = ()
    positions: tuple[PositionsRead, ...] = ()
    open_orders: tuple[OpenOrdersRead, ...] = ()
    candidates: tuple[CandidateEvidence, ...] = ()
    broker_orders: tuple[BrokerOrderObservation, ...] = ()
    order_reviews: tuple[OrderReviewObservation, ...] = ()
    cancel_requests: tuple[CancelRequestObservation, ...] = ()
    pending_option_positions: tuple[PendingOptionPositions, ...] = ()
    gaps: tuple[str, ...] = ()

    def _observed(self) -> tuple[_Observed, ...]:
        return (
            *self.broker_orders,
            *self.order_reviews,
            *self.cancel_requests,
            *self.pending_option_positions,
        )

    def evidence_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(i.evidence_id for i in self.instruments),
            *(q.quote_id for q in self.option_quotes),
            *(u.evidence_id for u in self.underlying_quotes),
            *(a.snapshot_id for a in self.account_snapshots),
            *(p.evidence_id for p in self.positions),
            *(o.evidence_id for o in self.open_orders),
            *(o.evidence_id for o in self._observed()),
        )

    def source_tool_call_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(t for i in self.instruments for t in i.source_tool_call_ids),
            *(t for q in self.option_quotes for t in q.source_tool_call_ids),
            *(t for u in self.underlying_quotes for t in u.source_tool_call_ids),
            *(t for a in self.account_snapshots for t in a.tool_call_ids),
            *(t for p in self.positions for t in p.source_tool_call_ids),
            *(t for o in self.open_orders for t in o.source_tool_call_ids),
            *(t for o in self._observed() for t in o.source_tool_call_ids),
        )


class MappingRequest(_Model):
    """Input to an `EvidenceMapper`: the parsed (redacted) payload of one successful call."""

    tool_call_id: uuid.UUID
    server: str
    tool: str
    effective_input: dict[str, JsonValue]
    payload: JsonValue
    retrieved_at: AwareDatetime
    # This run's trusted Agentic-eligibility check passed (agent/session.py, CLAUDE.md §9).
    account_eligible: bool = False


class EvidenceMapper(Protocol):
    """Map one tool's verified result schema to typed evidence. Raise on any schema mismatch.

    `new_id` issues evidence IDs and candidate refs, so every identity is code-issued.
    """

    def __call__(
        self, request: MappingRequest, new_id: Callable[[], uuid.UUID]
    ) -> MappedEvidence: ...
