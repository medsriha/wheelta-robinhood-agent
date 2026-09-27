"""Typed evidence contract between remote-tool result mappers and the result boundary.

Split out of `result_boundary.py` so verified mappers (`robinhood_mappers.py`) can build
`MappedEvidence` while `result_boundary.VERIFIED_MAPPERS` registers them, without an import
cycle. `result_boundary` re-exports every name here.
"""

import uuid
from collections.abc import Callable
from typing import Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import NonEmptyStr, Ref
from wheelta_robinhood_agent.domain.enums import CandidateOrigin
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


class MappedEvidence(_Model):
    """Normalized, typed evidence produced from one validated tool result."""

    instruments: tuple[OptionInstrument, ...] = ()
    option_quotes: tuple[Quote, ...] = ()
    underlying_quotes: tuple[UnderlyingQuote, ...] = ()
    account_snapshots: tuple[AccountSnapshot, ...] = ()
    positions: tuple[PositionsRead, ...] = ()
    open_orders: tuple[OpenOrdersRead, ...] = ()
    candidates: tuple[CandidateEvidence, ...] = ()
    gaps: tuple[str, ...] = ()

    def evidence_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(i.evidence_id for i in self.instruments),
            *(q.quote_id for q in self.option_quotes),
            *(u.evidence_id for u in self.underlying_quotes),
            *(a.snapshot_id for a in self.account_snapshots),
            *(p.evidence_id for p in self.positions),
            *(o.evidence_id for o in self.open_orders),
        )

    def source_tool_call_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(t for i in self.instruments for t in i.source_tool_call_ids),
            *(t for q in self.option_quotes for t in q.source_tool_call_ids),
            *(t for u in self.underlying_quotes for t in u.source_tool_call_ids),
            *(t for a in self.account_snapshots for t in a.tool_call_ids),
            *(t for p in self.positions for t in p.source_tool_call_ids),
            *(t for o in self.open_orders for t in o.source_tool_call_ids),
        )


class MappingRequest(_Model):
    """Input to an `EvidenceMapper`: the parsed (redacted) payload of one successful call."""

    tool_call_id: uuid.UUID
    server: str
    tool: str
    effective_input: dict[str, JsonValue]
    payload: JsonValue
    retrieved_at: AwareDatetime


class EvidenceMapper(Protocol):
    """Map one tool's verified result schema to typed evidence. Raise on any schema mismatch.

    `new_id` issues evidence IDs and candidate refs, so every identity is code-issued.
    """

    def __call__(
        self, request: MappingRequest, new_id: Callable[[], uuid.UUID]
    ) -> MappedEvidence: ...
