"""AssemblyContext: the immutable input bundle for `assemble_run_record` (OUTPUT_ASSEMBLY.md).

The bundle is loaded outside the pure assembler (agent/ledger code). It contains only
validated, redacted, Agentic-account data: trusted run metadata, the ordered tool-call
records, order projections, fact sets, and the registry of code-issued references that were
delivered to the model. Nothing in it is authored by the model.

Supporting models defined here exist because `run_record.py` does not carry them (see the
module docstring of `assembly.py` for the list of model gaps):

- `DeliveredRef`: a code-issued reference with its kind, run/account scope, delivery flag,
  and the instrument/position facts needed for compatibility checks;
- `AttemptEvidence`: the snapshot/quote evidence the loader linked to a live place call;
- `ReservationBaseline` / `ReservationRequirement`: the single dry-run reservation baseline
  and each proposal's per-contract resource use (VALIDATION.md "Dry-run reservations");
- `RankingKey`: which DecisionFacts metric implements each fixed `selection.ranking` key.
"""

from enum import StrEnum
from typing import Final, Self
from uuid import UUID

from pydantic import model_validator

from wheelta_robinhood_agent.domain.base import (
    Dec,
    DomainModel,
    NonEmptyStr,
    PosDec,
    Ref,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import AppEnv, CandidateOrigin, ExecutionMode
from wheelta_robinhood_agent.domain.facts import DecisionFacts
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import OrderRecord
from wheelta_robinhood_agent.domain.run_record import (
    AssemblyFinding,
    BoardComparisonRecord,
    Quote,
)
from wheelta_robinhood_agent.domain.tool_calls import ToolCallRecord

ASSEMBLER_VERSION: Final = "assembler.v1"
"""Version of the deterministic assembly algorithm; bump with any output-affecting change."""


class RefKind(StrEnum):
    """Kinds of code-issued references the model may select (OUTPUT_ASSEMBLY.md)."""

    CANDIDATE = "candidate"
    POSITION = "position"
    FACTS = "facts"
    EVIDENCE = "evidence"
    TOOL_CALL = "tool_call"


class DeliveredRef(DomainModel):
    """A code-issued reference and the trusted facts code needs to validate its use.

    - `run_id` is None only for durable references (positions) that are not run-scoped.
    - `account_scope_id` is None only for references that are not account-scoped (e.g. web
      evidence). Otherwise it must equal the run's Agentic account scope.
    - `delivered` records whether the reference was actually delivered to the model in this
      run's context or tool results; an undelivered reference cannot be selected.
    - Candidate/position refs carry the instrument identity (`occ_symbol`,
      `broker_instrument_id`) from validated source records; `tool_call_id` identifies the
      recorded call a TOOL_CALL ref names.
    """

    ref: Ref
    kind: RefKind
    run_id: UUID | None
    account_scope_id: NonEmptyStr | None
    delivered: bool
    underlying: NonEmptyStr | None = None
    occ_symbol: OccSymbol | None = None
    broker_instrument_id: NonEmptyStr | None = None
    position_id: UUID | None = None
    candidate_origin: CandidateOrigin | None = None
    board_comparison: BoardComparisonRecord | None = None
    source_evidence_ids: tuple[UUID, ...] = ()
    tool_call_id: UUID | None = None

    @model_validator(mode="after")
    def _check_ref(self) -> Self:
        if (self.kind is RefKind.TOOL_CALL) != (self.tool_call_id is not None):
            raise ValueError("tool_call_id is required exactly for tool_call refs")
        if self.kind is RefKind.POSITION and self.position_id is None:
            raise ValueError("a position ref needs a position_id")
        if self.kind is not RefKind.POSITION and self.run_id is None:
            raise ValueError("only durable position refs may omit run_id")
        require_unique(self.source_evidence_ids, "source evidence id")
        return self


class AttemptEvidence(DomainModel):
    """Loader-resolved pre-order evidence for one live place call (by verified evidence IDs)."""

    place_tool_call_id: UUID
    snapshot_ref: UUID | None
    quote_refs: tuple[UUID, ...] = ()


class ResourceAmount(DomainModel):
    """Baseline headroom for one resource key, or None when it cannot be established.

    Keys are opaque strings chosen by the loader, e.g. `cash` (min of settled cash after the
    reserve and the total-ratio headroom) or `shares:AAPL` (uncovered shares within the
    coverage ratio). The assembler only subtracts; it never derives the headroom itself.
    """

    key: NonEmptyStr
    amount: Dec | None
    evidence_ids: tuple[UUID, ...] = ()


class ReservationBaseline(DomainModel):
    """The single validated baseline for the dry-run reservation simulation."""

    snapshot_ref: UUID | None
    resources: tuple[ResourceAmount, ...]

    @model_validator(mode="after")
    def _check_baseline(self) -> Self:
        require_unique(tuple(r.key for r in self.resources), "resource key")
        return self


class ResourceUse(DomainModel):
    key: NonEmptyStr
    per_contract: PosDec


class ReservationRequirement(DomainModel):
    """Per-contract resource use of the proposal computed from `facts_ref`."""

    facts_ref: Ref
    uses: tuple[ResourceUse, ...]

    @model_validator(mode="after")
    def _check_requirement(self) -> Self:
        require_unique(tuple(u.key for u in self.uses), "resource key")
        return self


class RankingKey(DomainModel):
    """One fixed `selection.ranking` key, implemented by a named DecisionFacts metric."""

    metric_name: NonEmptyStr
    descending: bool


class AssemblyContext(DomainModel):
    """Immutable, complete input bundle for one run's assembly.

    `tool_calls` and `orders` are recorded facts for this run; `facts` are persisted
    DecisionFacts; `refs` is the registry of code-issued references. `prior_findings` carries
    parsing/reference findings computed earlier (passed through verbatim).
    """

    run_id: UUID
    environment: AppEnv
    slot: UtcDatetime
    terminated_at: UtcDatetime
    requested_execution_mode: ExecutionMode
    effective_execution_mode: ExecutionMode
    account_scope_id: NonEmptyStr
    rules_version: NonEmptyStr
    rules_hash: NonEmptyStr
    prompt_id: NonEmptyStr | None
    prompt_hash: NonEmptyStr | None
    model_id: NonEmptyStr | None
    assembler_version: NonEmptyStr = ASSEMBLER_VERSION
    output_record_id: UUID | None = None
    order_type: NonEmptyStr = "limit"
    time_in_force: NonEmptyStr | None = None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    orders: tuple[OrderRecord, ...] = ()
    facts: tuple[DecisionFacts, ...] = ()
    refs: tuple[DeliveredRef, ...] = ()
    quotes: tuple[Quote, ...] = ()
    attempt_evidence: tuple[AttemptEvidence, ...] = ()
    reservation_baseline: ReservationBaseline | None = None
    reservation_requirements: tuple[ReservationRequirement, ...] = ()
    ranking_keys: tuple[RankingKey, ...] = ()
    prior_findings: tuple[AssemblyFinding, ...] = ()

    @model_validator(mode="after")
    def _check_context(self) -> Self:
        require_unique(tuple(t.identity.tool_call_id for t in self.tool_calls), "tool call id")
        require_unique(tuple(f.facts_ref for f in self.facts), "facts ref")
        require_unique(tuple(r.ref for r in self.refs), "ref")
        require_unique(tuple(q.quote_id for q in self.quotes), "quote id")
        require_unique(
            tuple(e.place_tool_call_id for e in self.attempt_evidence), "attempt evidence"
        )
        require_unique(
            tuple(r.facts_ref for r in self.reservation_requirements), "reservation requirement"
        )
        return self


__all__ = [
    "ASSEMBLER_VERSION",
    "AssemblyContext",
    "AttemptEvidence",
    "DeliveredRef",
    "RankingKey",
    "RefKind",
    "ReservationBaseline",
    "ReservationRequirement",
    "ResourceAmount",
    "ResourceUse",
]
