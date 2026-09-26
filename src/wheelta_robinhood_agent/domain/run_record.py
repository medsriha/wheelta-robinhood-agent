"""RunRecord v4 models: the code-assembled run JSON (OUTPUT_ASSEMBLY.md "Final record assembly").

Models only; `assemble_run_record` is a separate, later task. Named `DecisionRecord` and
`LegRecord` to avoid confusion with the model-authored `decision_output.Decision`.

Invariants enforced here (INTERFACES.md RunRecord; OUTPUT_ASSEMBLY.md):
- missing/invalid model output means no decisions are synthesized, but every attempt,
  cancellation, and unassociated action is still recorded;
- every place call and cancel call appears exactly once across the record;
- unknown targets stay None (with a gap) and never erase an actual attempt quantity.
"""

from datetime import date
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import model_validator

from wheelta_robinhood_agent.domain.base import (
    Count,
    Dec,
    DomainModel,
    NonEmptyStr,
    NonNegDec,
    PosDec,
    Ref,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    BoardComparison,
    CandidateOrigin,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts import DerivedMetric
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import Attempt, Cancellation, ReasonCode
from wheelta_robinhood_agent.domain.sanity import (
    is_delta_plausible,
    is_mark_within_bid_ask,
    is_valid_bid_ask,
)

RUN_RECORD_SCHEMA_VERSION: Literal[4] = 4


class DecisionOutputStatus(StrEnum):
    """Whether a valid AgentDecisionOutput was available to the assembler."""

    PARSED = "parsed"
    INVALID = "invalid"
    MISSING = "missing"


class Quote(DomainModel):
    """A live option quote from a validated source (INTERFACES.md "Quotes carry …").

    Greeks, OI, and volume are optional; sanity: `0 <= bid <= ask`, `ask > 0`, mark within
    the spread when present, `|delta| <= 1`.
    """

    quote_id: UUID
    broker_instrument_id: NonEmptyStr
    bid: NonNegDec
    ask: PosDec
    mark: PosDec | None = None
    delta: Dec | None = None
    gamma: Dec | None = None
    theta: Dec | None = None
    vega: Dec | None = None
    implied_volatility_ratio: NonNegDec | None = None
    open_interest: Count | None = None
    volume: Count | None = None
    as_of: UtcDatetime
    source_tool_call_ids: tuple[UUID, ...]

    @model_validator(mode="after")
    def _check_quote(self) -> Self:
        if not self.source_tool_call_ids:
            raise ValueError("a quote needs a source tool call")
        if not is_valid_bid_ask(self.bid, self.ask):
            raise ValueError("quote requires 0 <= bid <= ask and ask > 0")
        if self.mark is not None and not is_mark_within_bid_ask(self.mark, self.bid, self.ask):
            raise ValueError("mark must lie within [bid, ask]")
        if self.delta is not None and not is_delta_plausible(self.delta):
            raise ValueError("|delta| must be <= 1")
        return self


class BoardComparisonRecord(DomainModel):
    """Board screen vs live quote comparison (INTERFACES.md "Board comparison …").

    `compared` requires the board row/build/call references, premium measure, and both
    premiums. Other statuses may omit them.
    """

    status: BoardComparison
    board_row_ref: Ref | None = None
    build_id: NonEmptyStr | None = None
    tool_call_ids: tuple[UUID, ...] = ()
    premium_measure: NonEmptyStr | None = None
    board_premium: NonNegDec | None = None
    live_premium: NonNegDec | None = None

    @model_validator(mode="after")
    def _check_board(self) -> Self:
        if self.status is BoardComparison.COMPARED and (
            self.board_row_ref is None
            or self.build_id is None
            or not self.tool_call_ids
            or self.premium_measure is None
            or self.board_premium is None
            or self.live_premium is None
        ):
            raise ValueError("a compared board comparison needs full provenance and premiums")
        return self


class LegRecord(DomainModel):
    """One leg of a decision (INTERFACES.md "Leg"). Attempts are in chronological order."""

    leg_ref: Ref
    depends_on_leg_ref: Ref | None = None
    side: OrderSide
    occ_symbol: OccSymbol | None
    broker_instrument_id: NonEmptyStr | None
    right: OptionRight | None
    strike: PosDec | None
    expiration: date | None
    target_quantity: Count | None
    candidate_origin: CandidateOrigin | None = None
    facts_ref: Ref | None = None
    source_evidence_ids: tuple[UUID, ...] = ()
    quotes: tuple[Quote, ...] = ()
    board_comparison: BoardComparisonRecord | None = None
    conditional: bool = False
    attempts: tuple[Attempt, ...] = ()
    reason_codes: tuple[ReasonCode, ...] = ()
    gaps: tuple[Gap, ...] = ()

    @model_validator(mode="after")
    def _check_leg(self) -> Self:
        if self.depends_on_leg_ref == self.leg_ref:
            raise ValueError("a leg cannot depend on itself")
        if self.occ_symbol is not None and (
            (self.right is not None and self.right is not self.occ_symbol.right)
            or (self.strike is not None and self.strike != self.occ_symbol.strike)
            or (self.expiration is not None and self.expiration != self.occ_symbol.expiration)
        ):
            raise ValueError("leg right/strike/expiration disagree with its OCC symbol")
        if [a.index for a in self.attempts] != list(range(len(self.attempts))):
            raise ValueError("attempt indices must be 0..n-1 in chronological order")
        if sum(1 for a in self.attempts if a.proposal_ref is not None) > 1:
            raise ValueError("at most one unsubmitted proposal attempt per leg")
        if self.target_quantity is None and "target_quantity" not in {g.field for g in self.gaps}:
            raise ValueError("an unknown target_quantity needs a gap")
        require_unique(self.reason_codes, "reason code")
        return self


class DecisionRecord(DomainModel):
    """A code-enriched decision (INTERFACES.md "Decision").

    `priority` is derived by code (execution order + fixed ranking + array order); the
    rationale, thesis, and invalidation conditions are the agent's judgments, verbatim.
    """

    decision_ref: Ref
    action: DecisionAction
    priority: Count | None
    target_ref: Ref
    replacement_ref: Ref | None
    underlying: NonEmptyStr | None
    position_id: UUID | None
    depends_on_decision_refs: tuple[Ref, ...] = ()
    rationale: NonEmptyStr
    thesis: NonEmptyStr | None
    invalidation_conditions: tuple[NonEmptyStr, ...] = ()
    evidence_refs: tuple[Ref, ...] = ()
    legs: tuple[LegRecord, ...] = ()
    metrics: tuple[DerivedMetric, ...] = ()
    gaps: tuple[Gap, ...] = ()

    @model_validator(mode="after")
    def _check_decision_record(self) -> Self:
        if self.decision_ref in self.depends_on_decision_refs:
            raise ValueError("a decision cannot depend on itself")
        require_unique(self.depends_on_decision_refs, "dependency")
        require_unique(tuple(m.name for m in self.metrics), "metric name")
        return self


class CancellationRationaleRecord(DomainModel):
    """The agent's cancellation rationale, kept verbatim (AgentDecisionOutput v5).

    `cancel_tool_call_id` is set by code only when `cancel_call_ref` resolved to a recorded
    cancel call that the rationale validly selected; otherwise it is None and the assembly
    findings say why. The text is the agent's judgment, never evidence of execution.
    """

    cancel_call_ref: Ref
    cancel_tool_call_id: UUID | None
    rationale: NonEmptyStr
    evidence_refs: tuple[Ref, ...] = ()

    @model_validator(mode="after")
    def _check_cancellation_rationale(self) -> Self:
        require_unique(self.evidence_refs, "evidence ref")
        return self


class UnresolvedQuestionRecord(DomainModel):
    """A research question the agent left open this run, kept verbatim with its refs."""

    target_ref: Ref | None
    question: NonEmptyStr
    evidence_refs: tuple[Ref, ...] = ()

    @model_validator(mode="after")
    def _check_unresolved_question(self) -> Self:
        require_unique(self.evidence_refs, "evidence ref")
        return self


class UnassociatedActionKind(StrEnum):
    PLACE = "place"
    CANCEL = "cancel"


class UnassociatedAction(DomainModel):
    """A recorded place/cancel not associated with any valid decision.

    Retains the same factual detail as an associated action (OUTPUT_ASSEMBLY.md step 1).
    """

    kind: UnassociatedActionKind
    attempt: Attempt | None = None
    cancellation: Cancellation | None = None
    occ_symbol: OccSymbol | None = None
    broker_instrument_id: NonEmptyStr | None = None
    side_raw: NonEmptyStr | None = None
    reason_codes: tuple[ReasonCode, ...] = ()

    @model_validator(mode="after")
    def _check_unassociated(self) -> Self:
        if self.kind is UnassociatedActionKind.PLACE:
            if self.attempt is None or self.cancellation is not None:
                raise ValueError("an unassociated place carries exactly an attempt")
            if self.attempt.place_tool_call_id is None:
                raise ValueError("an unassociated place must be a recorded place call")
        elif self.cancellation is None or self.attempt is not None:
            raise ValueError("an unassociated cancel carries exactly a cancellation")
        return self


class AssemblyFinding(DomainModel):
    """A typed assembly/association finding (never a reason to drop an action).

    `code` is a machine-readable identifier; its closed set is **TBD** (not defined in the
    docs yet) and should become an enum when specified.
    """

    code: NonEmptyStr
    detail: NonEmptyStr
    decision_ref: Ref | None = None
    leg_ref: Ref | None = None
    tool_call_ids: tuple[UUID, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()


class RunRecord(DomainModel):
    """RunRecord v4: the complete, code-assembled run JSON."""

    schema_version: Literal[4]
    assembler_version: NonEmptyStr
    input_hash: NonEmptyStr
    run_id: UUID
    environment: AppEnv
    slot: UtcDatetime
    terminated_at: UtcDatetime
    requested_execution_mode: ExecutionMode
    effective_execution_mode: ExecutionMode
    rules_version: NonEmptyStr
    rules_hash: NonEmptyStr
    prompt_id: NonEmptyStr | None
    prompt_hash: NonEmptyStr | None
    model_id: NonEmptyStr | None
    decision_output_status: DecisionOutputStatus
    decisions: tuple[DecisionRecord, ...] = ()
    cancellations: tuple[Cancellation, ...] = ()
    cancellation_rationales: tuple[CancellationRationaleRecord, ...] = ()
    unresolved_questions: tuple[UnresolvedQuestionRecord, ...] = ()
    unassociated_actions: tuple[UnassociatedAction, ...] = ()
    metrics: tuple[DerivedMetric, ...] = ()
    gaps: tuple[Gap, ...] = ()
    findings: tuple[AssemblyFinding, ...] = ()
    summary: NonEmptyStr

    @model_validator(mode="after")
    def _check_run_record(self) -> Self:
        if self.effective_execution_mode is ExecutionMode.LIVE and (
            self.requested_execution_mode is not ExecutionMode.LIVE
        ):
            raise ValueError("effective live requires requested live")
        if self.decision_output_status is not DecisionOutputStatus.PARSED and (
            self.decisions or self.cancellation_rationales or self.unresolved_questions
        ):
            raise ValueError("model choices are never synthesized without parsed model output")
        decision_refs = tuple(d.decision_ref for d in self.decisions)
        require_unique(decision_refs, "decision ref")
        legs = tuple(leg for d in self.decisions for leg in d.legs)
        leg_refs = tuple(leg.leg_ref for leg in legs)
        require_unique(leg_refs, "leg ref")
        for decision in self.decisions:
            for dep in decision.depends_on_decision_refs:
                if dep not in decision_refs:
                    raise ValueError(f"unknown decision dependency {dep!r}")
        for leg in legs:
            if leg.depends_on_leg_ref is not None and leg.depends_on_leg_ref not in leg_refs:
                raise ValueError(f"unknown leg dependency {leg.depends_on_leg_ref!r}")
        attempts = [a for leg in legs for a in leg.attempts] + [
            u.attempt for u in self.unassociated_actions if u.attempt is not None
        ]
        require_unique(
            tuple(a.place_tool_call_id for a in attempts if a.place_tool_call_id is not None),
            "place call (recorded more than once)",
        )
        require_unique(
            tuple(a.proposal_ref for a in attempts if a.proposal_ref is not None), "proposal ref"
        )
        cancels = list(self.cancellations) + [
            u.cancellation for u in self.unassociated_actions if u.cancellation is not None
        ]
        require_unique(
            tuple(c.cancel_tool_call_id for c in cancels), "cancel call (recorded more than once)"
        )
        require_unique(tuple(m.name for m in self.metrics), "metric name")
        return self
