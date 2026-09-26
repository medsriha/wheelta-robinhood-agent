"""Deterministic RunRecord v4 assembly (OUTPUT_ASSEMBLY.md "Final record assembly"; ADR-0011).

`assemble_run_record(context, decisions)` is pure: no clock, randomness, fresh UUIDs, I/O,
or LLM calls. Identical inputs give byte-identical `canonical_json` and `record_hash`.

Order of work:
1. Rebuild every place/cancel action from recorded events (`assembly_events`).
2. Validate model choices: every selected reference must resolve to a delivered code-issued
   reference of the right kind, run, and account, and be compatible with the action. Errors
   become `AssemblyFinding`s; nothing is repaired by ticker/price/time matching.
3. Bind live attempts to decision legs by recorded place-call identity; anything that cannot
   be attributed stays in `unassociated_actions`. Every place/cancel call appears once.
4. Off mode: build at most one not_placed/DRY_RUN attempt per valid proposal with a known
   positive quantity, simulating reservations in priority order against one baseline.
5. Priorities: closes/rolls before opens (`orders.execution_order`), opens by the fixed
   ranking metrics, array order breaking discretionary ties.

Refs are deterministic strings: `decision:<output id or run id>:<index>`,
`<decision_ref>:leg:<close|open>`, and `proposal:<leg_ref>`.

Rationale text is copied verbatim and never parsed for facts.
"""

import hashlib
import json
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Context, Decimal
from typing import Final, Literal
from uuid import UUID

from pydantic import JsonValue

from wheelta_robinhood_agent.domain.assembly_context import (
    AssemblyContext,
    DeliveredRef,
    RefKind,
    ReservationRequirement,
)
from wheelta_robinhood_agent.domain.assembly_events import (
    CANCEL_TOOL,
    PLACE_TOOL,
    REVIEW_TOOL,
    EventIndex,
    PlaceAction,
    build_event_index,
    call_sort_key,
    tool_name,
)
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    Decision,
    DecisionOutputParsed,
    DecisionOutputParseFailure,
    ProposedLeg,
)
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts import DecisionFacts, DerivedMetric, FactsPurpose
from wheelta_robinhood_agent.domain.orders import Attempt, Cancellation, ReasonCode
from wheelta_robinhood_agent.domain.run_record import (
    RUN_RECORD_SCHEMA_VERSION,
    AssemblyFinding,
    CancellationRationaleRecord,
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    RunRecord,
    UnassociatedAction,
    UnassociatedActionKind,
    UnresolvedQuestionRecord,
)

Role = Literal["close", "open"]
_OPEN_ACTIONS: Final = frozenset({DecisionAction.OPEN_CSP, DecisionAction.OPEN_CC})
_EVIDENCE_KINDS: Final = frozenset({RefKind.EVIDENCE, RefKind.FACTS})
_SUBJECT_KINDS: Final = frozenset({RefKind.CANDIDATE, RefKind.POSITION})
_DIVISION: Final = Context(prec=60, rounding=ROUND_FLOOR)

DecisionsInput = DecisionOutputParsed | DecisionOutputParseFailure | None


# --------------------------------------------------------------------------------------------
# Canonical serialization
# --------------------------------------------------------------------------------------------


def _canonical(data: JsonValue) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_json(record: RunRecord) -> bytes:
    """Canonical bytes: sorted keys, compact separators, Decimals as strings, UTC ISO times."""
    return _canonical(record.model_dump(mode="json"))


def record_hash(record: RunRecord) -> str:
    """SHA-256 hex digest of `canonical_json(record)`."""
    return hashlib.sha256(canonical_json(record)).hexdigest()


def input_hash(context: AssemblyContext, decisions: DecisionsInput) -> str:
    """SHA-256 of the canonical complete input (context plus parsed/failed/missing output)."""
    decisions_json: JsonValue
    if decisions is None:
        decisions_json = None
    elif isinstance(decisions, DecisionOutputParsed):
        decisions_json = {
            "ok": True,
            "schema_version": decisions.schema_version,
            "output": decisions.output.model_dump(mode="json"),
        }
    else:
        decisions_json = {
            "ok": False,
            "schema_version": decisions.schema_version,
            "raw_text": decisions.raw_text,
            "issues": [[i.loc, i.message, i.kind] for i in decisions.issues],
        }
    payload: JsonValue = {
        "context": context.model_dump(mode="json"),
        "decisions": decisions_json,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


# --------------------------------------------------------------------------------------------
# Working state
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Proposal:
    leg: ProposedLeg
    facts: DecisionFacts
    position: int


@dataclass(slots=True)
class _Plan:
    """One model decision after reference validation (mutable only during assembly)."""

    index: int
    decision: Decision
    decision_ref: str
    target: DeliveredRef | None
    replacement: DeliveredRef | None
    roles: tuple[Role, ...]
    proposals: dict[Role, _Proposal | None] = field(default_factory=dict)
    attempts: dict[Role, list[PlaceAction]] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    evidence_facts: list[DecisionFacts] = field(default_factory=list)
    funding: list[str] = field(default_factory=list)
    funding_invalid: bool = False
    gaps: list[Gap] = field(default_factory=list)
    priority: int | None = None

    def leg_ref(self, role: Role) -> str:
        return f"{self.decision_ref}:leg:{role}"

    def subject(self, role: Role) -> DeliveredRef | None:
        if role == "close" or self.decision.action in _OPEN_ACTIONS:
            return self.target
        return self.replacement


@dataclass(slots=True)
class _Sim:
    """Dry-run reservation simulation state (VALIDATION.md "Dry-run reservations")."""

    available: dict[str, Decimal | None]
    snapshot_ref: UUID | None
    has_baseline: bool
    reserved_any: bool = False


def _gap(field_name: str, detail: str, kind: DataQuality = DataQuality.MISSING) -> Gap:
    return Gap(field=field_name, kind=kind, detail=detail)


def _floor_div(amount: Decimal, per: Decimal) -> int:
    if amount <= 0:
        return 0
    return int(_DIVISION.divide(amount, per).to_integral_value(rounding=ROUND_FLOOR))


# --------------------------------------------------------------------------------------------
# Assembler
# --------------------------------------------------------------------------------------------


class _Assembler:
    def __init__(self, ctx: AssemblyContext, output: AgentDecisionOutput | None) -> None:
        self.ctx = ctx
        self.output = output
        self.refs = {r.ref: r for r in ctx.refs}
        self.facts = {f.facts_ref: f for f in ctx.facts}
        self.quotes = {q.quote_id: q for q in ctx.quotes}
        evidence = {
            e.place_tool_call_id: (e.snapshot_ref, e.quote_refs) for e in ctx.attempt_evidence
        }
        self.idx: EventIndex = build_event_index(ctx.run_id, ctx.tool_calls, ctx.orders, evidence)
        self.findings: list[AssemblyFinding] = list(ctx.prior_findings) + self.idx.findings
        self.requirements: dict[str, ReservationRequirement] = {
            r.facts_ref: r for r in ctx.reservation_requirements
        }
        key = str(ctx.output_record_id) if ctx.output_record_id else str(ctx.run_id)
        self.output_key = key
        self.off = ctx.effective_execution_mode is ExecutionMode.OFF
        # cancellation_rationales index -> the recorded cancel call it validly selected.
        self.rationale_calls: dict[int, UUID] = {}

    # -- findings -------------------------------------------------------------------------

    def find(
        self,
        code: str,
        detail: str,
        decision_ref: str | None = None,
        leg_ref: str | None = None,
        call_ids: tuple[UUID, ...] = (),
    ) -> None:
        self.findings.append(
            AssemblyFinding(
                code=code,
                detail=detail,
                decision_ref=decision_ref,
                leg_ref=leg_ref,
                tool_call_ids=call_ids,
            )
        )

    # -- reference resolution -------------------------------------------------------------

    def resolve(
        self, ref: str, kinds: frozenset[RefKind], what: str, decision_ref: str | None
    ) -> DeliveredRef | None:
        """Resolve a model-selected ref; any scope/kind/delivery failure is a finding."""
        found = self.refs.get(ref)
        problem: tuple[str, str] | None = None
        if found is None:
            problem = ("unknown_reference", f"{what} was never issued by code")
        elif found.kind not in kinds:
            problem = ("wrong_reference_kind", f"{what} is a {found.kind.value} reference")
        elif found.run_id is not None and found.run_id != self.ctx.run_id:
            problem = ("wrong_run_reference", f"{what} belongs to another run")
        elif (
            found.account_scope_id is not None
            and found.account_scope_id != self.ctx.account_scope_id
        ):
            problem = ("wrong_account_reference", f"{what} belongs to another account")
        elif not found.delivered:
            problem = ("undelivered_reference", f"{what} was never delivered to the model")
        if problem is not None:
            self.find(problem[0], f"{problem[1]}: {ref!r}", decision_ref)
            return None
        return found

    def resolve_facts(self, ref: str, what: str, decision_ref: str | None) -> DecisionFacts | None:
        if self.resolve(ref, frozenset({RefKind.FACTS}), what, decision_ref) is None:
            return None
        facts = self.facts.get(ref)
        if facts is None:
            self.find(
                "facts_not_recorded", f"{what} has no recorded fact set: {ref!r}", decision_ref
            )
            return None
        if facts.run_id != self.ctx.run_id:
            self.find("wrong_run_reference", f"{what} was computed in another run", decision_ref)
            return None
        if facts.rules_version != self.ctx.rules_version or facts.rules_hash != self.ctx.rules_hash:
            self.find("facts_rules_mismatch", f"{what} used other trading rules", decision_ref)
            return None
        return facts

    # -- decision validation ---------------------------------------------------------------

    def plan(self, index: int, decision: Decision) -> _Plan:
        action = decision.action
        decision_ref = f"decision:{self.output_key}:{index}"
        target_kind = RefKind.CANDIDATE if action in _OPEN_ACTIONS else RefKind.POSITION
        target = self.resolve(
            decision.target_ref, frozenset({target_kind}), "target_ref", decision_ref
        )
        if target is not None and action in _OPEN_ACTIONS and target.occ_symbol is not None:
            wanted = OptionRight.PUT if action is DecisionAction.OPEN_CSP else OptionRight.CALL
            if target.occ_symbol.right is not wanted:
                self.find(
                    "incompatible_reference",
                    f"{action.value} target is a {target.occ_symbol.right.value} contract",
                    decision_ref,
                )
                target = None
        replacement = self._replacement(decision, target, decision_ref)
        roles: tuple[Role, ...]
        if target is None:
            roles = ()
        elif action in _OPEN_ACTIONS:
            roles = ("open",)
        elif action is DecisionAction.CLOSE:
            roles = ("close",)
        elif action is DecisionAction.ROLL:
            roles = ("close", "open") if replacement is not None else ("close",)
        else:
            roles = ()
        plan = _Plan(index, decision, decision_ref, target, replacement, roles)
        for ref in decision.evidence_refs:
            found = self.resolve(ref, _EVIDENCE_KINDS, "evidence_ref", decision_ref)
            if found is None:
                continue
            plan.evidence_refs.append(ref)
            if found.kind is RefKind.FACTS and ref in self.facts:
                plan.evidence_facts.append(self.facts[ref])
        self._proposals(plan)
        if action in (DecisionAction.OPEN_CSP, DecisionAction.OPEN_CC, DecisionAction.ROLL) and (
            decision.thesis is None or not decision.invalidation_conditions
        ):
            plan.gaps.append(_gap("thesis", "the agent supplied no entry thesis/invalidation"))
        return plan

    def _replacement(
        self, decision: Decision, target: DeliveredRef | None, decision_ref: str
    ) -> DeliveredRef | None:
        if decision.replacement_ref is None:
            if decision.action is DecisionAction.ROLL:
                self.find("missing_replacement", "a ROLL needs a replacement_ref", decision_ref)
            return None
        if decision.action is not DecisionAction.ROLL:
            self.find(
                "incompatible_reference", "only a ROLL may select a replacement", decision_ref
            )
            return None
        found = self.resolve(
            decision.replacement_ref,
            frozenset({RefKind.CANDIDATE}),
            "replacement_ref",
            decision_ref,
        )
        if found is None or target is None:
            return found
        t_occ, r_occ = target.occ_symbol, found.occ_symbol
        if (
            target.underlying is not None
            and found.underlying is not None
            and target.underlying != found.underlying
        ) or (t_occ is not None and r_occ is not None and t_occ.right is not r_occ.right):
            self.find(
                "incompatible_reference",
                "replacement differs from the position's underlying or right",
                decision_ref,
            )
            return None
        return found

    def _proposals(self, plan: _Plan) -> None:
        decision = plan.decision
        if not plan.roles:
            if decision.proposed_legs:
                self.find(
                    "incompatible_proposal",
                    "proposed legs given for a decision without executable legs",
                    plan.decision_ref,
                )
            return
        for position, leg in enumerate(decision.proposed_legs):
            facts = self.resolve_facts(leg.facts_ref, "proposed leg facts_ref", plan.decision_ref)
            if facts is None:
                continue
            role = self._role_for(plan, facts)
            if role is None:
                self.find(
                    "incompatible_proposal",
                    "fact set subject/purpose does not match any leg of this action",
                    plan.decision_ref,
                )
                continue
            leg_ref = plan.leg_ref(role)
            if role in plan.proposals:
                self.find(
                    "duplicate_proposal",
                    "more than one proposal for the same leg",
                    plan.decision_ref,
                    leg_ref,
                )
                plan.proposals[role] = None
                continue
            if facts.limit_price is not None and facts.limit_price != leg.limit_price:
                self.find(
                    "facts_price_mismatch",
                    "the fact set was computed for a different limit price",
                    plan.decision_ref,
                    leg_ref,
                )
                plan.proposals[role] = None
                continue
            if not self._facts_match_subject(facts, plan.subject(role)):
                self.find(
                    "mismatched_contract",
                    "fact set belongs to a different contract/position",
                    plan.decision_ref,
                    leg_ref,
                )
                plan.proposals[role] = None
                continue
            plan.proposals[role] = _Proposal(leg, facts, position)
        close, open_ = plan.proposals.get("close"), plan.proposals.get("open")
        if close is not None and open_ is not None and open_.position < close.position:
            self.find(
                "proposal_order",
                "a roll proposal lists the open before the close",
                plan.decision_ref,
            )
        if self.off:
            missing = [r for r in plan.roles if r not in plan.proposals]
            if missing:
                self.find(
                    "proposal_shape_incomplete",
                    f"no proposal for leg(s): {', '.join(missing)}",
                    plan.decision_ref,
                )

    @staticmethod
    def _role_for(plan: _Plan, facts: DecisionFacts) -> Role | None:
        for role in plan.roles:
            subject = plan.subject(role)
            wanted = FactsPurpose.CLOSE if role == "close" else FactsPurpose.OPEN
            if subject is not None and facts.subject_ref == subject.ref and facts.purpose is wanted:
                return role
        return None

    @staticmethod
    def _facts_match_subject(facts: DecisionFacts, subject: DeliveredRef | None) -> bool:
        if subject is None:
            return False
        if facts.candidate_ref is not None and facts.candidate_ref != subject.ref:
            return False
        return not (
            facts.position_id is not None
            and subject.position_id is not None
            and facts.position_id != subject.position_id
        )

    # -- execution associations ------------------------------------------------------------

    def associate(self, plans: list[_Plan], output: AgentDecisionOutput) -> set[UUID]:
        """Bind place calls to legs; return cancel calls validly selected by a model choice."""
        claims: dict[UUID, list[str]] = {}
        selections: list[tuple[_Plan | None, UUID, int | None]] = []
        for plan in plans:
            if plan.decision.execution_refs and self.off:
                self.find(
                    "execution_refs_in_off_mode", "off mode has no executions", plan.decision_ref
                )
            for ref in plan.decision.execution_refs:
                found = self.resolve(
                    ref, frozenset({RefKind.TOOL_CALL}), "execution_ref", plan.decision_ref
                )
                if found is None or found.tool_call_id is None:
                    continue
                selections.append((plan, found.tool_call_id, None))
                claims.setdefault(found.tool_call_id, []).append(plan.decision_ref)
        if output.cancellation_rationales and self.off:
            self.find("cancellation_rationale_in_off_mode", "off mode has no cancellations")
        for rationale_index, rationale in enumerate(output.cancellation_rationales):
            for ref in rationale.evidence_refs:
                self.resolve(ref, _EVIDENCE_KINDS, "cancellation evidence_ref", None)
            found = self.resolve(
                rationale.cancel_call_ref, frozenset({RefKind.TOOL_CALL}), "cancel_call_ref", None
            )
            if found is None or found.tool_call_id is None:
                continue
            selections.append((None, found.tool_call_id, rationale_index))
            claims.setdefault(found.tool_call_id, []).append("cancellation_rationale")
        for question in output.unresolved_questions:
            if question.target_ref is not None:
                self.resolve(question.target_ref, _SUBJECT_KINDS, "question target_ref", None)
            for ref in question.evidence_refs:
                self.resolve(ref, _EVIDENCE_KINDS, "question evidence_ref", None)

        selected_cancels: set[UUID] = set()
        for owner, call_id, selected_rationale in selections:
            decision_ref = owner.decision_ref if owner is not None else None
            if len(claims[call_id]) > 1:
                self.find(
                    "duplicate_execution_association",
                    "the same recorded call is associated more than once; left unassociated",
                    decision_ref,
                    call_ids=(call_id,),
                )
                continue
            call = self.idx.calls_by_id.get(call_id)
            name = tool_name(call) if call is not None else None
            if owner is None:
                if name != CANCEL_TOOL:
                    self.find(
                        "wrong_reference_kind",
                        "cancel_call_ref is not a cancel call",
                        call_ids=(call_id,),
                    )
                else:
                    selected_cancels.add(call_id)
                    if selected_rationale is not None:
                        self.rationale_calls[selected_rationale] = call_id
                continue
            action = self.idx.place(call_id)
            if name == PLACE_TOOL and action is not None:
                self._bind_place(owner, action)
            elif name == CANCEL_TOOL:
                selected_cancels.add(call_id)
                self._check_cancel_owner(owner, call_id, plans)
            elif name != REVIEW_TOOL:
                self.find(
                    "wrong_reference_kind",
                    "execution_ref is not a review/place/cancel call",
                    decision_ref,
                    call_ids=(call_id,),
                )
        return selected_cancels

    def _bind_place(self, plan: _Plan, action: PlaceAction) -> None:
        call_id = action.call.identity.tool_call_id
        intent = action.intent
        side = intent.side if intent is not None else None
        role: Role | None = None
        if side is OrderSide.BUY_TO_CLOSE:
            role = "close"
        elif side is OrderSide.SELL_TO_OPEN:
            role = "open"
        if role is None or intent is None or role not in plan.roles:
            self.find(
                "incompatible_execution_ref",
                "the place call's side does not fit this decision's legs",
                plan.decision_ref,
                call_ids=(call_id,),
            )
            return
        subject = plan.subject(role)
        same: bool | None = None
        if subject is not None:
            if subject.broker_instrument_id is not None and intent.broker_instrument_id:
                same = subject.broker_instrument_id == intent.broker_instrument_id
            elif subject.occ_symbol is not None and intent.occ_symbol is not None:
                same = subject.occ_symbol == intent.occ_symbol
        if same is None:
            self.find(
                "execution_ref_unverifiable",
                "instrument identity cannot be compared; left unassociated",
                plan.decision_ref,
                plan.leg_ref(role),
                (call_id,),
            )
            return
        if not same:
            self.find(
                "mismatched_contract",
                "the place call is for a different contract",
                plan.decision_ref,
                plan.leg_ref(role),
                (call_id,),
            )
            return
        plan.attempts.setdefault(role, []).append(action)

    def _check_cancel_owner(self, plan: _Plan, call_id: UUID, plans: list[_Plan]) -> None:
        place_id = self.idx.order_place_call_of_cancel.get(call_id)
        if place_id is None:
            return
        for other in plans:
            if other is plan:
                continue
            if any(
                a.call.identity.tool_call_id == place_id
                for acts in other.attempts.values()
                for a in acts
            ):
                self.find(
                    "contradictory_cancel_association",
                    "the cancelled order belongs to another decision; broker evidence wins",
                    plan.decision_ref,
                    call_ids=(call_id,),
                )

    # -- priorities ------------------------------------------------------------------------

    def prioritize(self, plans: list[_Plan]) -> list[_Plan]:
        closes = [
            p for p in plans if p.decision.action in (DecisionAction.CLOSE, DecisionAction.ROLL)
        ]
        opens = [p for p in plans if p.decision.action in _OPEN_ACTIONS]
        keyed: list[tuple[tuple[Decimal, ...], _Plan]] = []
        incomplete: list[_Plan] = []
        for plan in opens:
            key = self._ranking_key(plan)
            if key is None:
                incomplete.append(plan)
            else:
                keyed.append((key, plan))
        if incomplete:
            for plan in incomplete:
                plan.gaps.append(
                    _gap("ranking", "fixed ranking inputs unavailable; array order used")
                )
            if keyed:
                self.find("ranking_incomplete", "opens ranked by agent array order only")
            ordered_opens = opens
        else:
            keyed.sort(key=lambda kp: (kp[0], kp[1].index))
            ordered_opens = [p for _k, p in keyed]
        ordered = closes + ordered_opens
        for number, plan in enumerate(ordered, start=1):
            plan.priority = number
        return ordered

    def _ranking_key(self, plan: _Plan) -> tuple[Decimal, ...] | None:
        facts = self._leg_facts(plan, "open")
        if facts is None or not self.ctx.ranking_keys:
            return None
        key: list[Decimal] = []
        for rk in self.ctx.ranking_keys:
            metric = facts.metric(rk.metric_name)
            if metric is None or not metric.value.usable or metric.value.value is None:
                return None
            key.append(-metric.value.value if rk.descending else metric.value.value)
        return tuple(key)

    def _leg_facts(self, plan: _Plan, role: Role) -> DecisionFacts | None:
        proposal = plan.proposals.get(role)
        if proposal is not None:
            return proposal.facts
        subject = plan.subject(role)
        if subject is None or role in plan.proposals:
            return None
        wanted = FactsPurpose.CLOSE if role == "close" else FactsPurpose.OPEN
        matches = [
            f for f in plan.evidence_facts if f.subject_ref == subject.ref and f.purpose is wanted
        ]
        return matches[0] if len(matches) == 1 else None

    # -- dry-run simulation ----------------------------------------------------------------

    def simulate(self, facts: DecisionFacts, sim: _Sim) -> tuple[int | None, list[Gap]]:
        base = facts.initial_quantity
        if base is None:
            return None, [_gap("target_quantity", "the fact set has no computed quantity")]
        requirement = self.requirements.get(facts.facts_ref)
        if requirement is None:
            if sim.reserved_any:
                return None, [
                    _gap(
                        "target_quantity",
                        "resource use unknown; cannot recompute after earlier dry-run reservations",
                    )
                ]
            return base, []
        quantity = base
        for use in requirement.uses:
            amount = sim.available.get(use.key)
            if amount is None:
                return None, [
                    _gap("target_quantity", f"baseline headroom for {use.key} unavailable")
                ]
            quantity = min(quantity, _floor_div(amount, use.per_contract))
        return quantity, []

    @staticmethod
    def reserve(requirement: ReservationRequirement | None, quantity: int, sim: _Sim) -> None:
        sim.reserved_any = True
        if requirement is None:
            return
        for use in requirement.uses:
            amount = sim.available.get(use.key)
            if amount is not None:
                sim.available[use.key] = amount - use.per_contract * quantity

    # -- leg and decision records ----------------------------------------------------------

    def legs(self, plan: _Plan, ordered: list[_Plan], sim: _Sim) -> list[LegRecord]:
        records: list[LegRecord] = []
        for role in plan.roles:
            if plan.proposals.get(role) is None and not plan.attempts.get(role):
                continue
            records.append(self._leg(plan, role, ordered, sim))
        return records

    def _dependency(self, plan: _Plan, role: Role, ordered: list[_Plan]) -> str | None:
        if role == "open" and plan.decision.action is DecisionAction.ROLL:
            return plan.leg_ref("close")
        if role == "open" and plan.funding:
            funders = [p for p in ordered if p.decision_ref in plan.funding]
            if funders and "close" in funders[0].roles:
                return funders[0].leg_ref("close")
        return None

    def _leg(self, plan: _Plan, role: Role, ordered: list[_Plan], sim: _Sim) -> LegRecord:
        leg_ref = plan.leg_ref(role)
        subject = plan.subject(role)
        facts = self._leg_facts(plan, role)
        proposal = plan.proposals.get(role)
        live_actions = sorted(plan.attempts.get(role, []), key=lambda a: call_sort_key(a.call))
        dependent = role == "open" and (
            plan.decision.action is DecisionAction.ROLL
            or bool(plan.funding)
            or plan.funding_invalid
        )
        conditional = dependent and not live_actions
        gaps: list[Gap] = []
        reasons: list[ReasonCode] = []
        attempts: list[Attempt] = [
            a.attempt.model_copy(update={"index": i}) for i, a in enumerate(live_actions)
        ]
        target: int | None = None
        if facts is not None:
            target = facts.initial_quantity
            if target is None:
                gaps.append(_gap("target_quantity", "the fact set has no computed quantity"))
            if (
                role == "close"
                and plan.decision.action is DecisionAction.CLOSE
                and facts.close_quantity is None
            ):
                target = None
                gaps.append(_gap("target_quantity", "no discretionary close quantity was supplied"))
        else:
            gaps.append(_gap("target_quantity", "no fact set is associated with this leg"))
        if conditional:
            gaps.append(
                _gap("funding", "conditional on a hypothetical close; funding unverifiable")
            )
        if self.off and proposal is not None and not live_actions:
            target, attempt = self._dry_attempt(
                plan, role, proposal, conditional, sim, gaps, reasons
            )
            if attempt is not None:
                attempts.append(attempt)
        elif target is not None and live_actions:
            self._check_actual_quantities(plan, role, target, live_actions)
        if target is None and "target_quantity" not in {g.field for g in gaps}:
            gaps.append(_gap("target_quantity", "target quantity unavailable"))
        quote_ids: dict[UUID, None] = {}
        for qid in facts.quote_refs if facts else ():
            quote_ids[qid] = None
        for attempt in attempts:
            for qid in attempt.quote_refs:
                quote_ids[qid] = None
        quotes = sorted(
            (self.quotes[q] for q in quote_ids if q in self.quotes),
            key=lambda q: (q.as_of, str(q.quote_id)),
        )
        evidence_ids = dict.fromkeys(
            (subject.source_evidence_ids if subject else ())
            + (facts.input_evidence_ids if facts else ())
        )
        occ = subject.occ_symbol if subject else None
        if occ is None:
            gaps.append(_gap("occ_symbol", "instrument identity unavailable for this leg"))
        return LegRecord(
            leg_ref=leg_ref,
            depends_on_leg_ref=self._dependency(plan, role, ordered),
            side=OrderSide.BUY_TO_CLOSE if role == "close" else OrderSide.SELL_TO_OPEN,
            occ_symbol=occ,
            broker_instrument_id=subject.broker_instrument_id if subject else None,
            right=occ.right if occ else None,
            strike=occ.strike if occ else None,
            expiration=occ.expiration if occ else None,
            target_quantity=target,
            candidate_origin=subject.candidate_origin if subject else None,
            facts_ref=facts.facts_ref if facts else None,
            source_evidence_ids=tuple(evidence_ids),
            quotes=tuple(quotes),
            board_comparison=subject.board_comparison if subject and role == "open" else None,
            conditional=conditional,
            attempts=tuple(attempts),
            reason_codes=tuple(dict.fromkeys(reasons)),
            gaps=tuple(gaps),
        )

    def _dry_attempt(
        self,
        plan: _Plan,
        role: Role,
        proposal: _Proposal,
        conditional: bool,
        sim: _Sim,
        gaps: list[Gap],
        reasons: list[ReasonCode],
    ) -> tuple[int | None, Attempt | None]:
        facts = proposal.facts
        leg_ref = plan.leg_ref(role)
        if (
            role == "close"
            and plan.decision.action is DecisionAction.CLOSE
            and facts.close_quantity is None
        ):
            reasons.append(ReasonCode.QUANTITY_UNAVAILABLE)
            return None, None
        if conditional:
            quantity = facts.initial_quantity
            sim_gaps: list[Gap] = []
        else:
            if sim.has_baseline and facts.snapshot_ref != sim.snapshot_ref:
                self.find(
                    "facts_snapshot_not_baseline",
                    "fact set uses another snapshot; the dry-run baseline is kept",
                    plan.decision_ref,
                    leg_ref,
                )
            quantity, sim_gaps = self.simulate(facts, sim)
        for gap in sim_gaps:
            if gap.field not in {g.field for g in gaps}:
                gaps.append(gap)
        if quantity is None:
            reasons.append(ReasonCode.QUANTITY_UNAVAILABLE)
            return None, None
        if quantity == 0:
            reasons.append(ReasonCode.NO_CAPACITY)
            self.find(
                "no_capacity",
                "computed quantity is zero; no intended order",
                plan.decision_ref,
                leg_ref,
            )
            return 0, None
        if proposal.leg.limit_price <= 0:
            self.find(
                "invalid_limit_price",
                "proposal limit price must be positive",
                plan.decision_ref,
                leg_ref,
            )
            return quantity, None
        if not conditional:
            self.reserve(self.requirements.get(facts.facts_ref), quantity, sim)
        attempt = Attempt(
            index=0,
            place_tool_call_id=None,
            proposal_ref=f"proposal:{leg_ref}",
            requested_quantity=quantity,
            order_type_raw=self.ctx.order_type,
            time_in_force_raw=self.ctx.time_in_force,
            limit_price=proposal.leg.limit_price,
            snapshot_ref=facts.snapshot_ref,
            quote_refs=facts.quote_refs,
            status=AttemptStatus.NOT_PLACED,
            broker_order_id=None,
            filled_quantity=None,
            reason_codes=(ReasonCode.DRY_RUN,),
        )
        return quantity, attempt

    def _check_actual_quantities(
        self, plan: _Plan, role: Role, target: int, actions: list[PlaceAction]
    ) -> None:
        """Flag (never correct) actual quantities that differ from the computed remainder."""
        leg_ref = plan.leg_ref(role)
        filled_before: int | None = 0
        for action in actions:
            attempt = action.attempt
            call_ids = (action.call.identity.tool_call_id,)
            if filled_before is not None and attempt.requested_quantity is not None:
                if attempt.requested_quantity != target - filled_before:
                    self.find(
                        "requested_quantity_differs_from_computed",
                        f"requested {attempt.requested_quantity}, computed remainder "
                        f"{target - filled_before}; actual arguments kept",
                        plan.decision_ref,
                        leg_ref,
                        call_ids,
                    )
            if filled_before is not None:
                filled_before = (
                    None
                    if attempt.filled_quantity is None
                    else filled_before + attempt.filled_quantity
                )
        if filled_before is not None and filled_before > target:
            self.find(
                "fills_exceed_target",
                "confirmed fills exceed the leg target",
                plan.decision_ref,
                leg_ref,
            )

    def decision_record(self, plan: _Plan, legs: list[LegRecord]) -> DecisionRecord:
        target = plan.target
        metrics: list[DerivedMetric] = []
        for role in plan.roles:
            facts = self._leg_facts(plan, role)
            if facts is None:
                continue
            for metric in facts.metrics:
                name = metric.name if len(plan.roles) == 1 else f"{role}.{metric.name}"
                metrics.append(metric.model_copy(update={"name": name}))
        return DecisionRecord(
            decision_ref=plan.decision_ref,
            action=plan.decision.action,
            priority=plan.priority,
            target_ref=plan.decision.target_ref,
            replacement_ref=plan.decision.replacement_ref,
            underlying=target.underlying if target else None,
            position_id=target.position_id if target else None,
            depends_on_decision_refs=tuple(plan.funding),
            rationale=plan.decision.rationale,
            thesis=plan.decision.thesis,
            invalidation_conditions=plan.decision.invalidation_conditions,
            evidence_refs=tuple(plan.evidence_refs),
            legs=tuple(legs),
            metrics=tuple(metrics),
            gaps=tuple(plan.gaps),
        )

    def funding(self, plans: list[_Plan]) -> None:
        """Derive reallocation dependencies from funding_close_refs (matching CLOSE decisions)."""
        for plan in plans:
            refs = plan.decision.funding_close_refs
            if not refs:
                continue
            if plan.decision.action not in _OPEN_ACTIONS:
                self.find(
                    "invalid_funding_dependency",
                    "only a new open may depend on funding closes",
                    plan.decision_ref,
                )
                continue
            for ref in refs:
                found = self.resolve(
                    ref, frozenset({RefKind.POSITION}), "funding_close_ref", plan.decision_ref
                )
                closes = [
                    p
                    for p in plans
                    if found is not None
                    and p.decision.action is DecisionAction.CLOSE
                    and p.target is not None
                    and p.target.ref == found.ref
                ]
                if len(closes) != 1:
                    self.find(
                        "invalid_funding_dependency",
                        "funding close does not match exactly one CLOSE decision",
                        plan.decision_ref,
                    )
                    plan.funding_invalid = True
                    continue
                plan.funding.append(closes[0].decision_ref)

    # -- verbatim model choices -------------------------------------------------------------

    def cancellation_rationales(self) -> list[CancellationRationaleRecord]:
        """The agent's cancellation rationales in output order (call after `run`)."""
        if self.output is None:
            return []
        return [
            CancellationRationaleRecord(
                cancel_call_ref=r.cancel_call_ref,
                cancel_tool_call_id=self.rationale_calls.get(i),
                rationale=r.rationale,
                evidence_refs=r.evidence_refs,
            )
            for i, r in enumerate(self.output.cancellation_rationales)
        ]

    def unresolved_questions(self) -> list[UnresolvedQuestionRecord]:
        """The agent's open research questions in output order."""
        if self.output is None:
            return []
        return [
            UnresolvedQuestionRecord(
                target_ref=q.target_ref, question=q.question, evidence_refs=q.evidence_refs
            )
            for q in self.output.unresolved_questions
        ]

    # -- main ------------------------------------------------------------------------------

    def run(self) -> tuple[list[DecisionRecord], list[Cancellation], list[UnassociatedAction]]:
        decisions: list[DecisionRecord] = []
        selected_cancels: set[UUID] = set()
        associated_places: set[UUID] = set()
        if self.output is not None:
            plans = [self.plan(i, d) for i, d in enumerate(self.output.decisions)]
            self.funding(plans)
            selected_cancels = self.associate(plans, self.output)
            ordered = self.prioritize(plans)
            baseline = self.ctx.reservation_baseline
            sim = _Sim(
                available={r.key: r.amount for r in baseline.resources} if baseline else {},
                snapshot_ref=baseline.snapshot_ref if baseline else None,
                has_baseline=baseline is not None,
            )
            built: dict[str, DecisionRecord] = {}
            sim_order = ordered + [p for p in plans if p.priority is None]
            for plan in sim_order:
                built[plan.decision_ref] = self.decision_record(plan, self.legs(plan, ordered, sim))
            decisions = [built[p.decision_ref] for p in plans]
            associated_places = {
                a.call.identity.tool_call_id
                for p in plans
                for acts in p.attempts.values()
                for a in acts
            }
        unassociated: list[UnassociatedAction] = []
        for action in self.idx.places:
            if action.call.identity.tool_call_id in associated_places:
                continue
            intent = action.intent
            unassociated.append(
                UnassociatedAction(
                    kind=UnassociatedActionKind.PLACE,
                    attempt=action.attempt,
                    occ_symbol=intent.occ_symbol if intent else None,
                    broker_instrument_id=intent.broker_instrument_id if intent else None,
                    side_raw=intent.side_raw if intent else None,
                    reason_codes=action.attempt.reason_codes,
                )
            )
        cancellations: list[Cancellation] = []
        for cancellation in self.idx.cancellations:
            call_id = cancellation.cancel_tool_call_id
            owner = self.idx.order_place_call_of_cancel.get(call_id)
            if call_id in selected_cancels or (owner is not None and owner in associated_places):
                cancellations.append(cancellation)
            else:
                unassociated.append(
                    UnassociatedAction(
                        kind=UnassociatedActionKind.CANCEL, cancellation=cancellation
                    )
                )
        return decisions, cancellations, unassociated


def _summary(
    ctx: AssemblyContext,
    status: DecisionOutputStatus,
    decisions: list[DecisionRecord],
    cancellations: list[Cancellation],
    unassociated: list[UnassociatedAction],
    findings: int,
) -> str:
    """Template-generated factual summary (never 'no trades' when actions were recorded)."""
    attempts = [a for d in decisions for leg in d.legs for a in leg.attempts] + [
        u.attempt for u in unassociated if u.attempt is not None
    ]
    dry = sum(1 for a in attempts if a.proposal_ref is not None)
    recorded = len(attempts) - dry
    total_cancels = len(cancellations) + sum(1 for u in unassociated if u.cancellation)
    coverage = (
        "decision coverage known"
        if status is DecisionOutputStatus.PARSED
        else ("decision coverage unknown")
    )
    return (
        f"Run {ctx.run_id} ({ctx.environment.value}, slot {ctx.slot.isoformat()}), "
        f"effective mode {ctx.effective_execution_mode.value}: decision output {status.value} "
        f"({coverage}); {len(decisions)} decision(s); {recorded} recorded place call(s); "
        f"{dry} dry-run proposal(s); {total_cancels} cancel call(s); "
        f"{len(unassociated)} unassociated action(s); {findings} assembly finding(s)."
    )


def assemble_run_record(context: AssemblyContext, decisions: DecisionsInput) -> RunRecord:
    """Assemble the complete RunRecord v4 from the immutable input bundle.

    Missing/invalid output yields no decisions but still records every place/cancel call.
    Performs no execution, retry, or replay; see the module docstring for the algorithm.
    """
    output: AgentDecisionOutput | None = None
    if isinstance(decisions, DecisionOutputParsed):
        output = decisions.output
        status = DecisionOutputStatus.PARSED
    elif isinstance(decisions, DecisionOutputParseFailure):
        status = DecisionOutputStatus.INVALID
    else:
        status = DecisionOutputStatus.MISSING
    assembler = _Assembler(context, output)
    if isinstance(decisions, DecisionOutputParseFailure):
        kinds = sorted({issue.kind for issue in decisions.issues}) or ["unknown"]
        assembler.find(
            "decision_output_invalid",
            f"{len(decisions.issues)} parse issue(s): {', '.join(kinds)}",
        )
    elif decisions is None:
        assembler.find("decision_output_missing", "no final model output was available")
    records, cancellations, unassociated = assembler.run()
    gaps: tuple[Gap, ...] = ()
    if status is not DecisionOutputStatus.PARSED:
        gaps = (_gap("decisions", f"decision output {status.value}; coverage unknown"),)
    findings = tuple(assembler.findings)
    return RunRecord(
        schema_version=RUN_RECORD_SCHEMA_VERSION,
        assembler_version=context.assembler_version,
        input_hash=input_hash(context, decisions),
        run_id=context.run_id,
        environment=context.environment,
        slot=context.slot,
        terminated_at=context.terminated_at,
        requested_execution_mode=context.requested_execution_mode,
        effective_execution_mode=context.effective_execution_mode,
        rules_version=context.rules_version,
        rules_hash=context.rules_hash,
        prompt_id=context.prompt_id,
        prompt_hash=context.prompt_hash,
        model_id=context.model_id,
        decision_output_status=status,
        decisions=tuple(records),
        cancellations=tuple(cancellations),
        cancellation_rationales=tuple(assembler.cancellation_rationales()),
        unresolved_questions=tuple(assembler.unresolved_questions()),
        unassociated_actions=tuple(unassociated),
        gaps=gaps,
        findings=findings,
        summary=_summary(context, status, records, cancellations, unassociated, len(findings)),
    )


__all__ = ["assemble_run_record", "canonical_json", "input_hash", "record_hash"]
