"""assemble_run_record: determinism, event authority, references, dry-run proposals, live.

Covers TESTING.md "Output ownership regressions required by ADR-0011" rows that apply to
assembly (the strict schema row belongs to the parser tests).
"""

import zlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.assembly import (
    assemble_run_record,
    canonical_json,
    input_hash,
    record_hash,
)
from wheelta_robinhood_agent.domain.assembly_context import (
    AssemblyContext,
    AttemptEvidence,
    DeliveredRef,
    RankingKey,
    RefKind,
    ReservationBaseline,
    ReservationRequirement,
    ResourceAmount,
    ResourceUse,
)
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    CancellationRationale,
    Decision,
    DecisionOutputParsed,
    DecisionOutputParseFailure,
    ParseIssue,
    ProposedLeg,
    ResearchQuestion,
)
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    CancelDispatchStatus,
    CancellationStatus,
    CandidateOrigin,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OrderSide,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.evidence import Derivation, Gap, SourcedValue
from wheelta_robinhood_agent.domain.facts import (
    DecisionFacts,
    DerivedMetric,
    FactsPurpose,
    FormulaVersion,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import (
    BrokerOrder,
    Cancellation,
    FillObservationKind,
    FillRecord,
    OrderIntent,
    OrderRecord,
    ReasonCode,
    StatusObservation,
)
from wheelta_robinhood_agent.domain.run_record import (
    AssemblyFinding,
    CancellationRationaleRecord,
    DecisionOutputStatus,
    Quote,
    RunRecord,
    UnassociatedActionKind,
    UnresolvedQuestionRecord,
)
from wheelta_robinhood_agent.domain.tool_calls import (
    ToolCallDecision,
    ToolCallIdentity,
    ToolCallRecord,
)

RUN = UUID(int=1)
OTHER_RUN = UUID(int=2)
ACCT = "acct-scope"
T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
SNAP = UUID(int=900)
PUT_A = OccSymbol.parse("AAPL  261016P00019000")
PUT_B = OccSymbol.parse("MSFT  261016P00019000")
PUT_A2 = OccSymbol.parse("AAPL  261023P00018000")
CALL_A = OccSymbol.parse("AAPL  261016C00021000")
SHORT_A = OccSymbol.parse("AAPL  261009P00020000")
POS_ID = UUID(int=700)
POS2_ID = UUID(int=701)


def uid(n: int) -> UUID:
    return UUID(int=n)


# ------------------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------------------


def call(
    n: int,
    tool: str,
    status: ToolCallStatus = ToolCallStatus.SUCCEEDED,
    t: int = 0,
    run_id: UUID = RUN,
) -> ToolCallRecord:
    requested = T0 + timedelta(seconds=t)
    identity = ToolCallIdentity(
        tool_call_id=uid(n),
        sdk_tool_use_id=f"tu{n:04d}",
        run_id=run_id,
        stage="agent",
        server="robinhood",
        tool=f"mcp__robinhood__{tool}",
        tier=ToolTier.X,
        requested_at=requested,
        arguments_redacted={},
    )
    if status is ToolCallStatus.DENIED:
        return ToolCallRecord(
            identity=identity,
            effective_arguments_redacted=None,
            decision=ToolCallDecision.DENIED,
            deny_reason="mode",
            status=status,
            dispatched_at=None,
            completed_at=None,
        )
    completed = requested + timedelta(seconds=1)
    terminal = status in (ToolCallStatus.SUCCEEDED, ToolCallStatus.FAILED, ToolCallStatus.TIMED_OUT)
    return ToolCallRecord(
        identity=identity,
        effective_arguments_redacted={},
        decision=ToolCallDecision.ALLOWED,
        status=status,
        dispatched_at=requested,
        completed_at=completed if terminal else None,
        result_ref=uid(n + 50_000) if status is ToolCallStatus.SUCCEEDED else None,
    )


def intent(
    place: int,
    occ: OccSymbol | None,
    qty: int | None,
    side: str = "sell_to_open",
    inst: str | None = None,
    price: str = "1.25",
) -> OrderIntent:
    return OrderIntent(
        intent_id=uid(place + 10_000),
        run_id=RUN,
        place_tool_call_id=uid(place),
        account_scope_id=ACCT,
        occ_symbol=occ,
        broker_instrument_id=inst,
        side_raw=side,
        quantity=qty,
        order_type_raw="limit",
        time_in_force_raw="day",
        limit_price=Decimal(price),
        requested_at=T0,
    )


def order(
    place: int,
    the_intent: OrderIntent | None,
    status: AttemptStatus | None = AttemptStatus.PLACED,
    broker: bool = True,
    fills: tuple[tuple[int, int, FillObservationKind], ...] = (),
    cancellations: tuple[Cancellation, ...] = (),
    status_t: int = 5,
) -> OrderRecord:
    order_id = uid(place + 20_000)
    history = (
        (
            StatusObservation(
                status=status,
                broker_status_raw=status.value,
                observed_at=T0 + timedelta(seconds=status_t),
                tool_call_id=uid(place),
            ),
        )
        if status is not None
        else ()
    )
    return OrderRecord(
        intent=the_intent,
        broker_order=BrokerOrder(
            order_id=order_id,
            account_scope_id=ACCT,
            broker_order_id=f"bo-{place}",
            intent_id=None,
            first_observed_at=T0,
        )
        if broker
        else None,
        review_tool_call_ids=(uid(place - 1),),
        status_history=history,
        cancellations=cancellations,
        fills=tuple(
            FillRecord(
                fill_id=uid(fid),
                order_id=order_id,
                kind=kind,
                broker_execution_id=f"ex-{fid}" if kind is FillObservationKind.EXECUTION else None,
                quantity=qty,
                price=Decimal("1.2"),
                executed_at=None,
                observed_at=T0 + timedelta(seconds=fid % 100),
                source_tool_call_id=uid(place),
            )
            for fid, qty, kind in fills
        ),
    )


def metric(name: str, value: str) -> DerivedMetric:
    return DerivedMetric(
        name=name,
        unit="ratio",
        value=SourcedValue[Decimal](
            evidence_id=uid(zlib.crc32(name.encode()) % 10_000 + 60_000),
            value=Decimal(value),
            source="code",
            as_of=T0,
            retrieved_at=T0,
            tool_call_id=None,
            quality=DataQuality.OK,
            derivation=Derivation(
                formula=name, formula_version="1", input_evidence_ids=(uid(800),)
            ),
        ),
    )


def facts(
    ref: str,
    subject: str,
    purpose: FactsPurpose = FactsPurpose.OPEN,
    qty: int | None = 2,
    snapshot: UUID | None = SNAP,
    limit_price: str | None = None,
    close_quantity: int | None = None,
    metrics: tuple[DerivedMetric, ...] = (),
    run_id: UUID = RUN,
    rules_version: str = "5",
    candidate_ref: str | None = None,
    position_id: UUID | None = None,
    quote_refs: tuple[UUID, ...] = (),
) -> DecisionFacts:
    gaps = (
        ()
        if qty is not None
        else (
            Gap(field="initial_quantity", kind=DataQuality.MISSING, detail="x"),
            Gap(field="remaining_quantity", kind=DataQuality.MISSING, detail="x"),
        )
    )
    return DecisionFacts(
        facts_id=uid(zlib.crc32(ref.encode()) % 10_000 + 30_000),
        facts_ref=ref,
        run_id=run_id,
        subject_ref=subject,
        purpose=purpose,
        observed_at=T0,
        rules_version=rules_version,
        rules_hash="rh",
        input_evidence_ids=(uid(800),),
        snapshot_ref=snapshot,
        quote_refs=quote_refs,
        candidate_ref=candidate_ref,
        position_id=position_id,
        limit_price=Decimal(limit_price) if limit_price else None,
        close_quantity=close_quantity,
        formula_versions=tuple(FormulaVersion(formula=m.name, version="1") for m in metrics),
        metrics=metrics,
        initial_quantity=qty,
        remaining_quantity=qty,
        quality=DataQuality.OK if qty is not None else DataQuality.MISSING,
        gaps=gaps,
    )


def cand(ref: str, occ: OccSymbol, inst: str | None = None, **kw: Any) -> DeliveredRef:
    base: dict[str, Any] = {
        "ref": ref,
        "kind": RefKind.CANDIDATE,
        "run_id": RUN,
        "account_scope_id": ACCT,
        "delivered": True,
        "underlying": occ.root,
        "occ_symbol": occ,
        "broker_instrument_id": inst,
        "candidate_origin": CandidateOrigin.ROBINHOOD,
    }
    base.update(kw)
    return DeliveredRef.model_validate(base)


def pos(
    ref: str, occ: OccSymbol, position_id: UUID = POS_ID, inst: str | None = None
) -> DeliveredRef:
    return DeliveredRef(
        ref=ref,
        kind=RefKind.POSITION,
        run_id=None,
        account_scope_id=ACCT,
        delivered=True,
        underlying=occ.root,
        occ_symbol=occ,
        broker_instrument_id=inst,
        position_id=position_id,
    )


def fref(ref: str, **kw: Any) -> DeliveredRef:
    base: dict[str, Any] = {
        "ref": ref,
        "kind": RefKind.FACTS,
        "run_id": RUN,
        "account_scope_id": ACCT,
        "delivered": True,
    }
    base.update(kw)
    return DeliveredRef.model_validate(base)


def cref(ref: str, call_id: int) -> DeliveredRef:
    return DeliveredRef(
        ref=ref,
        kind=RefKind.TOOL_CALL,
        run_id=RUN,
        account_scope_id=ACCT,
        delivered=True,
        tool_call_id=uid(call_id),
    )


def decision(
    action: DecisionAction,
    target: str,
    legs: tuple[tuple[str, str], ...] = (),
    replacement: str | None = None,
    exec_refs: tuple[str, ...] = (),
    funding: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
    thesis: str | None = "thesis",
    rationale: str = "because",
) -> Decision:
    return Decision(
        action=action,
        target_ref=target,
        replacement_ref=replacement,
        funding_close_refs=funding,
        proposed_legs=tuple(ProposedLeg(facts_ref=f, limit_price=p) for f, p in legs),
        execution_refs=exec_refs,
        rationale=rationale,
        thesis=thesis,
        invalidation_conditions=("breaks 17",) if thesis else (),
        evidence_refs=evidence,
    )


def parsed(
    *decisions: Decision,
    cancels: tuple[CancellationRationale, ...] = (),
    questions: tuple[ResearchQuestion, ...] = (),
) -> DecisionOutputParsed:
    return DecisionOutputParsed(
        ok=True,
        output=AgentDecisionOutput(
            decisions=decisions, cancellation_rationales=cancels, unresolved_questions=questions
        ),
    )


def ctx(mode: ExecutionMode = ExecutionMode.OFF, **kw: Any) -> AssemblyContext:
    base: dict[str, Any] = {
        "run_id": RUN,
        "environment": AppEnv.STAGING,
        "slot": T0,
        "terminated_at": T0 + timedelta(minutes=20),
        "requested_execution_mode": mode,
        "effective_execution_mode": mode,
        "account_scope_id": ACCT,
        "rules_version": "5",
        "rules_hash": "rh",
        "prompt_id": "wheel_agent.v5",
        "prompt_hash": "ph",
        "model_id": "model",
        "time_in_force": "day",
    }
    base.update(kw)
    return AssemblyContext.model_validate(base)


def codes(record: RunRecord) -> list[str]:
    return [f.code for f in record.findings]


def all_attempts(record: RunRecord) -> list[Any]:
    return [a for d in record.decisions for leg in d.legs for a in leg.attempts]


def cash(amount: str) -> ReservationBaseline:
    return ReservationBaseline(
        snapshot_ref=SNAP, resources=(ResourceAmount(key="cash", amount=Decimal(amount)),)
    )


def csp_use(ref: str, per: str = "1900") -> ReservationRequirement:
    return ReservationRequirement(
        facts_ref=ref, uses=(ResourceUse(key="cash", per_contract=Decimal(per)),)
    )


# ------------------------------------------------------------------------------------------
# Off mode: dry-run proposals
# ------------------------------------------------------------------------------------------


def test_single_dry_run_proposal() -> None:
    quote = Quote(
        quote_id=uid(400),
        broker_instrument_id="inst-a",
        bid=Decimal("1.2"),
        ask=Decimal("1.3"),
        as_of=T0,
        source_tool_call_ids=(uid(401),),
    )
    context = ctx(
        refs=(cand("cand:a", PUT_A, "inst-a"), fref("facts:a")),
        facts=(facts("facts:a", "cand:a", quote_refs=(uid(400),)),),
        quotes=(quote,),
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)))
    )
    assert record.decision_output_status is DecisionOutputStatus.PARSED
    (dec,) = record.decisions
    assert dec.decision_ref == f"decision:{RUN}:0"
    assert dec.priority == 1
    assert dec.underlying == "AAPL"
    (leg,) = dec.legs
    assert leg.leg_ref == f"decision:{RUN}:0:leg:open"
    assert leg.side is OrderSide.SELL_TO_OPEN
    assert leg.strike == Decimal("19") and leg.occ_symbol == PUT_A
    assert leg.target_quantity == 2
    assert leg.quotes == (quote,)
    assert leg.conditional is False
    (attempt,) = leg.attempts
    assert attempt.proposal_ref == f"proposal:{leg.leg_ref}"
    assert attempt.status is AttemptStatus.NOT_PLACED
    assert attempt.reason_codes == (ReasonCode.DRY_RUN,)
    assert attempt.requested_quantity == 2
    assert attempt.limit_price == Decimal("1.25")
    assert attempt.time_in_force_raw == "day" and attempt.order_type_raw == "limit"
    assert attempt.broker_order_id is None and attempt.fill_ids == ()
    assert attempt.place_tool_call_id is None and attempt.filled_quantity is None
    assert record.findings == () and record.cancellations == ()
    assert "1 dry-run proposal" in record.summary


def test_output_record_id_keys_refs() -> None:
    context = ctx(
        output_record_id=uid(77),
        refs=(cand("cand:a", PUT_A), fref("facts:a")),
        facts=(facts("facts:a", "cand:a"),),
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)))
    )
    assert record.decisions[0].decision_ref == f"decision:{uid(77)}:0"


def test_dry_proposals_compete_for_one_cash_baseline() -> None:
    # Baseline cash 5000; each contract reserves 1900. Facts each say 2 contracts.
    context = ctx(
        refs=(cand("cand:a", PUT_A), cand("cand:b", PUT_B), fref("facts:a"), fref("facts:b")),
        facts=(facts("facts:a", "cand:a"), facts("facts:b", "cand:b")),
        reservation_baseline=cash("5000"),
        reservation_requirements=(csp_use("facts:a"), csp_use("facts:b")),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)),
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
        ),
    )
    first, second = record.decisions
    assert first.legs[0].attempts[0].requested_quantity == 2
    assert second.legs[0].attempts == ()
    assert second.legs[0].target_quantity == 0
    assert ReasonCode.NO_CAPACITY in second.legs[0].reason_codes
    assert "no_capacity" in codes(record)


def test_ranking_decides_reservation_order() -> None:
    yield_key = (RankingKey(metric_name="yield", descending=True),)
    context = ctx(
        refs=(cand("cand:a", PUT_A), cand("cand:b", PUT_B), fref("facts:a"), fref("facts:b")),
        facts=(
            facts("facts:a", "cand:a", metrics=(metric("yield", "0.10"),)),
            facts("facts:b", "cand:b", metrics=(metric("yield", "0.20"),)),
        ),
        reservation_baseline=cash("5000"),
        reservation_requirements=(csp_use("facts:a"), csp_use("facts:b")),
        ranking_keys=yield_key,
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)),
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
        ),
    )
    a, b = record.decisions  # output order preserved; priorities derived
    assert (a.priority, b.priority) == (2, 1)
    assert b.legs[0].attempts[0].requested_quantity == 2
    assert a.legs[0].attempts == ()
    assert a.metrics[0].name == "yield"


def test_array_order_breaks_ranking_ties() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), cand("cand:b", PUT_B), fref("facts:a"), fref("facts:b")),
        facts=(
            facts("facts:a", "cand:a", metrics=(metric("yield", "0.10"),)),
            facts("facts:b", "cand:b", metrics=(metric("yield", "0.10"),)),
        ),
        ranking_keys=(RankingKey(metric_name="yield", descending=False),),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
            decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)),
        ),
    )
    assert [d.priority for d in record.decisions] == [1, 2]


def test_incomplete_ranking_falls_back_to_array_order() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), cand("cand:b", PUT_B), fref("facts:a"), fref("facts:b")),
        facts=(
            facts("facts:a", "cand:a", metrics=(metric("yield", "0.10"),)),
            facts("facts:b", "cand:b"),
        ),
        ranking_keys=(RankingKey(metric_name="yield", descending=True),),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
            decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)),
        ),
    )
    assert [d.priority for d in record.decisions] == [1, 2]
    assert "ranking_incomplete" in codes(record)
    assert any(g.field == "ranking" for g in record.decisions[0].gaps)


def test_later_facts_do_not_reset_reservations() -> None:
    # facts:b was computed on a newer snapshot that shows more cash; baseline is kept.
    context = ctx(
        refs=(cand("cand:a", PUT_A), cand("cand:b", PUT_B), fref("facts:a"), fref("facts:b")),
        facts=(
            facts("facts:a", "cand:a", qty=2),
            facts("facts:b", "cand:b", qty=5, snapshot=uid(901)),
        ),
        reservation_baseline=cash("5700"),
        reservation_requirements=(csp_use("facts:a"), csp_use("facts:b")),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)),
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
        ),
    )
    assert record.decisions[1].legs[0].attempts[0].requested_quantity == 1
    assert "facts_snapshot_not_baseline" in codes(record)


def test_unknown_resource_use_after_reservation_is_unavailable() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), cand("cand:b", PUT_B), fref("facts:a"), fref("facts:b")),
        facts=(facts("facts:a", "cand:a"), facts("facts:b", "cand:b")),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)),
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
        ),
    )
    leg_b = record.decisions[1].legs[0]
    assert leg_b.attempts == () and leg_b.target_quantity is None
    assert ReasonCode.QUANTITY_UNAVAILABLE in leg_b.reason_codes


def test_missing_baseline_headroom_is_unavailable() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), fref("facts:a")),
        facts=(facts("facts:a", "cand:a"),),
        reservation_baseline=ReservationBaseline(
            snapshot_ref=SNAP, resources=(ResourceAmount(key="cash", amount=None),)
        ),
        reservation_requirements=(csp_use("facts:a"),),
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)))
    )
    leg = record.decisions[0].legs[0]
    assert leg.attempts == () and leg.target_quantity is None


def test_negative_headroom_is_zero_capacity() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), fref("facts:a")),
        facts=(facts("facts:a", "cand:a"),),
        reservation_baseline=cash("-5"),
        reservation_requirements=(csp_use("facts:a"),),
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)))
    )
    assert record.decisions[0].legs[0].target_quantity == 0


def test_zero_and_unavailable_capacity_make_no_attempt() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), cand("cand:b", PUT_B), fref("facts:a"), fref("facts:b")),
        facts=(facts("facts:a", "cand:a", qty=0), facts("facts:b", "cand:b", qty=None)),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)),
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
        ),
    )
    zero, none = (d.legs[0] for d in record.decisions)
    assert zero.attempts == () and zero.target_quantity == 0
    assert zero.reason_codes == (ReasonCode.NO_CAPACITY,)
    assert none.attempts == () and none.target_quantity is None
    assert none.reason_codes == (ReasonCode.QUANTITY_UNAVAILABLE,)
    assert any(g.field == "target_quantity" for g in none.gaps)
    assert all_attempts(record) == []


def test_non_positive_limit_price_makes_no_attempt() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), fref("facts:a")), facts=(facts("facts:a", "cand:a"),)
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "0"),)))
    )
    assert record.decisions[0].legs[0].attempts == ()
    assert "invalid_limit_price" in codes(record)


def test_close_requires_discretionary_close_quantity() -> None:
    context = ctx(
        refs=(pos("pos:a", SHORT_A), fref("facts:c1"), fref("facts:c2")),
        facts=(
            facts("facts:c1", "pos:a", FactsPurpose.CLOSE, qty=1),
            facts("facts:c2", "pos:a", FactsPurpose.CLOSE, qty=1, close_quantity=1),
        ),
    )
    missing = assemble_run_record(
        context, parsed(decision(DecisionAction.CLOSE, "pos:a", (("facts:c1", "0.40"),)))
    )
    leg = missing.decisions[0].legs[0]
    assert leg.attempts == () and leg.target_quantity is None
    assert ReasonCode.QUANTITY_UNAVAILABLE in leg.reason_codes
    ok = assemble_run_record(
        context, parsed(decision(DecisionAction.CLOSE, "pos:a", (("facts:c2", "0.40"),)))
    )
    leg = ok.decisions[0].legs[0]
    assert leg.side is OrderSide.BUY_TO_CLOSE and leg.attempts[0].requested_quantity == 1
    assert ok.decisions[0].position_id == POS_ID


def test_dry_roll_open_is_conditional_and_reserves_nothing() -> None:
    context = ctx(
        refs=(
            pos("pos:a", SHORT_A),
            cand("cand:r", PUT_A2),
            cand("cand:b", PUT_B),
            fref("facts:close"),
            fref("facts:open"),
            fref("facts:b"),
        ),
        facts=(
            facts("facts:close", "pos:a", FactsPurpose.CLOSE, qty=1),
            facts("facts:open", "cand:r", qty=1),
            facts("facts:b", "cand:b", qty=2),
        ),
        reservation_baseline=cash("3800"),
        reservation_requirements=(csp_use("facts:open"), csp_use("facts:b")),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),)),
            decision(
                DecisionAction.ROLL,
                "pos:a",
                (("facts:close", "0.40"), ("facts:open", "1.10")),
                replacement="cand:r",
            ),
        ),
    )
    open_dec, roll = record.decisions
    assert (roll.priority, open_dec.priority) == (1, 2)
    close_leg, open_leg = roll.legs
    assert close_leg.attempts[0].reason_codes == (ReasonCode.DRY_RUN,)
    assert open_leg.conditional is True
    assert open_leg.depends_on_leg_ref == close_leg.leg_ref
    assert any(g.field == "funding" for g in open_leg.gaps)
    assert open_leg.attempts[0].requested_quantity == 1
    assert open_leg.board_comparison is None
    # The conditional roll open did not consume the 3800: the CSP still gets 2 contracts.
    assert open_dec.legs[0].attempts[0].requested_quantity == 2
    assert {m.name for m in roll.metrics} == set()


def test_reallocation_open_depends_on_close() -> None:
    context = ctx(
        refs=(pos("pos:a", SHORT_A), cand("cand:b", PUT_B), fref("facts:c"), fref("facts:b")),
        facts=(
            facts("facts:c", "pos:a", FactsPurpose.CLOSE, qty=1, close_quantity=1),
            facts("facts:b", "cand:b", qty=1),
        ),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),), funding=("pos:a",)),
            decision(DecisionAction.CLOSE, "pos:a", (("facts:c", "0.40"),)),
        ),
    )
    open_dec, close_dec = record.decisions
    assert open_dec.depends_on_decision_refs == (close_dec.decision_ref,)
    assert open_dec.legs[0].conditional is True
    assert open_dec.legs[0].depends_on_leg_ref == close_dec.legs[0].leg_ref


def test_invalid_funding_dependency_stays_conditional() -> None:
    context = ctx(
        refs=(pos("pos:a", SHORT_A), cand("cand:b", PUT_B), fref("facts:b")),
        facts=(facts("facts:b", "cand:b", qty=1),),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:b", (("facts:b", "1.25"),), funding=("pos:a",)),
            decision(DecisionAction.HOLD, "pos:a", funding=("pos:a",)),
        ),
    )
    assert codes(record).count("invalid_funding_dependency") == 2
    leg = record.decisions[0].legs[0]
    assert leg.conditional is True and leg.depends_on_leg_ref is None
    assert record.decisions[1].priority is None and record.decisions[1].legs == ()


def test_off_mode_rejects_execution_links_and_incomplete_shapes() -> None:
    context = ctx(
        refs=(pos("pos:a", SHORT_A), cand("cand:r", PUT_A2), fref("facts:close"), cref("c:x", 9)),
        facts=(facts("facts:close", "pos:a", FactsPurpose.CLOSE, qty=1),),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.ROLL,
                "pos:a",
                (("facts:close", "0.40"),),
                replacement="cand:r",
                exec_refs=("c:x",),
            ),
            cancels=(
                CancellationRationale(cancel_call_ref="c:x", rationale="r", evidence_refs=()),
            ),
        ),
    )
    found = codes(record)
    assert "execution_refs_in_off_mode" in found
    assert "cancellation_rationale_in_off_mode" in found
    assert "proposal_shape_incomplete" in found
    assert "duplicate_execution_association" in found


def test_proposal_problems_are_findings() -> None:
    context = ctx(
        refs=(
            pos("pos:a", SHORT_A),
            cand("cand:r", PUT_A2),
            fref("facts:close"),
            fref("facts:open"),
            fref("facts:priced"),
            fref("facts:other"),
        ),
        facts=(
            facts("facts:close", "pos:a", FactsPurpose.CLOSE, qty=1),
            facts("facts:open", "cand:r", qty=1),
            facts("facts:priced", "cand:r", qty=1, limit_price="1.00"),
            facts("facts:other", "cand:r", qty=1, candidate_ref="cand:zzz"),
        ),
    )
    roll = decision(
        DecisionAction.ROLL,
        "pos:a",
        (("facts:open", "1.10"), ("facts:close", "0.40")),
        replacement="cand:r",
    )
    assert "proposal_order" in codes(assemble_run_record(context, parsed(roll)))
    dup = decision(
        DecisionAction.ROLL,
        "pos:a",
        (("facts:close", "0.40"), ("facts:open", "1.10"), ("facts:open", "1.10")),
        replacement="cand:r",
    )
    rec = assemble_run_record(context, parsed(dup))
    assert "duplicate_proposal" in codes(rec)
    assert len(rec.decisions[0].legs) == 1  # the invalid open leg is not fabricated
    priced = decision(DecisionAction.OPEN_CSP, "cand:r", (("facts:priced", "1.10"),))
    assert "facts_price_mismatch" in codes(assemble_run_record(context, parsed(priced)))
    other = decision(DecisionAction.OPEN_CSP, "cand:r", (("facts:other", "1.10"),))
    assert "mismatched_contract" in codes(assemble_run_record(context, parsed(other)))
    wrong_purpose = decision(DecisionAction.OPEN_CSP, "cand:r", (("facts:close", "1.10"),))
    assert "incompatible_proposal" in codes(assemble_run_record(context, parsed(wrong_purpose)))
    hold = decision(DecisionAction.HOLD, "pos:a", (("facts:close", "0.40"),))
    rec = assemble_run_record(context, parsed(hold))
    assert "incompatible_proposal" in codes(rec) and rec.decisions[0].priority is None


def test_position_id_mismatch_in_facts() -> None:
    context = ctx(
        refs=(pos("pos:a", SHORT_A), fref("facts:c")),
        facts=(
            facts(
                "facts:c",
                "pos:a",
                FactsPurpose.CLOSE,
                qty=1,
                close_quantity=1,
                position_id=POS2_ID,
            ),
        ),
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.CLOSE, "pos:a", (("facts:c", "0.40"),)))
    )
    assert "mismatched_contract" in codes(record) and record.decisions[0].legs == ()


# ------------------------------------------------------------------------------------------
# Determinism and missing/invalid output
# ------------------------------------------------------------------------------------------


def _live_bundle() -> AssemblyContext:
    place = call(11, "place_option_order", t=10)
    return ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(10, "review_option_order", t=5), place),
        orders=(order(11, intent(11, PUT_A, 2, inst="inst-a")),),
        refs=(cand("cand:a", PUT_A, "inst-a"), cref("call:11", 11), fref("facts:a")),
        facts=(facts("facts:a", "cand:a"),),
    )


def test_determinism_with_and_without_output() -> None:
    context = _live_bundle()
    out = parsed(decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",)))
    for decisions in (out, None):
        first = assemble_run_record(context, decisions)
        second = assemble_run_record(context.model_copy(deep=True), decisions)
        assert canonical_json(first) == canonical_json(second)
        assert record_hash(first) == record_hash(second)
        assert first.input_hash == input_hash(context, decisions)
    assert input_hash(context, out) != input_hash(context, None)
    changed = context.model_copy(update={"rules_hash": "other"})
    assert input_hash(changed, None) != input_hash(context, None)
    raw = canonical_json(assemble_run_record(context, out))
    assert raw.startswith(
        b'{"assembler_version":"assembler.v1","cancellation_rationales":[],"cancellations":[]'
    )


def test_missing_output_still_records_every_action() -> None:
    context = _live_bundle()
    record = assemble_run_record(context, None)
    assert record.decision_output_status is DecisionOutputStatus.MISSING
    assert record.decisions == ()
    (action,) = record.unassociated_actions
    assert action.kind is UnassociatedActionKind.PLACE
    assert action.attempt is not None and action.attempt.requested_quantity == 2
    assert action.occ_symbol == PUT_A and action.side_raw == "sell_to_open"
    assert record.gaps[0].field == "decisions"
    assert "decision_output_missing" in codes(record)
    assert "1 recorded place call" in record.summary and "coverage unknown" in record.summary


def test_invalid_output_is_recorded_as_invalid() -> None:
    failure = DecisionOutputParseFailure(
        ok=False,
        raw_text="{bad",
        issues=(ParseIssue(loc="", message="m", kind="invalid_json"),),
    )
    record = assemble_run_record(_live_bundle(), failure)
    assert record.decision_output_status is DecisionOutputStatus.INVALID
    assert record.decisions == () and len(record.unassociated_actions) == 1
    finding = next(f for f in record.findings if f.code == "decision_output_invalid")
    assert "invalid_json" in finding.detail
    empty = DecisionOutputParseFailure(ok=False, raw_text="", issues=())
    assert "unknown" in next(f.detail for f in assemble_run_record(_live_bundle(), empty).findings)
    assert input_hash(_live_bundle(), failure) != input_hash(_live_bundle(), empty)


def test_prior_findings_pass_through() -> None:
    prior = AssemblyFinding(code="parse_semantic", detail="from the validator")
    record = assemble_run_record(ctx(prior_findings=(prior,)), parsed())
    assert record.findings == (prior,)
    assert "0 decision(s)" in record.summary


# ------------------------------------------------------------------------------------------
# Live: event authority
# ------------------------------------------------------------------------------------------


def test_live_attempt_bound_by_place_call() -> None:
    context = _live_bundle().model_copy(
        update={
            "attempt_evidence": (
                AttemptEvidence(place_tool_call_id=uid(11), snapshot_ref=SNAP, quote_refs=()),
            )
        }
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.OPEN_CSP,
                "cand:a",
                exec_refs=("call:11",),
                evidence=("facts:a",),
            )
        ),
    )
    (leg,) = record.decisions[0].legs
    (attempt,) = leg.attempts
    assert attempt.place_tool_call_id == uid(11)
    assert attempt.status is AttemptStatus.PLACED and attempt.broker_order_id == "bo-11"
    assert attempt.review_tool_call_ids == (uid(10),)
    assert attempt.snapshot_ref == SNAP
    assert leg.target_quantity == 2 and leg.facts_ref == "facts:a"
    assert leg.conditional is False
    assert record.unassociated_actions == ()
    assert record.findings == ()


def test_agent_omits_actions_all_survive() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(
            call(11, "place_option_order", t=1),
            call(12, "place_option_order", ToolCallStatus.DENIED, t=2),
            call(13, "place_option_order", ToolCallStatus.TIMED_OUT, t=3),
            call(14, "place_option_order", ToolCallStatus.FAILED, t=4),
            call(15, "place_option_order", ToolCallStatus.UNKNOWN, t=5),
            call(16, "cancel_option_order", ToolCallStatus.DENIED, t=6),
            call(17, "cancel_option_order", ToolCallStatus.SUCCEEDED, t=7),
        ),
        orders=(
            order(11, intent(11, PUT_A, 1), status=AttemptStatus.REJECTED),
            order(12, intent(12, PUT_A, 1), status=None, broker=False),
            order(13, intent(13, PUT_A, 1), status=None, broker=False),
        ),
        refs=(cand("cand:a", PUT_A),),
    )
    record = assemble_run_record(context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a")))
    places = [u.attempt for u in record.unassociated_actions if u.attempt is not None]
    by_call = {a.place_tool_call_id: a for a in places}
    assert len(places) == 5
    assert by_call[uid(11)].status is AttemptStatus.REJECTED
    assert by_call[uid(11)].reason_codes == (ReasonCode.BROKER_REJECTED,)
    assert by_call[uid(12)].status is AttemptStatus.NOT_PLACED
    assert by_call[uid(12)].reason_codes == (ReasonCode.NOT_DISPATCHED,)
    assert by_call[uid(12)].filled_quantity == 0
    assert by_call[uid(13)].status is AttemptStatus.UNKNOWN
    assert by_call[uid(13)].reason_codes == (ReasonCode.OUTCOME_UNKNOWN,)
    assert by_call[uid(14)].reason_codes == (ReasonCode.TOOL_ERROR, ReasonCode.OUTCOME_UNKNOWN)
    assert by_call[uid(15)].requested_quantity is None
    cancels = [u.cancellation for u in record.unassociated_actions if u.cancellation]
    assert [c.status for c in cancels] == [CancellationStatus.UNKNOWN, CancellationStatus.PENDING]
    found = codes(record)
    assert "cancel_not_dispatched" in found and "place_intent_missing" in found
    assert record.decisions[0].legs == ()


def test_contradictory_associations_fail_visibly() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(
            call(11, "place_option_order", t=1),
            call(21, "place_option_order", t=2),
            call(31, "place_option_order", t=3),
            call(41, "get_option_orders", t=4),
            call(51, "place_option_order", t=5),
        ),
        orders=(
            order(11, intent(11, PUT_A, 1, inst="inst-a")),
            order(21, intent(21, PUT_B, 1, inst="inst-b")),
            order(31, intent(31, PUT_A, 1, side="buy_to_open", inst="inst-a")),
            order(51, intent(51, None, 1)),
        ),
        refs=(
            cand("cand:a", PUT_A, "inst-a"),
            cand("cand:x", PUT_A, None, occ_symbol=None),
            pos("pos:a", SHORT_A),
            cref("call:11", 11),
            cref("call:21", 21),
            cref("call:31", 31),
            cref("call:41", 41),
            cref("call:51", 51),
        ),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.CLOSE, "pos:a", exec_refs=("call:11",)),
            decision(
                DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:21", "call:31", "call:41")
            ),
            decision(DecisionAction.OPEN_CSP, "cand:x", exec_refs=("call:51",)),
        ),
    )
    found = codes(record)
    assert "incompatible_execution_ref" in found  # STO call on a CLOSE; buy_to_open
    assert "mismatched_contract" in found  # MSFT call on AAPL decision
    assert "wrong_reference_kind" in found  # an order read is not an execution
    assert "execution_ref_unverifiable" in found  # no comparable identity
    assert len(record.unassociated_actions) == 4
    assert all_attempts(record) == []


def test_duplicate_execution_association_is_not_guessed() -> None:
    context = _live_bundle()
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",)),
            decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",)),
        ),
    )
    assert codes(record).count("duplicate_execution_association") == 2
    assert all_attempts(record) == [] and len(record.unassociated_actions) == 1


def test_actual_quantity_differs_from_computed() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(11, "place_option_order"),),
        orders=(order(11, intent(11, PUT_A, 5, inst="inst-a")),),
        refs=(cand("cand:a", PUT_A, "inst-a"), cref("call:11", 11), fref("facts:a")),
        facts=(facts("facts:a", "cand:a", qty=3),),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",), evidence=("facts:a",)
            )
        ),
    )
    leg = record.decisions[0].legs[0]
    assert leg.target_quantity == 3
    assert leg.attempts[0].requested_quantity == 5  # actual arguments unchanged
    finding = next(
        f for f in record.findings if f.code == "requested_quantity_differs_from_computed"
    )
    assert finding.tool_call_ids == (uid(11),) and finding.leg_ref == leg.leg_ref


def _price_walk(confirm_t: int) -> AssemblyContext:
    """Step 1: 3 contracts, 1 fills (seen twice), cancel, 1 more fills during cancel;
    step 2: the remaining 1 contract."""
    cancellation = Cancellation(
        cancel_tool_call_id=uid(13),
        broker_order_id="bo-11",
        confirmation_tool_call_ids=(uid(14),),
        status=CancellationStatus.CONFIRMED,
        fill_ids=(uid(3),),
    )
    first = order(
        11,
        intent(11, PUT_A, 3, inst="inst-a"),
        status=AttemptStatus.CANCELLED,
        fills=(
            (1, 1, FillObservationKind.CUMULATIVE),
            (2, 1, FillObservationKind.CUMULATIVE),
            (3, 2, FillObservationKind.CUMULATIVE),
        ),
        cancellations=(cancellation,),
    )
    second = order(21, intent(21, PUT_A, 1, inst="inst-a", price="1.20"))
    return ctx(
        ExecutionMode.LIVE,
        tool_calls=(
            call(10, "review_option_order", t=1),
            call(11, "place_option_order", t=2),
            call(13, "cancel_option_order", t=4),
            call(14, "get_option_orders", t=confirm_t),
            call(20, "review_option_order", t=7),
            call(21, "place_option_order", t=8),
        ),
        orders=(second, first),
        refs=(
            cand("cand:a", PUT_A, "inst-a"),
            cref("call:11", 11),
            cref("call:21", 21),
            fref("facts:a"),
        ),
        facts=(facts("facts:a", "cand:a", qty=3),),
    )


def test_price_walk_with_fill_during_cancel() -> None:
    out = parsed(
        decision(
            DecisionAction.OPEN_CSP,
            "cand:a",
            exec_refs=("call:21", "call:11"),
            evidence=("facts:a",),
        )
    )
    record = assemble_run_record(_price_walk(confirm_t=5), out)
    leg = record.decisions[0].legs[0]
    step1, step2 = leg.attempts
    assert (step1.index, step1.place_tool_call_id) == (0, uid(11))
    assert (step2.index, step2.place_tool_call_id) == (1, uid(21))
    assert step1.filled_quantity == 2  # latest cumulative total, never summed
    assert step1.fill_ids == (uid(1), uid(2), uid(3))
    assert step1.cancel_tool_call_ids == (uid(13),)
    assert step1.confirmation_tool_call_ids == (uid(14),)
    (cancel,) = record.cancellations
    assert cancel.status is CancellationStatus.CONFIRMED and cancel.fill_ids == (uid(3),)
    assert record.unassociated_actions == ()
    assert "requested_quantity_differs_from_computed" not in codes(record)


def test_confirmation_read_before_cancel_is_not_confirmation() -> None:
    record = assemble_run_record(
        _price_walk(confirm_t=0),
        parsed(decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11", "call:21"))),
    )
    (cancel,) = record.cancellations
    assert cancel.status is CancellationStatus.PENDING
    assert "cancel_confirmation_unverified" in codes(record)


def test_confirmation_needs_terminal_order_and_successful_read() -> None:
    cancellation = Cancellation(
        cancel_tool_call_id=uid(13),
        broker_order_id="bo-11",
        confirmation_tool_call_ids=(uid(14),),
        status=CancellationStatus.CONFIRMED,
    )
    working = order(11, intent(11, PUT_A, 1), cancellations=(cancellation,))
    record = assemble_run_record(
        ctx(
            ExecutionMode.LIVE,
            tool_calls=(call(11, "place_option_order"), call(13, "cancel_option_order", t=2)),
            orders=(working,),
        ),
        None,
    )
    cancel = next(u.cancellation for u in record.unassociated_actions if u.cancellation)
    assert cancel.status is CancellationStatus.PENDING
    unknown_cancel = call(13, "cancel_option_order", ToolCallStatus.UNKNOWN, t=2)
    terminal = order(
        11, intent(11, PUT_A, 1), status=AttemptStatus.CANCELLED, cancellations=(cancellation,)
    )
    reconciled = assemble_run_record(
        ctx(
            ExecutionMode.LIVE,
            tool_calls=(
                call(11, "place_option_order"),
                unknown_cancel,
                call(14, "get_option_orders", t=4),
            ),
            orders=(terminal,),
        ),
        None,
    )
    # An unknown cancel outcome later reconciled by a terminal order read is confirmed.
    cancel = next(u.cancellation for u in reconciled.unassociated_actions if u.cancellation)
    assert cancel.status is CancellationStatus.CONFIRMED
    for tool_calls in (
        (
            call(11, "place_option_order"),
            call(13, "cancel_option_order", ToolCallStatus.DENIED, t=2),
            call(14, "get_option_orders", t=4),
        ),
        (
            call(11, "place_option_order"),
            call(13, "cancel_option_order", t=2),
            call(14, "get_option_orders", ToolCallStatus.FAILED, t=4),
        ),
    ):
        terminal = order(
            11, intent(11, PUT_A, 1), status=AttemptStatus.CANCELLED, cancellations=(cancellation,)
        )
        rec = assemble_run_record(
            ctx(ExecutionMode.LIVE, tool_calls=tool_calls, orders=(terminal,)), None
        )
        cancel = next(u.cancellation for u in rec.unassociated_actions if u.cancellation)
        assert cancel.status is not CancellationStatus.CONFIRMED


def test_partial_roll_close_without_replacement() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(11, "place_option_order"), call(13, "cancel_option_order", t=3)),
        orders=(
            order(
                11,
                intent(11, SHORT_A, 2, side="buy_to_close"),
                status=AttemptStatus.PARTIALLY_FILLED,
                fills=((5, 1, FillObservationKind.EXECUTION),),
                cancellations=(
                    Cancellation(
                        cancel_tool_call_id=uid(13),
                        broker_order_id="bo-11",
                        status=CancellationStatus.PENDING,
                    ),
                ),
            ),
        ),
        refs=(
            pos("pos:a", SHORT_A),
            cand("cand:r", PUT_A2),
            cref("call:11", 11),
            cref("call:13", 13),
            fref("facts:open"),
        ),
        facts=(facts("facts:open", "cand:r", qty=2),),
    )
    closed_only = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.ROLL, "pos:a", replacement="cand:r", exec_refs=("call:11", "call:13")
            )
        ),
    )
    (close_leg,) = closed_only.decisions[0].legs  # no fictional replacement leg
    assert close_leg.attempts[0].filled_quantity == 1
    assert close_leg.target_quantity is None
    assert len(closed_only.cancellations) == 1  # selected by the decision
    with_proposal = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.ROLL,
                "pos:a",
                (("facts:open", "1.10"),),
                replacement="cand:r",
                exec_refs=("call:11",),
            )
        ),
    )
    close_leg, open_leg = with_proposal.decisions[0].legs
    assert open_leg.attempts == () and open_leg.conditional is True
    assert open_leg.depends_on_leg_ref == close_leg.leg_ref
    assert open_leg.target_quantity == 2
    # The cancel is linked through its order's attempt even though the model omitted it.
    assert len(with_proposal.cancellations) == 1 and with_proposal.unassociated_actions == ()


def test_fills_exceeding_target_and_unknown_fill_totals() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(11, "place_option_order"), call(21, "place_option_order", t=5)),
        orders=(
            order(
                11,
                intent(11, PUT_A, 1, inst="inst-a"),
                status=AttemptStatus.FILLED,
                fills=((5, 2, FillObservationKind.EXECUTION),),
            ),
            order(
                21,
                intent(21, PUT_A, 1, inst="inst-a"),
                fills=(
                    (6, 2, FillObservationKind.CUMULATIVE),
                    (7, 1, FillObservationKind.CUMULATIVE),
                ),
            ),
        ),
        refs=(
            cand("cand:a", PUT_A, "inst-a"),
            cref("call:11", 11),
            cref("call:21", 21),
            fref("facts:a"),
        ),
        facts=(facts("facts:a", "cand:a", qty=1),),
    )
    single = parsed(
        decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",), evidence=("facts:a",))
    )
    assert "fills_exceed_target" in codes(assemble_run_record(context, single))
    both = parsed(
        decision(
            DecisionAction.OPEN_CSP,
            "cand:a",
            exec_refs=("call:21", "call:11"),
            evidence=("facts:a",),
        )
    )
    record = assemble_run_record(context, both)
    assert "inconsistent_fills" in codes(record)
    assert record.decisions[0].legs[0].attempts[1].filled_quantity is None


def test_standalone_cancellation_rationale() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(13, "cancel_option_order"), call(14, "get_option_orders", t=3)),
        refs=(cref("call:13", 13), cref("call:14", 14)),
    )
    explained = assemble_run_record(
        context,
        parsed(
            cancels=(
                CancellationRationale(
                    cancel_call_ref="call:13", rationale="stale", evidence_refs=()
                ),
                CancellationRationale(cancel_call_ref="call:14", rationale="x", evidence_refs=()),
            )
        ),
    )
    assert len(explained.cancellations) == 1 and explained.unassociated_actions == ()
    assert "wrong_reference_kind" in codes(explained)
    silent = assemble_run_record(context, parsed())
    assert silent.cancellations == () and len(silent.unassociated_actions) == 1


def test_cancel_owned_by_another_decision() -> None:
    cancellation = Cancellation(
        cancel_tool_call_id=uid(13), broker_order_id="bo-11", status=CancellationStatus.PENDING
    )
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(11, "place_option_order"), call(13, "cancel_option_order", t=3)),
        orders=(order(11, intent(11, PUT_A, 1, inst="inst-a"), cancellations=(cancellation,)),),
        refs=(
            cand("cand:a", PUT_A, "inst-a"),
            pos("pos:a", SHORT_A),
            cref("call:11", 11),
            cref("call:13", 13),
        ),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",)),
            decision(DecisionAction.CLOSE, "pos:a", exec_refs=("call:13",)),
        ),
    )
    assert "contradictory_cancel_association" in codes(record)
    # pending: releases nothing; assembly stamps the cancel's dispatch status.
    assert record.cancellations == (
        cancellation.model_copy(update={"dispatch_status": CancelDispatchStatus.DISPATCHED}),
    )


# ------------------------------------------------------------------------------------------
# Event projection edge cases
# ------------------------------------------------------------------------------------------


def test_event_projection_anomalies_are_findings() -> None:
    orphan = Cancellation(
        cancel_tool_call_id=uid(99), broker_order_id=None, status=CancellationStatus.UNKNOWN
    )
    c1 = Cancellation(
        cancel_tool_call_id=uid(13), broker_order_id=None, status=CancellationStatus.PENDING
    )
    c2 = Cancellation(
        cancel_tool_call_id=uid(13), broker_order_id=None, status=CancellationStatus.UNKNOWN
    )
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(
            call(11, "place_option_order"),
            call(21, "place_option_order", ToolCallStatus.DENIED, t=1),
            call(31, "place_option_order", t=2),
            call(41, "place_option_order", t=3),
            call(13, "cancel_option_order", t=4),
            call(90, "place_option_order", run_id=OTHER_RUN),
        ),
        orders=(
            order(11, intent(11, PUT_A, 1), cancellations=(orphan, c1)),
            order(11, intent(11, PUT_A, 2), cancellations=(c2,)),
            order(21, intent(21, PUT_A, 1)),
            order(31, intent(31, PUT_A, 1), status=AttemptStatus.NOT_PLACED),
            order(41, intent(41, PUT_A, 1), status=None, broker=True),
        ),
    )
    record = assemble_run_record(context, None)
    found = codes(record)
    for code in (
        "ambiguous_order_record",
        "denied_call_has_broker_order",
        "contradictory_order_status",
        "cancellation_without_call",
        "ambiguous_cancellation_record",
        "foreign_tool_call",
    ):
        assert code in found, code
    by_call = {
        u.attempt.place_tool_call_id: u.attempt for u in record.unassociated_actions if u.attempt
    }
    assert uid(90) not in by_call
    assert by_call[uid(11)].status is AttemptStatus.UNKNOWN  # ambiguous record ignored
    assert by_call[uid(21)].status is AttemptStatus.PLACED  # broker observation wins
    assert by_call[uid(31)].status is AttemptStatus.UNKNOWN
    assert by_call[uid(41)].status is AttemptStatus.PLACED
    cancel = next(u.cancellation for u in record.unassociated_actions if u.cancellation)
    assert cancel.status is CancellationStatus.PENDING
    # Duplicate identical order records are deduplicated silently.
    same = order(41, intent(41, PUT_A, 1))
    rec = assemble_run_record(
        ctx(ExecutionMode.LIVE, tool_calls=(call(41, "place_option_order"),), orders=(same, same)),
        None,
    )
    assert "ambiguous_order_record" not in codes(rec)


# ------------------------------------------------------------------------------------------
# References and ownership
# ------------------------------------------------------------------------------------------


def test_reference_validation_findings() -> None:
    context = ctx(
        refs=(
            cand("cand:other_run", PUT_A, run_id=OTHER_RUN),
            cand("cand:other_acct", PUT_A, account_scope_id="acct-other"),
            cand("cand:undelivered", PUT_A, delivered=False),
            cand("cand:call", CALL_A),
            pos("pos:a", SHORT_A),
            fref("facts:unrecorded"),
            fref("facts:other_run"),
            fref("facts:other_rules"),
            cand("cand:a", PUT_A),
            cand("cand:msft", PUT_B),
        ),
        facts=(
            facts("facts:other_run", "cand:a", run_id=OTHER_RUN),
            facts("facts:other_rules", "cand:a", rules_version="4"),
        ),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(DecisionAction.OPEN_CSP, "cand:fabricated"),
            decision(DecisionAction.OPEN_CSP, "pos:a"),
            decision(DecisionAction.OPEN_CSP, "cand:other_run"),
            decision(DecisionAction.OPEN_CSP, "cand:other_acct"),
            decision(DecisionAction.OPEN_CSP, "cand:undelivered"),
            decision(DecisionAction.OPEN_CSP, "cand:call"),
            decision(
                DecisionAction.OPEN_CSP,
                "cand:a",
                (
                    ("facts:unrecorded", "1"),
                    ("facts:other_run", "1"),
                    ("facts:other_rules", "1"),
                ),
                evidence=("facts:nope",),
            ),
            decision(DecisionAction.CLOSE, "pos:a", replacement="cand:a"),
            decision(DecisionAction.ROLL, "pos:a"),
            decision(DecisionAction.ROLL, "pos:a", replacement="cand:msft"),
            questions=(
                ResearchQuestion(target_ref="cand:ghost", question="q", evidence_refs=("ev:x",)),
                ResearchQuestion(target_ref=None, question="q2", evidence_refs=()),
            ),
        ),
    )
    found = codes(record)
    for code in (
        "unknown_reference",
        "wrong_reference_kind",
        "wrong_run_reference",
        "wrong_account_reference",
        "undelivered_reference",
        "incompatible_reference",
        "facts_not_recorded",
        "facts_rules_mismatch",
        "missing_replacement",
    ):
        assert code in found, code
    assert found.count("incompatible_reference") == 3  # call target, CLOSE replacement, MSFT roll
    # Invalid targets keep the decision but derive no legs or underlying by guessing.
    assert record.decisions[0].underlying is None and record.decisions[0].legs == ()
    assert record.decisions[6].evidence_refs == ()
    assert all_attempts(record) == []


def test_rationale_numbers_never_become_facts() -> None:
    context = ctx(
        refs=(cand("cand:a", PUT_A), fref("facts:a")),
        facts=(facts("facts:a", "cand:a", qty=1, metrics=(metric("yield", "0.12"),)),),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.OPEN_CSP,
                "cand:a",
                (("facts:a", "1.25"),),
                evidence=("facts:a",),
                rationale="Sell 50 contracts at $9.99; yield 45%; cash 1000000; expires 2031-01-01",
                thesis=None,
            )
        ),
    )
    (dec,) = record.decisions
    assert dec.rationale.startswith("Sell 50")
    assert [(m.name, m.value.value) for m in dec.metrics] == [("yield", Decimal("0.12"))]
    assert dec.legs[0].attempts[0].requested_quantity == 1
    assert dec.legs[0].attempts[0].limit_price == Decimal("1.25")
    assert dec.evidence_refs == ("facts:a",)
    assert any(g.field == "thesis" for g in dec.gaps)


def test_roll_metrics_are_prefixed_by_leg() -> None:
    context = ctx(
        refs=(pos("pos:a", SHORT_A), cand("cand:r", PUT_A2), fref("facts:c"), fref("facts:o")),
        facts=(
            facts("facts:c", "pos:a", FactsPurpose.CLOSE, qty=1, metrics=(metric("dte", "3"),)),
            facts("facts:o", "cand:r", qty=1, metrics=(metric("dte", "28"),)),
        ),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.ROLL,
                "pos:a",
                (("facts:c", "0.40"), ("facts:o", "1.10")),
                replacement="cand:r",
            )
        ),
    )
    assert [m.name for m in record.decisions[0].metrics] == ["close.dte", "open.dte"]


def test_no_replay_reassembly_adds_no_actions() -> None:
    context = _live_bundle()
    linked = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",)))
    )
    unlinked = assemble_run_record(context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a")))
    again = assemble_run_record(context, None)

    def place_ids(record: RunRecord) -> list[UUID]:
        attempts = all_attempts(record) + [
            u.attempt for u in record.unassociated_actions if u.attempt
        ]
        return sorted(a.place_tool_call_id for a in attempts if a.place_tool_call_id)

    assert place_ids(linked) == place_ids(unlinked) == place_ids(again) == [uid(11)]
    assert len(context.tool_calls) == 2  # the frozen input is untouched
    assert all(a.proposal_ref is None for a in all_attempts(linked))  # no dry proposal in live


def test_live_unsubmitted_proposal_has_no_dry_attempt() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        refs=(cand("cand:a", PUT_A, board_comparison=None), fref("facts:a")),
        facts=(facts("facts:a", "cand:a"),),
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", (("facts:a", "1.25"),)))
    )
    leg = record.decisions[0].legs[0]
    assert leg.attempts == () and leg.target_quantity == 2


def test_leg_without_instrument_identity_has_gap() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(11, "place_option_order"),),
        orders=(order(11, intent(11, PUT_A, 1, inst="inst-a")),),
        refs=(cand("cand:a", PUT_A, "inst-a", occ_symbol=None), cref("call:11", 11)),
    )
    record = assemble_run_record(
        context, parsed(decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",)))
    )
    leg = record.decisions[0].legs[0]
    assert leg.occ_symbol is None and any(g.field == "occ_symbol" for g in leg.gaps)
    assert any(g.field == "target_quantity" for g in leg.gaps)


def test_invalid_execution_and_cancel_refs() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(13, "cancel_option_order"),),
        refs=(cand("cand:a", PUT_A), cref("call:13", 13), fref("facts:a")),
        facts=(facts("facts:a", "cand:a"),),
    )
    record = assemble_run_record(
        context,
        parsed(
            decision(
                DecisionAction.OPEN_CSP,
                "cand:a",
                (("cand:a", "1.25"),),
                exec_refs=("call:ghost", "call:13"),
            ),
            cancels=(
                CancellationRationale(
                    cancel_call_ref="facts:a", rationale="r", evidence_refs=("ev:none",)
                ),
            ),
        ),
    )
    found = codes(record)
    assert found.count("unknown_reference") == 2  # execution ref and cancellation evidence
    assert found.count("wrong_reference_kind") == 2  # candidate as facts, facts as cancel
    assert len(record.cancellations) == 1  # a selected cancel with no order record


def test_context_and_ref_validation() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        DeliveredRef(
            ref="c", kind=RefKind.TOOL_CALL, run_id=RUN, account_scope_id=None, delivered=True
        )
    with pytest.raises(ValidationError, match="position_id"):
        DeliveredRef(
            ref="p", kind=RefKind.POSITION, run_id=None, account_scope_id=None, delivered=True
        )
    with pytest.raises(ValidationError, match="durable"):
        DeliveredRef(
            ref="e", kind=RefKind.EVIDENCE, run_id=None, account_scope_id=None, delivered=True
        )
    with pytest.raises(ValidationError, match="duplicate ref"):
        ctx(refs=(fref("facts:a"), fref("facts:a")))
    with pytest.raises(ValidationError, match="resource key"):
        ReservationBaseline(
            snapshot_ref=None,
            resources=(ResourceAmount(key="k", amount=None), ResourceAmount(key="k", amount=None)),
        )
    with pytest.raises(ValidationError, match="resource key"):
        ReservationRequirement(
            facts_ref="f",
            uses=(
                ResourceUse(key="k", per_contract=Decimal(1)),
                ResourceUse(key="k", per_contract=Decimal(1)),
            ),
        )


# ------------------------------------------------------------------------------------------
# Verbatim model choices: cancellation rationales and unresolved questions
# ------------------------------------------------------------------------------------------
def test_cancellation_rationales_and_questions_are_kept_verbatim() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(13, "cancel_option_order"), call(14, "get_option_orders", t=3)),
        refs=(cref("call:13", 13), cref("call:14", 14), fref("facts:a")),
    )
    output = parsed(
        cancels=(
            CancellationRationale(
                cancel_call_ref="call:13", rationale="stale", evidence_refs=("facts:a",)
            ),
            CancellationRationale(cancel_call_ref="call:14", rationale="x", evidence_refs=()),
            CancellationRationale(cancel_call_ref="call:99", rationale="y", evidence_refs=()),
        ),
        questions=(
            ResearchQuestion(target_ref=None, question="earnings date?", evidence_refs=()),
            ResearchQuestion(target_ref="cand:zz", question="halted?", evidence_refs=("facts:a",)),
        ),
    )
    record = assemble_run_record(context, output)
    assert record.cancellation_rationales == (
        CancellationRationaleRecord(
            cancel_call_ref="call:13",
            cancel_tool_call_id=uid(13),
            rationale="stale",
            evidence_refs=("facts:a",),
        ),
        # Not a cancel call / never issued: kept verbatim, not resolved to a call.
        CancellationRationaleRecord(
            cancel_call_ref="call:14", cancel_tool_call_id=None, rationale="x"
        ),
        CancellationRationaleRecord(
            cancel_call_ref="call:99", cancel_tool_call_id=None, rationale="y"
        ),
    )
    assert record.unresolved_questions == (
        UnresolvedQuestionRecord(target_ref=None, question="earnings date?"),
        UnresolvedQuestionRecord(
            target_ref="cand:zz", question="halted?", evidence_refs=("facts:a",)
        ),
    )
    assert {"wrong_reference_kind", "unknown_reference"} <= set(codes(record))
    again = assemble_run_record(context, output)
    assert canonical_json(again) == canonical_json(record)


def test_duplicate_cancel_claims_leave_the_rationale_unresolved() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(13, "cancel_option_order"),),
        refs=(cref("call:13", 13),),
    )
    record = assemble_run_record(
        context,
        parsed(
            cancels=(
                CancellationRationale(cancel_call_ref="call:13", rationale="a", evidence_refs=()),
                CancellationRationale(cancel_call_ref="call:13", rationale="b", evidence_refs=()),
            )
        ),
    )
    assert [r.cancel_tool_call_id for r in record.cancellation_rationales] == [None, None]
    assert "duplicate_execution_association" in codes(record)


def test_no_model_choices_without_parsed_output() -> None:
    context = ctx(ExecutionMode.LIVE, tool_calls=(call(13, "cancel_option_order"),))
    record = assemble_run_record(context, None)
    assert record.cancellation_rationales == () and record.unresolved_questions == ()
    with pytest.raises(ValidationError, match="synthesized"):
        RunRecord.model_validate(
            {
                **record.model_dump(),
                "unresolved_questions": ({"target_ref": None, "question": "q"},),
            }
        )


def test_denied_cancel_records_dispatch_status_and_reason() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(
            call(13, "cancel_option_order", status=ToolCallStatus.DENIED),
            call(14, "cancel_option_order", status=ToolCallStatus.REQUESTED, t=1),
            call(15, "cancel_option_order", t=2),
        ),
    )
    record = assemble_run_record(context, None)
    cancels = {
        u.cancellation.cancel_tool_call_id: u.cancellation
        for u in record.unassociated_actions
        if u.cancellation
    }
    denied, dispatched_pending, succeeded = cancels[uid(13)], cancels[uid(14)], cancels[uid(15)]
    assert denied.dispatch_status is CancelDispatchStatus.DENIED
    assert denied.dispatch_reason == "mode"
    assert denied.status is CancellationStatus.UNKNOWN
    # call(..., REQUESTED) is dispatched with no outcome: in flight, not pending.
    assert dispatched_pending.dispatch_status is CancelDispatchStatus.DISPATCHED
    assert dispatched_pending.status is CancellationStatus.UNKNOWN
    assert succeeded.dispatch_status is CancelDispatchStatus.DISPATCHED
    assert succeeded.status is CancellationStatus.PENDING and succeeded.dispatch_reason is None
    assert "cancel_not_dispatched" in codes(record)


def test_undispatched_cancel_is_never_pending() -> None:
    requested_only = ToolCallRecord(
        identity=call(13, "cancel_option_order").identity,
        effective_arguments_redacted=None,
        decision=None,
        status=ToolCallStatus.REQUESTED,
        dispatched_at=None,
        completed_at=None,
    )
    pending = Cancellation(
        cancel_tool_call_id=uid(13), broker_order_id="bo-11", status=CancellationStatus.PENDING
    )
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(11, "place_option_order"), requested_only),
        orders=(order(11, intent(11, PUT_A, 1, inst="inst-a"), cancellations=(pending,)),),
    )
    record = assemble_run_record(context, None)
    (cancel,) = [u.cancellation for u in record.unassociated_actions if u.cancellation]
    assert cancel.dispatch_status is CancelDispatchStatus.NO_DISPATCH_RECORDED
    assert cancel.status is CancellationStatus.UNKNOWN
    with pytest.raises(ValidationError, match="undispatched"):
        Cancellation.model_validate({**cancel.model_dump(), "status": CancellationStatus.PENDING})
    with pytest.raises(ValidationError, match="dispatch_reason"):
        Cancellation(
            cancel_tool_call_id=uid(13),
            broker_order_id=None,
            status=CancellationStatus.UNKNOWN,
            dispatch_status=CancelDispatchStatus.DENIED,
        )
