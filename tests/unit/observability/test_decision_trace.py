"""Decision trace (observability/decision_trace.py): pure, resolves every reference it can."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    AuditCheck,
    AuditOutcome,
    CancellationStatus,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
    OrderVenue,
    RunStatus,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts import DecisionFacts, FactsPurpose
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import Attempt, Cancellation
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import (
    RUN_RECORD_SCHEMA_VERSION,
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    Quote,
    RunRecord,
    UnassociatedAction,
    UnassociatedActionKind,
    UnresolvedQuestionRecord,
)
from wheelta_robinhood_agent.domain.tool_calls import (
    ToolCallDecision,
    ToolCallIdentity,
    ToolCallRecord,
)
from wheelta_robinhood_agent.observability.decision_trace import (
    TraceInput,
    build_decision_trace,
    decision_log_events,
    render_trace_markdown,
)

T0 = datetime(2026, 9, 28, 15, 0, tzinfo=UTC)
RUN = uuid4()
OCC = OccSymbol.parse("SPY   261016P00740000")


def _call(tool: str, minute: int, tier: ToolTier = ToolTier.X, **kw: Any) -> ToolCallRecord:
    at = T0 + timedelta(minutes=minute)
    identity = ToolCallIdentity(
        tool_call_id=uuid4(),
        sdk_tool_use_id=f"toolu_{minute}",
        run_id=RUN,
        stage="agent",
        server=kw.pop("server", "robinhood"),
        tool=tool,
        tier=tier,
        requested_at=at,
        arguments_redacted={"account_number": "…1234"},
        agent_id=kw.pop("agent_id", None),
        agent_type=kw.pop("agent_type", None),
    )
    if kw.pop("denied", False):
        return ToolCallRecord(
            identity=identity,
            effective_arguments_redacted=None,
            decision=ToolCallDecision.DENIED,
            deny_reason="not available",
            status=ToolCallStatus.DENIED,
            dispatched_at=None,
            completed_at=None,
        )
    return ToolCallRecord(
        identity=identity,
        effective_arguments_redacted={"account_number": "…1234", "price": "1.79"},
        decision=ToolCallDecision.ALLOWED,
        status=ToolCallStatus.SUCCEEDED,
        dispatched_at=at,
        completed_at=at + timedelta(seconds=1),
        latency_ms=1000,
        result_ref=uuid4(),
    )


QUOTE = _call("get_option_quotes", 1, ToolTier.R, agent_id="a1", agent_type="mignon-market--m")
SPAWN = _call("Agent", 0, ToolTier.D, server="builtin")
REVIEW, PLACE, CANCEL = (
    _call("review_option_order", 2),
    _call("place_option_order", 3),
    _call("cancel_option_order", 4),
)
STRAY_CANCEL = _call("cancel_option_order", 5)
DENIED = _call("place_equity_order", 6, denied=True)
STRAY_PLACE = _call("place_option_order", 7)
FACTS = DecisionFacts(
    facts_id=uuid4(),
    facts_ref="facts:1",
    run_id=RUN,
    subject_ref="candidate:1",
    purpose=FactsPurpose.OPEN,
    observed_at=T0,
    rules_version="8",
    rules_hash="sha256:r",
    input_evidence_ids=(uuid4(),),
    snapshot_ref=None,
    limit_price=Decimal("1.79"),
    initial_quantity=1,
    remaining_quantity=None,
    quality=DataQuality.MISSING,
    gaps=(Gap(field="remaining_quantity", kind=DataQuality.MISSING, detail="no fills yet"),),
)


def _attempt(place: ToolCallRecord = PLACE) -> Attempt:
    linked = place is PLACE
    return Attempt(
        index=0,
        place_tool_call_id=place.identity.tool_call_id,
        proposal_ref=None,
        requested_quantity=1,
        order_type_raw="limit",
        time_in_force_raw="gfd",
        limit_price=Decimal("1.79"),
        snapshot_ref=None,
        status=AttemptStatus.CANCELLED,
        broker_order_id="order-1",
        review_tool_call_ids=(REVIEW.identity.tool_call_id,) if linked else (),
        cancel_tool_call_ids=(CANCEL.identity.tool_call_id,) if linked else (),
        filled_quantity=0,
    )


def _record(**kw: Any) -> RunRecord:
    leg = LegRecord(
        leg_ref="decision:0:leg:open",
        side=OrderSide.SELL_TO_OPEN,
        occ_symbol=OCC,
        broker_instrument_id="inst",
        right=OptionRight.PUT,
        strike=Decimal("740"),
        expiration=date(2026, 10, 16),
        target_quantity=1,
        facts_ref="facts:1",
        quotes=(
            Quote(
                quote_id=uuid4(),
                broker_instrument_id="inst",
                bid=Decimal("1.78"),
                ask=Decimal("1.80"),
                as_of=T0,
                source_tool_call_ids=(QUOTE.identity.tool_call_id,),
            ),
        ),
        attempts=(_attempt(),),
        gaps=(Gap(field="tick", kind=DataQuality.MISSING, detail="tick unverified"),),
    )
    decision = DecisionRecord(
        decision_ref="decision:0",
        action=DecisionAction.OPEN_CSP,
        priority=0,
        target_ref="candidate:1",
        replacement_ref=None,
        underlying="SPY",
        position_id=None,
        rationale="Cash covers it.",
        thesis="Index holds its range.",
        invalidation_conditions=("A close below support.",),
        evidence_refs=(
            f"evidence:{QUOTE.identity.tool_call_id}",
            f"evidence:{uuid4()}",  # never recorded
            "evidence:not-a-uuid",
            "facts:1",
            "facts:missing",
            "candidate:1",
        ),
        legs=(leg,),
    )
    base: dict[str, Any] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "assembler_version": "1",
        "input_hash": "sha256:x",
        "run_id": RUN,
        "environment": AppEnv.PRODUCTION,
        "slot": T0,
        "terminated_at": T0,
        "requested_execution_mode": ExecutionMode.OFF,
        "effective_execution_mode": ExecutionMode.OFF,
        "order_venue": OrderVenue.SIMULATED,
        "rules_version": "8",
        "rules_hash": "sha256:r",
        "prompt_id": "wheel_agent",
        "prompt_hash": "sha256:p",
        "model_id": "claude-x",
        "decision_output_status": DecisionOutputStatus.PARSED,
        "decisions": (decision,),
        "unassociated_actions": (
            UnassociatedAction(
                kind=UnassociatedActionKind.CANCEL,
                cancellation=Cancellation(
                    cancel_tool_call_id=STRAY_CANCEL.identity.tool_call_id,
                    broker_order_id="order-9",
                    status=CancellationStatus.PENDING,
                ),
            ),
            UnassociatedAction(
                kind=UnassociatedActionKind.PLACE,
                occ_symbol=OCC,
                side_raw="sell_to_open",
                attempt=_attempt(STRAY_PLACE),
            ),
        ),
        "unresolved_questions": (
            UnresolvedQuestionRecord(target_ref=None, question="When is CPI?", evidence_refs=()),
        ),
        "summary": "1 decision(s).",
    }
    base.update(kw)
    return RunRecord.model_validate(base)


def _finding(outcome: AuditOutcome, check: AuditCheck = AuditCheck.V3, **kw: Any) -> AuditFinding:
    return AuditFinding(
        finding_id=uuid4(),
        run_id=RUN,
        check_id=check,
        sub_item=kw.pop("sub_item", "1"),
        outcome=outcome,
        effective_execution_mode=ExecutionMode.OFF,
        detail=kw.pop("detail", "checked"),
        audit_version="1",
        context_hash="h",
        **kw,
    )


def _input(**kw: Any) -> TraceInput:
    base: dict[str, Any] = {
        "run_id": RUN,
        "environment": AppEnv.PRODUCTION,
        "slot": T0,
        "status": RunStatus.COMPLETED,
        "reason": "completed",
        "effective_execution_mode": ExecutionMode.OFF,
        "order_venue": OrderVenue.SIMULATED,
        "prompt_execution_mode": "live",
        "record": _record(),
        "tool_calls": (PLACE, REVIEW, SPAWN, QUOTE, CANCEL, STRAY_CANCEL, DENIED, STRAY_PLACE),
        "facts": (FACTS,),
        "findings": (
            _finding(AuditOutcome.PASS, decision_ref="decision:0"),
            _finding(
                AuditOutcome.VIOLATION,
                decision_ref="decision:0",
                leg_ref="decision:0:leg:open",
                attempt_index=0,
                detail="limit outside the quote",
            ),
            _finding(
                AuditOutcome.UNVERIFIABLE,
                AuditCheck.V2,
                decision_ref="decision:0",
                leg_ref="decision:0:leg:open",
                detail="no attempt evidence",
            ),
            _finding(AuditOutcome.PASS, AuditCheck.V6, sub_item=None),
            _finding(
                AuditOutcome.UNVERIFIABLE,
                AuditCheck.V4,
                sub_item="3",
                detail="stray cancel",
                tool_call_ids=(STRAY_CANCEL.identity.tool_call_id,),
            ),
        ),
        "next_run": {"applied": False, "reason": "dry_run_on_demand"},
    }
    base.update(kw)
    return TraceInput(**base)


def _ids(calls: Any) -> list[UUID]:
    return [c.tool_call_id for c in calls]


def test_decision_trace_resolves_refs_facts_attempts_and_findings() -> None:
    trace = build_decision_trace(_input())
    assert trace.order_venue is OrderVenue.SIMULATED and trace.prompt_execution_mode == "live"
    assert trace.model_id == "claude-x" and trace.rules_version == "8"
    (decision,) = trace.decisions
    assert [(e.kind, e.resolved) for e in decision.evidence] == [
        ("evidence", True),
        ("evidence", False),
        ("evidence", False),
        ("facts", True),
        ("facts", False),
        ("other", True),
    ]
    assert decision.evidence[0].call is not None
    assert decision.evidence[0].call.caller == "mignon-market--m"
    assert decision.findings.passed == 1
    (leg,) = decision.legs
    assert leg.facts is not None and leg.facts.initial_quantity == 1
    assert leg.facts.gaps == ("remaining_quantity: no fills yet",)
    assert leg.findings.unverifiable == ("V2.1: no attempt evidence",)
    (attempt,) = leg.attempts
    assert attempt.place is not None and attempt.place.tool == "place_option_order"
    assert _ids(attempt.reviews) == [REVIEW.identity.tool_call_id]
    assert _ids(attempt.cancels) == [CANCEL.identity.tool_call_id]
    assert attempt.findings.violations == ("V3.1: limit outside the quote",)
    assert [u.kind for u in trace.unassociated] == ["cancel", "place"]
    cancel, stray = trace.unassociated
    assert cancel.status == "pending" and cancel.broker_order_id == "order-9"
    # A run-level finding about an unlinked action's call sits under that action.
    assert cancel.findings.unverifiable == ("V4.3: stray cancel",)
    assert stray.broker_order_id == "order-1"
    assert trace.run_findings.passed == 1 and trace.run_findings.unverifiable == ()
    assert trace.audit_counts == {"pass": 2, "unverifiable": 2, "violation": 1}
    assert trace.tool_call_counts == {"denied": 1, "succeeded": 7}
    assert trace.mignon_spawns == 1
    assert [c.tool for c in trace.timeline][:2] == ["Agent", "get_option_quotes"]
    assert trace.unresolved_questions == ("When is CPI?",)


def test_trace_is_deterministic_and_renders_every_part() -> None:
    inp = _input()
    first, second = build_decision_trace(inp), build_decision_trace(inp)
    assert first == second
    text = render_trace_markdown(first)
    for fragment in (
        "effective mode **off**, order venue **simulated**, prompt told `live`",
        "### OPEN_CSP SPY",
        "- rationale: Cash covers it.",
        "- thesis: Index holds its range.",
        "- invalidated if: A close below support.",
        "(**unresolved**)",
        "facts `facts:1` (open, quality missing): limit 1.79, quantity 1",
        "quote bid 1.78 / ask 1.80",
        "attempt 0: **cancelled** qty 1 @ 1.79, filled 0, order `order-1`",
        "- review `15:02:00` orchestrator → `robinhood.review_option_order`",
        "⚠ violation V3.1: limit outside the quote",
        "? unverifiable V2.1: no attempt evidence",
        "## Actions not linked to a decision",
        "- cancel order `order-9`: **pending**",
        "- place SPY   261016P00740000 sell_to_open order `order-1`: **cancelled**",
        "- open question: When is CPI?",
        "mignon-market--m → `robinhood.get_option_quotes`",
        "denied: not available",
        "next run: {'applied': False",
        "gap tick: tick unverified",
    ):
        assert fragment in text, fragment


def test_log_events_carry_one_compact_line_per_decision() -> None:
    (event,) = decision_log_events(build_decision_trace(_input()))
    assert event["action"] == "OPEN_CSP" and event["underlying"] == "SPY"
    assert event["attempts"] == [
        {
            "status": "cancelled",
            "requested_quantity": 1,
            "limit_price": "1.79",
            "broker_order_id": "order-1",
        }
    ]
    assert event["unresolved_refs"] == 3
    assert event["audit_violations"] == ["V3.1: limit outside the quote"]


def test_a_run_without_a_record_still_traces_its_calls() -> None:
    trace = build_decision_trace(
        _input(record=None, findings=(), next_run=None, status=None, reason=None)
    )
    assert trace.decisions == () and trace.decision_output_status is None
    assert trace.audit_counts == {}
    text = render_trace_markdown(trace)
    assert "status **unknown**" in text and "None." in text and "audit: not run" in text


def test_a_proposal_only_leg_says_no_attempt() -> None:
    record = _record(order_venue=OrderVenue.NONE, unassociated_actions=())
    leg = record.decisions[0].legs[0].model_copy(update={"attempts": (), "facts_ref": None})
    decision = record.decisions[0].model_copy(update={"legs": (leg,), "thesis": None})
    trace = build_decision_trace(
        _input(
            record=record.model_copy(update={"decisions": (decision,)}),
            order_venue=OrderVenue.NONE,
            tool_calls=(),
        )
    )
    text = render_trace_markdown(trace)
    assert "- no order attempt" in text and "No tool calls." in text
    assert trace.decisions[0].legs[0].facts is None
