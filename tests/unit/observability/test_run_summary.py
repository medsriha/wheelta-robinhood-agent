"""Run-summary email content (ADR-0029, ADR-0064): pure, deterministic, redacted, short."""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pydantic import SecretStr

from wheelta_robinhood_agent.domain.enums import (
    AgentRole,
    AppEnv,
    AttemptStatus,
    CancellationStatus,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
    OrderVenue,
    RunStatus,
)
from wheelta_robinhood_agent.domain.mignon_report import MignonReport
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import Attempt, Cancellation, ReasonCode
from wheelta_robinhood_agent.domain.run_record import (
    RUN_RECORD_SCHEMA_VERSION,
    CancellationRationaleRecord,
    DecisionOutputStatus,
    DecisionRecord,
    Gap,
    LegRecord,
    RunRecord,
    UnassociatedAction,
    UnassociatedActionKind,
    UnresolvedQuestionRecord,
)
from wheelta_robinhood_agent.observability.redaction import Redactor
from wheelta_robinhood_agent.observability.run_summary import (
    FACTS_HEADING,
    PROSE_UNAVAILABLE,
    ConsideredOption,
    RunSummaryInput,
    SlotSummaryInput,
    build_subject,
    placed_count,
    proposal_count,
    render_bodies,
    render_slot_facts_text,
    slot_summary_facts,
)

T0 = datetime(2026, 9, 28, 14, 5, tzinfo=UTC)
OCC = OccSymbol.parse("AAPL  261016P00190000")
ACCOUNT = "5RH123456789"
REDACTOR = Redactor(account_number=SecretStr(ACCOUNT))


def _proposal() -> Attempt:
    return Attempt(
        index=0,
        place_tool_call_id=None,
        proposal_ref="proposal:leg:0",
        requested_quantity=2,
        order_type_raw="limit",
        time_in_force_raw=None,
        limit_price=Decimal("1.25"),
        snapshot_ref=None,
        status=AttemptStatus.NOT_PLACED,
        broker_order_id=None,
        filled_quantity=None,
        reason_codes=(ReasonCode.DRY_RUN,),
    )


def _placed(status: AttemptStatus = AttemptStatus.FILLED) -> Attempt:
    return Attempt(
        index=0,
        place_tool_call_id=uuid4(),
        proposal_ref=None,
        requested_quantity=2,
        order_type_raw="limit",
        time_in_force_raw="gfd",
        limit_price=Decimal("1.30"),
        snapshot_ref=None,
        status=status,
        broker_order_id="b-1",
        filled_quantity=2,
    )


def _decision(attempt: Attempt, **kw: Any) -> DecisionRecord:
    leg = LegRecord(
        leg_ref="leg:0",
        side=OrderSide.SELL_TO_OPEN,
        occ_symbol=OCC,
        broker_instrument_id="inst-1",
        right=OptionRight.PUT,
        strike=Decimal("190"),
        expiration=date(2026, 10, 16),
        target_quantity=2,
        attempts=(attempt,),
    )
    base: dict[str, Any] = {
        "decision_ref": "decision:0",
        "action": DecisionAction.OPEN_CSP,
        "priority": 0,
        "target_ref": "candidate:1",
        "replacement_ref": None,
        "underlying": "AAPL",
        "position_id": None,
        "rationale": f"Cash covers it; account {ACCOUNT} has room.",
        "thesis": "Durable business.",
        "legs": (leg,),
    }
    base.update(kw)
    return DecisionRecord.model_validate(base)


def _record(*decisions: DecisionRecord, **kw: Any) -> RunRecord:
    base: dict[str, Any] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "assembler_version": "1",
        "input_hash": "sha256:x",
        "run_id": uuid4(),
        "environment": AppEnv.LOCAL,
        "slot": T0,
        "terminated_at": T0,
        "requested_execution_mode": ExecutionMode.OFF,
        "effective_execution_mode": ExecutionMode.OFF,
        "rules_version": "6",
        "rules_hash": "sha256:r",
        "prompt_id": "wheel_agent.v8",
        "prompt_hash": "sha256:p",
        "model_id": "claude-x",
        "decision_output_status": DecisionOutputStatus.PARSED,
        "decisions": decisions,
        "summary": f"{len(decisions)} decision(s).",
    }
    base.update(kw)
    return RunRecord.model_validate(base)


def _input(record: RunRecord | None, **kw: Any) -> RunSummaryInput:
    base: dict[str, Any] = {
        "run_id": "run-1",
        "environment": AppEnv.LOCAL,
        "slot": T0,
        "status": RunStatus.COMPLETED,
        "reason": "completed",
        "requested_execution_mode": ExecutionMode.OFF,
        "effective_execution_mode": ExecutionMode.OFF,
        "record": record,
        "audit_status": "completed",
    }
    base.update(kw)
    return RunSummaryInput.model_validate(base)


def _text(*agents: RunSummaryInput, **kw: Any) -> str:
    tick = SlotSummaryInput(environment=AppEnv.LOCAL, slot=T0, agents=agents, **kw)
    return render_slot_facts_text(tick, REDACTOR)


def _facts(*agents: RunSummaryInput) -> dict[str, Any]:
    tick = SlotSummaryInput(environment=AppEnv.LOCAL, slot=T0, agents=agents)
    return slot_summary_facts(tick, REDACTOR)  # type: ignore[return-value]


def test_dry_run_subject_counts_proposals() -> None:
    summary = _input(_record(_decision(_proposal())))
    assert proposal_count(summary.record) == 1
    assert build_subject(summary) == (
        "[Wheelta agent] completed · dry run · 1 proposal · 2026-09-28 14:05 UTC"
    )


def test_no_trade_run_says_so() -> None:
    hold = _decision(_proposal(), action=DecisionAction.HOLD, legs=())
    summary = _input(_record(hold))
    assert "no trades" in build_subject(summary)
    assert "- Hold AAPL\n  Why: Cash covers it" in _text(summary)
    empty = _input(_record())
    assert "no trades" in build_subject(empty)
    assert "- No decisions." in _text(empty)


def test_live_subject_counts_placed_orders_including_unassociated() -> None:
    stray = UnassociatedAction(kind=UnassociatedActionKind.PLACE, attempt=_placed())
    record = _record(
        _decision(_placed()),
        requested_execution_mode=ExecutionMode.LIVE,
        effective_execution_mode=ExecutionMode.LIVE,
        unassociated_actions=(stray,),
    )
    summary = _input(
        record,
        requested_execution_mode=ExecutionMode.LIVE,
        effective_execution_mode=ExecutionMode.LIVE,
    )
    assert placed_count(record) == 2
    assert "· live · 2 orders placed ·" in build_subject(summary)


def test_simulated_dry_run_calls_orders_simulated() -> None:
    """ADR-0038: a dry run on the simulated venue reports its simulated placements."""
    venue = {"order_venue": OrderVenue.SIMULATED}
    summary = _input(_record(_decision(_placed()), **venue), **venue)
    assert "· dry run (simulated orders) · 1 simulated order placed ·" in build_subject(summary)
    assert "  Simulated order filled: 2 at limit 1.30" in _text(summary)
    (agent,) = _facts(summary)["agents"]
    assert agent["mode"] == "dry run (simulated orders)"
    assert agent["orders_sent_to_broker"] is False
    idle = _input(_record(**venue), **venue)
    assert "no trades" in build_subject(idle)


def test_a_decision_reads_as_action_contract_outcome_and_why() -> None:
    """ADR-0064: plain words; no refs, IDs, codes, thesis, metrics or gaps."""
    decision = _decision(
        _proposal(),
        invalidation_conditions=("Thesis weakens.",),
        gaps=(Gap(field="iv_rank", kind=DataQuality.MISSING, detail="not returned"),),
    )
    text = _text(_input(_record(decision)))
    assert text == (
        "Wheel agent: completed\n"
        "- Sell a cash-secured put AAPL\n"
        "  Sell to open 2 × AAPL 2026-10-16 190 put\n"
        "  Proposed, not sent: 2 at limit 1.25\n"
        "  Why: Cash covers it; account ****6789 has room."
    )
    for noise in ("decision:0", "candidate:1", "DRY_RUN", "Durable", "Thesis", "iv_rank", "run-1"):
        assert noise not in text


def test_partial_fills_and_unknown_outcomes_are_stated() -> None:
    partial = _placed(AttemptStatus.PARTIALLY_FILLED).model_copy(update={"filled_quantity": 1})
    text = _text(_input(_record(_decision(partial))))
    assert "Order partially filled: 1 of 2 at limit 1.30" in text
    assert "Order unknown: 2 at limit 1.30" in _text(
        _input(_record(_decision(_placed(AttemptStatus.UNKNOWN))))
    )


def test_cancellations_carry_the_agents_reason() -> None:
    call = uuid4()
    cancel = Cancellation(
        cancel_tool_call_id=call,
        broker_order_id="order-9",
        status=CancellationStatus.CONFIRMED,
        confirmation_tool_call_ids=(uuid4(),),
    )
    record = _record(
        cancellations=(cancel,),
        cancellation_rationales=(
            CancellationRationaleRecord(
                cancel_call_ref="call:1", cancel_tool_call_id=call, rationale="Bid moved away."
            ),
        ),
    )
    text = _text(_input(record))
    assert "- Cancel a working order (confirmed)\n  Why: Bid moved away." in text
    assert "order-9" not in text and "No decisions" not in text


def test_unlinked_actions_need_attention_without_broker_ids() -> None:
    cancel = UnassociatedAction(
        kind=UnassociatedActionKind.CANCEL,
        cancellation=Cancellation(
            cancel_tool_call_id=uuid4(),
            broker_order_id="order-9",
            status=CancellationStatus.CONFIRMED,
            confirmation_tool_call_ids=(uuid4(),),
        ),
    )
    place = UnassociatedAction(
        kind=UnassociatedActionKind.PLACE,
        occ_symbol=OCC,
        side_raw="sell_to_open",
        attempt=_placed(),
    )
    text = _text(_input(_record(unassociated_actions=(cancel, place))))
    assert "Needs attention:" in text
    assert "- A cancel no decision accounts for: confirmed" in text
    assert (
        "- An order no decision accounts for (sell_to_open AAPL  261016P00190000): "
        "Order filled: 2 at limit 1.30"
    ) in text
    assert "order-9" not in text and "b-1" not in text


def test_a_clean_run_has_no_attention_section_and_hides_routine_alerts() -> None:
    summary = _input(_record(_decision(_placed())), alerts=("order_activity",))
    text = _text(summary)
    assert "Needs attention" not in text and "Alert" not in text
    assert "Audit" not in text


def test_failures_audit_and_alerts_are_listed_once_and_redacted() -> None:
    summary = _input(
        None,
        status=RunStatus.FAILED,
        reason="invalid_agent_output",
        diagnostic_details=(f"Parse: <bad account {ACCOUNT}>", f"Parse: <bad account {ACCOUNT}>"),
        audit_violations=1,
        audit_details=("V4: placed price differs from review",),
        alerts=("invalid_agent_output", "order_activity", "audit_violation"),
    )
    text = _text(summary)
    assert "Decisions unknown: no valid final output." in text
    assert "- The agent's final decision output was missing or invalid." in text
    assert text.count("Parse: <bad account") == 1
    assert "- The run record could not be assembled" in text
    assert "- The post-run audit found 1 violation." in text
    assert "- V4: placed price differs from review" in text
    assert "- Alerts sent: invalid agent output, audit violation" in text
    assert ACCOUNT not in text and ACCOUNT not in repr(_facts(summary))
    _, html = render_bodies(None, text, REDACTOR)
    assert "<bad account" not in html and "&lt;bad account" in html


def test_audit_failure_is_reported_once() -> None:
    summary = _input(
        _record(), status=RunStatus.FAILED, reason="audit_failed", audit_status="failed"
    )
    assert _text(summary).count("The post-run audit could not complete.") == 1
    other = _input(_record(), audit_status="failed")
    assert "- The post-run audit could not complete." in _text(other)


def test_next_runs_are_short_and_the_agents_reason_is_kept() -> None:
    summary = _input(
        _record(),
        next_run_at=datetime(2026, 9, 28, 15, 0, tzinfo=UTC),
        next_run_source="agent",
        next_run_rationale="Wait for refreshed quotes.",
    )
    text = _text(summary, next_run_at=datetime(2026, 9, 28, 15, 0, tzinfo=UTC))
    assert "Asked to run next at 2026-09-28 15:00 UTC: Wait for refreshed quotes." in text
    assert text.endswith("\n\nNext run: 2026-09-28 15:00 UTC")
    fallback = _text(
        _input(_record()),
        next_run_at=datetime(2026, 9, 28, 15, 0, tzinfo=UTC),
        next_run_source="fallback",
    )
    assert "Asked to run next" not in fallback
    assert fallback.endswith("Next run: 2026-09-28 15:00 UTC (hourly fallback)")


def test_skipped_agent_gets_one_line() -> None:
    skipped = _input(
        None,
        agent=AgentRole.CLOSE,
        status=RunStatus.SKIPPED_NO_OPEN_SHORTS,
        reason="no open short option positions",
        session_started=False,
        audit_status=None,
    )
    sell = _input(_record(), agent=AgentRole.SELL)
    assert _text(skipped, sell) == (
        "Buy-to-Close agent: did not run (no open short option positions)\n\n"
        "Sell Options agent: completed\n- No decisions."
    )


def test_writer_facts_keep_numbers_verbatim_and_carry_no_identifiers() -> None:
    summary = _input(_record(_decision(_proposal())))
    facts = _facts(summary)
    (agent,) = facts["agents"]
    (action,) = agent["actions"]
    assert action["legs"] == [
        {
            "contract": "Sell to open 2 × AAPL 2026-10-16 190 put",
            "orders": ["Proposed, not sent: 2 at limit 1.25"],
        }
    ]
    assert agent["decisions_known"] is True and agent["needs_attention"] == []
    dumped = repr(facts)
    for noise in ("run-1", "decision:0", "candidate:1", "claude-x", "sha256", ACCOUNT):
        assert noise not in dumped


def test_bodies_carry_prose_then_actions_and_escape_html() -> None:
    text, html = render_bodies("All quiet.\n\n<b>no</b> trades.", "Wheel agent", REDACTOR)
    assert text.index("All quiet.") < text.index(FACTS_HEADING) < text.index("Wheel agent")
    assert "&lt;b&gt;no&lt;/b&gt;" in html and "<b>no</b>" not in html
    assert "<pre" not in html


def test_bodies_fall_back_when_prose_is_missing() -> None:
    for prose in (None, "   "):
        text, _ = render_bodies(prose, "Wheel agent", REDACTOR)
        assert text.startswith(PROSE_UNAVAILABLE)


def test_prose_is_redacted() -> None:
    text, html = render_bodies(f"Account {ACCOUNT} is fine.", "x", REDACTOR)
    assert ACCOUNT not in text and ACCOUNT not in html


MSFT = "MSFT  261016P00400000"


def _research(*findings: dict[str, Any]) -> MignonReport:
    return MignonReport(task="Compare puts", findings=findings, gaps=(), follow_up_questions=())


def test_hold_names_the_held_contract() -> None:
    hold = _decision(_proposal(), action=DecisionAction.HOLD, legs=(), target_occ_symbol=OCC)
    assert "- Hold AAPL 2026-10-16 190 put\n  Why:" in _text(_input(_record(hold)))


def test_candidates_not_selected_show_research_notes_once_labelled() -> None:
    summary = _input(
        _record(_decision(_proposal())),
        candidates=(
            ConsideredOption(candidate_ref="candidate:1", underlying="AAPL", occ_symbol=str(OCC)),
            ConsideredOption(candidate_ref="candidate:2", underlying="MSFT", occ_symbol=MSFT),
            ConsideredOption(candidate_ref="candidate:3", underlying="MSFT", occ_symbol=MSFT),
            ConsideredOption(
                candidate_ref="candidate:4",
                underlying="NVDA",
                occ_symbol="NVDA  261016P00100000",
            ),
        ),
        research_reports=(
            _research(
                {"claim": "AAPL looks fine.", "refs": ("candidate:1",), "web_urls": ()},
                {"claim": "Earnings are close.", "refs": ("candidate:2",), "web_urls": ()},
            ),
            _research(
                {"claim": "Earnings are close.", "refs": ("candidate:3",), "web_urls": ()},
            ),
        ),
    )
    text = _text(summary)
    assert (
        "Not selected (research notes, not the agent's stated reasons):\n"
        "- MSFT 2026-10-16 400 put: Earnings are close."
    ) in text
    assert text.count("Earnings are close.") == 1
    assert "AAPL looks fine." not in text  # selected: its decision already explains it
    assert "NVDA" not in text  # no research note, nothing to say
    assert "candidate:" not in text
    (agent,) = _facts(summary)["agents"]
    assert agent["passed_over_research_notes"] == [
        {"contract": "MSFT 2026-10-16 400 put", "notes": "Earnings are close."}
    ]


def test_no_passed_over_list_when_decisions_are_unknown() -> None:
    summary = _input(
        None,
        status=RunStatus.FAILED,
        reason="invalid_agent_output",
        candidates=(
            ConsideredOption(candidate_ref="candidate:2", underlying="MSFT", occ_symbol=MSFT),
        ),
        research_reports=(
            _research({"claim": "Earnings are close.", "refs": ("candidate:2",), "web_urls": ()}),
        ),
    )
    assert "Not selected" not in _text(summary)


def test_open_questions_are_listed_once() -> None:
    question = UnresolvedQuestionRecord(target_ref=None, question="Recheck after earnings?")
    text = _text(_input(_record(unresolved_questions=(question, question))))
    assert "Open questions:\n- Recheck after earnings?" in text
    assert text.count("Recheck after earnings?") == 1


def test_research_load_failure_needs_attention() -> None:
    text = _text(_input(_record(), research_unavailable="RuntimeError"))
    assert "- Research notes could not be loaded (RuntimeError)." in text
