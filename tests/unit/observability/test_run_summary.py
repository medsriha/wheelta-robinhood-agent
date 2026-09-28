"""Run-summary email content (ADR-0029): pure, deterministic, redacted."""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pydantic import SecretStr

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
    RunStatus,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import Attempt, ReasonCode
from wheelta_robinhood_agent.domain.run_record import (
    RUN_RECORD_SCHEMA_VERSION,
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    RunRecord,
    UnassociatedAction,
    UnassociatedActionKind,
)
from wheelta_robinhood_agent.observability.redaction import Redactor
from wheelta_robinhood_agent.observability.run_summary import (
    FACTS_HEADING,
    PROSE_UNAVAILABLE,
    RunSummaryInput,
    build_subject,
    placed_count,
    proposal_count,
    render_bodies,
    render_facts_text,
    summary_facts,
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
    empty = _input(_record())
    assert "no trades" in build_subject(empty)
    assert "Decisions: none" in render_facts_text(empty, REDACTOR)


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


def test_facts_text_lists_every_decision_leg_and_attempt() -> None:
    summary = _input(
        _record(_decision(_proposal())),
        alerts=("audit_violation",),
        audit_violations=1,
        next_run_at=datetime(2026, 9, 28, 15, 0, tzinfo=UTC),
        next_run_source="agent",
    )
    text = render_facts_text(summary, REDACTOR)
    assert "- OPEN_CSP AAPL [decision:0]" in text
    assert "sell_to_open · AAPL  261016P00190000 · strike 190 · exp 2026-10-16 · qty 2" in text
    assert "proposal (not sent): not_placed · qty 2 · limit 1.25 · codes DRY_RUN" in text
    assert "Audit: completed · 1 violation(s)" in text
    assert "Alerts: audit_violation" in text
    assert "Next run not before: 2026-09-28T15:00:00+00:00 (agent)" in text


def test_missing_record_is_stated_not_invented() -> None:
    summary = _input(None, status=RunStatus.FAILED, reason="error:RuntimeError", audit_status=None)
    text = render_facts_text(summary, REDACTOR)
    assert "Run record: not assembled" in text
    assert "Audit: not run" in text
    assert summary_facts(summary, REDACTOR)["record"] is None


def test_facts_json_keeps_numbers_verbatim_and_redacts_the_account() -> None:
    summary = _input(_record(_decision(_proposal())))
    facts = summary_facts(summary, REDACTOR)
    record = facts["record"]
    assert isinstance(record, dict)
    decisions = record["decisions"]
    assert isinstance(decisions, list) and isinstance(decisions[0], dict)
    legs = decisions[0]["legs"]
    assert isinstance(legs, list) and isinstance(legs[0], dict)
    assert legs[0]["strike"] == "190"
    attempts = legs[0]["attempts"]
    assert isinstance(attempts, list) and isinstance(attempts[0], dict)
    assert attempts[0]["limit_price"] == "1.25"
    assert attempts[0]["kind"] == "dry_run_proposal"
    run = facts["run"]
    assert isinstance(run, dict) and run["orders_sent_to_broker"] is False
    assert ACCOUNT not in repr(facts)
    assert ACCOUNT not in render_facts_text(summary, REDACTOR)


def test_bodies_carry_prose_then_facts_and_escape_html() -> None:
    text, html = render_bodies("All quiet.\n\n<b>no</b> trades.", "Run run-1", REDACTOR)
    assert text.index("All quiet.") < text.index(FACTS_HEADING) < text.index("Run run-1")
    assert "&lt;b&gt;no&lt;/b&gt;" in html and "<b>no</b>" not in html


def test_bodies_fall_back_when_prose_is_missing() -> None:
    for prose in (None, "   "):
        text, _ = render_bodies(prose, "Run run-1", REDACTOR)
        assert text.startswith(PROSE_UNAVAILABLE)


def test_prose_is_redacted() -> None:
    text, html = render_bodies(f"Account {ACCOUNT} is fine.", "x", REDACTOR)
    assert ACCOUNT not in text and ACCOUNT not in html
