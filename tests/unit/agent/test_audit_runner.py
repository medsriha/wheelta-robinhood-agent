"""run_audit, context validation/hash, repeat-unverifiable alerts, and cross-check scenarios."""

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_audit_builders import (
    PUT,
    RUN_ID,
    STO,
    T0,
    LegSpec,
    Scenario,
    live_csp,
    off_csp,
    outcomes,
    reasons,
    t,
)

from wheelta_robinhood_agent.agent.audit import (
    CHECKS,
    AuditContext,
    DayHistory,
    WorkingOrderObservation,
    repeat_unverifiable_alerts,
    run_audit,
    unverifiable_checks,
)
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditCheck,
    AuditOutcome,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OrderSide,
    StrategyKind,
)
from wheelta_robinhood_agent.domain.positions import (
    PositionBook,
    PositionBookEntry,
    PositionInstrument,
)
from wheelta_robinhood_agent.domain.run import AuditStatus
from wheelta_robinhood_agent.domain.run_record import DecisionOutputStatus

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE


def test_run_audit_live_clean_run() -> None:
    s = Scenario()
    live_csp(s)
    result = run_audit(s.ctx())
    assert result.status is AuditStatus.COMPLETED
    assert result.violations == ()
    assert {f.check_id for f in result.findings} == set(AuditCheck)
    assert all(f.context_hash == result.context_hash for f in result.findings)
    assert len({f.finding_id for f in result.findings}) == len(result.findings)
    assert result.unverifiable_checks == ()


def test_run_audit_is_deterministic() -> None:
    s = Scenario()
    live_csp(s)
    ctx = s.ctx()
    a, b = run_audit(ctx), run_audit(AuditContext.model_validate_json(ctx.model_dump_json()))
    assert a.context_hash == b.context_hash
    assert [f.finding_id for f in a.findings] == [f.finding_id for f in b.findings]


def test_exception_in_one_check_does_not_suppress_others() -> None:
    def boom(_: AuditContext) -> tuple[()]:
        raise RuntimeError("bad data")

    s = Scenario()
    live_csp(s)
    checks = [(AuditCheck.V1, boom), *CHECKS[1:]]
    result = run_audit(s.ctx(), checks)
    assert result.status is AuditStatus.FAILED
    assert [(e.check_id, e.error_type) for e in result.errors] == [(AuditCheck.V1, "RuntimeError")]
    assert AuditCheck.V1 not in {f.check_id for f in result.findings}
    assert AuditCheck.V7 in {f.check_id for f in result.findings}


def test_dry_run_audit_marks_execution_checks_unverifiable() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    result = run_audit(s.ctx())
    assert result.violations == ()
    assert set(result.unverifiable_checks) == {AuditCheck.V3, AuditCheck.V6}
    assert all(
        f.detail.startswith("dry_run_no_execution") for f in result.findings if f.outcome is U
    )


def test_missing_final_output_live_keeps_event_checks() -> None:
    s = Scenario()
    live_csp(s)
    s.output_status = DecisionOutputStatus.MISSING
    result = run_audit(s.ctx())
    by_check = {c: [f for f in result.findings if f.check_id is c] for c in AuditCheck}
    assert outcomes(tuple(by_check[AuditCheck.V1]), "1") == [P]
    assert outcomes(tuple(by_check[AuditCheck.V6]), "3") == [P]
    assert reasons(tuple(by_check[AuditCheck.V4]), "6") == ["missing_final_output"]
    assert reasons(tuple(by_check[AuditCheck.V4]), "3") == ["unassociated_action"]


def test_older_entry_history_outside_recent_decisions() -> None:
    """A lineage opened in an earlier run is covered via the PositionBook, not recent runs."""
    s = Scenario()
    entry = PositionBookEntry(
        position_id=uuid4(),
        position_ref="position:old",
        underlying="AAPL",
        strategy=StrategyKind.CASH_SECURED_PUT,
        current_instruments=(
            PositionInstrument(occ_symbol=PUT, broker_instrument_id="put-1", short_quantity=1),
        ),
        entry_fill_ids=(uuid4(),),
        entry_date=date(2026, 6, 1),
        entry_weighted_credit=Decimal("1.20"),
        thesis="durable franchise",
        roll_count=0,
        history_quality=DataQuality.OK,
    )
    s.book = PositionBook(as_of=T0, entries=(entry,))
    result = run_audit(s.ctx())
    assert outcomes(tuple(f for f in result.findings if f.check_id is AuditCheck.V4), "1") == [V]
    s.decide(DecisionAction.HOLD, target_ref="position:old")
    result = run_audit(s.ctx())
    assert outcomes(tuple(f for f in result.findings if f.check_id is AuditCheck.V4), "1") == [P]


def test_repeat_unverifiable_alerts() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    dry = run_audit(s.ctx()).findings
    live = Scenario()
    live_csp(live)
    clean = run_audit(live.ctx()).findings
    assert repeat_unverifiable_alerts([dry, dry], dry, 3) == (AuditCheck.V3, AuditCheck.V6)
    assert repeat_unverifiable_alerts([dry, clean], dry, 3) == ()
    assert repeat_unverifiable_alerts([dry], dry, 3) == ()
    assert repeat_unverifiable_alerts([], dry, 1) == (AuditCheck.V3, AuditCheck.V6)
    assert unverifiable_checks(clean) == ()
    for bad in (0, True, -1):
        with pytest.raises(ValueError):
            repeat_unverifiable_alerts([], dry, bad)


def test_context_validation() -> None:
    s = Scenario()
    live_csp(s)
    ctx = s.ctx()
    data = ctx.model_dump(mode="json")
    with pytest.raises(ValidationError, match="another run"):
        AuditContext.model_validate({**data, "run_id": str(uuid4()), "tool_calls": ()})
    with pytest.raises(ValidationError, match="audited run"):
        AuditContext.model_validate({**data, "run_id": str(uuid4()), "run_record": None})
    with pytest.raises(ValidationError, match="execution mode"):
        AuditContext.model_validate({**data, "effective_execution_mode": "off"})
    with pytest.raises(ValidationError, match="duplicate"):
        AuditContext.model_validate({**data, "quotes": data["quotes"] * 2})


def test_evidence_model_validation() -> None:
    with pytest.raises(ValidationError, match="filled_quantity"):
        WorkingOrderObservation(
            broker_order_id="o",
            broker_instrument_id="i",
            side_raw=STO,
            quantity=1,
            filled_quantity=2,
            owned=True,
        )
    lineage = uuid4()
    with pytest.raises(ValidationError, match="duplicate"):
        DayHistory(trading_date=T0.date(), prior_new_lineage_ids=(lineage, lineage))
    s = Scenario()
    state = s.state(90)
    with pytest.raises(ValidationError, match="no earlier"):
        state.model_validate({**state.model_dump(), "completed_at": t(-500)})


def test_run_id_in_findings() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.state(90)
    p = s.place(100, "put-1", PUT, STO, 1, "1.10")
    s.status(p, 110, AttemptStatus.FILLED, filled=1)
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [p]))
    result = run_audit(s.ctx())
    assert {f.run_id for f in result.findings} == {RUN_ID}
    assert all(f.audit_version == "audit-v1" for f in result.findings)
