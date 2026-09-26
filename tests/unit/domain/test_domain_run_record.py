from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    BoardComparison,
    CancellationStatus,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import Attempt, Cancellation, ReasonCode
from wheelta_robinhood_agent.domain.run_record import (
    RUN_RECORD_SCHEMA_VERSION,
    AssemblyFinding,
    BoardComparisonRecord,
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    Quote,
    RunRecord,
    UnassociatedAction,
    UnassociatedActionKind,
)

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
OCC = OccSymbol.parse("AAPL  261016P00190000")


def _quote(**kw: object) -> Quote:
    base: dict[str, object] = {
        "quote_id": uuid4(),
        "broker_instrument_id": "inst-1",
        "bid": Decimal("1.20"),
        "ask": Decimal("1.30"),
        "mark": Decimal("1.25"),
        "delta": Decimal("-0.25"),
        "as_of": T0,
        "source_tool_call_ids": (uuid4(),),
    }
    base.update(kw)
    return Quote.model_validate(base)


def test_quote_sanity() -> None:
    _quote()
    _quote(bid=Decimal("0"), mark=None)
    with pytest.raises(ValidationError, match="bid"):
        _quote(bid=Decimal("1.40"))
    with pytest.raises(ValidationError, match="mark"):
        _quote(mark=Decimal("1.50"))
    with pytest.raises(ValidationError, match="delta"):
        _quote(delta=Decimal("1.2"))
    with pytest.raises(ValidationError, match="source"):
        _quote(source_tool_call_ids=())
    with pytest.raises(ValidationError):
        _quote(ask=1.3)


def test_board_comparison() -> None:
    BoardComparisonRecord(status=BoardComparison.NOT_APPLICABLE)
    BoardComparisonRecord(
        status=BoardComparison.COMPARED,
        board_row_ref="board:AAPL",
        build_id="b-1",
        tool_call_ids=(uuid4(),),
        premium_measure="bid",
        board_premium=Decimal("1.10"),
        live_premium=Decimal("1.20"),
    )
    with pytest.raises(ValidationError, match="provenance"):
        BoardComparisonRecord(status=BoardComparison.COMPARED, build_id="b-1")


def _attempt(index: int = 0, **kw: object) -> Attempt:
    base: dict[str, object] = {
        "index": index,
        "place_tool_call_id": None,
        "proposal_ref": f"proposal:{uuid4()}",
        "requested_quantity": 1,
        "order_type_raw": "limit",
        "time_in_force_raw": None,
        "limit_price": Decimal("1.25"),
        "snapshot_ref": None,
        "status": AttemptStatus.NOT_PLACED,
        "broker_order_id": None,
        "filled_quantity": None,
        "reason_codes": (ReasonCode.DRY_RUN,),
    }
    base.update(kw)
    return Attempt.model_validate(base)


def _placed(index: int = 0) -> Attempt:
    return _attempt(
        index,
        place_tool_call_id=uuid4(),
        proposal_ref=None,
        status=AttemptStatus.PLACED,
        broker_order_id="b1",
        reason_codes=(),
    )


def _leg(**kw: object) -> LegRecord:
    base: dict[str, object] = {
        "leg_ref": "leg:0:0",
        "side": OrderSide.SELL_TO_OPEN,
        "occ_symbol": OCC,
        "broker_instrument_id": "inst-1",
        "right": OptionRight.PUT,
        "strike": Decimal("190"),
        "expiration": date(2026, 10, 16),
        "target_quantity": 1,
        "quotes": (_quote(),),
        "attempts": (_attempt(),),
    }
    base.update(kw)
    return LegRecord.model_validate(base)


def test_leg() -> None:
    _leg()
    _leg(attempts=(_placed(0), _placed(1)))
    with pytest.raises(ValidationError, match="OCC"):
        _leg(strike=Decimal("195"))
    with pytest.raises(ValidationError, match="OCC"):
        _leg(right=OptionRight.CALL)
    with pytest.raises(ValidationError, match="OCC"):
        _leg(expiration=date(2026, 10, 23))
    with pytest.raises(ValidationError, match="indices"):
        _leg(attempts=(_placed(1),))
    with pytest.raises(ValidationError, match="one unsubmitted"):
        _leg(attempts=(_attempt(0), _attempt(1)))
    with pytest.raises(ValidationError, match="itself"):
        _leg(depends_on_leg_ref="leg:0:0")
    with pytest.raises(ValidationError, match="target_quantity"):
        _leg(target_quantity=None, attempts=())
    gap = Gap(field="target_quantity", kind=DataQuality.MISSING, detail="multiplier unverified")
    leg = _leg(
        target_quantity=None,
        attempts=(),
        gaps=(gap,),
        reason_codes=(ReasonCode.QUANTITY_UNAVAILABLE,),
    )
    assert leg.target_quantity is None


def _decision(ref: str = "decision:0", **kw: object) -> DecisionRecord:
    base: dict[str, object] = {
        "decision_ref": ref,
        "action": DecisionAction.OPEN_CSP,
        "priority": 0,
        "target_ref": "candidate:1",
        "replacement_ref": None,
        "underlying": "AAPL",
        "position_id": None,
        "rationale": "Supported by evidence.",
        "thesis": "Durable business.",
        "legs": (_leg(leg_ref=f"leg:{ref}"),),
    }
    base.update(kw)
    return DecisionRecord.model_validate(base)


def test_decision_record() -> None:
    with pytest.raises(ValidationError, match="itself"):
        _decision(depends_on_decision_refs=("decision:0",))
    with pytest.raises(ValidationError, match="duplicate"):
        _decision(depends_on_decision_refs=("d:1", "d:1"))


def _cancel(**kw: object) -> Cancellation:
    base: dict[str, object] = {
        "cancel_tool_call_id": uuid4(),
        "broker_order_id": "b1",
        "status": CancellationStatus.UNKNOWN,
    }
    base.update(kw)
    return Cancellation.model_validate(base)


def test_unassociated_action() -> None:
    UnassociatedAction(kind=UnassociatedActionKind.PLACE, attempt=_placed())
    UnassociatedAction(kind=UnassociatedActionKind.CANCEL, cancellation=_cancel())
    with pytest.raises(ValidationError, match="exactly an attempt"):
        UnassociatedAction(kind=UnassociatedActionKind.PLACE)
    with pytest.raises(ValidationError, match="recorded place call"):
        UnassociatedAction(kind=UnassociatedActionKind.PLACE, attempt=_attempt())
    with pytest.raises(ValidationError, match="exactly a cancellation"):
        UnassociatedAction(
            kind=UnassociatedActionKind.CANCEL, cancellation=_cancel(), attempt=_placed()
        )


def _record(**kw: object) -> RunRecord:
    base: dict[str, object] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "assembler_version": "1",
        "input_hash": "sha256:x",
        "run_id": uuid4(),
        "environment": AppEnv.STAGING,
        "slot": T0,
        "terminated_at": T0,
        "requested_execution_mode": ExecutionMode.OFF,
        "effective_execution_mode": ExecutionMode.OFF,
        "rules_version": "5",
        "rules_hash": "sha256:r",
        "prompt_id": "wheel_agent.v5",
        "prompt_hash": "sha256:p",
        "model_id": "claude-x",
        "decision_output_status": DecisionOutputStatus.PARSED,
        "decisions": (_decision(),),
        "findings": (AssemblyFinding(code="unresolved_reference", detail="x"),),
        "summary": "1 decision.",
    }
    base.update(kw)
    return RunRecord.model_validate(base)


def test_run_record_valid() -> None:
    r = _record()
    assert r.schema_version == 4
    with pytest.raises(ValidationError):
        _record(schema_version=3)


def test_no_synthesized_decisions_without_output() -> None:
    placed = UnassociatedAction(kind=UnassociatedActionKind.PLACE, attempt=_placed())
    r = _record(
        decision_output_status=DecisionOutputStatus.INVALID,
        decisions=(),
        unassociated_actions=(placed,),
    )
    assert r.unassociated_actions == (placed,)
    with pytest.raises(ValidationError, match="synthesized"):
        _record(decision_output_status=DecisionOutputStatus.MISSING)


def test_mode_consistency() -> None:
    with pytest.raises(ValidationError, match="requested live"):
        _record(effective_execution_mode=ExecutionMode.LIVE)


def test_refs_unique_and_dependencies_resolve() -> None:
    with pytest.raises(ValidationError, match="decision ref"):
        _record(decisions=(_decision(), _decision()))
    d1 = _decision("decision:1")
    d1_dup_leg = _decision("decision:2", legs=(_leg(leg_ref="leg:decision:1"),))
    with pytest.raises(ValidationError, match="leg ref"):
        _record(decisions=(d1, d1_dup_leg))
    with pytest.raises(ValidationError, match="decision dependency"):
        _record(decisions=(_decision(depends_on_decision_refs=("decision:9",)),))
    _record(decisions=(d1, _decision("decision:3", depends_on_decision_refs=("decision:1",))))
    bad_leg = _decision(legs=(_leg(leg_ref="leg:a", depends_on_leg_ref="leg:zz"),))
    with pytest.raises(ValidationError, match="leg dependency"):
        _record(decisions=(bad_leg,))


def test_each_action_recorded_once() -> None:
    placed = _placed()
    leg = _leg(attempts=(placed,))
    dup = UnassociatedAction(kind=UnassociatedActionKind.PLACE, attempt=placed)
    with pytest.raises(ValidationError, match="place call"):
        _record(decisions=(_decision(legs=(leg,)),), unassociated_actions=(dup,))
    cancel = _cancel()
    dup_cancel = UnassociatedAction(kind=UnassociatedActionKind.CANCEL, cancellation=cancel)
    with pytest.raises(ValidationError, match="cancel call"):
        _record(cancellations=(cancel,), unassociated_actions=(dup_cancel,))
    prop = _attempt(proposal_ref="proposal:same")
    d_a = _decision("decision:a", legs=(_leg(leg_ref="leg:a", attempts=(prop,)),))
    d_b = _decision("decision:b", legs=(_leg(leg_ref="leg:b", attempts=(prop,)),))
    with pytest.raises(ValidationError, match="proposal ref"):
        _record(decisions=(d_a, d_b))
