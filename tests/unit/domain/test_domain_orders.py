from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import AttemptStatus, CancellationStatus, OrderSide
from wheelta_robinhood_agent.domain.orders import (
    Attempt,
    BrokerOrder,
    Cancellation,
    FillObservationKind,
    FillRecord,
    OrderIntent,
    OrderRecord,
    ReasonCode,
    StatusObservation,
    WorkspaceObject,
    WorkspaceObjectKind,
    WorkspaceObjectState,
    filled_quantity,
    is_owned_name,
)

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
OID = UUID(int=11)


def _intent(**kw: object) -> OrderIntent:
    base: dict[str, object] = {
        "intent_id": uuid4(),
        "run_id": uuid4(),
        "place_tool_call_id": uuid4(),
        "account_scope_id": "acct-scope",
        "occ_symbol": None,
        "broker_instrument_id": "inst-1",
        "side_raw": "sell_to_open",
        "quantity": 1,
        "order_type_raw": "limit",
        "time_in_force_raw": "gfd",
        "limit_price": Decimal("1.25"),
        "requested_at": T0,
    }
    base.update(kw)
    return OrderIntent.model_validate(base)


def test_intent_preserves_raw_side() -> None:
    assert _intent().side is OrderSide.SELL_TO_OPEN
    assert _intent(side_raw="buy_to_open").side is None
    assert _intent(side_raw=None).side is None
    assert _intent(side_raw="buy_to_open").side_raw == "buy_to_open"
    with pytest.raises(ValidationError):
        _intent(quantity=1.0)
    with pytest.raises(ValidationError):
        _intent(limit_price=1.25)


def _fill(kind: FillObservationKind, qty: int, **kw: object) -> FillRecord:
    base: dict[str, object] = {
        "fill_id": uuid4(),
        "order_id": OID,
        "kind": kind,
        "broker_execution_id": "x1" if kind is FillObservationKind.EXECUTION else None,
        "quantity": qty,
        "price": Decimal("1.20"),
        "executed_at": None,
        "observed_at": T0,
        "source_tool_call_id": uuid4(),
    }
    base.update(kw)
    return FillRecord.model_validate(base)


EX = FillObservationKind.EXECUTION
CU = FillObservationKind.CUMULATIVE


def test_fill_validation() -> None:
    with pytest.raises(ValidationError, match="execution id"):
        _fill(EX, 1, broker_execution_id=None)
    with pytest.raises(ValidationError, match="positive"):
        _fill(EX, 0)
    with pytest.raises(ValidationError, match="no broker execution id"):
        _fill(CU, 1, broker_execution_id="x")


def test_filled_quantity_executions_deduplicated() -> None:
    a = _fill(EX, 2, broker_execution_id="a")
    a_again = _fill(EX, 2, broker_execution_id="a")
    b = _fill(EX, 1, broker_execution_id="b")
    assert filled_quantity([a, a_again, b]) == 3
    assert filled_quantity([]) == 0
    assert filled_quantity([a, _fill(EX, 5, broker_execution_id="a")]) is None


def test_filled_quantity_cumulative_never_summed() -> None:
    c1 = _fill(CU, 1, observed_at=T0)
    c2 = _fill(CU, 3, observed_at=T0 + timedelta(minutes=1))
    c2_again = _fill(CU, 3, observed_at=T0 + timedelta(minutes=2))
    assert filled_quantity([c2_again, c1, c2]) == 3
    decreasing = [_fill(CU, 3, observed_at=T0), _fill(CU, 1, observed_at=T0 + timedelta(1))]
    assert filled_quantity(decreasing) is None


def test_filled_quantity_mixed_and_foreign() -> None:
    ex = _fill(EX, 2)
    assert filled_quantity([ex, _fill(CU, 2)]) == 2
    assert filled_quantity([ex, _fill(CU, 3)]) is None
    assert filled_quantity([ex, _fill(EX, 1, order_id=uuid4(), broker_execution_id="z")]) is None


def _attempt(**kw: object) -> Attempt:
    base: dict[str, object] = {
        "index": 0,
        "place_tool_call_id": None,
        "proposal_ref": "proposal:0:0",
        "requested_quantity": 1,
        "order_type_raw": "limit",
        "time_in_force_raw": None,
        "limit_price": Decimal("1.25"),
        "snapshot_ref": uuid4(),
        "status": AttemptStatus.NOT_PLACED,
        "broker_order_id": None,
        "filled_quantity": None,
        "reason_codes": (ReasonCode.DRY_RUN,),
    }
    base.update(kw)
    return Attempt.model_validate(base)


def test_dry_run_attempt() -> None:
    assert _attempt().reason_codes == (ReasonCode.DRY_RUN,)
    with pytest.raises(ValidationError, match="exactly one"):
        _attempt(place_tool_call_id=uuid4())
    with pytest.raises(ValidationError, match="exactly one"):
        _attempt(proposal_ref=None)
    with pytest.raises(ValidationError, match="not_placed"):
        _attempt(status=AttemptStatus.PLACED)
    with pytest.raises(ValidationError, match="broker-action"):
        _attempt(review_tool_call_ids=(uuid4(),))
    with pytest.raises(ValidationError, match="positive quantity"):
        _attempt(requested_quantity=0)
    with pytest.raises(ValidationError, match="positive quantity"):
        _attempt(requested_quantity=None)
    with pytest.raises(ValidationError, match="duplicate"):
        _attempt(reason_codes=(ReasonCode.DRY_RUN, ReasonCode.DRY_RUN))


def test_live_attempts() -> None:
    live: dict[str, object] = {"place_tool_call_id": uuid4(), "proposal_ref": None}
    a = _attempt(
        **live,
        status=AttemptStatus.FILLED,
        broker_order_id="b1",
        fill_ids=(uuid4(),),
        filled_quantity=1,
    )
    assert a.status is AttemptStatus.FILLED
    # A denied place call: not placed, no broker id; actual quantity kept even if 0.
    _attempt(**live, requested_quantity=0, reason_codes=(ReasonCode.NOT_DISPATCHED,))
    with pytest.raises(ValidationError, match="no broker order id"):
        _attempt(**live, broker_order_id="b1")
    with pytest.raises(ValidationError, match="no fills"):
        _attempt(**live, filled_quantity=1)


def test_cancellation() -> None:
    Cancellation(
        cancel_tool_call_id=uuid4(), broker_order_id=None, status=CancellationStatus.PENDING
    )
    Cancellation(
        cancel_tool_call_id=uuid4(),
        broker_order_id="b1",
        status=CancellationStatus.CONFIRMED,
        confirmation_tool_call_ids=(uuid4(),),
    )
    with pytest.raises(ValidationError, match="confirming"):
        Cancellation(
            cancel_tool_call_id=uuid4(), broker_order_id="b1", status=CancellationStatus.CONFIRMED
        )


def _broker(**kw: object) -> BrokerOrder:
    base: dict[str, object] = {
        "order_id": OID,
        "account_scope_id": "acct-scope",
        "broker_order_id": "b1",
        "intent_id": None,
        "first_observed_at": T0,
    }
    base.update(kw)
    return BrokerOrder.model_validate(base)


def test_order_record() -> None:
    rec = OrderRecord(intent=_intent(), broker_order=None)
    assert rec.status is AttemptStatus.UNKNOWN
    assert rec.filled_quantity == 0
    obs = (
        StatusObservation(
            status=AttemptStatus.FILLED,
            broker_status_raw="filled",
            observed_at=T0 + timedelta(minutes=1),
            tool_call_id=uuid4(),
        ),
        StatusObservation(
            status=AttemptStatus.PLACED,
            broker_status_raw=None,
            observed_at=T0,
            tool_call_id=uuid4(),
        ),
    )
    rec = OrderRecord(
        intent=None, broker_order=_broker(), status_history=obs, fills=(_fill(EX, 1),)
    )
    assert rec.status is AttemptStatus.FILLED
    assert rec.filled_quantity == 1
    with pytest.raises(ValidationError, match="intent or a broker"):
        OrderRecord(intent=None, broker_order=None)
    with pytest.raises(ValidationError, match="different intent"):
        OrderRecord(intent=_intent(), broker_order=_broker(intent_id=uuid4()))
    with pytest.raises(ValidationError, match="broker order identity"):
        OrderRecord(intent=_intent(), broker_order=None, fills=(_fill(EX, 1),))
    with pytest.raises(ValidationError, match="belong"):
        OrderRecord(intent=None, broker_order=_broker(), fills=(_fill(EX, 1, order_id=uuid4()),))


def test_workspace() -> None:
    ident = WorkspaceObject(
        object_id=uuid4(),
        account_scope_id="acct-scope",
        kind=WorkspaceObjectKind.WATCHLIST,
        broker_id="wl-1",
        created_run_id=uuid4(),
        created_at=T0,
    )
    state = WorkspaceObjectState(
        identity=ident,
        current_name="agent-csp",
        ledger_recorded=True,
        last_modified_at=T0,
        deleted=False,
    )
    assert state.owned("agent-")
    assert not state.owned("other-")
    assert not state.model_copy(update={"ledger_recorded": False}).owned("agent-")
    assert not state.model_copy(update={"deleted": True}).owned("agent-")
    assert not state.model_copy(update={"current_name": None}).owned("agent-")
    assert not is_owned_name("anything", "")
