"""Order, review, cancel, and non-empty positions mappers (ADR-0034).

Payloads follow the output schemas Robinhood publishes in `tools/list`
(tests/fixtures/robinhood/output_schemas_orders_2026-09-28.json); a test checks that every
field the mappers read is declared there.
"""

import copy
import json
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, ValidationRequest
from wheelta_robinhood_agent.agent.result_boundary import (
    BoundaryValidator,
    MappingRequest,
    _check_provenance,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.robinhood_mappers import (
    map_equity_positions,
    map_option_orders,
    map_option_positions,
    map_order_cancel,
    map_order_placement,
    map_order_review,
)
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    OptionRight,
    OrderSide,
    PositionsCoverage,
    ToolTier,
)
from wheelta_robinhood_agent.observability.redaction import Redactor

SCHEMAS = (
    Path(__file__).parents[2] / "fixtures" / "robinhood" / "output_schemas_orders_2026-09-28.json"
)
CALL = uuid.UUID("0190a0a0-0000-7000-8000-000000000002")
RETRIEVED = datetime(2026, 9, 29, 14, 0, 5, tzinfo=UTC)
ORDER_ID = "4b3c0f4e-1d2a-4e5f-9a8b-7c6d5e4f3a2b"
OPTION_ID = "d17decae-92f6-430e-b4c0-3772e5dd27ab"

ORDER: dict[str, Any] = {
    "id": ORDER_ID,
    "chain_id": "c1c2c3c4-0000-4000-8000-000000000001",
    "chain_symbol": "SPY",
    "state": "confirmed",
    "type": "limit",
    "trigger": "immediate",
    "direction": "credit",
    "quantity": "2.00000",
    "processed_quantity": "0.00000",
    "pending_quantity": "2.00000",
    "canceled_quantity": "0.00000",
    "price": "1.25000000",
    "stop_price": None,
    "premium": "250.00000000",
    "processed_premium": "0",
    "trade_value_multiplier": "100.0000",
    "time_in_force": "gfd",
    "market_hours": "regular_hours",
    "opening_strategy": "short_put",
    "closing_strategy": None,
    "placed_agent": "agentic",
    "created_at": "2026-09-29T14:00:01.123456789Z",
    "updated_at": "2026-09-29T14:00:02.5Z",
    "last_transaction_at": None,
    "is_replaceable": True,
    "legs": [
        {
            "id": "1e1e1e1e-0000-4000-8000-000000000001",
            "option_id": OPTION_ID,
            "side": "sell",
            "position_effect": "open",
            "ratio_quantity": 1,
            "expiration_date": "2026-10-16",
            "strike_price": "740.0000",
            "option_type": "put",
            "executions": [],
        }
    ],
}


def _ids() -> Callable[[], uuid.UUID]:
    counter: Iterator[int] = iter(range(1, 1_000))
    return lambda: uuid.UUID(int=next(counter))


def _request(tool: str, data: Any, **effective_input: Any) -> MappingRequest:
    return MappingRequest(
        tool_call_id=CALL,
        server="robinhood",
        tool=tool,
        effective_input=effective_input,
        payload={"data": data, "guide": "prose"},
        retrieved_at=RETRIEVED,
    )


def _order(**changes: Any) -> dict[str, Any]:
    order = copy.deepcopy(ORDER)
    order.update(changes)
    return order


def _schema_fields(tool: str, *path: str) -> set[str]:
    node: Any = json.loads(SCHEMAS.read_text())["tools"][tool]["output_schema"]
    for key in path:
        node = node["properties"][key]
        if "items" in node:
            node = node["items"]
    return set(node["properties"])


def test_fixture_order_uses_only_schema_fields() -> None:
    assert set(ORDER) <= _schema_fields("get_option_orders", "data", "orders")
    assert set(ORDER["legs"][0]) <= _schema_fields("get_option_orders", "data", "orders", "legs")
    assert _schema_fields("place_option_order", "data", "order") == _schema_fields(
        "get_option_orders", "data", "orders"
    )


# ----------------------------------------------------------------------------- order reads


def test_working_order_maps_to_observation_and_open_orders_read() -> None:
    out = map_option_orders(_request("get_option_orders", {"orders": [ORDER]}), _ids())
    _check_provenance(out, CALL)
    (obs,) = out.broker_orders
    assert obs.broker_order_id == ORDER_ID and obs.status is AttemptStatus.PLACED
    assert obs.quantity == 2 and obs.pending_quantity == 2 and obs.limit_price == Decimal("1.25")
    assert obs.multiplier == 100 and obs.placed_agent == "agentic"
    assert obs.as_of == datetime(2026, 9, 29, 14, 0, 2, 500000, tzinfo=UTC)
    (leg,) = obs.legs
    assert leg.side_raw == "sell_to_open" and leg.broker_instrument_id == OPTION_ID
    assert leg.occ_symbol is not None and leg.occ_symbol.right is OptionRight.PUT
    (read,) = out.open_orders
    (working,) = read.orders
    assert working.side is OrderSide.SELL_TO_OPEN and working.unfilled_quantity == 2
    assert working.broker_order_ref == ORDER_ID


def test_filled_order_is_not_working_and_carries_executions() -> None:
    execution = {
        "id": "e0e0e0e0-0000-4000-8000-000000000001",
        "price": "1.25000000",
        "quantity": "2.00000",
        "settlement_date": "2026-09-30",
        "trade_date": "2026-09-29",
        "timestamp": "2026-09-29T14:00:03Z",
    }
    leg = {**ORDER["legs"][0], "executions": [execution]}
    order = _order(state="filled", processed_quantity="2", pending_quantity="0", legs=[leg])
    out = map_option_orders(_request("get_option_orders", {"orders": [order]}), _ids())
    (obs,) = out.broker_orders
    assert obs.status is AttemptStatus.FILLED
    (ex,) = obs.executions
    assert ex.quantity == 2 and ex.price == Decimal("1.25")
    assert out.open_orders[0].orders == ()


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("unconfirmed", AttemptStatus.PLACED),
        ("queued", AttemptStatus.PLACED),
        ("pending_cancelled", AttemptStatus.PLACED),
        ("partially_filled", AttemptStatus.PARTIALLY_FILLED),
        ("cancelled", AttemptStatus.CANCELLED),
        ("voided", AttemptStatus.CANCELLED),
        ("rejected", AttemptStatus.REJECTED),
        ("failed", AttemptStatus.REJECTED),
    ],
)
def test_order_states_map(state: str, status: AttemptStatus) -> None:
    out = map_option_orders(
        _request("get_option_orders", {"orders": [_order(state=state)]}), _ids()
    )
    assert out.broker_orders[0].status is status


@pytest.mark.parametrize(
    ("effective_input", "data"),
    [
        ({"state": "confirmed"}, {"orders": [ORDER]}),
        ({"order_id": ORDER_ID}, {"orders": [ORDER]}),
        ({}, {"orders": [ORDER], "next": "opaque"}),
    ],
)
def test_narrowed_or_paged_order_reads_are_not_complete(
    effective_input: dict[str, Any], data: Any
) -> None:
    out = map_option_orders(_request("get_option_orders", data, **effective_input), _ids())
    assert out.open_orders == () and len(out.broker_orders) == 1
    assert "not a complete read" in out.gaps[0]


@pytest.mark.parametrize(
    "order",
    [
        _order(state="mystery"),
        _order(legs=[ORDER["legs"][0], {**ORDER["legs"][0], "option_id": str(uuid.uuid4())}]),
        _order(quantity="1.5"),
        _order(pending_quantity="3"),
    ],
    ids=["unknown-state", "multi-leg", "fractional", "inconsistent"],
)
def test_unsupported_orders_raise(order: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        map_option_orders(_request("get_option_orders", {"orders": [order]}), _ids())


def test_working_order_with_an_unpermitted_side_raises() -> None:
    leg = {**ORDER["legs"][0], "side": "buy", "position_effect": "open"}
    with pytest.raises(ValueError):
        map_option_orders(_request("get_option_orders", {"orders": [_order(legs=[leg])]}), _ids())


# ------------------------------------------------------------------------------ order tools


def test_place_result_maps_to_broker_order() -> None:
    out = map_order_placement(_request("place_option_order", {"order": ORDER}), _ids())
    _check_provenance(out, CALL)
    (obs,) = out.broker_orders
    assert obs.broker_order_id == ORDER_ID and out.open_orders == ()


def test_unconfirmed_place_result_is_a_working_order() -> None:
    """A just-created order reported `unconfirmed` keeps its identity (2026-10-01 HIMS)."""
    order = _order(state="unconfirmed")
    out = map_order_placement(_request("place_option_order", {"order": order}), _ids())
    (obs,) = out.broker_orders
    assert obs.broker_order_id == ORDER_ID and obs.status is AttemptStatus.PLACED
    read = map_option_orders(_request("get_option_orders", {"orders": [order]}), _ids())
    (working,) = read.open_orders[0].orders
    assert working.broker_order_ref == ORDER_ID and working.unfilled_quantity == 2


def test_place_result_without_order_raises() -> None:
    with pytest.raises(ValueError):
        map_order_placement(_request("place_option_order", {"order": None}), _ids())


REVIEW: dict[str, Any] = {
    "account_number": "****1234",
    "type": "limit",
    "direction": "credit",
    "quantity": "2",
    "price": "1.25",
    "time_in_force": "gfd",
    "market_hours": "regular_hours",
    "legs": [{"option_id": OPTION_ID, "side": "sell", "position_effect": "open"}],
    "order_checks": {},
    "option_quotes": [
        {
            "instrument_id": OPTION_ID,
            "ask_price": "1.30",
            "ask_size": 10,
            "bid_price": "1.20",
            "bid_size": 12,
            "mark_price": "1.25",
            "delta": "-0.2",
            "gamma": "0.01",
            "theta": "-0.05",
            "vega": "0.3",
            "rho": "-0.01",
            "implied_volatility": "0.18",
            "open_interest": 100,
            "volume": 20,
            "updated_at": "2026-09-29T14:00:00Z",
        }
    ],
}


def test_clean_review_maps_with_quote() -> None:
    assert set(REVIEW) <= _schema_fields("review_option_order", "data")
    out = map_order_review(_request("review_option_order", REVIEW), _ids())
    _check_provenance(out, CALL)
    (review,) = out.order_reviews
    assert review.clean and review.alert_type is None and review.quantity == 2
    assert review.legs[0].side_raw == "sell_to_open"
    (quote,) = out.option_quotes
    assert quote.bid == Decimal("1.20") and quote.ask == Decimal("1.30")
    assert out.gaps == ()


def test_review_alert_is_carried_and_named_in_a_gap() -> None:
    data = {**REVIEW, "order_checks": {"alertType": "insufficient_buying_power", "details": {}}}
    out = map_order_review(_request("review_option_order", data), _ids())
    (review,) = out.order_reviews
    assert not review.clean and review.alert_type == "insufficient_buying_power"
    assert any("insufficient_buying_power" in g and "do not place" in g for g in out.gaps)


def test_cancel_ack_maps_to_request_not_cancellation() -> None:
    out = map_order_cancel(
        _request("cancel_option_order", {"accepted": True}, order_id=ORDER_ID), _ids()
    )
    (ack,) = out.cancel_requests
    assert ack.accepted and ack.broker_order_id == ORDER_ID
    assert "not a cancellation" in out.gaps[0]


def test_cancel_without_order_id_raises() -> None:
    with pytest.raises(ValueError):
        map_order_cancel(_request("cancel_option_order", {"accepted": True}), _ids())


# ------------------------------------------------------------------------------- positions


def test_share_positions_count_whole_shares() -> None:
    data = {
        "positions": [
            {"symbol": "SPY", "quantity": "100.5", "type": "long"},
            {"symbol": "QQQ", "quantity": "0.4", "type": "long"},
        ]
    }
    out = map_equity_positions(_request("get_equity_positions", data), _ids())
    (read,) = out.positions
    assert read.covers == {PositionsCoverage.SHARES}
    assert [(h.symbol, h.quantity) for h in read.share_holdings] == [("SPY", 100)]


def test_short_share_position_raises() -> None:
    data = {"positions": [{"symbol": "SPY", "quantity": "-1", "type": "short"}]}
    with pytest.raises(ValueError):
        map_equity_positions(_request("get_equity_positions", data), _ids())


def test_short_option_rows_are_pending_until_resolved() -> None:
    row = {
        "option_id": OPTION_ID,
        "chain_id": "c1",
        "chain_symbol": "SPY",
        "type": "short",
        "quantity": "2.0000",
        "trade_value_multiplier": "100.0000",
    }
    closed = {**row, "option_id": str(uuid.uuid4()), "quantity": "0.0000"}
    out = map_option_positions(
        _request("get_option_positions", {"positions": [row, closed]}), _ids()
    )
    _check_provenance(out, CALL)
    assert out.positions == ()
    (pending,) = out.pending_option_positions
    (held,) = pending.rows
    assert held.broker_instrument_id == OPTION_ID and held.short_quantity == 2
    assert OPTION_ID in out.gaps[0]


def test_long_option_position_raises() -> None:
    row = {
        "option_id": OPTION_ID,
        "chain_symbol": "SPY",
        "type": "long",
        "quantity": "1",
        "trade_value_multiplier": "100",
    }
    with pytest.raises(ValueError):
        map_option_positions(_request("get_option_positions", {"positions": [row]}), _ids())


def test_filtered_positions_are_not_complete() -> None:
    out = map_option_positions(
        _request("get_option_positions", {"positions": []}, option_type="put"), _ids()
    )
    assert out.positions == () and "not a complete read" in out.gaps[0]
    out = map_equity_positions(
        _request("get_equity_positions", {"positions": [], "next": "c"}), _ids()
    )
    assert out.positions == ()


# ---------------------------------------------------------------------------- the boundary


def test_boundary_validates_a_place_result() -> None:
    text = json.dumps({"data": {"order": ORDER}, "guide": "prose"})
    envelope = BoundaryValidator(redactor=Redactor())(
        ValidationRequest(
            tool_call_id=CALL,
            server="robinhood",
            tool="place_option_order",
            tier=ToolTier.X,
            effective_input={"account_number": "AGENTIC_ACCOUNT"},
            tool_response={"content": [{"type": "text", "text": text}]},
            retrieved_at=RETRIEVED,
        )
    ).envelope
    assert envelope.kind is EnvelopeKind.VALIDATED
    mapped = mapped_evidence_of(envelope.model_dump(mode="json"))
    assert mapped is not None and mapped.broker_orders[0].broker_order_id == ORDER_ID
