"""The simulated broker (ADR-0038): order tools answered in-process, never forwarded."""

import asyncio
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from wheelta_robinhood_agent.agent.mapped_evidence import MappingRequest
from wheelta_robinhood_agent.agent.result_boundary import extract_mcp_payload
from wheelta_robinhood_agent.agent.robinhood_mappers import (
    map_option_orders,
    map_order_cancel,
    map_order_placement,
    map_order_review,
)
from wheelta_robinhood_agent.agent.simulated_broker import SimulatedBroker, simulated_scope_id
from wheelta_robinhood_agent.domain.enums import OptionRight
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.integrations.mcp_upstream import (
    UpstreamResult,
    UpstreamTool,
    UpstreamUnavailable,
)
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=UTC)
IID = "d17decae-92f6-430e-b4c0-3772e5dd27ab"
REAL_ORDER = "0e0e0e0e-0000-4000-8000-000000000001"
INSTRUMENT = OptionInstrument(
    evidence_id=uuid.uuid4(),
    as_of=NOW,
    source_tool_call_ids=(uuid.uuid4(),),
    occ_symbol=OccSymbol(
        root="SPY", expiration=date(2026, 10, 16), right=OptionRight.PUT, strike=Decimal("740")
    ),
    broker_instrument_id=IID,
    underlying="SPY",
    multiplier=100,
)
ORDER = {
    "account_number": "5550001234",
    "legs": [{"option_id": IID, "side": "sell", "position_effect": "open"}],
    "quantity": "2",
    "price": "1.79",
    "type": "limit",
    "time_in_force": "gfd",
}


def _order_json(order_id: str, state: str, pending: str) -> dict[str, Any]:
    return {
        "id": order_id,
        "chain_symbol": "SPY",
        "state": state,
        "type": "limit",
        "trigger": "immediate",
        "time_in_force": "gfd",
        "quantity": "1",
        "processed_quantity": "0",
        "pending_quantity": pending,
        "canceled_quantity": "0",
        "price": "2.00",
        "trade_value_multiplier": "100",
        "created_at": (NOW - timedelta(days=1)).isoformat(),
        "updated_at": (NOW - timedelta(days=1)).isoformat(),
        "legs": [
            {
                "option_id": IID,
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


class FakeUpstream:
    """A real-shape Robinhood upstream that fails the test if an order tool reaches it."""

    server = "robinhood"
    tools = (UpstreamTool("get_option_orders", None, {}),)

    def __init__(self, orders: list[dict[str, Any]] | None = None) -> None:
        self.orders = orders or []
        self.calls: list[str] = []

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_seconds: float
    ) -> UpstreamResult:
        assert not name.endswith("_order") and not name.startswith("exercise"), name
        self.calls.append(name)
        payload: dict[str, Any] = {"data": {"orders": [dict(o) for o in self.orders]}}
        if name != "get_option_orders":
            payload = {"data": {"other": True}}
        text = json.dumps(payload)
        return UpstreamResult({"content": [{"type": "text", "text": text}]}, len(text))


def _broker(upstream: FakeUpstream | None = None, **kw: Any) -> SimulatedBroker:
    ids = iter(uuid.UUID(int=i) for i in range(1, 1000))
    return SimulatedBroker(
        upstream=upstream or FakeUpstream(),
        registry=ROBINHOOD_REGISTRY,
        instruments=kw.get("instruments", lambda iid: INSTRUMENT if iid == IID else None),
        clock=lambda: NOW,
        id_factory=lambda: next(ids),
    )


def call(broker: SimulatedBroker, name: str, args: Mapping[str, Any]) -> Any:
    result = asyncio.run(broker.call_tool(name, args, timeout_seconds=5))
    _, payload = extract_mcp_payload(result.response)
    return payload


def mapped(tool: str, args: Mapping[str, Any], payload: Any) -> Any:
    mapper = {
        "review_option_order": map_order_review,
        "place_option_order": map_order_placement,
        "cancel_option_order": map_order_cancel,
        "get_option_orders": map_option_orders,
    }[tool]
    request = MappingRequest(
        tool_call_id=uuid.uuid4(),
        server="robinhood",
        tool=tool,
        effective_input=dict(args),
        payload=payload,
        retrieved_at=NOW,
    )
    return mapper(request, uuid.uuid4)


def test_review_is_clean_and_passes_the_verified_mapper() -> None:
    upstream = FakeUpstream()
    evidence = mapped(
        "review_option_order", ORDER, call(_broker(upstream), "review_option_order", ORDER)
    )
    (review,) = evidence.order_reviews
    assert review.clean and review.quantity == 2 and review.limit_price == Decimal("1.79")
    assert review.legs[0].side_raw == "sell_to_open"
    assert upstream.calls == []


def test_place_stays_working_and_reads_back_until_cancelled() -> None:
    upstream = FakeUpstream()
    broker = _broker(upstream)
    placed = mapped("place_option_order", ORDER, call(broker, "place_option_order", ORDER))
    (order,) = placed.broker_orders
    assert order.state_raw == "confirmed" and order.pending_quantity == 2
    assert order.processed_quantity == 0 and order.executions == ()
    assert str(order.legs[0].occ_symbol) == str(INSTRUMENT.occ_symbol)
    account = {"account_number": ORDER["account_number"]}
    read = mapped("get_option_orders", account, call(broker, "get_option_orders", account))
    (working,) = read.open_orders[0].orders
    assert working.broker_order_ref == order.broker_order_id and working.unfilled_quantity == 2
    cancel_args = {**account, "order_id": order.broker_order_id}
    ack = mapped(
        "cancel_option_order", cancel_args, call(broker, "cancel_option_order", cancel_args)
    )
    assert ack.cancel_requests[0].accepted
    after = mapped("get_option_orders", account, call(broker, "get_option_orders", account))
    (listed,) = after.broker_orders
    assert listed.state_raw == "cancelled" and listed.canceled_quantity == 2
    assert after.open_orders[0].orders == ()
    with pytest.raises(UpstreamUnavailable, match="not open"):
        call(broker, "cancel_option_order", cancel_args)
    assert upstream.calls == ["get_option_orders", "get_option_orders"]


def test_place_ref_id_is_idempotent() -> None:
    broker = _broker()
    args = {**ORDER, "ref_id": "r-1"}
    first = call(broker, "place_option_order", args)
    assert call(broker, "place_option_order", args) == first
    account = {"account_number": ORDER["account_number"]}
    assert len(call(broker, "get_option_orders", account)["data"]["orders"]) == 1


def test_a_working_real_order_can_be_cancelled_in_simulation_only() -> None:
    upstream = FakeUpstream([_order_json(REAL_ORDER, "confirmed", "1")])
    broker = _broker(upstream)
    account = {"account_number": ORDER["account_number"]}
    args = {**account, "order_id": REAL_ORDER}
    with pytest.raises(UpstreamUnavailable, match="no open order"):
        call(broker, "cancel_option_order", args)  # not read yet this run
    call(broker, "get_option_orders", account)
    assert call(broker, "cancel_option_order", args) == {"data": {"accepted": True}}
    (listed,) = call(broker, "get_option_orders", account)["data"]["orders"]
    assert listed["state"] == "cancelled" and listed["canceled_quantity"] == "1"
    assert upstream.orders[0]["state"] == "confirmed"  # the real order is untouched
    with pytest.raises(UpstreamUnavailable, match="no open order"):
        call(broker, "cancel_option_order", args)


def test_an_unchanged_read_passes_through_verbatim() -> None:
    upstream = FakeUpstream([_order_json(REAL_ORDER, "filled", "0")])
    broker = _broker(upstream)
    result = asyncio.run(
        broker.call_tool("get_option_orders", {"account_number": "x"}, timeout_seconds=5)
    )
    assert "structuredContent" not in result.response


@pytest.mark.parametrize(
    ("args", "listed"),
    [
        ({"state": "confirmed"}, True),
        ({"state": "filled"}, False),
        ({"placed_agent": "agentic"}, True),
        ({"placed_agent": "user"}, False),
        ({"created_at_gte": (NOW - timedelta(hours=1)).isoformat()}, True),
        ({"created_at_gte": (NOW + timedelta(hours=1)).isoformat()}, False),
        ({"created_at_gte": "yesterday"}, False),
        ({"created_at_gte": 5}, False),
        ({"cursor": "next-page"}, False),
        ({"chain_ids": "c1"}, False),  # cannot be matched: no simulated order listed
        ({"state": ""}, True),
    ],
)
def test_narrowed_reads_list_simulated_orders_only_when_they_match(
    args: dict[str, Any], listed: bool
) -> None:
    broker = _broker()
    placed = call(broker, "place_option_order", ORDER)["data"]["order"]
    orders = call(broker, "get_option_orders", {"account_number": "x", **args})["data"]["orders"]
    assert (orders == [placed]) is listed


def test_order_id_filter() -> None:
    broker = _broker()
    placed = call(broker, "place_option_order", ORDER)["data"]["order"]
    read = call(broker, "get_option_orders", {"order_id": placed["id"]})["data"]["orders"]
    assert read == [placed]
    assert call(broker, "get_option_orders", {"order_id": REAL_ORDER})["data"]["orders"] == []


@pytest.mark.parametrize(
    "tool",
    [
        "place_equity_order",
        "review_equity_order",
        "cancel_equity_order",
        "exercise_option",
        "replace_option_order",
        "preview_option_order",  # unregistered
        "preview_crypto_order",  # excluded
        "some_new_tool",  # unregistered
    ],
)
def test_other_order_actions_are_never_forwarded(tool: str) -> None:
    upstream = FakeUpstream()
    with pytest.raises(UpstreamUnavailable, match="not available on the simulated broker"):
        call(_broker(upstream), tool, ORDER)
    assert upstream.calls == []


def test_reads_pass_through() -> None:
    upstream = FakeUpstream()
    assert call(_broker(upstream), "get_portfolio", {}) == {"data": {"other": True}}
    assert upstream.calls == ["get_portfolio"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"account_number": ""}, "account_number"),
        ({"legs": []}, "single-leg"),
        ({"legs": ["x"]}, "single-leg"),
        ({"legs": [{"option_id": IID, "side": "short", "position_effect": "open"}]}, "leg needs"),
        (
            {
                "legs": [
                    {
                        "option_id": IID,
                        "side": "sell",
                        "position_effect": "open",
                        "ratio_quantity": 2,
                    }
                ]
            },
            "ratio_quantity",
        ),
        ({"type": "market"}, "limit orders"),
        ({"stop_price": "1.00"}, "limit orders"),
        ({"quantity": "0"}, "positive"),
        ({"quantity": "1.5"}, "positive"),
        ({"price": "-1"}, "positive"),
        ({"price": "abc"}, "positive"),
        ({"price": "NaN"}, "positive"),
        ({"price": None}, "positive"),
        (
            {"legs": [{"option_id": "other", "side": "sell", "position_effect": "open"}]},
            "not read this run",
        ),
    ],
)
@pytest.mark.parametrize("tool", ["review_option_order", "place_option_order"])
def test_malformed_requests_are_refused(tool: str, change: dict[str, Any], reason: str) -> None:
    with pytest.raises(UpstreamUnavailable, match=reason):
        call(_broker(), tool, {**ORDER, **change})


def test_an_instrument_without_a_verified_multiplier_is_refused() -> None:
    unverified = INSTRUMENT.model_copy(update={"multiplier": None})
    broker = _broker(instruments=lambda iid: unverified)
    with pytest.raises(UpstreamUnavailable, match="multiplier"):
        call(broker, "place_option_order", ORDER)


@pytest.mark.parametrize(
    ("args", "reason"),
    [({"order_id": REAL_ORDER}, "account_number"), ({"account_number": "x"}, "order_id")],
)
def test_cancel_needs_account_and_order(args: dict[str, Any], reason: str) -> None:
    with pytest.raises(UpstreamUnavailable, match=reason):
        call(_broker(), "cancel_option_order", args)


def test_the_upstream_must_be_the_registry_server() -> None:
    upstream = FakeUpstream()
    upstream.server = "wheelta"  # type: ignore[misc]
    with pytest.raises(ValueError, match="other servers"):
        _broker(upstream)
    broker = _broker()
    assert broker.server == "robinhood" and broker.tools == FakeUpstream.tools


def test_unusable_real_reads_are_left_to_the_validator() -> None:
    class Odd(FakeUpstream):
        def __init__(self, response: dict[str, Any]) -> None:
            super().__init__()
            self.response = response

        async def call_tool(
            self, name: str, arguments: Mapping[str, Any], *, timeout_seconds: float
        ) -> UpstreamResult:
            return UpstreamResult(self.response, 1)

    for response in (
        {"isError": True, "content": [{"type": "text", "text": "boom"}]},
        {"content": [{"type": "image"}]},
        {"content": [{"type": "text", "text": "[1]"}]},
        {"content": [{"type": "text", "text": '{"data": 1}'}]},
        {"content": [{"type": "text", "text": '{"data": {"orders": 1}}'}]},
    ):
        broker = _broker(Odd(response))
        call(broker, "place_option_order", ORDER)
        result = asyncio.run(broker.call_tool("get_option_orders", {}, timeout_seconds=5))
        assert result.response == response


def test_simulated_scope_is_per_run() -> None:
    run = uuid.uuid4()
    assert simulated_scope_id(run) == f"simulated:{run}"


def test_a_real_row_without_an_id_is_left_for_the_mapper() -> None:
    upstream = FakeUpstream([{"state": "confirmed"}])
    broker = _broker(upstream)
    call(broker, "place_option_order", ORDER)
    orders = call(broker, "get_option_orders", {})["data"]["orders"]
    assert orders[-1] == {"state": "confirmed"} and len(orders) == 2
