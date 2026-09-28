"""BrokerLedger against a real ledger (ADR-0034): intents, orders, fills, lineages, closes.

Broker results go through the real `BoundaryValidator` and mappers, are stored as validated
results like the proxy stores them, then handed to `BrokerLedger` like the proxy does.
"""

import copy
import json
import uuid
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

import psycopg
import pytest

from wheelta_robinhood_agent.agent.broker_ledger import BrokerLedger
from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, ValidationRequest
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyCall
from wheelta_robinhood_agent.agent.result_boundary import BoundaryValidator
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    CancellationStatus,
    StrategyKind,
    ToolTier,
)
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger.orders import (
    order_record,
    owned_unresolved_orders,
    run_order_records,
)
from wheelta_robinhood_agent.ledger.positions import position_book
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import record_tool_call_requested
from wheelta_robinhood_agent.observability.redaction import Redactor

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 29, 14, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=1)
ACCOUNT = "acct-scope-1"
ORDER_ID = "4b3c0f4e-1d2a-4e5f-9a8b-7c6d5e4f3a2b"
OPTION_ID = "d17decae-92f6-430e-b4c0-3772e5dd27ab"
_sdk = count(1)

LEG = {
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
ORDER: dict[str, Any] = {
    "id": ORDER_ID,
    "chain_symbol": "SPY",
    "state": "confirmed",
    "type": "limit",
    "trigger": "immediate",
    "quantity": "2",
    "processed_quantity": "0",
    "pending_quantity": "2",
    "canceled_quantity": "0",
    "price": "1.25",
    "trade_value_multiplier": "100.0000",
    "time_in_force": "gfd",
    "placed_agent": "agentic",
    "created_at": "2026-09-29T14:00:30Z",
    "updated_at": "2026-09-29T14:00:30Z",
    "legs": [LEG],
}
REVIEW = {
    "type": "limit",
    "quantity": "2",
    "price": "1.25",
    "time_in_force": "gfd",
    "legs": [{"option_id": OPTION_ID, "side": "sell", "position_effect": "open"}],
    "order_checks": {},
    "option_quotes": [],
}
PLACE_ARGS = {
    "account_number": "AGENTIC_ACCOUNT",
    "legs": [{"option_id": OPTION_ID, "side": "sell", "position_effect": "open"}],
    "quantity": "2",
    "type": "limit",
    "price": "1.25",
    "time_in_force": "gfd",
}


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


@pytest.fixture
def broker(conn: Conn, run_id: uuid.UUID) -> BrokerLedger:
    return BrokerLedger(conn, run_id, ACCOUNT)


def _call(conn: Conn, run_id: uuid.UUID, tool: str, args: dict[str, Any]) -> ProxyCall:
    tier = ToolTier.R if tool.startswith("get_") else ToolTier.X
    tool_call_id = record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id=f"toolu_{next(_sdk)}",
        stage="agent",
        server="robinhood",
        tool=tool,
        tier=tier,
        arguments_redacted=args,
        requested_at=T0,
    ).tool_call_id
    return ProxyCall(tool_call_id, "robinhood", tool, tier, dict(args))


def _result(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger, call: ProxyCall, data: Any
) -> dict[str, Any]:
    text = json.dumps({"data": data, "guide": "prose"})
    envelope = BoundaryValidator(redactor=Redactor())(
        ValidationRequest(
            tool_call_id=call.tool_call_id,
            server=call.server,
            tool=call.tool,
            tier=call.tier,
            effective_input=call.effective_input,
            tool_response={"content": [{"type": "text", "text": text}]},
            retrieved_at=T0 + timedelta(seconds=40),
        )
    ).envelope
    assert envelope.kind is EnvelopeKind.VALIDATED, envelope.gaps
    payload = envelope.model_dump(mode="json")
    ledger_evidence.insert_result(
        conn,
        run_id=run_id,
        kind=ledger_evidence.ResultKind.VALIDATED,
        payload=payload,
        tool_call_id=call.tool_call_id,
    )
    broker.after_validated(call, payload)
    return payload


def _place(conn: Conn, run_id: uuid.UUID, broker: BrokerLedger, order: dict[str, Any]) -> None:
    review = _call(conn, run_id, "review_option_order", PLACE_ARGS)
    _result(conn, run_id, broker, review, REVIEW)
    place = _call(conn, run_id, "place_option_order", PLACE_ARGS)
    broker.before_dispatch(place)
    _result(conn, run_id, broker, place, {"order": order})


def _read_orders(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger, orders: list[dict[str, Any]]
) -> None:
    read = _call(conn, run_id, "get_option_orders", {"account_number": "AGENTIC_ACCOUNT"})
    _result(conn, run_id, broker, read, {"orders": orders})


def _filled(quantity: int, execution: str) -> dict[str, Any]:
    ex = {
        "id": execution,
        "price": "1.25",
        "quantity": str(quantity),
        "settlement_date": "2026-09-30",
        "trade_date": "2026-09-29",
        "timestamp": "2026-09-29T14:01:00Z",
    }
    leg = {**LEG, "executions": [ex]}
    state = "filled" if quantity == 2 else "partially_filled"
    return {
        **copy.deepcopy(ORDER),
        "state": state,
        "processed_quantity": str(quantity),
        "pending_quantity": str(2 - quantity),
        "updated_at": "2026-09-29T14:01:00Z",
        "legs": [leg],
    }


def test_place_records_intent_review_order_and_status(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger
) -> None:
    _place(conn, run_id, broker, ORDER)
    (record,) = run_order_records(conn, run_id)
    assert record.intent is not None and record.broker_order is not None
    assert record.broker_order.broker_order_id == ORDER_ID
    assert record.intent.side_raw == "sell_to_open" and record.intent.quantity == 2
    assert len(record.review_tool_call_ids) == 1
    assert record.status is AttemptStatus.PLACED
    (owned,) = owned_unresolved_orders(conn, ACCOUNT)
    assert owned.broker_order is not None


def test_fill_opens_a_csp_lineage_and_a_positions_read_closes_it(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger
) -> None:
    _place(conn, run_id, broker, ORDER)
    _read_orders(conn, run_id, broker, [_filled(2, "e0e0e0e0-0000-4000-8000-000000000001")])
    # The same read again must not duplicate fills or lineages.
    _read_orders(conn, run_id, broker, [_filled(2, "e0e0e0e0-0000-4000-8000-000000000001")])
    (record,) = run_order_records(conn, run_id)
    assert record.status is AttemptStatus.FILLED and record.filled_quantity == 2
    book = position_book(conn, ACCOUNT, as_of=T0 + timedelta(minutes=5))
    (entry,) = book.entries
    assert entry.strategy is StrategyKind.CASH_SECURED_PUT and entry.underlying == "SPY"
    assert entry.current_instruments[0].broker_instrument_id == OPTION_ID
    assert len(entry.entry_fill_ids) == 1
    assert owned_unresolved_orders(conn, ACCOUNT) == ()

    held = {
        "option_id": OPTION_ID,
        "chain_symbol": "SPY",
        "type": "short",
        "quantity": "2",
        "trade_value_multiplier": "100",
    }
    positions = _call(conn, run_id, "get_option_positions", {"account_number": "AGENTIC_ACCOUNT"})
    _result(conn, run_id, broker, positions, {"positions": [held]})
    assert len(position_book(conn, ACCOUNT, as_of=T0 + timedelta(minutes=6)).entries) == 1

    later = _call(conn, run_id, "get_option_positions", {"account_number": "AGENTIC_ACCOUNT"})
    _result(conn, run_id, broker, later, {"positions": []})
    assert position_book(conn, ACCOUNT, as_of=T0 + timedelta(minutes=7)).entries == ()


def test_partial_fill_then_positions_read_reconciles_quantity(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger
) -> None:
    _place(conn, run_id, broker, ORDER)
    _read_orders(conn, run_id, broker, [_filled(1, "e0e0e0e0-0000-4000-8000-000000000002")])
    (entry,) = position_book(conn, ACCOUNT, as_of=T0 + timedelta(minutes=5)).entries
    assert entry.current_instruments[0].short_quantity == 1
    (owned,) = owned_unresolved_orders(conn, ACCOUNT)
    assert owned.status is AttemptStatus.PARTIALLY_FILLED


def test_cancel_request_is_pending_until_a_terminal_read_confirms_it(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger
) -> None:
    _place(conn, run_id, broker, ORDER)
    cancel = _call(
        conn,
        run_id,
        "cancel_option_order",
        {"account_number": "AGENTIC_ACCOUNT", "order_id": ORDER_ID},
    )
    _result(conn, run_id, broker, cancel, {"accepted": True})
    (record,) = run_order_records(conn, run_id)
    assert [c.status for c in record.cancellations] == [CancellationStatus.PENDING]
    cancelled = {
        **ORDER,
        "state": "cancelled",
        "pending_quantity": "0",
        "canceled_quantity": "2",
        "updated_at": "2026-09-29T14:02:00Z",
    }
    _read_orders(conn, run_id, broker, [cancelled])
    record = order_record(conn, record.broker_order.order_id)  # type: ignore[union-attr]
    assert record.status is AttemptStatus.CANCELLED
    assert CancellationStatus.CONFIRMED in {c.status for c in record.cancellations}
    assert owned_unresolved_orders(conn, ACCOUNT) == ()


def test_unanswered_place_stays_unknown_until_a_complete_read_shows_no_order(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger
) -> None:
    place = _call(conn, run_id, "place_option_order", PLACE_ARGS)
    broker.before_dispatch(place)  # the broker never answered
    (owned,) = owned_unresolved_orders(conn, ACCOUNT)
    assert owned.broker_order is None and owned.status is AttemptStatus.UNKNOWN
    _read_orders(conn, run_id, broker, [])
    assert owned_unresolved_orders(conn, ACCOUNT) == ()


def test_a_newer_order_keeps_an_unanswered_place_unresolved(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger
) -> None:
    place = _call(conn, run_id, "place_option_order", PLACE_ARGS)
    broker.before_dispatch(place)
    _read_orders(conn, run_id, broker, [ORDER])  # an order exists but is not linked
    assert len(owned_unresolved_orders(conn, ACCOUNT)) == 1


def test_orders_placed_by_others_open_no_lineage(
    conn: Conn, run_id: uuid.UUID, broker: BrokerLedger
) -> None:
    _read_orders(conn, run_id, broker, [_filled(2, "e0e0e0e0-0000-4000-8000-000000000003")])
    assert position_book(conn, ACCOUNT, as_of=T0 + timedelta(minutes=5)).entries == ()
