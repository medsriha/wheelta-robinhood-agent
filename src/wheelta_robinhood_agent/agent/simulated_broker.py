"""The simulated broker behind the Robinhood proxy in a dry run (ADR-0038).

A dry run with order venue `simulated` gives the agent the same three option-order tools, the
same prompt, and the same procedure as armed live. `SimulatedBroker` wraps the real Robinhood
upstream and answers those tools in-process, so a dry run exercises the live order path
without any order reaching Robinhood:

- **Forwarded only:** registered Tier R/S tools not named like an order action (`review_` /
  `place_` / `replace_` / `cancel_` / `exercise_`). Everything else (Tier X, excluded, an
  unregistered name) is answered here or refused with an `UpstreamError`. Nothing else in the
  run can send one: the proxy is the only path to the upstream (ADR-0023).
- `review_option_order`: the request echoed back, `order_checks` `{}` (clean), no quotes.
- `place_option_order`: the order fills in full at its limit price, with one execution
  (ADR-0046, owner decision 2026-09-29; it replaced "stays working"). The contract comes from
  this run's `get_option_instruments` evidence; an instrument not read this run (or without a
  verified multiplier) is refused, and so is a buy-to-close for more than the short quantity
  known this run (a positions read plus this run's fills).
- `cancel_option_order`: a simulated order is already filled, so cancelling one is refused,
  as the broker would. A working real order a `get_option_orders` read showed this run is
  accepted and then reads `cancelled`. Anything else is refused.
- `get_option_orders` is read from Robinhood and overlaid: this run's simulated orders are
  listed first (the list is newest first), and simulated cancellations of real orders are
  applied.
- `get_option_positions` is read from Robinhood and overlaid with this run's fills: a
  sell-to-open adds to (or creates) the contract's short row, a buy-to-close reduces it.
  Only a first-page read lists a new row. `get_portfolio` passes through unchanged: whether
  the broker's `cash` nets short-put collateral is unverified, and the facts service already
  treats CSP reserved cash as unavailable while any short put is held.

State lives only in memory, in a `SimulatedState` (owner decisions 2026-09-28 and
2026-09-29). ADR-0057: one state serves the two runs of a dry-run tick (Buy-to-Close, then
Sell Options), so the second sees the first's fills; nothing carries over to another tick.
The ledger records every call, and `BrokerLedger` records the simulated orders, fills, and
lineages under a simulated scope (`simulated_scope_id`, one per tick: the close run's id),
never the account's own scope.

Responses follow the captured output schemas
(`tests/fixtures/robinhood/output_schemas_orders_2026-09-28.json`) so they pass the same
mappers as live results. A refusal carries fixed text only.
"""

import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from pydantic import JsonValue

from wheelta_robinhood_agent.agent.result_boundary import (
    PayloadError,
    PayloadKind,
    extract_mcp_payload,
)
from wheelta_robinhood_agent.domain.enums import OptionRight, ToolTier
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument
from wheelta_robinhood_agent.integrations.mcp_upstream import (
    McpUpstream,
    UpstreamResult,
    UpstreamTool,
    UpstreamUnavailable,
)
from wheelta_robinhood_agent.integrations.registry import ToolRegistry

REVIEW_TOOL: Final = "review_option_order"
PLACE_TOOL: Final = "place_option_order"
CANCEL_TOOL: Final = "cancel_option_order"
ORDERS_TOOL: Final = "get_option_orders"
POSITIONS_TOOL: Final = "get_option_positions"
SIMULATED_SCOPE_PREFIX: Final = "simulated:"
# Names that act on orders; never forwarded whatever their registry tier.
_ORDER_ACTION: Final = re.compile(r"^(review|place|replace|cancel|exercise)_")
_WORKING: Final = frozenset({"unconfirmed", "queued", "confirmed", "partially_filled"})
# get_option_orders arguments a simulated order can be matched against.
_MATCHABLE: Final = frozenset({"account_number", "order_id", "state", "placed_agent"})
_CREATED_GTE: Final = "created_at_gte"
_CURSOR: Final = "cursor"

InstrumentLookup = Callable[[str], OptionInstrument | None]

__all__ = [
    "SIMULATED_SCOPE_PREFIX",
    "InstrumentLookup",
    "SimulatedBroker",
    "SimulatedState",
    "simulated_scope_id",
]


def simulated_scope_id(run_id: uuid.UUID) -> str:
    """The ledger account scope of one dry-run tick's simulated orders (never a real
    account's). ADR-0057: `run_id` is the tick's first run, shared by both of its runs."""
    return f"{SIMULATED_SCOPE_PREFIX}{run_id}"


class _Refused(Exception):
    """The simulated broker rejects the request (fixed text, safe to deliver)."""


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _count(value: object) -> int | None:
    text = _text(value)
    return int(text) if text is not None and text.isdigit() and int(text) > 0 else None


def _price(value: object) -> str | None:
    text = _text(value)
    if text is None:
        return None
    try:
        price = Decimal(text)
    except InvalidOperation:
        return None
    return text if price.is_finite() and price > 0 else None


def _result(payload: dict[str, JsonValue]) -> UpstreamResult:
    """A `CallToolResult`-shaped result, as the HTTP upstream returns it."""
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    response: dict[str, JsonValue] = {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
    }
    return UpstreamResult(response=response, size_bytes=len(text.encode()))


@dataclass(frozen=True)
class _Leg:
    option_id: str
    side: str
    position_effect: str


@dataclass(frozen=True)
class _Request:
    """A single-leg limit order request, validated the way the broker would refuse it."""

    account_number: str
    leg: _Leg
    quantity: int
    price: str
    time_in_force: str
    market_hours: str
    instrument: OptionInstrument
    multiplier: int


@dataclass
class SimulatedState:
    """What the simulated broker has done. ADR-0057: one object serves both runs of a dry-run
    tick, so the Sell Options agent's reads see the Buy-to-Close agent's simulated fills."""

    # Simulated orders by id, oldest first, in the broker's order JSON shape.
    orders: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    # place ref_id -> simulated order id (the tool's idempotency key).
    ref_ids: dict[str, str] = field(default_factory=dict)
    # Real orders a read showed working, and those cancelled in simulation.
    real_working: set[str] = field(default_factory=set)
    real_cancelled: dict[str, str] = field(default_factory=dict)
    # ADR-0046: net short contracts the simulated fills added (+) or closed (-), by option
    # id, and the instrument each names.
    short_delta: dict[str, int] = field(default_factory=dict)
    filled_instruments: dict[str, OptionInstrument] = field(default_factory=dict)
    # Short quantity per option id in the latest real positions read.
    real_short: dict[str, int] = field(default_factory=dict)


@dataclass
class SimulatedBroker:
    """`McpUpstream` for the Robinhood proxy in a simulated-venue dry run (module docstring)."""

    upstream: McpUpstream
    registry: ToolRegistry
    instruments: InstrumentLookup
    clock: Callable[[], datetime]
    id_factory: Callable[[], uuid.UUID] = uuid.uuid4
    # ADR-0057: shared by the close and sell runs of one dry-run tick.
    state: SimulatedState = field(default_factory=SimulatedState)

    def __post_init__(self) -> None:
        if self.upstream.server != self.registry.server:
            raise ValueError("the simulated broker's upstream and registry name other servers")

    @property
    def server(self) -> str:
        return self.upstream.server

    @property
    def tools(self) -> tuple[UpstreamTool, ...]:
        return self.upstream.tools

    def _forwardable(self, name: str) -> bool:
        """Only registered reads and workspace writes reach Robinhood (fail closed)."""
        spec = self.registry.get(name)
        return (
            spec is not None
            and spec.tier in (ToolTier.R, ToolTier.S)
            and not _ORDER_ACTION.match(name)
        )

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_seconds: float
    ) -> UpstreamResult:
        if not self._forwardable(name):
            try:
                if name == REVIEW_TOOL:
                    return _result(self._review(arguments))
                if name == PLACE_TOOL:
                    return _result(self._place(arguments))
                if name == CANCEL_TOOL:
                    return _result(self._cancel(arguments))
            except _Refused as exc:
                raise UpstreamUnavailable(f"{name}: simulated broker refused: {exc}") from None
            raise UpstreamUnavailable(f"{name}: not available on the simulated broker")
        result = await self.upstream.call_tool(name, arguments, timeout_seconds=timeout_seconds)
        if name == ORDERS_TOOL:
            return self._overlay_orders(arguments, result)
        if name == POSITIONS_TOOL:
            return self._overlay_positions(arguments, result)
        return result

    # ---------------------------------------------------------------------------- requests
    def _request(self, arguments: Mapping[str, Any]) -> _Request:
        account = _text(arguments.get("account_number"))
        if account is None:
            raise _Refused("account_number is required")
        legs = arguments.get("legs")
        if not isinstance(legs, list) or len(legs) != 1 or not isinstance(legs[0], dict):
            raise _Refused("only single-leg orders are simulated")
        raw = legs[0]
        option_id, side, effect = (
            _text(raw.get("option_id")),
            raw.get("side"),
            raw.get("position_effect"),
        )
        if option_id is None or side not in ("buy", "sell") or effect not in ("open", "close"):
            raise _Refused("the leg needs option_id, side buy|sell, and position_effect")
        if raw.get("ratio_quantity", 1) != 1:
            raise _Refused("a single-leg order has ratio_quantity 1")
        if arguments.get("type", "limit") != "limit" or arguments.get("stop_price"):
            raise _Refused("only limit orders are simulated")
        quantity = _count(arguments.get("quantity"))
        price = _price(arguments.get("price"))
        if quantity is None or price is None:
            raise _Refused("quantity must be a positive integer and price a positive decimal")
        instrument = self.instruments(option_id)
        if instrument is None:
            raise _Refused("the option instrument was not read this run")
        if instrument.multiplier is None:
            raise _Refused("the option instrument has no verified multiplier")
        return _Request(
            account_number=account,
            leg=_Leg(option_id=option_id, side=side, position_effect=effect),
            quantity=quantity,
            price=price,
            time_in_force=_text(arguments.get("time_in_force")) or "gfd",
            market_hours=_text(arguments.get("market_hours")) or "regular_hours",
            instrument=instrument,
            multiplier=instrument.multiplier,
        )

    @staticmethod
    def _leg_json(leg: _Leg) -> dict[str, JsonValue]:
        return {
            "option_id": leg.option_id,
            "side": leg.side,
            "position_effect": leg.position_effect,
            "ratio_quantity": 1,
        }

    def _review(self, arguments: Mapping[str, Any]) -> dict[str, JsonValue]:
        req = self._request(arguments)
        return {
            "data": {
                "account_number": req.account_number,
                "type": "limit",
                "direction": "credit" if req.leg.side == "sell" else "debit",
                "quantity": str(req.quantity),
                "price": req.price,
                "time_in_force": req.time_in_force,
                "market_hours": req.market_hours,
                "legs": [self._leg_json(req.leg)],
                "order_checks": {},
                "option_quotes": [],
            }
        }

    def _place(self, arguments: Mapping[str, Any]) -> dict[str, JsonValue]:
        req = self._request(arguments)
        ref_id = _text(arguments.get("ref_id"))
        if ref_id is not None and ref_id in self.state.ref_ids:
            return {"data": {"order": dict(self.state.orders[self.state.ref_ids[ref_id]])}}
        opening = req.leg.side == "sell" and req.leg.position_effect == "open"
        closing = req.leg.side == "buy" and req.leg.position_effect == "close"
        if not (opening or closing):
            raise _Refused("only sell-to-open and buy-to-close are simulated")
        option_id = req.leg.option_id
        if closing and self._short_held(option_id) < req.quantity:
            raise _Refused("no short position of that size to close")
        now = self.clock().isoformat()
        occ = req.instrument.occ_symbol
        order_id = str(self.id_factory())
        premium = Decimal(req.price) * req.quantity * req.multiplier
        order: dict[str, JsonValue] = {
            "id": order_id,
            "chain_symbol": occ.root,
            "state": "filled",
            "type": "limit",
            "trigger": "immediate",
            "direction": "credit" if req.leg.side == "sell" else "debit",
            "quantity": str(req.quantity),
            "processed_quantity": str(req.quantity),
            "pending_quantity": "0",
            "canceled_quantity": "0",
            "price": req.price,
            "stop_price": None,
            "processed_premium": str(premium),
            "trade_value_multiplier": str(req.multiplier),
            "time_in_force": req.time_in_force,
            "market_hours": req.market_hours,
            "placed_agent": "agentic",
            "created_at": now,
            "updated_at": now,
            "last_transaction_at": now,
            "is_replaceable": False,
            "legs": [
                {
                    **self._leg_json(req.leg),
                    "id": str(self.id_factory()),
                    "expiration_date": occ.expiration.isoformat(),
                    "strike_price": str(occ.strike),
                    "option_type": "put" if occ.right is OptionRight.PUT else "call",
                    "executions": [
                        {
                            "id": str(self.id_factory()),
                            "price": req.price,
                            "quantity": str(req.quantity),
                            "timestamp": now,
                        }
                    ],
                }
            ],
        }
        self.state.orders[order_id] = order
        if ref_id is not None:
            self.state.ref_ids[ref_id] = order_id
        change = req.quantity if opening else -req.quantity
        self.state.short_delta[option_id] = self.state.short_delta.get(option_id, 0) + change
        self.state.filled_instruments[option_id] = req.instrument
        return {"data": {"order": dict(order)}}

    def _short_held(self, option_id: str) -> int:
        """Short contracts known this run: the latest real read plus this run's fills."""
        return self.state.real_short.get(option_id, 0) + self.state.short_delta.get(option_id, 0)

    def _cancel(self, arguments: Mapping[str, Any]) -> dict[str, JsonValue]:
        if _text(arguments.get("account_number")) is None:
            raise _Refused("account_number is required")
        order_id = _text(arguments.get("order_id"))
        if order_id is None:
            raise _Refused("order_id is required")
        now = self.clock().isoformat()
        order = self.state.orders.get(order_id)
        if order is not None:
            if order["state"] not in _WORKING:
                raise _Refused("the order is not open")  # every simulated order is filled
            pending = order["pending_quantity"]
            order.update(
                state="cancelled",
                pending_quantity="0",
                canceled_quantity=pending,
                updated_at=now,
                last_transaction_at=now,
                is_replaceable=False,
            )
        elif order_id in self.state.real_working and order_id not in self.state.real_cancelled:
            self.state.real_cancelled[order_id] = now
        else:
            raise _Refused("no open order with this order_id")
        return {"data": {"accepted": True}}

    # ---------------------------------------------------------------------------- reads
    def _matches(self, order: Mapping[str, JsonValue], arguments: Mapping[str, Any]) -> bool:
        """Whether a simulated order belongs in a read with these arguments. A read narrowed
        by a filter it cannot be matched against (e.g. chain_ids) lists no simulated order."""
        for key, value in arguments.items():
            if value in (None, ""):
                continue
            if key == _CURSOR:
                return False  # later pages: simulated orders are listed on the first one
            if key == _CREATED_GTE:
                if not isinstance(value, str) or not isinstance(order["created_at"], str):
                    return False
                try:
                    since = datetime.fromisoformat(value)
                    created = datetime.fromisoformat(order["created_at"])
                    if created < since:
                        return False
                except (TypeError, ValueError):
                    return False
                continue
            if key not in _MATCHABLE:
                return False
            if key == "order_id" and value != order["id"]:
                return False
            if key in ("state", "placed_agent") and value != order[key]:
                return False
        return True

    def _observe_real(self, order: dict[str, JsonValue]) -> dict[str, JsonValue]:
        order_id = order.get("id")
        if not isinstance(order_id, str):
            return order
        if order.get("state") in _WORKING:
            self.state.real_working.add(order_id)
        when = self.state.real_cancelled.get(order_id)
        if when is None or order.get("state") not in _WORKING:
            return order
        return {
            **order,
            "state": "cancelled",
            "pending_quantity": "0",
            "canceled_quantity": order.get("pending_quantity"),
            "updated_at": when,
            "last_transaction_at": when,
            "is_replaceable": False,
        }

    def _overlay_orders(
        self, arguments: Mapping[str, Any], result: UpstreamResult
    ) -> UpstreamResult:
        try:
            kind, payload = extract_mcp_payload(result.response)
        except (PayloadError, TypeError, ValueError):
            return result  # the validator reports the unusable result
        if kind is not PayloadKind.OK or not isinstance(payload, dict):
            return result
        data = payload.get("data")
        if not isinstance(data, dict):
            return result
        listed = data.get("orders")
        if listed is not None and not isinstance(listed, list):
            return result
        real = [self._observe_real(o) if isinstance(o, dict) else o for o in listed or []]
        simulated: list[JsonValue] = [
            dict(o) for o in reversed(self.state.orders.values()) if self._matches(o, arguments)
        ]
        if not simulated and real == (listed or []):
            return result
        orders: list[JsonValue] = [*simulated, *real]
        return _result({**payload, "data": {**data, "orders": orders}})

    def _new_short_row(self, option_id: str, quantity: int) -> dict[str, JsonValue]:
        instrument = self.state.filled_instruments[option_id]
        occ = instrument.occ_symbol
        return {
            "option_id": option_id,
            "chain_symbol": occ.root,
            "type": "short",
            "quantity": str(quantity),
            "trade_value_multiplier": str(instrument.multiplier),
            "expiration_date": occ.expiration.isoformat(),
        }

    def _overlay_positions(
        self, arguments: Mapping[str, Any], result: UpstreamResult
    ) -> UpstreamResult:
        """This run's fills applied to a real positions read (module docstring)."""
        try:
            kind, payload = extract_mcp_payload(result.response)
        except (PayloadError, TypeError, ValueError):
            return result
        if kind is not PayloadKind.OK or not isinstance(payload, dict):
            return result
        data = payload.get("data")
        if not isinstance(data, dict):
            return result
        listed = data.get("positions")
        if listed is not None and not isinstance(listed, list):
            return result
        rows: list[JsonValue] = []
        seen: set[str] = set()
        for row in listed or []:
            option_id = row.get("option_id") if isinstance(row, dict) else None
            if not isinstance(row, dict) or not isinstance(option_id, str):
                rows.append(row)
                continue
            seen.add(option_id)
            real = _count(row.get("quantity")) if row.get("type") == "short" else None
            if real is not None:
                self.state.real_short[option_id] = real
            delta = self.state.short_delta.get(option_id, 0)
            if delta == 0 or real is None:
                rows.append(row)
                continue
            rows.append({**row, "quantity": str(max(real + delta, 0))})
        first_page = not arguments.get(_CURSOR)
        added = [
            self._new_short_row(option_id, delta)
            for option_id, delta in self.state.short_delta.items()
            if first_page and delta > 0 and option_id not in seen
        ]
        if not added and rows == (listed or []):
            return result
        return _result({**payload, "data": {**data, "positions": [*added, *rows]}})
