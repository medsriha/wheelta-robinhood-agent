"""Record live broker orders, fills, cancellations, and position lineages (ADR-0034).

The validating proxy calls this module around proxied Robinhood calls, so the ledger, not the
broker's own lists, stays the system of record (CLAUDE.md §9, §17):

- **Before a place call is sent** (`before_dispatch`): the order intent, keyed by the place
  tool call, plus the reviews of this run that match it. If that write fails, the call is not
  forwarded (fail closed).
- **After a validated result** (`after_validated`), from the typed `MappedEvidence` only:
  - every broker order observation: its identity, the link to this run's intent (place
    result), its status, and one fill per broker execution ID;
  - sell-to-open fills open a lineage (put -> CSP, call -> CC) or link to the one already
    opened for that order; buy-to-close fills link to the active lineage holding that
    contract;
  - a cancel request records a `pending` (accepted) or `unknown` cancellation; a later read
    showing the order cancelled confirms it;
  - a complete options-positions read closes every active lineage whose contract it no longer
    lists, and reconciles a lineage whose held quantity changed;
  - a complete order read with no order created since an unlinked intent resolves that intent
    as "no order" (the place call failed or timed out and nothing reached the broker).

Nothing here is matched by price, ticker, or time other than the documented rules above, and
nothing gates an order: this only records what the broker reported.
"""

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Final

import psycopg

from wheelta_robinhood_agent.agent.facts_tool import load_run_evidence
from wheelta_robinhood_agent.agent.mapped_evidence import (
    BrokerOrderObservation,
    CancelRequestObservation,
    MappedEvidence,
)
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyCall
from wheelta_robinhood_agent.agent.result_boundary import mapped_evidence_of
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    CancellationStatus,
    OptionRight,
    OrderSide,
    StrategyKind,
)
from wheelta_robinhood_agent.domain.orders import OrderIntent
from wheelta_robinhood_agent.domain.positions import PositionInstrument
from wheelta_robinhood_agent.ledger import orders as ledger_orders
from wheelta_robinhood_agent.ledger import positions as ledger_positions
from wheelta_robinhood_agent.ledger.ids import new_id

Conn = psycopg.Connection[tuple[object, ...]]

PLACE_TOOL: Final = "place_option_order"
ORDER_TOOLS: Final = frozenset(
    {PLACE_TOOL, "review_option_order", "cancel_option_order", "get_option_orders"}
)
POSITION_TOOLS: Final = frozenset({"get_option_positions"})
# An unlinked intent is resolved as "no order" only by a complete order read listing no
# order created after (intent time - this margin): broker and local clocks may differ.
NO_ORDER_CLOCK_MARGIN: Final = timedelta(minutes=5)
_TERMINAL: Final = frozenset(
    {AttemptStatus.FILLED, AttemptStatus.CANCELLED, AttemptStatus.REJECTED, AttemptStatus.EXPIRED}
)


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _price(value: object) -> Decimal | None:
    if not isinstance(value, str):
        return None
    try:
        price = Decimal(value)
    except ArithmeticError:
        return None
    return price if price.is_finite() and price > 0 else None


def _single_leg(arguments: Mapping[str, Any]) -> Mapping[str, Any] | None:
    legs = arguments.get("legs")
    if isinstance(legs, list) and len(legs) == 1 and isinstance(legs[0], dict):
        return legs[0]
    return None


def _leg_side(leg: Mapping[str, Any]) -> str | None:
    side, effect = leg.get("side"), leg.get("position_effect")
    if isinstance(side, str) and side and isinstance(effect, str) and effect:
        return f"{side}_to_{effect}"
    return None


@dataclass(frozen=True)
class BrokerLedger:
    """Ledger writes for one live run's proxied broker calls (see module docstring)."""

    conn: Conn
    run_id: uuid.UUID
    account_scope_id: str
    id_factory: Callable[[], uuid.UUID] = new_id

    # ------------------------------------------------------------------ before dispatch
    def before_dispatch(self, call: ProxyCall) -> None:
        """Record the place intent (and its matching reviews) before the call is sent.
        Raises on any ledger failure, so the caller does not forward the call."""
        if call.tool != PLACE_TOOL:
            return
        args = call.effective_input
        leg = _single_leg(args) or {}
        option_id = leg.get("option_id")
        instrument_id = option_id if isinstance(option_id, str) and option_id else None
        evidence = load_run_evidence(self.conn, self.run_id)
        instrument = evidence.instrument(instrument_id) if instrument_id else None
        row = self.conn.execute(
            "SELECT requested_at FROM tool_calls WHERE tool_call_id = %s", (call.tool_call_id,)
        ).fetchone()
        if row is None or not isinstance(row[0], datetime):
            raise LookupError("the place tool call is not recorded")
        side = _leg_side(leg)
        quantity = _int(args.get("quantity"))
        price = _price(args.get("price"))
        order_type = args.get("type", "limit")
        tif = args.get("time_in_force", "gfd")
        intent = OrderIntent(
            intent_id=self.id_factory(),
            run_id=self.run_id,
            place_tool_call_id=call.tool_call_id,
            account_scope_id=self.account_scope_id,
            occ_symbol=instrument.occ_symbol if instrument else None,
            broker_instrument_id=instrument_id,
            side_raw=side,
            quantity=quantity,
            order_type_raw=order_type if isinstance(order_type, str) and order_type else None,
            time_in_force_raw=tif if isinstance(tif, str) and tif else None,
            limit_price=price,
            requested_at=row[0],
        )
        intent_id = ledger_orders.record_order_intent(self.conn, intent)
        for item in evidence.items:
            for review in item.order_reviews:
                (rleg,) = review.legs or (None,)
                if (
                    rleg is not None
                    and rleg.broker_instrument_id == instrument_id
                    and rleg.side_raw == side
                    and review.quantity == quantity
                    and review.limit_price == price
                ):
                    ledger_orders.record_intent_reviewed(
                        self.conn,
                        intent_id,
                        run_id=self.run_id,
                        review_tool_call_id=review.source_tool_call_ids[0],
                        observed_at=review.as_of,
                    )

    # ------------------------------------------------------------------ after a result
    def after_validated(self, call: ProxyCall, envelope: Mapping[str, Any]) -> None:
        """Record what a validated order/position result reports. Raises on ledger failure."""
        if call.tool not in ORDER_TOOLS | POSITION_TOOLS:
            return
        mapped = mapped_evidence_of(envelope)
        if mapped is None:
            return
        intent_id = self._intent_for(call.tool_call_id) if call.tool == PLACE_TOOL else None
        for observation in mapped.broker_orders:
            self._record_order(observation, call.tool_call_id, intent_id)
        for request in mapped.cancel_requests:
            self._record_cancel_request(request, call.tool_call_id)
        if call.tool == "get_option_orders" and mapped.open_orders:
            self._resolve_unlinked_intents(mapped, call.tool_call_id)
        if call.tool in POSITION_TOOLS:
            self._reconcile_lineages(mapped, call.tool_call_id)

    def _intent_for(self, place_tool_call_id: uuid.UUID) -> uuid.UUID | None:
        row = self.conn.execute(
            "SELECT intent_id FROM order_intents WHERE place_tool_call_id = %s",
            (place_tool_call_id,),
        ).fetchone()
        return row[0] if row is not None and isinstance(row[0], uuid.UUID) else None

    def _record_order(
        self,
        obs: BrokerOrderObservation,
        tool_call_id: uuid.UUID,
        intent_id: uuid.UUID | None,
    ) -> None:
        order_id = ledger_orders.record_broker_order(
            self.conn,
            run_id=self.run_id,
            account_scope_id=self.account_scope_id,
            broker_order_id=obs.broker_order_id,
        )
        if intent_id is not None:
            reviews = self.conn.execute(
                "SELECT e.source_tool_call_ids FROM order_intent_events e "
                "WHERE e.entity_id = %s AND e.event_type = 'reviewed' ORDER BY e.sequence",
                (intent_id,),
            ).fetchall()
            ledger_orders.link_intent(
                self.conn,
                order_id,
                intent_id,
                run_id=self.run_id,
                observed_at=obs.as_of,
                source_tool_call_ids=(tool_call_id,),
                review_tool_call_ids=[i for (ids,) in reviews for i in ids],  # type: ignore[attr-defined]
            )
        ledger_orders.observe_status(
            self.conn,
            order_id,
            run_id=self.run_id,
            status=obs.status,
            observed_at=obs.as_of,
            source_tool_call_id=tool_call_id,
            broker_status_raw=obs.state_raw,
            source_as_of=obs.as_of,
        )
        fill_events: list[uuid.UUID] = []
        if obs.executions:
            for ex in obs.executions:
                appended = ledger_orders.observe_fill(
                    self.conn,
                    order_id,
                    run_id=self.run_id,
                    quantity=ex.quantity,
                    price=ex.price,
                    broker_execution_id=ex.broker_execution_id,
                    observed_at=obs.as_of,
                    source_tool_call_id=tool_call_id,
                    executed_at=ex.executed_at,
                )
                fill_events.append(appended.event_id)
        elif obs.processed_quantity > 0:
            appended = ledger_orders.observe_fill(
                self.conn,
                order_id,
                run_id=self.run_id,
                quantity=obs.processed_quantity,
                price=None,
                broker_execution_id=None,
                observed_at=obs.as_of,
                source_tool_call_id=tool_call_id,
            )
            fill_events.append(appended.event_id)
        if obs.status in _TERMINAL:
            self._confirm_cancellations(order_id, obs, tool_call_id)
        if fill_events and self._owned(order_id):
            self._link_fills(order_id, obs, fill_events)

    def _owned(self, order_id: uuid.UUID) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM order_events WHERE entity_id = %s AND event_type = 'intent_linked' "
            "LIMIT 1",
            (order_id,),
        ).fetchone()
        return row is not None

    def _link_fills(
        self, order_id: uuid.UUID, obs: BrokerOrderObservation, fill_events: list[uuid.UUID]
    ) -> None:
        leg = obs.legs[0]
        if leg.occ_symbol is None:
            return
        side = leg.side_raw
        if side == OrderSide.SELL_TO_OPEN:
            position_id = self._lineage_of_order(order_id)
            if position_id is None:
                position_id = ledger_positions.open_position(
                    self.conn,
                    run_id=self.run_id,
                    account_scope_id=self.account_scope_id,
                    underlying=obs.underlying,
                    strategy=(
                        StrategyKind.CASH_SECURED_PUT
                        if leg.occ_symbol.right is OptionRight.PUT
                        else StrategyKind.COVERED_CALL
                    ),
                    instruments=[
                        PositionInstrument(
                            occ_symbol=leg.occ_symbol,
                            broker_instrument_id=leg.broker_instrument_id,
                            short_quantity=max(obs.processed_quantity, 1),
                        )
                    ],
                    observed_at=obs.as_of,
                    source_tool_call_ids=obs.source_tool_call_ids,
                )
            role = ledger_positions.FillRole.ENTRY
        elif side == OrderSide.BUY_TO_CLOSE:
            position_id = self._lineage_holding(leg.broker_instrument_id)
            if position_id is None:
                return
            role = ledger_positions.FillRole.CLOSE
        else:
            return
        for fill_event_id in fill_events:
            ledger_positions.link_fill(
                self.conn,
                position_id,
                run_id=self.run_id,
                fill_event_id=fill_event_id,
                role=role,
                observed_at=obs.as_of,
            )

    def _lineage_of_order(self, order_id: uuid.UUID) -> uuid.UUID | None:
        row = self.conn.execute(
            "SELECT entity_id FROM position_events WHERE order_id = %s "
            "AND event_type = 'fill_linked' ORDER BY recorded_at LIMIT 1",
            (order_id,),
        ).fetchone()
        return row[0] if row is not None and isinstance(row[0], uuid.UUID) else None

    def _active_entries(self, as_of: datetime) -> list[Any]:
        book = ledger_positions.position_book(self.conn, self.account_scope_id, as_of=as_of)
        return list(book.entries)

    def _lineage_holding(self, broker_instrument_id: str) -> uuid.UUID | None:
        """The one active lineage holding this contract, or None if none or ambiguous."""
        matches = [
            e.position_id
            for e in self._active_entries(self._now())
            if any(i.broker_instrument_id == broker_instrument_id for i in e.current_instruments)
        ]
        return matches[0] if len(matches) == 1 else None

    def _now(self) -> datetime:
        row = self.conn.execute("SELECT now()").fetchone()
        if row is None or not isinstance(row[0], datetime):
            raise LookupError("database clock unavailable")
        return row[0]

    def _record_cancel_request(
        self, req: CancelRequestObservation, tool_call_id: uuid.UUID
    ) -> None:
        order_id = ledger_orders.record_broker_order(
            self.conn,
            run_id=self.run_id,
            account_scope_id=self.account_scope_id,
            broker_order_id=req.broker_order_id,
        )
        ledger_orders.observe_cancellation(
            self.conn,
            order_id,
            run_id=self.run_id,
            cancel_tool_call_id=tool_call_id,
            status=CancellationStatus.PENDING if req.accepted else CancellationStatus.UNKNOWN,
            observed_at=req.as_of,
        )

    def _confirm_cancellations(
        self, order_id: uuid.UUID, obs: BrokerOrderObservation, read_tool_call_id: uuid.UUID
    ) -> None:
        """A terminal read confirms every earlier cancel request on this order."""
        rows = self.conn.execute(
            "SELECT DISTINCT payload->>'cancel_tool_call_id' FROM order_events "
            "WHERE entity_id = %s AND event_type = 'cancellation_observed'",
            (order_id,),
        ).fetchall()
        for (cancel_id,) in rows:
            if not isinstance(cancel_id, str) or uuid.UUID(cancel_id) == read_tool_call_id:
                continue
            ledger_orders.observe_cancellation(
                self.conn,
                order_id,
                run_id=self.run_id,
                cancel_tool_call_id=uuid.UUID(cancel_id),
                status=CancellationStatus.CONFIRMED,
                observed_at=obs.as_of,
                confirmation_tool_call_ids=(read_tool_call_id,),
            )

    def _resolve_unlinked_intents(self, mapped: MappedEvidence, tool_call_id: uuid.UUID) -> None:
        """A complete order read proves "no order" for an unlinked intent only when it lists
        no order created after the intent (minus the clock margin)."""
        created = [o.created_at for o in mapped.broker_orders]
        for record in ledger_orders.owned_unresolved_orders(self.conn, self.account_scope_id):
            intent = record.intent
            if intent is None or record.broker_order is not None:
                continue
            since = intent.requested_at - NO_ORDER_CLOCK_MARGIN
            if any(c >= since for c in created):
                continue
            ledger_orders.resolve_intent_no_order(
                self.conn,
                intent.intent_id,
                run_id=self.run_id,
                observed_at=mapped.open_orders[0].as_of,
                source_tool_call_ids=(tool_call_id,),
                detail="complete order read lists no order created since the intent",
            )

    def _reconcile_lineages(self, mapped: MappedEvidence, tool_call_id: uuid.UUID) -> None:
        """Close lineages a complete options read no longer lists; update changed quantities."""
        if mapped.pending_option_positions:
            pending = mapped.pending_option_positions[0]
            held = {r.broker_instrument_id: r.short_quantity for r in pending.rows}
            as_of = pending.as_of
        elif mapped.positions:
            held = {}
            as_of = mapped.positions[0].as_of
        else:
            return  # an incomplete read proves nothing
        for entry in self._active_entries(as_of):
            current = entry.current_instruments
            if not current:
                continue
            if all(i.broker_instrument_id not in held for i in current):
                ledger_positions.close_position(
                    self.conn,
                    entry.position_id,
                    run_id=self.run_id,
                    observed_at=as_of,
                    source_tool_call_ids=(tool_call_id,),
                )
                continue
            updated = [
                PositionInstrument(
                    occ_symbol=i.occ_symbol,
                    broker_instrument_id=i.broker_instrument_id,
                    short_quantity=held[i.broker_instrument_id],
                )
                for i in current
                if i.broker_instrument_id in held
            ]
            if [(i.broker_instrument_id, i.short_quantity) for i in updated] != [
                (i.broker_instrument_id, i.short_quantity) for i in current
            ]:
                ledger_positions.record_reconciliation(
                    self.conn,
                    entry.position_id,
                    run_id=self.run_id,
                    dedup_key=f"positions:{tool_call_id}",
                    detail="held quantity from a complete options-positions read",
                    observed_at=as_of,
                    source_tool_call_ids=(tool_call_id,),
                    current_instruments=updated,
                )
