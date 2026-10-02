"""The rows a run's filled orders add to the owner's Google Sheet trade log (ADR-0073).

One entry per broker order with at least one filled contract, from the ledger's order records
and this run's validated evidence only: the fill price is the broker's (contract-weighted
over executions, else the latest cumulative average), the stock price and IV are the latest
underlying and option quotes this run recorded, or None. Nothing is estimated. In a
Buy-to-Close run a sell-to-open fill is a roll's replacement (ADR-0057), and a buy-to-close
fill on the same underlying and right is the rolled leg.

Informational only: the sheet is a convenience copy, never a record the agent reads.
"""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from uuid import UUID
from zoneinfo import ZoneInfo

from wheelta_robinhood_agent.agent.facts_tool import RunEvidence
from wheelta_robinhood_agent.domain.enums import AgentRole, OptionRight, OrderSide
from wheelta_robinhood_agent.domain.orders import FillObservationKind, FillRecord, OrderRecord

_NY = ZoneInfo("America/New_York")


def average_fill_price(fills: Sequence[FillRecord]) -> Decimal | None:
    """Contract-weighted price of executions (deduplicated by execution ID), else the latest
    cumulative observation's price; None when any execution lacks a price."""
    executions = {
        f.broker_execution_id: f for f in fills if f.kind is FillObservationKind.EXECUTION
    }
    if executions:
        if any(f.price is None for f in executions.values()):
            return None
        quantity = sum(f.quantity for f in executions.values())
        notional = sum(
            ((f.price or Decimal(0)) * f.quantity for f in executions.values()), Decimal(0)
        )
        return notional / quantity if quantity else None
    cumulative = [f for f in fills if f.kind is FillObservationKind.CUMULATIVE]
    return max(cumulative, key=lambda f: f.observed_at).price if cumulative else None


def sheet_fills(
    records: Sequence[OrderRecord],
    evidence: RunEvidence,
    role: AgentRole,
    opened_by: Mapping[UUID, tuple[str, ...]],
) -> list[dict[str, object]]:
    """The sheet entries for a run's filled orders, buy-to-close first (a roll's replacement
    takes the cycle of the leg it rolled). `opened_by` maps a closing order to the broker
    order IDs that opened its lineage: the sheet closes the open row of their Cycle #."""
    entries: list[dict[str, object]] = []
    for record in records:
        intent, order = record.intent, record.broker_order
        filled = record.filled_quantity
        if intent is None or order is None or intent.occ_symbol is None or not filled:
            continue
        if intent.side is None:
            continue
        occ = intent.occ_symbol
        instrument = (
            evidence.instrument(intent.broker_instrument_id)
            if intent.broker_instrument_id
            else None
        )
        ticker = instrument.underlying if instrument is not None else occ.root
        quote = (
            evidence.option_quote(intent.broker_instrument_id)
            if intent.broker_instrument_id
            else None
        )
        underlying = evidence.underlying_quote(ticker)
        first = min(record.fills, key=lambda f: f.executed_at or f.observed_at)
        price = average_fill_price(record.fills)
        entries.append(
            {
                "order_id": order.broker_order_id,
                "side": "STO" if intent.side is OrderSide.SELL_TO_OPEN else "BTC",
                "type": "CSP" if occ.right is OptionRight.PUT else "CC",
                "ticker": ticker,
                "trade_date": (first.executed_at or first.observed_at)
                .astimezone(_NY)
                .date()
                .isoformat(),
                "expiry": occ.expiration.isoformat(),
                "strike": str(occ.strike),
                "contracts": filled,
                "price": str(price) if price is not None else None,
                "stock_price": str(underlying.price) if underlying is not None else None,
                "iv": str(quote.implied_volatility_ratio)
                if quote is not None and quote.implied_volatility_ratio is not None
                else None,
                "roll": False,
                "open_order_ids": list(opened_by.get(order.order_id, ()))
                if intent.side is OrderSide.BUY_TO_CLOSE
                else [],
            }
        )
    if role is AgentRole.CLOSE:
        opened = {(e["ticker"], e["type"]) for e in entries if e["side"] == "STO"}
        for entry in entries:
            entry["roll"] = entry["side"] == "STO" or (entry["ticker"], entry["type"]) in opened
    entries.sort(key=lambda e: e["side"] != "BTC")
    return entries
