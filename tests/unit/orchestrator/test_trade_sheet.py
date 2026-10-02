"""ADR-0073: filled orders -> Google Sheet trade-log entries, and the Apps Script post."""

from datetime import UTC, datetime
from decimal import Decimal as D
from uuid import UUID

import httpx
from pydantic import SecretStr

from wheelta_robinhood_agent.agent.facts_tool import RunEvidence
from wheelta_robinhood_agent.domain.enums import AgentRole
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import (
    BrokerOrder,
    FillObservationKind,
    FillRecord,
    OrderIntent,
    OrderRecord,
)
from wheelta_robinhood_agent.integrations.notifications.trade_sheet import post_trade_fills
from wheelta_robinhood_agent.orchestrator.trade_sheet import sheet_fills

# 02:00 UTC on 2026-10-03 is 2026-10-02 in New York: the trade date is the ET date.
T0 = datetime(2026, 10, 3, 2, 0, tzinfo=UTC)


def record(n: int, occ: str, side: str, fills: tuple[tuple[int, str | None], ...]) -> OrderRecord:
    order_id = UUID(int=100 + n)
    return OrderRecord(
        intent=OrderIntent(
            intent_id=UUID(int=n),
            run_id=UUID(int=1),
            place_tool_call_id=UUID(int=50 + n),
            account_scope_id="acct:1234",
            occ_symbol=OccSymbol.parse(occ),
            broker_instrument_id=f"inst-{n}",
            side_raw=side,
            quantity=sum(q for q, _ in fills) or 1,
            order_type_raw="limit",
            time_in_force_raw="gfd",
            limit_price=D("1.00"),
            requested_at=T0,
        ),
        broker_order=BrokerOrder(
            order_id=order_id,
            account_scope_id="acct:1234",
            broker_order_id=f"ord-{n}",
            intent_id=UUID(int=n),
            first_observed_at=T0,
        ),
        fills=tuple(
            FillRecord(
                fill_id=UUID(int=1000 * n + i),
                order_id=order_id,
                kind=FillObservationKind.EXECUTION,
                broker_execution_id=f"exec-{n}-{i}",
                quantity=q,
                price=D(p) if p is not None else None,
                executed_at=T0,
                observed_at=T0,
                source_tool_call_id=UUID(int=9),
            )
            for i, (q, p) in enumerate(fills)
        ),
    )


NO_EVIDENCE = RunEvidence(())


def test_sell_run_maps_fills_and_skips_unfilled() -> None:
    entries = sheet_fills(
        (
            record(1, "OSCR  261016P00028000", "sell_to_open", ((1, "0.50"), (1, "0.60"))),
            record(2, "DHT   261016P00022000", "sell_to_open", ()),
        ),
        NO_EVIDENCE,
        AgentRole.SELL,
        {},
    )
    assert entries == [
        {
            "order_id": "ord-1",
            "side": "STO",
            "type": "CSP",
            "ticker": "OSCR",
            "trade_date": "2026-10-02",
            "expiry": "2026-10-16",
            "strike": "28",
            "contracts": 2,
            "price": "0.55",
            "stock_price": None,
            "iv": None,
            "roll": False,
            "open_order_ids": [],
        }
    ]


def test_close_run_marks_roll_and_puts_closes_first() -> None:
    entries = sheet_fills(
        (
            record(1, "XOM   261023C00160000", "sell_to_open", ((1, "1.20"),)),
            record(2, "XOM   261016C00160000", "buy_to_close", ((1, "0.30"),)),
            record(3, "GOOG  261016C00370000", "buy_to_close", ((1, None),)),
        ),
        NO_EVIDENCE,
        AgentRole.CLOSE,
        {UUID(int=102): ("ord-0",)},
    )
    assert [
        (e["order_id"], e["side"], e["roll"], e["price"], e["open_order_ids"]) for e in entries
    ] == [
        ("ord-2", "BTC", True, "0.30", ["ord-0"]),
        ("ord-3", "BTC", False, None, []),
        ("ord-1", "STO", True, "1.20", []),
    ]


def _post(handler: httpx.MockTransport) -> object:
    with httpx.Client(transport=handler) as client:
        return post_trade_fills(
            [{"order_id": "ord-1"}],
            client=client,
            url=SecretStr("https://x/exec"),
            timeout_seconds=1,
        )


def test_post_follows_apps_script_redirect_and_reads_statuses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "x":
            return httpx.Response(302, headers={"Location": "https://echo/out"})
        return httpx.Response(200, json={"ok": True, "results": [{"status": "written"}]})

    result = _post(httpx.MockTransport(handler))
    assert result.delivered and result.statuses == ("written",)  # type: ignore[attr-defined]


def test_post_reports_script_error_and_retries_transport_once() -> None:
    script = _post(
        httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": False, "error": "boom"}))
    )
    assert not script.delivered and script.error == "script_error: boom"  # type: ignore[attr-defined]

    calls = []

    def down(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("down")

    result = _post(httpx.MockTransport(down))
    assert not result.delivered and len(calls) == 2  # type: ignore[attr-defined]
