"""ADR-0057: the trusted reads behind each agent's start condition (`agent/start_probe.py`).

Payload shapes follow the captured results in tests/fixtures/robinhood/results.
"""

import asyncio
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from wheelta_robinhood_agent.agent.simulated_broker import SimulatedState
from wheelta_robinhood_agent.agent.start_probe import overlay_simulated, probe_start_condition
from wheelta_robinhood_agent.domain.enums import AgentRole
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.start_conditions import ShortRow, StartOutcome
from wheelta_robinhood_agent.integrations.mcp_upstream import (
    UpstreamResult,
    UpstreamTool,
    UpstreamUnavailable,
)

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=UTC)
ACCOUNT = SecretStr("123456789")
MIN = Decimal("1000.00")
RESULTS = Path(__file__).resolve().parents[2] / "fixtures/robinhood/results"


def _fixture(name: str) -> dict[str, Any]:
    data = json.loads((RESULTS / name).read_text())["data"]
    assert isinstance(data, dict)
    return data


def _short(option_id: str, symbol: str, quantity: str = "1") -> dict[str, Any]:
    return {
        "option_id": option_id,
        "chain_symbol": symbol,
        "type": "short",
        "quantity": quantity,
        "trade_value_multiplier": "100.0000",
    }


class Upstream:
    """Answers the three probe reads with captured-shape payloads; anything else fails."""

    server = "robinhood"
    tools = (UpstreamTool("get_option_positions", None, {}),)

    def __init__(
        self,
        *,
        cash: str = "20000",
        shares: list[dict[str, Any]] | None = None,
        shorts: list[dict[str, Any]] | None = None,
        fail: str | None = None,
    ) -> None:
        portfolio = _fixture("get_portfolio.funded_no_positions.json")
        portfolio["cash"] = cash
        self.payloads: dict[str, dict[str, Any]] = {
            "get_portfolio": portfolio,
            "get_equity_positions": {"positions": shares or []},
            "get_option_positions": {"positions": shorts or []},
        }
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_seconds: float
    ) -> UpstreamResult:
        self.calls.append((name, dict(arguments)))
        assert name in self.payloads, name
        if name == self.fail:
            raise UpstreamUnavailable(f"{name}: down")
        text = json.dumps({"data": self.payloads[name]})
        return UpstreamResult({"content": [{"type": "text", "text": text}]}, len(text))


def probe(role: AgentRole, upstream: Upstream, **kw: Any) -> Any:
    return asyncio.run(
        probe_start_condition(
            role,
            upstream,
            account_number=ACCOUNT,
            min_settled_cash_usd=MIN,
            clock=lambda: NOW,
            timeout_seconds=5,
            **kw,
        )
    )


def test_close_reads_option_positions_with_the_configured_account() -> None:
    upstream = Upstream(shorts=[_short(str(uuid.UUID(int=1)), "AAPL")])
    result = probe(AgentRole.CLOSE, upstream)
    assert result.outcome is StartOutcome.MET and result.short_positions == 1
    assert upstream.calls == [("get_option_positions", {"account_number": "123456789"})]
    assert "123456789" not in result.model_dump_json()


def test_close_without_shorts_is_not_met() -> None:
    assert probe(AgentRole.CLOSE, Upstream()).outcome is StartOutcome.NOT_MET


def test_a_failed_read_is_unavailable_and_named() -> None:
    result = probe(AgentRole.CLOSE, Upstream(fail="get_option_positions"))
    assert result.outcome is StartOutcome.UNAVAILABLE
    assert "get_option_positions: UpstreamUnavailable" in result.reason


def test_sell_with_enough_cash_reads_only_the_portfolio() -> None:
    upstream = Upstream(cash="1000")
    result = probe(AgentRole.SELL, upstream)
    assert result.outcome is StartOutcome.MET
    assert result.settled_cash_usd == Decimal("1000")
    assert [name for name, _ in upstream.calls] == ["get_portfolio"]


def test_sell_below_the_minimum_falls_back_to_uncovered_shares() -> None:
    lot = [{"symbol": "AAPL", "quantity": "150.0000", "type": "long"}]
    upstream = Upstream(cash="999.99", shares=lot)
    result = probe(AgentRole.SELL, upstream)
    assert result.outcome is StartOutcome.MET and result.coverable_symbols == ("AAPL",)
    assert [name for name, _ in upstream.calls] == [
        "get_portfolio",
        "get_equity_positions",
        "get_option_positions",
    ]
    covered = Upstream(cash="999.99", shares=lot, shorts=[_short(str(uuid.UUID(int=2)), "AAPL")])
    assert probe(AgentRole.SELL, covered).outcome is StartOutcome.NOT_MET


def test_sell_without_cash_or_positions_reads_is_unavailable() -> None:
    result = probe(AgentRole.SELL, Upstream(cash="10", fail="get_equity_positions"))
    assert result.outcome is StartOutcome.UNAVAILABLE
    assert "get_equity_positions" in result.reason


def test_simulated_fills_overlay_the_option_rows() -> None:
    """A dry-run tick: the close agent's simulated buy-to-close removes the row it closed."""
    held = str(uuid.UUID(int=3))
    state = SimulatedState()
    state.short_delta[held] = -1
    upstream = Upstream(shorts=[_short(held, "AAPL")])
    assert probe(AgentRole.CLOSE, upstream, simulated=state).outcome is StartOutcome.NOT_MET
    assert probe(AgentRole.CLOSE, upstream).outcome is StartOutcome.MET


def test_overlay_adds_a_simulated_open_and_fails_closed_on_an_unknown_contract() -> None:
    state = SimulatedState()
    state.short_delta["new"] = 2
    assert overlay_simulated((), state) is None  # no instrument known for "new"
    state.filled_instruments["new"] = OptionInstrument(
        evidence_id=uuid.UUID(int=9),
        as_of=NOW,
        source_tool_call_ids=(uuid.UUID(int=8),),
        occ_symbol=OccSymbol.parse("MSFT  261016P00400000"),
        broker_instrument_id="new",
        underlying="MSFT",
        multiplier=100,
    )
    assert overlay_simulated((), state) == (
        ShortRow(underlying="MSFT", short_quantity=2, multiplier=100),
    )
    assert overlay_simulated(None, state) is None


def _debit(premium: Any, state: str = "filled") -> dict[str, Any]:
    return {"direction": "debit", "state": state, "processed_premium": premium}


def test_simulated_debits_reduce_settled_cash_in_a_dry_run() -> None:
    """The close agent's simulated buy-to-close spent cash the broker read cannot show."""
    state = SimulatedState()
    state.orders["btc-1"] = _debit("150.00")
    state.orders["sto-1"] = {"direction": "credit", "state": "filled", "processed_premium": "90"}
    state.orders["working"] = _debit("999", state="queued")
    result = probe(AgentRole.SELL, Upstream(cash="1100"), simulated=state)
    assert result.settled_cash_usd == Decimal("950.00")  # below the minimum after the debit
    assert result.outcome is StartOutcome.NOT_MET
    assert probe(AgentRole.SELL, Upstream(cash="1100")).outcome is StartOutcome.MET


def test_an_unreadable_simulated_debit_makes_cash_unknown() -> None:
    state = SimulatedState()
    state.orders["btc-1"] = _debit(None)
    result = probe(AgentRole.SELL, Upstream(cash="5000"), simulated=state)
    # Unknown cash and no complete shares read: fail closed, never a guess.
    assert result.settled_cash_usd is None and result.outcome is StartOutcome.UNAVAILABLE
    assert "simulated debits: ValueError" in result.reason


def test_without_a_trusted_upstream_only_a_proposal_only_dry_run_starts() -> None:
    """ADR-0019 direct Robinhood: no trusted channel. A dry run without order tools starts
    unchecked; any venue that places orders fails closed."""
    from types import SimpleNamespace
    from typing import cast

    from wheelta_robinhood_agent.agent.session import SessionDeps, _check_start_condition
    from wheelta_robinhood_agent.domain.enums import OrderVenue

    for venue, expected in (
        (OrderVenue.NONE, StartOutcome.UNCHECKED),
        (OrderVenue.SIMULATED, StartOutcome.UNAVAILABLE),
        (OrderVenue.BROKER, StartOutcome.UNAVAILABLE),
    ):
        deps = cast(
            SessionDeps,
            SimpleNamespace(role=AgentRole.CLOSE, plan=SimpleNamespace(order_venue=venue)),
        )
        result = asyncio.run(_check_start_condition(deps, None))
        assert result.outcome is expected, venue
