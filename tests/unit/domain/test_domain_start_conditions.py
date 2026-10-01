"""ADR-0057: the Buy-to-Close and Sell Options agents' start conditions (pure)."""

from decimal import Decimal

from wheelta_robinhood_agent.domain.enums import AgentRole
from wheelta_robinhood_agent.domain.start_conditions import (
    Shares,
    ShortRow,
    StartOutcome,
    coverable_symbols,
    evaluate_close_start,
    evaluate_sell_start,
)

MIN = Decimal("1000.00")


def _put(symbol: str = "AAPL", contracts: int = 1) -> ShortRow:
    return ShortRow(underlying=symbol, short_quantity=contracts, multiplier=100)


def test_close_starts_only_with_a_short_option() -> None:
    held = evaluate_close_start((_put(), _put("MSFT")))
    assert held.outcome is StartOutcome.MET and held.short_positions == 2
    assert held.role is AgentRole.CLOSE
    empty = evaluate_close_start(())
    assert empty.outcome is StartOutcome.NOT_MET and empty.short_positions == 0
    unknown = evaluate_close_start(None)
    assert unknown.outcome is StartOutcome.UNAVAILABLE and unknown.short_positions is None


def test_sell_starts_on_cash_at_or_above_the_minimum() -> None:
    for cash in (Decimal("1000.00"), Decimal("25000")):
        result = evaluate_sell_start(
            settled_cash_usd=cash, min_settled_cash_usd=MIN, shares=None, short_rows=None
        )
        assert result.outcome is StartOutcome.MET
        assert result.settled_cash_usd == cash and result.min_settled_cash_usd == MIN


def test_sell_below_the_minimum_needs_an_uncovered_lot() -> None:
    low = Decimal("999.99")
    lot = (Shares(symbol="AAPL", quantity=100),)
    met = evaluate_sell_start(
        settled_cash_usd=low, min_settled_cash_usd=MIN, shares=lot, short_rows=()
    )
    assert met.outcome is StartOutcome.MET and met.coverable_symbols == ("AAPL",)
    covered = evaluate_sell_start(
        settled_cash_usd=low, min_settled_cash_usd=MIN, shares=lot, short_rows=(_put(),)
    )
    assert covered.outcome is StartOutcome.NOT_MET and covered.coverable_symbols == ()
    none = evaluate_sell_start(
        settled_cash_usd=low, min_settled_cash_usd=MIN, shares=(), short_rows=()
    )
    assert none.outcome is StartOutcome.NOT_MET


def test_sell_fails_closed_without_the_facts_it_needs() -> None:
    low = Decimal("10")
    no_reads = evaluate_sell_start(
        settled_cash_usd=low, min_settled_cash_usd=MIN, shares=None, short_rows=()
    )
    assert no_reads.outcome is StartOutcome.UNAVAILABLE
    no_options = evaluate_sell_start(
        settled_cash_usd=low,
        min_settled_cash_usd=MIN,
        shares=(Shares(symbol="AAPL", quantity=100),),
        short_rows=None,
    )
    assert no_options.outcome is StartOutcome.UNAVAILABLE
    no_cash = evaluate_sell_start(
        settled_cash_usd=None, min_settled_cash_usd=MIN, shares=(), short_rows=()
    )
    assert no_cash.outcome is StartOutcome.UNAVAILABLE
    # Unknown cash does not hide an uncovered lot: the covered-call path decides alone.
    lot_only = evaluate_sell_start(
        settled_cash_usd=None,
        min_settled_cash_usd=MIN,
        shares=(Shares(symbol="AAPL", quantity=200),),
        short_rows=(_put(),),
    )
    assert lot_only.outcome is StartOutcome.MET and lot_only.coverable_symbols == ("AAPL",)


def test_every_short_row_reserves_its_shares() -> None:
    """Rows carry no call/put, so each counts against the symbol's shares (never over-counts)."""
    shares = (
        Shares(symbol="AAPL", quantity=250),
        Shares(symbol="MSFT", quantity=199),
        Shares(symbol="NVDA", quantity=99),
    )
    assert coverable_symbols(shares, (_put("AAPL"),)) == ("AAPL", "MSFT")
    assert coverable_symbols(shares, (_put("AAPL", 2), _put("MSFT"))) == ()
    assert coverable_symbols(shares, ()) == ("AAPL", "MSFT")


def test_only_met_and_unchecked_start_a_session() -> None:
    from wheelta_robinhood_agent.domain.start_conditions import STARTS_SESSION, unchecked_start

    assert STARTS_SESSION == {StartOutcome.MET, StartOutcome.UNCHECKED}
    condition = unchecked_start(AgentRole.SELL)
    assert condition.outcome is StartOutcome.UNCHECKED and condition.role is AgentRole.SELL
    assert "served directly" in condition.reason
