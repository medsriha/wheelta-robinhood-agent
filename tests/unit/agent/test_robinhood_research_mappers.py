"""Research-read mappers (ADR-0042) against results recorded on 2026-09-29."""

import copy
import json
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, ValidationRequest
from wheelta_robinhood_agent.agent.model_view import model_view
from wheelta_robinhood_agent.agent.result_boundary import (
    BoundaryValidator,
    MappingRequest,
    _check_provenance,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.robinhood_mappers import (
    map_earnings_calendar,
    map_earnings_results,
    map_equity_analyst_ratings,
    map_equity_fundamentals,
    map_financials,
    map_option_chains,
    map_sec_filing_index,
)
from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.observability.redaction import Redactor

FIXTURES = Path(__file__).parents[2] / "fixtures" / "robinhood" / "results"
CALL = uuid.UUID("0190a0a0-0000-7000-8000-000000000042")
RETRIEVED = datetime(2026, 9, 29, 14, 20, tzinfo=UTC)


def _fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


def _data(name: str) -> Any:
    return copy.deepcopy(_fixture(name)["data"])


def _args(name: str) -> dict[str, Any]:
    args: dict[str, Any] = _fixture(name)["provenance"]["arguments"]
    return args


def _ids() -> Callable[[], uuid.UUID]:
    counter: Iterator[int] = iter(range(1, 10_000))
    return lambda: uuid.UUID(int=next(counter))


def _request(tool: str, payload: Any, **effective_input: Any) -> MappingRequest:
    return MappingRequest(
        tool_call_id=CALL,
        server="robinhood",
        tool=tool,
        effective_input=effective_input,
        payload={"data": payload, "guide": "tool guidance prose"},
        retrieved_at=RETRIEVED,
    )


def _fixture_request(tool: str, name: str) -> MappingRequest:
    return _request(tool, _data(name), **_args(name))


FIXTURE_CASES = [
    ("get_earnings_results", "get_earnings_results.LYFT.json"),
    ("get_earnings_calendar", "get_earnings_calendar.window_18d.json"),
    ("get_sec_filing_index", "get_sec_filing_index.NCLH_paged.json"),
    ("get_financials", "get_financials.DHT_LUV_NCLH.json"),
    ("get_equity_fundamentals", "get_equity_fundamentals.DHT_LUV_NCLH.json"),
    ("get_equity_fundamentals", "get_equity_fundamentals.EWZ_ETHA.json"),
    ("get_equity_analyst_ratings", "get_equity_analyst_ratings.DHT_LUV_NCLH.json"),
    ("get_option_chains", "get_option_chains.T.json"),
    ("get_option_chains", "get_option_chains.SPY.json"),
]


@pytest.mark.parametrize(("tool", "fixture"), FIXTURE_CASES)
def test_boundary_validates_recorded_results_as_citable_evidence(tool: str, fixture: str) -> None:
    """Through the real boundary: a text-block MCP result becomes a validated, cited envelope
    whose model view fits the proxy's delivery cap."""
    response = {
        "content": [{"type": "text", "text": json.dumps({"data": _data(fixture)})}],
        "isError": False,
    }
    outcome = BoundaryValidator(Redactor())(
        ValidationRequest(
            tool_call_id=CALL,
            server="robinhood",
            tool=tool,
            tier=ToolTier.R,
            effective_input=_args(fixture),
            tool_response=response,
            retrieved_at=RETRIEVED,
        )
    )
    envelope = outcome.envelope.model_dump(mode="json")
    assert outcome.envelope.kind is EnvelopeKind.VALIDATED
    assert envelope["data"]["evidence_ref"] == f"evidence:{CALL}"
    mapped = mapped_evidence_of(envelope)
    assert mapped is not None and mapped.evidence_ids()
    _check_provenance(mapped, CALL)
    assert len(json.dumps(model_view(envelope))) < 30_000


def test_earnings_results_keep_tentative_dates_and_unreported_eps() -> None:
    out = map_earnings_results(
        _fixture_request("get_earnings_results", "get_earnings_results.LYFT.json"), _ids()
    )
    assert len(out.earnings_reports) == 8
    upcoming = out.earnings_reports[-1]
    assert upcoming.symbol == "LYFT"
    assert (upcoming.fiscal_year, upcoming.fiscal_quarter) == (2026, 3)
    assert upcoming.report_date == date(2026, 11, 4)
    assert upcoming.timing == "pm" and not upcoming.verified
    assert upcoming.eps_actual is None and upcoming.eps_estimate == Decimal("0.210000")
    assert all(r.as_of == RETRIEVED for r in out.earnings_reports)
    assert any("tentative" in g for g in out.gaps)


def test_earnings_results_reject_rows_for_another_symbol() -> None:
    with pytest.raises(ValueError, match="another symbol"):
        map_earnings_results(
            _request(
                "get_earnings_results", _data("get_earnings_results.LYFT.json"), symbol="UBER"
            ),
            _ids(),
        )


def test_earnings_results_accept_lowercase_request_and_report_not_found() -> None:
    out = map_earnings_results(
        _request("get_earnings_results", {"results": [], "not_found": ["ZZZZ"]}, symbol=" zzzz "),
        _ids(),
    )
    assert out.earnings_reports == ()
    assert "get_earnings_results: no data for ZZZZ" in out.gaps


def test_earnings_calendar_keeps_null_timing() -> None:
    out = map_earnings_calendar(
        _fixture_request("get_earnings_calendar", "get_earnings_calendar.window_18d.json"),
        _ids(),
    )
    assert out.earnings_reports
    assert any(r.timing is None for r in out.earnings_reports)


def test_earnings_calendar_empty_window_is_a_gap() -> None:
    out = map_earnings_calendar(_request("get_earnings_calendar", {"results": []}), _ids())
    assert out.earnings_reports == ()
    assert out.gaps == ("get_earnings_calendar: no reports in the window",)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["report"].update(timing="midday"),
        lambda r: r["report"].update(date="2026-11-04T00:00:00Z"),
        lambda r: r["eps"].update(estimate=0.21),
        lambda r: r.update(quarter=5),
        lambda r: r["report"].update(verified="false"),
    ],
)
def test_earnings_unrecognized_values_raise(mutate: Callable[[dict[str, Any]], None]) -> None:
    data = _data("get_earnings_results.LYFT.json")
    mutate(data["results"][0])
    with pytest.raises(ValidationError):
        map_earnings_results(_request("get_earnings_results", data, symbol="LYFT"), _ids())


def test_sec_filing_index_lists_filings_and_flags_more_pages() -> None:
    out = map_sec_filing_index(
        _fixture_request("get_sec_filing_index", "get_sec_filing_index.NCLH_paged.json"), _ids()
    )
    assert out.sec_filings and all(f.symbol == "NCLH" for f in out.sec_filings)
    assert isinstance(out.sec_filings[0].date_filed, date)
    assert any("more filings exist" in g for g in out.gaps)


def test_sec_filing_index_empty_is_a_gap_and_symbol_must_match() -> None:
    out = map_sec_filing_index(
        _request("get_sec_filing_index", {"symbol": "LYFT", "filings": []}, symbol="LYFT"),
        _ids(),
    )
    assert out.gaps == ("get_sec_filing_index: no matching filings for LYFT",)
    with pytest.raises(ValueError, match="another symbol"):
        map_sec_filing_index(
            _request("get_sec_filing_index", {"symbol": "LYFT", "filings": []}, symbol="UBER"),
            _ids(),
        )


def test_financials_convert_percent_margin_and_keep_null_entries_as_gaps() -> None:
    out = map_financials(
        _fixture_request("get_financials", "get_financials.DHT_LUV_NCLH.json"), _ids()
    )
    assert "get_financials: no data for DHT" in out.gaps
    luv = [p for p in out.financial_periods if p.symbol == "LUV"]
    latest = luv[0]
    assert (latest.fiscal_year, latest.fiscal_quarter) == (2026, 2)
    assert latest.revenue_usd == Decimal("8432000000.000000")
    assert latest.gross_profit_usd is None
    assert latest.net_margin_ratio == Decimal("0.0276")
    # The ratio agrees with the reported amounts (net income / revenue).
    assert latest.net_income_usd is not None and latest.revenue_usd is not None
    ratio = latest.net_income_usd / latest.revenue_usd
    assert abs(ratio - latest.net_margin_ratio) < Decimal("0.0001")


def test_financials_must_align_with_requested_symbols() -> None:
    data = _data("get_financials.DHT_LUV_NCLH.json")
    with pytest.raises(ValueError, match="not aligned"):
        map_financials(_request("get_financials", data, symbols=["LUV", "NCLH"]), _ids())
    with pytest.raises(ValueError, match="another symbol"):
        map_financials(_request("get_financials", data, symbols=["DHT", "NCLH", "LUV"]), _ids())


def test_financials_annual_rows_have_no_quarter() -> None:
    row = {
        "fiscal_year": 2025,
        "fiscal_quarter": None,
        "period_end_date": "2025-12-31",
        "revenue": "100.000000",
        "gross_profit": None,
        "net_income": "10.000000",
        "net_margin": "10.000000",
    }
    entry = {"symbol": "LUV", "period": "annual", "financials": [row]}
    out = map_financials(_request("get_financials", {"results": [entry]}, symbols=["LUV"]), _ids())
    assert out.financial_periods[0].fiscal_quarter is None
    bad = {**entry, "period": "quarterly"}
    with pytest.raises(ValueError, match="fiscal_quarter"):
        map_financials(_request("get_financials", {"results": [bad]}, symbols=["LUV"]), _ids())


def test_fundamentals_keep_settled_fields_and_flag_dividend_dates() -> None:
    out = map_equity_fundamentals(
        _fixture_request("get_equity_fundamentals", "get_equity_fundamentals.EWZ_ETHA.json"),
        _ids(),
    )
    ewz, etha = out.fundamentals
    assert ewz.symbol == "EWZ" and ewz.market_date == date(2026, 9, 29)
    assert ewz.ex_dividend_date == date(2026, 12, 15)
    assert etha.pe_ratio is None and etha.ex_dividend_date is None
    dumped = ewz.model_dump()
    assert "dividend_yield" not in dumped and "dividend_per_share" not in dumped
    assert "open" not in dumped
    assert [g for g in out.gaps if "dividend dates" in g] == [
        "get_equity_fundamentals: EWZ dividend dates as reported; whether they describe the "
        "last or the next distribution is unverified"
    ]


def test_fundamentals_reject_unrequested_symbols() -> None:
    data = _data("get_equity_fundamentals.EWZ_ETHA.json")
    with pytest.raises(ValueError, match="not requested"):
        map_equity_fundamentals(_request("get_equity_fundamentals", data, symbols=["EWZ"]), _ids())


def test_analyst_ratings_optional_update_time_and_no_coverage_gap() -> None:
    data = _data("get_equity_analyst_ratings.DHT_LUV_NCLH.json")
    data["results"].append({"symbol": "LYFT", "ratings": None})
    data["results"].append(None)
    out = map_equity_analyst_ratings(
        _request(
            "get_equity_analyst_ratings", data, symbols=["DHT", "LUV", "NCLH", "LYFT", "ZZZZ"]
        ),
        _ids(),
    )
    dht, luv, _ = out.analyst_ratings
    assert dht.updated_at is None
    assert luv.updated_at == datetime(2026, 9, 28, 9, 23, 56, tzinfo=UTC)
    assert (luv.buy_ratings, luv.hold_ratings, luv.sell_ratings) == (12, 13, 3)
    assert luv.mean_price_target == Decimal("50.4500")
    assert out.gaps == (
        "get_equity_analyst_ratings: no analyst coverage for LYFT",
        "get_equity_analyst_ratings: no analyst coverage for ZZZZ",
    )


def test_option_chains_map_expirations_multiplier_and_ticks() -> None:
    out = map_option_chains(
        _fixture_request("get_option_chains", "get_option_chains.T.json"), _ids()
    )
    (chain,) = out.option_chains
    assert chain.symbol == "T" and chain.multiplier == 100
    assert chain.expiration_dates[0] == date(2026, 10, 2)
    assert (chain.above_tick, chain.below_tick, chain.tick_cutoff_price) == (
        Decimal("0.05"),
        Decimal("0.01"),
        Decimal("3.00"),
    )


def test_option_chains_skip_a_chain_with_a_cash_component() -> None:
    data = _data("get_option_chains.T.json")
    data["chains"][0]["cash_component"] = "12.5000"
    out = map_option_chains(_request("get_option_chains", data), _ids())
    assert out.option_chains == ()
    assert "cash component" in out.gaps[0]


def test_option_chains_reject_a_fractional_multiplier() -> None:
    data = _data("get_option_chains.T.json")
    data["chains"][0]["trade_value_multiplier"] = "100.5000"
    with pytest.raises(ValueError, match="multiplier"):
        map_option_chains(_request("get_option_chains", data), _ids())
