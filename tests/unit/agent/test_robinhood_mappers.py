"""Verified Robinhood result mappers against the real captured fixtures (ADR-0017)."""

import copy
import json
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, ValidationRequest
from wheelta_robinhood_agent.agent.result_boundary import (
    VERIFIED_MAPPERS,
    BoundaryValidator,
    MappedEvidence,
    MappingRequest,
    _check_provenance,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.robinhood_mappers import (
    map_equity_positions,
    map_equity_quotes,
    map_option_instruments,
    map_option_orders,
    map_option_positions,
    map_option_quotes,
    map_portfolio,
    parse_broker_timestamp,
)
from wheelta_robinhood_agent.domain.enums import DataQuality, OptionRight, ToolTier
from wheelta_robinhood_agent.observability.redaction import Redactor

FIXTURES = Path(__file__).parents[2] / "fixtures" / "robinhood" / "results"
CALL = uuid.UUID("0190a0a0-0000-7000-8000-000000000001")
RETRIEVED = datetime(2026, 9, 26, 0, 0, 5, tzinfo=UTC)
INSTRUMENT_ID = "d17decae-92f6-430e-b4c0-3772e5dd27ab"


def _fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


def _data(name: str) -> Any:
    return copy.deepcopy(_fixture(name)["data"])


def _ids() -> Callable[[], uuid.UUID]:
    counter: Iterator[int] = iter(range(1, 1_000))
    return lambda: uuid.UUID(int=next(counter))


def _request(tool: str, payload: Any, **effective_input: Any) -> MappingRequest:
    return MappingRequest(
        tool_call_id=CALL,
        server="robinhood",
        tool=tool,
        effective_input=effective_input,
        payload=payload,
        retrieved_at=RETRIEVED,
    )


def _wrapped(payload: Any) -> dict[str, Any]:
    return {"data": payload, "guide": "tool guidance prose"}


def test_verified_mappers_are_exactly_the_mapped_tools() -> None:
    assert set(VERIFIED_MAPPERS) == {
        ("robinhood", "get_equity_quotes"),
        ("robinhood", "get_option_instruments"),
        ("robinhood", "get_option_quotes"),
        ("robinhood", "get_portfolio"),
        ("robinhood", "get_option_orders"),
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-09-25T19:59:59.969143521Z", datetime(2026, 9, 25, 19, 59, 59, 969143, tzinfo=UTC)),
        # 999999999 ns truncates to 999999 us; rounding would carry into the next second.
        ("2026-09-25T19:59:59.999999999Z", datetime(2026, 9, 25, 19, 59, 59, 999999, tzinfo=UTC)),
        ("2026-10-16T19:45:00+00:00", datetime(2026, 10, 16, 19, 45, tzinfo=UTC)),
        ("2026-10-16T15:45:00.5-04:00", datetime(2026, 10, 16, 19, 45, 0, 500000, tzinfo=UTC)),
    ],
)
def test_broker_timestamps_truncate_to_microseconds(raw: str, expected: datetime) -> None:
    assert parse_broker_timestamp(raw) == expected


@pytest.mark.parametrize(
    "raw", ["2026-09-25T19:59:59", "2026-09-25 19:59:59Z", "2026-09-25T19:59:59.1234567890Z", 1]
)
def test_broker_timestamps_reject_other_forms(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_broker_timestamp(raw)


# ------------------------------------------------------------------------ get_equity_quotes


@pytest.mark.parametrize("wrap", [False, True])
def test_equity_quote_fixture(wrap: bool) -> None:
    data = _data("get_equity_quotes.SPY.json")
    out = map_equity_quotes(_request("get_equity_quotes", _wrapped(data) if wrap else data), _ids())
    (quote,) = out.underlying_quotes
    assert quote.symbol == "SPY"
    assert quote.price == Decimal("771.300000")
    assert quote.as_of == datetime(2026, 9, 25, 19, 59, 59, 969143, tzinfo=UTC)
    assert quote.source_tool_call_ids == (CALL,)
    assert out.gaps == ()
    _check_provenance(out, CALL)


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("state", "inactive", "not active"),
        ("has_traded", False, "has not traded"),
        ("last_trade_price", "0.000000", "no positive last trade price"),
        ("last_trade_price", None, "no positive last trade price"),
        ("venue_last_trade_time", None, "no last trade time"),
    ],
)
def test_equity_quote_unusable_becomes_gap(field: str, value: object, detail: str) -> None:
    data = _data("get_equity_quotes.SPY.json")
    data["results"][0]["quote"][field] = value
    out = map_equity_quotes(_request("get_equity_quotes", data), _ids())
    assert out.underlying_quotes == ()
    assert len(out.gaps) == 1 and detail in out.gaps[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [("last_trade_price", 771.3), ("bid_price", "abc"), ("symbol", "spy"), ("has_traded", "yes")],
)
def test_equity_quote_schema_mismatch_raises(field: str, value: object) -> None:
    data = _data("get_equity_quotes.SPY.json")
    data["results"][0]["quote"][field] = value
    with pytest.raises(ValidationError):
        map_equity_quotes(_request("get_equity_quotes", data), _ids())


def test_equity_quote_duplicate_symbol_raises() -> None:
    data = _data("get_equity_quotes.SPY.json")
    data["results"].append(copy.deepcopy(data["results"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        map_equity_quotes(_request("get_equity_quotes", data), _ids())


def test_equity_quote_empty_results_is_a_gap() -> None:
    out = map_equity_quotes(_request("get_equity_quotes", {"results": []}), _ids())
    assert out.underlying_quotes == () and out.gaps


# ------------------------------------------------------------------- get_option_instruments


def test_option_instrument_fixture() -> None:
    data = _data("get_option_instruments.SPY_20261016_P740.json")
    out = map_option_instruments(_request("get_option_instruments", _wrapped(data)), _ids())
    (inst,) = out.instruments
    assert str(inst.occ_symbol) == "SPY   261016P00740000"
    assert inst.occ_symbol.right is OptionRight.PUT
    assert inst.occ_symbol.strike == Decimal("740")
    assert inst.broker_instrument_id == INSTRUMENT_ID
    assert inst.underlying == "SPY"
    assert inst.multiplier == 100
    assert inst.as_of == RETRIEVED
    assert out.gaps == ()
    _check_provenance(out, CALL)


@pytest.mark.parametrize(
    ("field", "value"),
    [("state", "inactive"), ("tradability", "position_closing_only"), ("underlying_type", "etf")],
)
def test_option_instrument_unusable_becomes_gap(field: str, value: str) -> None:
    data = _data("get_option_instruments.SPY_20261016_P740.json")
    data["instruments"][0][field] = value
    out = map_option_instruments(_request("get_option_instruments", data), _ids())
    assert out.instruments == () and len(out.gaps) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trade_value_multiplier", "100.5000"),
        ("trade_value_multiplier", "0.0000"),
        ("strike_price", 740),
        ("strike_price", "740.00001"),
        ("type", "straddle"),
        ("chain_symbol", "spy"),
        ("expiration_date", "2026-10-16T00:00:00Z"),
        ("id", "not-a-uuid"),
        ("min_ticks", {"above_tick": "0", "below_tick": "0.01", "cutoff_price": "0.00"}),
    ],
)
def test_option_instrument_schema_mismatch_raises(field: str, value: object) -> None:
    data = _data("get_option_instruments.SPY_20261016_P740.json")
    data["instruments"][0][field] = value
    with pytest.raises(ValueError):
        map_option_instruments(_request("get_option_instruments", data), _ids())


def test_option_instrument_duplicate_and_empty() -> None:
    data = _data("get_option_instruments.SPY_20261016_P740.json")
    data["instruments"].append(copy.deepcopy(data["instruments"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        map_option_instruments(_request("get_option_instruments", data), _ids())
    out = map_option_instruments(_request("get_option_instruments", {"instruments": []}), _ids())
    assert out.instruments == () and out.gaps


# ------------------------------------------------------------------------ get_option_quotes


def test_option_quote_fixture() -> None:
    data = _data("get_option_quotes.SPY_20261016_P740.json")
    out = map_option_quotes(_request("get_option_quotes", _wrapped(data)), _ids())
    (q,) = out.option_quotes
    assert q.broker_instrument_id == INSTRUMENT_ID
    assert (q.bid, q.ask, q.mark) == (Decimal("1.79"), Decimal("1.80"), Decimal("1.795"))
    assert q.delta == Decimal("-0.122280")
    assert q.gamma == Decimal("0.006899")
    assert q.theta == Decimal("-0.139924")
    assert q.vega == Decimal("0.364017")
    assert q.implied_volatility_ratio == Decimal("0.163391")
    assert (q.open_interest, q.volume) == (45801, 9439)
    assert q.as_of == datetime(2026, 9, 25, 20, 14, 59, 909071, tzinfo=UTC)
    dumped = json.dumps(q.model_dump(mode="json"))
    assert "fill_rate" not in dumped and "chance_of_profit" not in dumped
    assert out.gaps == ()
    _check_provenance(out, CALL)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid_price", "0.000000"),
        ("ask_price", "0.000000"),
        ("bid_price", None),
        ("ask_price", None),
        ("updated_at", None),
    ],
)
def test_option_quote_zero_or_missing_becomes_gap(field: str, value: object) -> None:
    data = _data("get_option_quotes.SPY_20261016_P740.json")
    data["results"][0]["quote"][field] = value
    out = map_option_quotes(_request("get_option_quotes", data), _ids())
    assert out.option_quotes == ()
    assert len(out.gaps) == 1 and INSTRUMENT_ID in out.gaps[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid_price", "1.900000"),  # bid > ask
        ("mark_price", "2.000000"),  # mark outside the spread
        ("delta", "-1.500000"),
        ("implied_volatility", "-0.1"),
        ("open_interest", "45801"),
        ("delta", -0.12228),
    ],
)
def test_option_quote_schema_or_sanity_mismatch_raises(field: str, value: object) -> None:
    data = _data("get_option_quotes.SPY_20261016_P740.json")
    data["results"][0]["quote"][field] = value
    with pytest.raises(ValueError):
        map_option_quotes(_request("get_option_quotes", data), _ids())


def test_option_quote_duplicate_and_empty() -> None:
    data = _data("get_option_quotes.SPY_20261016_P740.json")
    data["results"].append(copy.deepcopy(data["results"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        map_option_quotes(_request("get_option_quotes", data), _ids())
    out = map_option_quotes(_request("get_option_quotes", {"results": []}), _ids())
    assert out.option_quotes == () and out.gaps


# ---------------------------------------------------------------------------- get_portfolio


def test_portfolio_fixture_keeps_unverified_cash_as_gaps() -> None:
    data = _data("get_portfolio.empty_account.json")
    out = map_portfolio(
        _request("get_portfolio", _wrapped(data), account_number="****1234"), _ids()
    )
    (snap,) = out.account_snapshots
    assert snap.account_ref == "****1234"
    assert snap.account_value_usd == Decimal("0")
    assert snap.available_settled_cash_usd is None
    assert snap.csp_reserved_cash_usd is None
    assert snap.csp_cash_base_usd is None
    assert snap.agentic_verified is False
    assert snap.quality is DataQuality.MISSING
    assert snap.as_of == snap.retrieved_at == RETRIEVED
    assert {g.field for g in snap.gaps} == {
        "available_settled_cash_usd",
        "csp_reserved_cash_usd",
        "csp_cash_base_usd",
        "agentic_verified",
    }
    assert len(out.gaps) == 4
    _check_provenance(out, CALL)


def test_portfolio_is_agentic_verified_only_after_the_trusted_check() -> None:
    """`account_eligible` comes from the session's get_accounts check, never the payload."""
    data = _data("get_portfolio.empty_account.json")
    request = _request("get_portfolio", _wrapped(data), account_number="****1234")
    out = map_portfolio(request.model_copy(update={"account_eligible": True}), _ids())
    (snap,) = out.account_snapshots
    assert snap.agentic_verified is True
    assert "agentic_verified" not in {g.field for g in snap.gaps}
    assert snap.quality is DataQuality.MISSING  # cash fields are still unverified
    assert len(out.gaps) == 3


def test_portfolio_does_not_use_buying_power_as_cash() -> None:
    data = _data("get_portfolio.empty_account.json")
    data["buying_power"]["buying_power"] = "5000.0000"
    data["total_value"] = "5000.00"
    out = map_portfolio(_request("get_portfolio", data, account_number="****1234"), _ids())
    (snap,) = out.account_snapshots
    assert snap.account_value_usd == Decimal("5000.00")
    assert snap.available_settled_cash_usd is None


@pytest.mark.parametrize(
    ("mutate", "effective_input"),
    [
        (lambda d: d.update(currency="EUR"), {"account_number": "****1234"}),
        (lambda d: d.update(total_value=0.0), {"account_number": "****1234"}),
        (lambda d: d.pop("buying_power"), {"account_number": "****1234"}),
        (lambda d: None, {}),
        (lambda d: None, {"account_number": "123456789"}),  # unredacted number
    ],
)
def test_portfolio_mismatch_raises(
    mutate: Callable[[dict[str, Any]], object], effective_input: dict[str, Any]
) -> None:
    data = _data("get_portfolio.empty_account.json")
    mutate(data)
    with pytest.raises(ValueError):
        map_portfolio(_request("get_portfolio", data, **effective_input), _ids())


# ----------------------------------------------------------------------- empty-only lists


def test_empty_option_orders_map_to_empty_read() -> None:
    data = _data("get_option_orders.empty_account.json")
    out = map_option_orders(_request("get_option_orders", _wrapped(data)), _ids())
    (read,) = out.open_orders
    assert read.orders == () and read.as_of == RETRIEVED
    _check_provenance(out, CALL)


@pytest.mark.parametrize(
    "mapper", [map_option_positions, map_equity_positions], ids=["option", "equity"]
)
@pytest.mark.parametrize(
    "fixture",
    ["get_option_positions.empty_account.json", "get_equity_positions.empty_account.json"],
)
def test_empty_positions_map_to_empty_read(
    mapper: Callable[[MappingRequest, Callable[[], uuid.UUID]], MappedEvidence], fixture: str
) -> None:
    out = mapper(_request("positions", _data(fixture)), _ids())
    (read,) = out.positions
    assert read.share_holdings == () and read.short_options == ()
    _check_provenance(out, CALL)


@pytest.mark.parametrize(
    ("mapper", "payload"),
    [
        (map_option_positions, {"positions": [{"quantity": "1.0000"}]}),
        (map_equity_positions, {"positions": [{"symbol": "SPY"}]}),
        (map_option_orders, {"orders": [{"id": "x"}]}),
        (map_option_orders, {"orders": [], "next": "cursor"}),
        (map_option_positions, {"results": []}),
    ],
)
def test_unverified_list_shapes_raise(
    mapper: Callable[[MappingRequest, Callable[[], uuid.UUID]], MappedEvidence], payload: Any
) -> None:
    with pytest.raises(ValueError):
        mapper(_request("list", payload), _ids())


# -------------------------------------------------------------------- through the boundary


def _validate(tool: str, payload: Any, **effective_input: Any) -> Any:
    text = json.dumps(_wrapped(payload))
    return BoundaryValidator(redactor=Redactor())(
        ValidationRequest(
            tool_call_id=CALL,
            server="robinhood",
            tool=tool,
            tier=ToolTier.R,
            effective_input=effective_input,
            tool_response={"content": [{"type": "text", "text": text}]},
            retrieved_at=RETRIEVED,
        )
    ).envelope


@pytest.mark.parametrize(
    ("tool", "fixture", "effective_input"),
    [
        ("get_equity_quotes", "get_equity_quotes.SPY.json", {}),
        ("get_option_instruments", "get_option_instruments.SPY_20261016_P740.json", {}),
        ("get_option_quotes", "get_option_quotes.SPY_20261016_P740.json", {}),
        ("get_portfolio", "get_portfolio.empty_account.json", {"account_number": "****1234"}),
        ("get_option_orders", "get_option_orders.empty_account.json", {}),
    ],
)
def test_boundary_validates_real_fixtures(
    tool: str, fixture: str, effective_input: dict[str, Any]
) -> None:
    envelope = _validate(tool, _data(fixture), **effective_input)
    assert envelope.kind is EnvelopeKind.VALIDATED
    mapped = mapped_evidence_of(envelope.model_dump(mode="json"))
    assert mapped is not None and mapped.evidence_ids()


@pytest.mark.parametrize(
    ("tool", "fixture"),
    [
        ("get_option_chains", "get_option_chains.SPY.json"),
        ("get_option_positions", "get_option_positions.empty_account.json"),
        ("get_equity_positions", "get_equity_positions.empty_account.json"),
    ],
)
def test_boundary_keeps_unregistered_tools_missing(tool: str, fixture: str) -> None:
    envelope = _validate(tool, _data(fixture), account_number="****1234")
    assert envelope.kind is EnvelopeKind.MISSING


def test_boundary_turns_non_empty_orders_into_missing() -> None:
    envelope = _validate("get_option_orders", {"orders": [{"id": "x"}]})
    assert envelope.kind is EnvelopeKind.MISSING
    assert envelope.data is None
