"""Verified result mappers for Robinhood read tools (robinhood-trading 1.6.0, ADR-0017).

Every mapper here is derived from a real captured response in
`tests/fixtures/robinhood/results/` and nothing else (CLAUDE.md §2.3, §9, §12, §13). A mapper
parses the payload with pydantic, normalizes it (Decimal from decimal strings, UTC datetimes,
OCC symbols), cites the request's `tool_call_id`, issues every evidence ID via `new_id`, and
raises on any shape it does not recognize so the boundary delivers a `missing` envelope.

Conventions shared by all mappers:

- **Envelope.** The tool's JSON text is `{"data": ..., "guide": ...}`. The fixtures dropped
  `guide`, so `_unwrap` accepts both that wrapper and the bare `data` value.
- **Decimals.** Money, prices, Greeks, and ratios arrive as decimal strings. They are parsed
  exactly (no float, no rounding) and keep the broker's precision.
- **Timestamps.** Robinhood sends RFC 3339 times with up to nine fractional digits
  (nanoseconds, e.g. `2026-09-25T19:59:59.969143521Z`). Python datetimes hold microseconds,
  so the fraction is **truncated** to six digits, never rounded: the stored time is never
  later than the broker's.
- **No source time.** Instrument metadata and the portfolio carry no timestamp, so their
  `as_of` is the boundary's `retrieved_at` (the time this run observed them).
- **Skipped rows.** A row that is well formed but unusable (inactive, untradable, zero
  bid/ask) produces no evidence and a named gap instead.

Tools deliberately NOT registered:

- `get_option_chains`: the captured result carries chain identity, expiration dates, the chain
  multiplier, and ticks. No fact computation or run loader reads a chain: the multiplier and
  identity come from `get_option_instruments` (per instrument), and quotes from
  `get_option_quotes`. `MappedEvidence` has no field for expiration lists, and a `validated`
  envelope with no evidence would claim a check it did not do, so the result stays `missing`.
- `get_option_positions` / `get_equity_positions`: `map_option_positions` and
  `map_equity_positions` exist but are not registered. `PositionsRead` is one complete read of
  both share holdings and short options ("absence of a holding means zero"); either tool alone
  attests only one half, so its read would assert zero holdings of the other kind.
"""

import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, JsonValue, StrictBool, StrictInt

from wheelta_robinhood_agent.agent.mapped_evidence import (
    EvidenceMapper,
    MappedEvidence,
    MappingRequest,
)
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.enums import DataQuality, OptionRight
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts_compute import (
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
    UnderlyingQuote,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.run_record import Quote

# --------------------------------------------------------------------------------------------
# Parsing primitives
# --------------------------------------------------------------------------------------------

_DECIMAL_RE: Final = re.compile(r"^-?\d+(\.\d+)?$")
_TIMESTAMP_RE: Final = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d{1,9}))?"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})$"
)
_MICROSECOND_DIGITS: Final = 6


def _decimal_string(value: object) -> Decimal:
    """A decimal string (`"1.790000"`) parsed exactly. Floats, ints, and other forms raise."""
    if not isinstance(value, str) or not _DECIMAL_RE.fullmatch(value):
        raise ValueError("expected a decimal string")
    return Decimal(value)


def parse_broker_timestamp(value: object) -> datetime:
    """Parse a Robinhood RFC 3339 timestamp into an aware UTC datetime.

    Up to nine fractional digits are accepted; digits beyond microseconds are truncated
    (never rounded), so the result is never later than the source time. A timestamp without
    an explicit offset raises.
    """
    if not isinstance(value, str):
        raise ValueError("expected a timestamp string")
    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValueError("not an RFC 3339 timestamp with an offset")
    frac = (match["frac"] or "")[:_MICROSECOND_DIGITS].ljust(_MICROSECOND_DIGITS, "0")
    tz = "+00:00" if match["tz"] == "Z" else match["tz"]
    return datetime.fromisoformat(f"{match['base']}.{frac}{tz}").astimezone(UTC)


def _iso_date(value: object) -> date:
    """An ISO `YYYY-MM-DD` string; timestamps and other forms raise."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("expected an ISO date string")
    return date.fromisoformat(value)


DecStr = Annotated[Decimal, BeforeValidator(_decimal_string)]
IsoDate = Annotated[date, BeforeValidator(_iso_date)]
BrokerTime = Annotated[datetime, BeforeValidator(parse_broker_timestamp)]


class _External(BaseModel):
    """External payload: unknown fields ignored, the fields we rely on validated."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class _Exact(BaseModel):
    """An external shape verified only as captured; any extra key (e.g. paging) raises."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _unwrap(payload: JsonValue) -> JsonValue:
    """The tool's `data` value from `{"data": ..., "guide": ...}`, or the bare value."""
    if isinstance(payload, dict) and "data" in payload and set(payload) <= {"data", "guide"}:
        return payload["data"]
    return payload


def _gap_text(tool: str, detail: str) -> str:
    return f"{tool}: {detail}"


# --------------------------------------------------------------------------------------------
# get_equity_quotes
# --------------------------------------------------------------------------------------------


def _require_symbol(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", value):
        raise ValueError("expected an uppercase ticker symbol")
    return value


_Symbol = Annotated[str, BeforeValidator(_require_symbol)]


class _EquityQuote(_External):
    symbol: _Symbol
    last_trade_price: DecStr | None = None
    venue_last_trade_time: BrokerTime | None = None
    bid_price: DecStr | None = None
    ask_price: DecStr | None = None
    has_traded: StrictBool
    state: str


class _EquityQuoteResult(_External):
    quote: _EquityQuote


class _EquityQuotes(_External):
    results: tuple[_EquityQuoteResult, ...]


def map_equity_quotes(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`get_equity_quotes` -> one `UnderlyingQuote` per usable symbol.

    `price` is the regular-session `last_trade_price` and `as_of` its `venue_last_trade_time`.
    `last_non_reg_trade_price` (extended hours) and the previous close are not used.
    `UnderlyingQuote` has no bid/ask, so `bid_price`/`ask_price` are only schema-checked.
    A symbol whose `state` is not `active`, that has not traded, or lacks a positive last
    price or its time yields a gap instead of a quote. A repeated symbol raises.
    """
    parsed = _EquityQuotes.model_validate(_unwrap(request.payload))
    quotes: list[UnderlyingQuote] = []
    gaps: list[str] = []
    seen: set[str] = set()
    for result in parsed.results:
        q = result.quote
        if q.symbol in seen:
            raise ValueError(f"duplicate equity quote for {q.symbol}")
        seen.add(q.symbol)
        if q.state != "active":
            gaps.append(_gap_text(request.tool, f"{q.symbol} state is {q.state!r}, not active"))
        elif not q.has_traded:
            gaps.append(_gap_text(request.tool, f"{q.symbol} has not traded"))
        elif q.last_trade_price is None or q.last_trade_price <= 0:
            gaps.append(_gap_text(request.tool, f"{q.symbol} has no positive last trade price"))
        elif q.venue_last_trade_time is None:
            gaps.append(_gap_text(request.tool, f"{q.symbol} has no last trade time"))
        else:
            quotes.append(
                UnderlyingQuote(
                    evidence_id=new_id(),
                    as_of=q.venue_last_trade_time,
                    source_tool_call_ids=(request.tool_call_id,),
                    symbol=q.symbol,
                    price=q.last_trade_price,
                )
            )
    if not parsed.results:
        gaps.append(_gap_text(request.tool, "no quotes returned"))
    return MappedEvidence(underlying_quotes=tuple(quotes), gaps=tuple(gaps))


# --------------------------------------------------------------------------------------------
# get_option_instruments
# --------------------------------------------------------------------------------------------


def _require_root(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9]{0,5}", value):
        raise ValueError("expected an OCC root symbol")
    return value


class _MinTicks(_External):
    above_tick: DecStr
    below_tick: DecStr
    cutoff_price: DecStr


class _Instrument(_External):
    id: uuid.UUID
    chain_symbol: Annotated[str, BeforeValidator(_require_root)]
    underlying_type: str
    expiration_date: IsoDate
    strike_price: DecStr
    type: Literal["put", "call"]
    state: str
    tradability: str
    trade_value_multiplier: DecStr
    min_ticks: _MinTicks


class _Instruments(_External):
    instruments: tuple[_Instrument, ...]


def _multiplier(value: Decimal) -> int:
    """`trade_value_multiplier` ("100.0000") as a positive integer; anything else raises."""
    if value <= 0 or value != value.to_integral_value():
        raise ValueError(f"multiplier is not a positive integer: {value}")
    return int(value)


def map_option_instruments(
    request: MappingRequest, new_id: Callable[[], uuid.UUID]
) -> MappedEvidence:
    """`get_option_instruments` -> one `OptionInstrument` per active, tradable equity option.

    OCC symbol from `chain_symbol`, `expiration_date`, `type`, and `strike_price`; broker ID
    from `id`; underlying is `chain_symbol`; `multiplier` from `trade_value_multiplier`
    (verified in the capture). `min_ticks` is schema-checked (positive) but not carried:
    `OptionInstrument` has no tick field. An instrument that is not `active`/`tradable` or
    whose `underlying_type` is not `equity` yields a gap, because `OptionInstrument` cannot
    record tradability. A repeated instrument ID raises.
    """
    parsed = _Instruments.model_validate(_unwrap(request.payload))
    instruments: list[OptionInstrument] = []
    gaps: list[str] = []
    seen: set[uuid.UUID] = set()
    for inst in parsed.instruments:
        if inst.id in seen:
            raise ValueError("duplicate option instrument id")
        seen.add(inst.id)
        ticks = inst.min_ticks
        if ticks.above_tick <= 0 or ticks.below_tick <= 0 or ticks.cutoff_price < 0:
            raise ValueError("min_ticks must be positive")
        occ = OccSymbol(
            root=inst.chain_symbol,
            expiration=inst.expiration_date,
            right=OptionRight.PUT if inst.type == "put" else OptionRight.CALL,
            strike=inst.strike_price,
        )
        label = str(occ)  # raises if the strike is not representable in OCC form
        multiplier = _multiplier(inst.trade_value_multiplier)
        if inst.underlying_type != "equity":
            gaps.append(_gap_text(request.tool, f"{label} underlying type is not equity"))
        elif inst.state != "active" or inst.tradability != "tradable":
            gaps.append(
                _gap_text(
                    request.tool,
                    f"{label} is {inst.state!r}/{inst.tradability!r}, not active/tradable",
                )
            )
        else:
            instruments.append(
                OptionInstrument(
                    evidence_id=new_id(),
                    as_of=request.retrieved_at,
                    source_tool_call_ids=(request.tool_call_id,),
                    occ_symbol=occ,
                    broker_instrument_id=str(inst.id),
                    underlying=inst.chain_symbol,
                    multiplier=multiplier,
                )
            )
    if not parsed.instruments:
        gaps.append(_gap_text(request.tool, "no instruments returned"))
    return MappedEvidence(instruments=tuple(instruments), gaps=tuple(gaps))


# --------------------------------------------------------------------------------------------
# get_option_quotes
# --------------------------------------------------------------------------------------------


class _OptionQuote(_External):
    instrument_id: uuid.UUID
    bid_price: DecStr | None = None
    ask_price: DecStr | None = None
    mark_price: DecStr | None = None
    implied_volatility: DecStr | None = None
    delta: DecStr | None = None
    gamma: DecStr | None = None
    theta: DecStr | None = None
    vega: DecStr | None = None
    open_interest: StrictInt | None = None
    volume: StrictInt | None = None
    updated_at: BrokerTime | None = None


class _OptionQuoteResult(_External):
    quote: _OptionQuote


class _OptionQuotes(_External):
    results: tuple[_OptionQuoteResult, ...]


def map_option_quotes(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`get_option_quotes` -> one `Quote` per instrument with a two-sided market.

    bid/ask/mark from `bid_price`/`ask_price`/`mark_price`; `delta`, `gamma`, `theta`, `vega`;
    `implied_volatility` (already a ratio) -> `implied_volatility_ratio`; `open_interest`,
    `volume`; `as_of` from `updated_at`. A zero or absent bid or ask means no quote per the
    tool's own guidance, so it yields a gap. `rho`, fill-rate prices, `chance_of_profit_*`,
    break-even, and previous close are not facts we use and are dropped. Sanity (bid <= ask,
    mark within the spread, |delta| <= 1) is enforced by `Quote` and raises. A repeated
    instrument ID raises.
    """
    parsed = _OptionQuotes.model_validate(_unwrap(request.payload))
    quotes: list[Quote] = []
    gaps: list[str] = []
    seen: set[uuid.UUID] = set()
    for result in parsed.results:
        q = result.quote
        if q.instrument_id in seen:
            raise ValueError("duplicate option quote instrument id")
        seen.add(q.instrument_id)
        iid = str(q.instrument_id)
        if q.bid_price is None or q.ask_price is None or q.bid_price == 0 or q.ask_price == 0:
            gaps.append(_gap_text(request.tool, f"{iid} has a zero or missing bid/ask"))
        elif q.updated_at is None:
            gaps.append(_gap_text(request.tool, f"{iid} has no quote time"))
        else:
            quotes.append(
                Quote(
                    quote_id=new_id(),
                    broker_instrument_id=iid,
                    bid=q.bid_price,
                    ask=q.ask_price,
                    mark=q.mark_price,
                    delta=q.delta,
                    gamma=q.gamma,
                    theta=q.theta,
                    vega=q.vega,
                    implied_volatility_ratio=q.implied_volatility,
                    open_interest=q.open_interest,
                    volume=q.volume,
                    as_of=q.updated_at,
                    source_tool_call_ids=(request.tool_call_id,),
                )
            )
    if not parsed.results:
        gaps.append(_gap_text(request.tool, "no quotes returned"))
    return MappedEvidence(option_quotes=tuple(quotes), gaps=tuple(gaps))


# --------------------------------------------------------------------------------------------
# get_portfolio
# --------------------------------------------------------------------------------------------


class _BuyingPower(_External):
    buying_power: DecStr
    unleveraged_buying_power: DecStr | None = None


class _Portfolio(_External):
    total_value: DecStr
    currency: Literal["USD"]
    buying_power: _BuyingPower


_PORTFOLIO_GAPS: Final = (
    (
        "available_settled_cash_usd",
        "get_portfolio reports buying_power, which excludes unsettled funds, but its treatment "
        "of other reservations is unverified; it is not settled cash",
    ),
    (
        "csp_reserved_cash_usd",
        "get_portfolio reports no cash reserved for short puts or working orders",
    ),
    (
        "csp_cash_base_usd",
        "requires available_settled_cash_usd and csp_reserved_cash_usd",
    ),
)
_ELIGIBILITY_GAP: Final = (
    "agentic_verified",
    "Agentic eligibility was not verified this run (the trusted get_accounts check did not "
    "run or did not pass); get_portfolio does not report it",
)


def map_portfolio(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`get_portfolio` -> an `AccountSnapshot` for the configured account.

    `total_value` -> `account_value_usd` (USD only; any other currency raises). `account_ref`
    is the redacted `account_number` argument (the hook has already required the full
    configured number). `buying_power.buying_power` is NOT available settled cash: it excludes
    unsettled funds, but whether it nets reservations is unverified, and `AccountSnapshot`
    has no buying-power field, so it is only schema-checked. `available_settled_cash_usd`,
    `csp_reserved_cash_usd`, and `csp_cash_base_usd` stay None with named gaps. The response
    proves no Agentic eligibility: `agentic_verified` is True only when this run's trusted
    `get_accounts` check passed (`request.account_eligible`), else False with a gap. The
    snapshot quality stays `missing` while any cash field is missing. The payload has no
    timestamp: `as_of` is `retrieved_at`.
    """
    parsed = _Portfolio.model_validate(_unwrap(request.payload))
    account_ref = request.effective_input.get("account_number")
    if not isinstance(account_ref, str) or not account_ref:
        raise ValueError("get_portfolio needs the account_number argument")
    named = _PORTFOLIO_GAPS if request.account_eligible else (*_PORTFOLIO_GAPS, _ELIGIBILITY_GAP)
    gaps = tuple(Gap(field=name, kind=DataQuality.MISSING, detail=detail) for name, detail in named)
    snapshot = AccountSnapshot(
        snapshot_id=new_id(),
        as_of=request.retrieved_at,
        retrieved_at=request.retrieved_at,
        tool_call_ids=(request.tool_call_id,),
        account_ref=account_ref,
        agentic_verified=request.account_eligible,
        account_value_usd=parsed.total_value,
        available_settled_cash_usd=None,
        csp_reserved_cash_usd=None,
        csp_cash_base_usd=None,
        positions_ref=None,
        open_orders_ref=None,
        tax_lots_ref=None,
        quality=DataQuality.MISSING,
        gaps=gaps,
    )
    return MappedEvidence(
        account_snapshots=(snapshot,),
        gaps=tuple(_gap_text(request.tool, f"{g.field}: {g.detail}") for g in gaps),
    )


# --------------------------------------------------------------------------------------------
# Account lists: only the empty shape was captured
# --------------------------------------------------------------------------------------------


class _PositionList(_Exact):
    positions: list[JsonValue]


class _OrderList(_Exact):
    orders: list[JsonValue]


def _require_empty(items: list[JsonValue], tool: str) -> None:
    if items:
        raise ValueError(f"{tool}: non-empty result schema is not verified")


def map_option_orders(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`get_option_orders` -> an empty `OpenOrdersRead`.

    Only `{"orders": []}` was captured (the account was empty), so an empty list maps to a
    read with no working orders and ANY non-empty list, or any extra key such as a paging
    cursor, raises: the order schema is not verified. `as_of` is `retrieved_at`.
    """
    parsed = _OrderList.model_validate(_unwrap(request.payload))
    _require_empty(parsed.orders, request.tool)
    read = OpenOrdersRead(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
    )
    return MappedEvidence(open_orders=(read,))


def _empty_positions(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    parsed = _PositionList.model_validate(_unwrap(request.payload))
    _require_empty(parsed.positions, request.tool)
    read = PositionsRead(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
    )
    return MappedEvidence(positions=(read,))


def map_option_positions(
    request: MappingRequest, new_id: Callable[[], uuid.UUID]
) -> MappedEvidence:
    """`get_option_positions` -> an empty `PositionsRead` (NOT registered; module docstring).

    Only `{"positions": []}` was captured; any non-empty list or extra key raises.
    """
    return _empty_positions(request, new_id)


def map_equity_positions(
    request: MappingRequest, new_id: Callable[[], uuid.UUID]
) -> MappedEvidence:
    """`get_equity_positions` -> an empty `PositionsRead` (NOT registered; module docstring).

    Only `{"positions": []}` was captured; any non-empty list or extra key raises.
    """
    return _empty_positions(request, new_id)


# Tool name -> mapper; `result_boundary.VERIFIED_MAPPERS` keys these by the Robinhood server.
ROBINHOOD_MAPPERS: Mapping[str, EvidenceMapper] = MappingProxyType(
    {
        "get_equity_quotes": map_equity_quotes,
        "get_option_instruments": map_option_instruments,
        "get_option_quotes": map_option_quotes,
        "get_portfolio": map_portfolio,
        "get_option_orders": map_option_orders,
    }
)
