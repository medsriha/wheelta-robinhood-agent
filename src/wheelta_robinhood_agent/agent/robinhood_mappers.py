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

Positions (ADR-0031): `get_equity_positions` and `get_option_positions` each yield a
`PositionsRead` that covers only its own kind (shares or options); the facts service combines
the two halves of one run into a complete read. Only the empty shape is captured, so any
non-empty list still raises until a real holding is captured.
"""

import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from decimal import ROUND_FLOOR, Decimal
from types import MappingProxyType
from typing import Annotated, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, JsonValue, StrictBool, StrictInt

from wheelta_robinhood_agent.agent.mapped_evidence import (
    CANDIDATE_REF_PREFIX,
    BrokerOrderObservation,
    CancelRequestObservation,
    CandidateEvidence,
    EvidenceMapper,
    Execution,
    HeldOptionRow,
    MappedEvidence,
    MappingRequest,
    OrderLeg,
    OrderReviewObservation,
    PendingOptionPositions,
)
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    CandidateOrigin,
    DataQuality,
    OptionRight,
    OrderSide,
    PositionsCoverage,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts_compute import (
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
    ShareHolding,
    UnderlyingQuote,
    WorkingOrder,
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
    (verified in the capture). `min_ticks` is schema-checked (positive); `tick_increment` is
    carried only when `above_tick == below_tick` (one tick at every price), else None, because
    a price-dependent tick cannot be stated as one increment. An instrument that is not
    `active`/`tradable` or whose `underlying_type` is not `equity` yields a gap, because
    `OptionInstrument` cannot record tradability. A repeated instrument ID raises.

    Each mapped instrument also gets a code-issued candidate reference (ADR-0041), the only
    subject the decision-facts tool sizes for an open. Its origin is `robinhood` here; the
    result boundary sets `board` when a current-build Wheelta board row of this run lists
    the same contract.
    """
    parsed = _Instruments.model_validate(_unwrap(request.payload))
    instruments: list[OptionInstrument] = []
    candidates: list[CandidateEvidence] = []
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
                    tick_increment=(
                        ticks.above_tick if ticks.above_tick == ticks.below_tick else None
                    ),
                )
            )
            inst_ev = instruments[-1]
            candidates.append(
                CandidateEvidence(
                    candidate_ref=f"{CANDIDATE_REF_PREFIX}{new_id()}",
                    origin=CandidateOrigin.ROBINHOOD,
                    underlying=inst_ev.underlying,
                    instrument_evidence_id=inst_ev.evidence_id,
                    broker_instrument_id=inst_ev.broker_instrument_id,
                    occ_symbol=inst_ev.occ_symbol,
                )
            )
    if not parsed.instruments:
        gaps.append(_gap_text(request.tool, "no instruments returned"))
    return MappedEvidence(
        instruments=tuple(instruments), candidates=tuple(candidates), gaps=tuple(gaps)
    )


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
    cash: DecStr
    currency: Literal["USD"]
    buying_power: _BuyingPower


_PORTFOLIO_GAPS: Final = (
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
    configured number). `cash` -> `available_settled_cash_usd` (owner decision, ADR-0031;
    a negative value raises). `buying_power` is only schema-checked. `csp_reserved_cash_usd`
    and `csp_cash_base_usd` stay None with named gaps: the portfolio reports no reservation,
    and the facts derive it from positions and orders reads. The response
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
        available_settled_cash_usd=parsed.cash,
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
# Orders and positions (ADR-0034)
#
# Non-empty shapes follow the output schemas the server itself publishes in `tools/list`
# (tests/fixtures/robinhood/output_schemas_orders_2026-09-28.json); the empty shapes were
# also captured as real results (ADR-0017). A read is complete only when it was not narrowed
# by a filter and has no further page: a narrowed or paged read still yields broker order
# observations, but never an `OpenOrdersRead`/`PositionsRead`, and a gap says why.
# --------------------------------------------------------------------------------------------

# Broker order `state` -> AttemptStatus. `pending_cancelled` is still working until a read
# shows a terminal state; `failed` and `voided` never filled on their own.
_ORDER_STATUS: Final[Mapping[str, AttemptStatus]] = MappingProxyType(
    {
        "queued": AttemptStatus.PLACED,
        "confirmed": AttemptStatus.PLACED,
        "pending_cancelled": AttemptStatus.PLACED,
        "partially_filled": AttemptStatus.PARTIALLY_FILLED,
        "filled": AttemptStatus.FILLED,
        "cancelled": AttemptStatus.CANCELLED,
        "voided": AttemptStatus.CANCELLED,
        "rejected": AttemptStatus.REJECTED,
        "failed": AttemptStatus.REJECTED,
    }
)
_WORKING_STATES: Final = frozenset({"queued", "confirmed", "partially_filled", "pending_cancelled"})
# Arguments that narrow a list read; any of them makes it incomplete.
_ORDER_FILTERS: Final = frozenset(
    {
        "chain_ids",
        "created_at_gte",
        "cursor",
        "order_id",
        "placed_agent",
        "state",
        "underlying_type",
    }
)
_OPTION_POSITION_FILTERS: Final = frozenset(
    {
        "chain_ids",
        "cursor",
        "expiration_date",
        "expiration_date_gte",
        "expiration_date_lte",
        "option_ids",
        "option_type",
        "type",
    }
)


def _count_string(value: object) -> int:
    """A whole, non-negative count sent as a decimal string ("1", "1.0000")."""
    number = _decimal_string(value)
    if number < 0 or number != number.to_integral_value():
        raise ValueError(f"expected a whole non-negative count, got {value!r}")
    return int(number)


CountStr = Annotated[int, BeforeValidator(_count_string)]


class _Execution(_External):
    id: str
    price: DecStr
    quantity: CountStr
    timestamp: BrokerTime


class _OrderLeg(_External):
    option_id: uuid.UUID
    side: Literal["buy", "sell"]
    position_effect: Literal["open", "close"]
    ratio_quantity: StrictInt
    expiration_date: IsoDate
    strike_price: DecStr
    option_type: Literal["put", "call"]
    executions: tuple[_Execution, ...] | None = None


class _Order(_External):
    id: uuid.UUID
    chain_symbol: Annotated[str, BeforeValidator(_require_root)]
    state: str
    type: str
    trigger: str
    time_in_force: str
    quantity: CountStr
    processed_quantity: CountStr
    pending_quantity: CountStr
    canceled_quantity: CountStr
    price: DecStr | None = None
    trade_value_multiplier: DecStr
    placed_agent: str | None = None
    created_at: BrokerTime
    updated_at: BrokerTime | None = None
    legs: tuple[_OrderLeg, ...] | None


def _side_raw(side: str, effect: str) -> str:
    return f"{side}_to_{effect}"


def _broker_order(
    order: _Order, request: MappingRequest, new_id: Callable[[], uuid.UUID]
) -> BrokerOrderObservation:
    """One broker order as evidence. Multi-leg orders and unknown states raise."""
    status = _ORDER_STATUS.get(order.state)
    if status is None:
        raise ValueError(f"unknown order state {order.state!r}")
    legs = order.legs or ()
    if len(legs) != 1 or legs[0].ratio_quantity != 1:
        raise ValueError("only single-leg option orders are supported")
    leg = legs[0]
    occ = OccSymbol(
        root=order.chain_symbol,
        expiration=leg.expiration_date,
        right=OptionRight.PUT if leg.option_type == "put" else OptionRight.CALL,
        strike=leg.strike_price,
    )
    str(occ)  # raises if the strike is not representable in OCC form
    executions = tuple(
        Execution(
            broker_execution_id=e.id,
            quantity=e.quantity,
            price=e.price,
            executed_at=e.timestamp,
        )
        for e in leg.executions or ()
    )
    if any(e.quantity <= 0 or e.price <= 0 or not e.broker_execution_id for e in executions):
        raise ValueError("executions need an id, a positive quantity, and a positive price")
    if (
        order.processed_quantity + order.pending_quantity + order.canceled_quantity
        > max(order.quantity, 0)
        or order.quantity <= 0
    ):
        raise ValueError("order quantities are inconsistent")
    return BrokerOrderObservation(
        evidence_id=new_id(),
        as_of=order.updated_at or order.created_at,
        source_tool_call_ids=(request.tool_call_id,),
        broker_order_id=str(order.id),
        state_raw=order.state,
        status=status,
        underlying=order.chain_symbol,
        order_type_raw=order.type,
        trigger_raw=order.trigger,
        time_in_force_raw=order.time_in_force,
        quantity=order.quantity,
        processed_quantity=order.processed_quantity,
        pending_quantity=order.pending_quantity,
        canceled_quantity=order.canceled_quantity,
        limit_price=order.price,
        multiplier=_multiplier(order.trade_value_multiplier),
        placed_agent=order.placed_agent,
        created_at=order.created_at,
        legs=(
            OrderLeg(
                broker_instrument_id=str(leg.option_id),
                side_raw=_side_raw(leg.side, leg.position_effect),
                occ_symbol=occ,
            ),
        ),
        executions=executions,
    )


def _working_order(order: BrokerOrderObservation) -> WorkingOrder:
    """The unfilled part of a working order. A side the agent may not use raises: the
    working-orders read could not be represented completely."""
    leg = order.legs[0]
    if leg.occ_symbol is None:  # unreachable: _broker_order always sets it
        raise ValueError("a working order leg needs its contract")
    return WorkingOrder(
        broker_order_ref=order.broker_order_id,
        underlying=order.underlying,
        occ_symbol=leg.occ_symbol,
        broker_instrument_id=leg.broker_instrument_id,
        side=OrderSide(leg.side_raw),
        unfilled_quantity=order.pending_quantity,
        multiplier=order.multiplier,
    )


class _OrderList(_External):
    orders: tuple[_Order, ...] | None
    next: str | None = None


def _narrowed(request: MappingRequest, filters: frozenset[str]) -> list[str]:
    return sorted(k for k in filters if request.effective_input.get(k) not in (None, ""))


def map_option_orders(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`get_option_orders` -> every listed order as a `BrokerOrderObservation`, and, for a
    complete read, an `OpenOrdersRead` of the working ones.

    Complete means no narrowing filter (`_ORDER_FILTERS`) and no `next` page: the list is
    newest first and includes closed orders, so only then is every working order present.
    A working order is one in `_WORKING_STATES` with contracts still pending. `as_of` of the
    read is `retrieved_at`.
    """
    parsed = _OrderList.model_validate(_unwrap(request.payload))
    observations = tuple(_broker_order(o, request, new_id) for o in parsed.orders or ())
    ids = [o.broker_order_id for o in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate order id")
    narrowed = _narrowed(request, _ORDER_FILTERS)
    gaps: list[str] = []
    if narrowed:
        gaps.append(_gap_text(request.tool, f"narrowed by {narrowed}: not a complete read"))
    if parsed.next:
        gaps.append(_gap_text(request.tool, "more pages exist: not a complete read"))
    if gaps:
        return MappedEvidence(broker_orders=observations, gaps=tuple(gaps))
    working = tuple(
        _working_order(o)
        for o in observations
        if o.state_raw in _WORKING_STATES and o.pending_quantity > 0
    )
    read = OpenOrdersRead(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
        orders=working,
    )
    return MappedEvidence(broker_orders=observations, open_orders=(read,))


class _OptionPosition(_External):
    option_id: uuid.UUID
    chain_symbol: Annotated[str, BeforeValidator(_require_root)]
    type: Literal["long", "short"]
    quantity: CountStr
    trade_value_multiplier: DecStr


class _OptionPositions(_External):
    positions: tuple[_OptionPosition, ...] | None
    next: str | None = None


def map_option_positions(
    request: MappingRequest, new_id: Callable[[], uuid.UUID]
) -> MappedEvidence:
    """`get_option_positions` -> the options half of a positions read (ADR-0031, ADR-0034).

    Zero-quantity rows are closed positions and are skipped. A long option raises: the
    Agentic account only writes options, and `PositionsRead` cannot hold a long. With no open
    short row the read is an options-only `PositionsRead`; otherwise it is a
    `PendingOptionPositions` that the facts service resolves with this run's instrument
    evidence, because the row has no strike or call/put. Narrowed or paged reads are not
    complete and yield only a gap.
    """
    parsed = _OptionPositions.model_validate(_unwrap(request.payload))
    narrowed = _narrowed(request, _OPTION_POSITION_FILTERS)
    if narrowed or parsed.next:
        why = f"narrowed by {narrowed}" if narrowed else "more pages exist"
        return MappedEvidence(gaps=(_gap_text(request.tool, f"{why}: not a complete read"),))
    rows: list[HeldOptionRow] = []
    for p in parsed.positions or ():
        if p.quantity == 0:
            continue
        if p.type != "short":
            raise ValueError("a long option position cannot be represented")
        rows.append(
            HeldOptionRow(
                broker_instrument_id=str(p.option_id),
                underlying=p.chain_symbol,
                short_quantity=p.quantity,
                multiplier=_multiplier(p.trade_value_multiplier),
            )
        )
    held = [r.broker_instrument_id for r in rows]
    if len(held) != len(set(held)):
        raise ValueError("duplicate option position")
    if not rows:
        read = PositionsRead(
            evidence_id=new_id(),
            as_of=request.retrieved_at,
            source_tool_call_ids=(request.tool_call_id,),
            covers=frozenset({PositionsCoverage.OPTIONS}),
        )
        return MappedEvidence(positions=(read,))
    pending = PendingOptionPositions(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
        rows=tuple(rows),
    )
    gap = _gap_text(
        request.tool,
        "short option rows carry no strike or call/put; read them with get_option_instruments "
        f"ids={','.join(held)} to complete the positions read",
    )
    return MappedEvidence(pending_option_positions=(pending,), gaps=(gap,))


class _EquityPosition(_External):
    symbol: _Symbol
    quantity: DecStr
    type: str


class _EquityPositions(_External):
    positions: tuple[_EquityPosition, ...] | None
    next: str | None = None


def map_equity_positions(
    request: MappingRequest, new_id: Callable[[], uuid.UUID]
) -> MappedEvidence:
    """`get_equity_positions` -> a shares-only `PositionsRead` (ADR-0031, ADR-0034).

    `quantity` (total shares, including today's fills) counts whole shares: a fractional
    remainder cannot cover a contract, so it is dropped (floor, never rounded up). A short
    (negative) or `boxed` position raises. Zero rows are skipped. A paged read or any
    narrowing argument is not complete and yields only a gap.
    """
    parsed = _EquityPositions.model_validate(_unwrap(request.payload))
    narrowed = sorted(
        k
        for k, v in request.effective_input.items()
        if k != "account_number" and v not in (None, "")
    )
    if narrowed or parsed.next:
        why = f"narrowed by {narrowed}" if narrowed else "more pages exist"
        return MappedEvidence(gaps=(_gap_text(request.tool, f"{why}: not a complete read"),))
    holdings: list[ShareHolding] = []
    for p in parsed.positions or ():
        if p.quantity < 0 or p.type == "boxed":
            raise ValueError(f"{p.symbol}: short or boxed share positions are not supported")
        whole = int(p.quantity.to_integral_value(rounding=ROUND_FLOOR))
        if whole > 0:
            holdings.append(ShareHolding(symbol=p.symbol, quantity=whole))
    symbols = [h.symbol for h in holdings]
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate share position")
    read = PositionsRead(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
        covers=frozenset({PositionsCoverage.SHARES}),
        share_holdings=tuple(holdings),
    )
    return MappedEvidence(positions=(read,))


# --------------------------------------------------------------------------------------------
# Order tools (Tier X, ADR-0034)
# --------------------------------------------------------------------------------------------


class _ReviewLeg(_External):
    option_id: uuid.UUID
    side: Literal["buy", "sell"]
    position_effect: Literal["open", "close"]


class _Review(_External):
    type: str
    quantity: CountStr
    price: DecStr | None = None
    time_in_force: str | None = None
    legs: tuple[_ReviewLeg, ...] | None
    order_checks: dict[str, JsonValue]
    option_quotes: tuple[_OptionQuote | None, ...] | None = None


def map_order_review(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`review_option_order` -> an `OrderReviewObservation`, plus the returned live quotes.

    `clean` is True only for an empty `order_checks` object; otherwise the broker's
    `alertType` is carried and a gap names it, so the model sees the warning. Quotes follow
    the `get_option_quotes` rules (a zero bid/ask or a missing time yields a gap).
    """
    parsed = _Review.model_validate(_unwrap(request.payload))
    legs = parsed.legs or ()
    if len(legs) != 1:
        raise ValueError("only single-leg reviews are supported")
    clean = not parsed.order_checks
    alert = parsed.order_checks.get("alertType")
    alert_type = alert if isinstance(alert, str) and alert else None
    gaps: list[str] = []
    if not clean:
        gaps.append(
            _gap_text(request.tool, f"pre-trade alert {alert_type or 'unnamed'}: do not place")
        )
    quotes: list[Quote] = []
    for q in parsed.option_quotes or ():
        if q is None:
            continue
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
    review = OrderReviewObservation(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
        legs=tuple(
            OrderLeg(
                broker_instrument_id=str(leg.option_id),
                side_raw=_side_raw(leg.side, leg.position_effect),
            )
            for leg in legs
        ),
        quantity=parsed.quantity,
        order_type_raw=parsed.type,
        time_in_force_raw=parsed.time_in_force,
        limit_price=parsed.price,
        clean=clean,
        alert_type=alert_type,
    )
    return MappedEvidence(order_reviews=(review,), option_quotes=tuple(quotes), gaps=tuple(gaps))


class _PlaceResult(_External):
    order: _Order


def map_order_placement(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`place_option_order` -> the created order as a `BrokerOrderObservation`.

    The tool says the order was submitted, not filled; `status` carries the broker state. A
    missing `order` raises, so the outcome stays unknown and the prompt stops placement.
    """
    parsed = _PlaceResult.model_validate(_unwrap(request.payload))
    return MappedEvidence(broker_orders=(_broker_order(parsed.order, request, new_id),))


class _CancelResult(_External):
    accepted: StrictBool


def map_order_cancel(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`cancel_option_order` -> a `CancelRequestObservation` for the requested `order_id`.

    `accepted=true` is not a cancellation: only a later `get_option_orders` read showing a
    terminal state confirms it (the tool's own guidance), and a gap says so.
    """
    parsed = _CancelResult.model_validate(_unwrap(request.payload))
    order_id = request.effective_input.get("order_id")
    if not isinstance(order_id, str) or not order_id:
        raise ValueError("cancel_option_order needs the order_id argument")
    ack = CancelRequestObservation(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
        broker_order_id=str(uuid.UUID(order_id)),
        accepted=parsed.accepted,
    )
    gap = _gap_text(
        request.tool,
        "a cancel request is not a cancellation; confirm the final state with get_option_orders"
        if parsed.accepted
        else "the broker did not accept the cancel request",
    )
    return MappedEvidence(cancel_requests=(ack,), gaps=(gap,))


# Tool name -> mapper; `result_boundary.VERIFIED_MAPPERS` keys these by the Robinhood server.
ROBINHOOD_MAPPERS: Mapping[str, EvidenceMapper] = MappingProxyType(
    {
        "get_equity_quotes": map_equity_quotes,
        "get_option_instruments": map_option_instruments,
        "get_option_quotes": map_option_quotes,
        "get_portfolio": map_portfolio,
        "get_option_orders": map_option_orders,
        "get_option_positions": map_option_positions,
        "get_equity_positions": map_equity_positions,
        "review_option_order": map_order_review,
        "place_option_order": map_order_placement,
        "cancel_option_order": map_order_cancel,
    }
)
