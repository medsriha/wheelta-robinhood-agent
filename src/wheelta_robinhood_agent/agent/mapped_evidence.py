"""Typed evidence contract between remote-tool result mappers and the result boundary.

Split out of `result_boundary.py` so verified mappers (`robinhood_mappers.py`) can build
`MappedEvidence` while `result_boundary.VERIFIED_MAPPERS` registers them, without an import
cycle. `result_boundary` re-exports every name here.
"""

import uuid
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import NonEmptyStr, Ref
from wheelta_robinhood_agent.domain.enums import AttemptStatus, CandidateOrigin
from wheelta_robinhood_agent.domain.facts_compute import (
    BoardScreen,
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
    UnderlyingQuote,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.run_record import Quote

CANDIDATE_REF_PREFIX = "candidate:"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateEvidence(_Model):
    """A code-issued candidate reference and the instrument it names (OUTPUT_ASSEMBLY.md).

    The model may select `candidate_ref`; it can never mint one from a ticker string.
    """

    candidate_ref: Ref
    origin: CandidateOrigin
    underlying: NonEmptyStr
    instrument_evidence_id: uuid.UUID
    broker_instrument_id: NonEmptyStr
    occ_symbol: OccSymbol


class _Observed(_Model):
    evidence_id: uuid.UUID
    as_of: AwareDatetime
    source_tool_call_ids: tuple[uuid.UUID, ...]


class OrderLeg(_Model):
    """One leg of a broker order or review, identified by the broker instrument ID.

    `side_raw` joins the broker's `side` and `position_effect` (`sell` + `open` ->
    `sell_to_open`), so it compares directly with `OrderSide` values; other combinations are
    kept verbatim and simply match no permitted side.
    """

    broker_instrument_id: NonEmptyStr
    side_raw: NonEmptyStr
    occ_symbol: OccSymbol | None = None


class Execution(_Model):
    """One broker execution (fill) of an order leg."""

    broker_execution_id: NonEmptyStr
    quantity: int
    price: Decimal
    executed_at: AwareDatetime


class BrokerOrderObservation(_Observed):
    """One option order as the broker reported it (place result or order read, ADR-0034).

    `status` is the broker `state` mapped onto `AttemptStatus`; `state_raw` keeps the
    original. Quantities are whole contracts. `placed_agent` says who placed it.
    """

    broker_order_id: NonEmptyStr
    state_raw: NonEmptyStr
    status: AttemptStatus
    underlying: NonEmptyStr
    order_type_raw: NonEmptyStr
    trigger_raw: NonEmptyStr
    time_in_force_raw: NonEmptyStr
    quantity: int
    processed_quantity: int
    pending_quantity: int
    canceled_quantity: int
    limit_price: Decimal | None
    multiplier: int
    placed_agent: str | None
    created_at: AwareDatetime
    legs: tuple[OrderLeg, ...]
    executions: tuple[Execution, ...] = ()


class OrderReviewObservation(_Observed):
    """A `review_option_order` result: the echoed order and the broker's pre-trade check.

    `clean` is True only when `order_checks` is the empty object, which the tool defines as
    "clean and safe to place". Otherwise `alert_type` names the issue.
    """

    legs: tuple[OrderLeg, ...]
    quantity: int
    order_type_raw: NonEmptyStr
    time_in_force_raw: str | None
    limit_price: Decimal | None
    clean: bool
    alert_type: str | None


class CancelRequestObservation(_Observed):
    """A `cancel_option_order` result. `accepted` means the broker accepted the request, not
    that the order is cancelled; only a later order read confirms that."""

    broker_order_id: NonEmptyStr
    accepted: bool


class HeldOptionRow(_Model):
    """A short option position row before its contract (strike, right) is resolved."""

    broker_instrument_id: NonEmptyStr
    underlying: NonEmptyStr
    short_quantity: int
    multiplier: int


class PendingOptionPositions(_Observed):
    """A complete options-positions read whose short rows still need their contracts.

    `get_option_positions` reports no strike or call/put, so the facts service resolves each
    row against this run's instrument evidence (`get_option_instruments ids=...`). Until every
    row resolves, no options half exists (ADR-0034)."""

    rows: tuple[HeldOptionRow, ...]


class EarningsReport(_Observed):
    """One earnings report event (`get_earnings_results`, `get_earnings_calendar`; ADR-0042).

    `verified` false means the broker marks `report_date` as tentative. `eps_actual` is None
    until the company has reported. Fiscal year and quarter are the company's.
    """

    symbol: NonEmptyStr
    fiscal_year: int
    fiscal_quarter: int
    report_date: date
    timing: Literal["am", "pm"] | None
    verified: bool
    eps_estimate: Decimal | None
    eps_actual: Decimal | None


class SecFilingListing(_Observed):
    """One SEC filing as `get_sec_filing_index` lists it (ADR-0042)."""

    symbol: NonEmptyStr
    filing_id: NonEmptyStr
    form_type: NonEmptyStr
    date_filed: date
    description: str


class SecFilingSectionEntry(_Model):
    """One entry of a filing's table of contents: the `section_id` to pass back to
    `get_sec_filing`, its title, and its heading level."""

    section_id: NonEmptyStr
    title: str
    level: int


class SecFilingContents(_Observed):
    """A filing's table of contents from `get_sec_filing` without `section` (ADR-0069)."""

    filing_id: NonEmptyStr
    form_type: NonEmptyStr
    sections: tuple[SecFilingSectionEntry, ...]


class SecFilingSection(_Observed):
    """One section's text from `get_sec_filing` (ADR-0069): the filer's own words, Markdown
    as Robinhood renders it, kept verbatim (including the server's mis-encoded characters).
    A tier-1 source; a Form 4 table is the insider's reported transaction."""

    filing_id: NonEmptyStr
    form_type: NonEmptyStr
    section_id: NonEmptyStr
    section_title: str
    content: str


class FinancialPeriod(_Observed):
    """One reported fiscal period from `get_financials` (ADR-0042). USD amounts as reported;
    `fiscal_quarter` is None for an annual period."""

    symbol: NonEmptyStr
    period: Literal["quarterly", "annual"]
    fiscal_year: int
    fiscal_quarter: int | None
    period_end_date: date
    revenue_usd: Decimal | None
    gross_profit_usd: Decimal | None
    net_income_usd: Decimal | None
    net_margin_ratio: Decimal | None


class EquityFundamentals(_Observed):
    """Company fundamentals from `get_equity_fundamentals` for one `market_date` (ADR-0042).

    Dividend dates are kept as reported; whether they describe the last or the next
    distribution is unverified (the mapper adds a gap when any is present).
    """

    symbol: NonEmptyStr
    market_date: date
    market_cap_usd: Decimal
    shares_outstanding: Decimal
    pe_ratio: Decimal | None
    pb_ratio: Decimal | None
    high_52_weeks: Decimal
    high_52_weeks_date: date
    low_52_weeks: Decimal
    low_52_weeks_date: date
    average_volume_30_days: Decimal
    ex_dividend_date: date | None
    record_date: date | None
    payable_date: date | None
    distribution_frequency: str | None
    sector: str
    industry: str
    description: str


class AnalystRatings(_Observed):
    """Analyst rating counts and price targets for one symbol (`get_equity_analyst_ratings`,
    ADR-0042). Targets can be absent while counts are present."""

    symbol: NonEmptyStr
    buy_ratings: int
    hold_ratings: int
    sell_ratings: int
    low_price_target: Decimal | None
    mean_price_target: Decimal | None
    high_price_target: Decimal | None
    updated_at: AwareDatetime | None


class PoliticianTrade(_Observed):
    """One disclosed politician trade from `get_politician_trades` (ADR-0069), as Tip Ranks
    reports STOCK Act disclosures. The amount is a disclosed range in USD, never an exact
    figure; `disclosure_date` lags `transaction_date` by up to 45 days."""

    politician_name: NonEmptyStr
    party: NonEmptyStr
    position: NonEmptyStr
    asset_type: NonEmptyStr
    symbol: NonEmptyStr
    transaction_type: NonEmptyStr
    amount_min_usd: Decimal
    amount_max_usd: Decimal
    transaction_date: date
    disclosure_date: date
    source: NonEmptyStr


class PopularWatchlist(_Observed):
    """One Robinhood-curated list from `get_popular_watchlists` (ADR-0069): its name and
    size only; the members are not part of this result. `is_badged` (Robinhood marks the list
    new) is None when the result omits it."""

    list_id: NonEmptyStr
    display_name: NonEmptyStr
    item_count: int
    is_badged: bool | None


class OptionChainObservation(_Observed):
    """One option chain: its listed expirations and contract terms (`get_option_chains`,
    ADR-0042). A contract still comes only from `get_option_instruments`."""

    chain_id: NonEmptyStr
    symbol: NonEmptyStr
    expiration_dates: tuple[date, ...]
    multiplier: int
    can_open_position: bool
    settle_on_open: bool
    above_tick: Decimal
    below_tick: Decimal
    tick_cutoff_price: Decimal


class MacroRegimeInput(_Model):
    """One series value the Wheelta regime classification used, as of its own observation."""

    series_id: NonEmptyStr
    value: Decimal
    observed_on: date


class MacroRegime(_Observed):
    """Wheelta's macro regime classification (`wheelta_macro_snapshot`, ADR-0045).

    A descriptive label with Wheelta's own summary and guidance prose, not a forecast.
    `snapshot_as_of` is when Wheelta built the snapshot.
    """

    snapshot_as_of: AwareDatetime
    tag: NonEmptyStr
    label: NonEmptyStr
    summary: str
    guidance: str
    inputs: tuple[MacroRegimeInput, ...]


class MacroIndicator(_Observed):
    """One macro indicator from `wheelta_macro_snapshot` (ADR-0045).

    `value` and `change` are in `unit` as Wheelta reports it (`pct` = percent, e.g. 3.63;
    `pct_points`, `index`, `count`, `thousands`, `usd`, `binary`). `change_ratio` is the
    relative change as a ratio (-0.0118 = -1.18%) over `change_period`. `observed_on` is the
    observation date; `observed_at` is set only when the source gives a time.
    `direction` is Wheelta's own reading for put sellers (good, bad, flat).
    """

    series_id: NonEmptyStr
    name: NonEmptyStr
    category: NonEmptyStr
    unit: NonEmptyStr
    value: Decimal
    change: Decimal | None
    change_ratio: Decimal | None
    change_period: str
    direction: str
    source: NonEmptyStr
    observed_on: date
    observed_at: AwareDatetime | None


class MappedEvidence(_Model):
    """Normalized, typed evidence produced from one validated tool result."""

    instruments: tuple[OptionInstrument, ...] = ()
    option_quotes: tuple[Quote, ...] = ()
    underlying_quotes: tuple[UnderlyingQuote, ...] = ()
    account_snapshots: tuple[AccountSnapshot, ...] = ()
    positions: tuple[PositionsRead, ...] = ()
    open_orders: tuple[OpenOrdersRead, ...] = ()
    candidates: tuple[CandidateEvidence, ...] = ()
    broker_orders: tuple[BrokerOrderObservation, ...] = ()
    order_reviews: tuple[OrderReviewObservation, ...] = ()
    cancel_requests: tuple[CancelRequestObservation, ...] = ()
    pending_option_positions: tuple[PendingOptionPositions, ...] = ()
    # ADR-0041: Wheelta board rows, a build-time screen (never a quote).
    board_screens: tuple[BoardScreen, ...] = ()
    # ADR-0042: research reads. Citable context; no decision fact is computed from them.
    earnings_reports: tuple[EarningsReport, ...] = ()
    sec_filings: tuple[SecFilingListing, ...] = ()
    financial_periods: tuple[FinancialPeriod, ...] = ()
    fundamentals: tuple[EquityFundamentals, ...] = ()
    analyst_ratings: tuple[AnalystRatings, ...] = ()
    option_chains: tuple[OptionChainObservation, ...] = ()
    # ADR-0069: filing contents and text, politician trades, curated lists. Citable context.
    sec_filing_contents: tuple[SecFilingContents, ...] = ()
    sec_filing_sections: tuple[SecFilingSection, ...] = ()
    politician_trades: tuple[PoliticianTrade, ...] = ()
    popular_watchlists: tuple[PopularWatchlist, ...] = ()
    # ADR-0045: Wheelta macro snapshot. Citable context; no decision fact uses it.
    macro_regimes: tuple[MacroRegime, ...] = ()
    macro_indicators: tuple[MacroIndicator, ...] = ()
    # Non-citable context delivered beside the evidence (the other selected board columns);
    # it carries no evidence identity and can back no number or decision.
    screen_context: tuple[dict[str, JsonValue], ...] = ()
    gaps: tuple[str, ...] = ()

    def _observed(self) -> tuple[_Observed, ...]:
        return (
            *self.broker_orders,
            *self.order_reviews,
            *self.cancel_requests,
            *self.pending_option_positions,
            *self.earnings_reports,
            *self.sec_filings,
            *self.financial_periods,
            *self.fundamentals,
            *self.analyst_ratings,
            *self.option_chains,
            *self.sec_filing_contents,
            *self.sec_filing_sections,
            *self.politician_trades,
            *self.popular_watchlists,
            *self.macro_regimes,
            *self.macro_indicators,
        )

    def evidence_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(i.evidence_id for i in self.instruments),
            *(q.quote_id for q in self.option_quotes),
            *(u.evidence_id for u in self.underlying_quotes),
            *(a.snapshot_id for a in self.account_snapshots),
            *(p.evidence_id for p in self.positions),
            *(o.evidence_id for o in self.open_orders),
            *(o.evidence_id for o in self._observed()),
            *(b.evidence_id for b in self.board_screens),
        )

    def source_tool_call_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(t for i in self.instruments for t in i.source_tool_call_ids),
            *(t for q in self.option_quotes for t in q.source_tool_call_ids),
            *(t for u in self.underlying_quotes for t in u.source_tool_call_ids),
            *(t for a in self.account_snapshots for t in a.tool_call_ids),
            *(t for p in self.positions for t in p.source_tool_call_ids),
            *(t for o in self.open_orders for t in o.source_tool_call_ids),
            *(t for o in self._observed() for t in o.source_tool_call_ids),
            *(t for b in self.board_screens for t in b.source_tool_call_ids),
        )


class MappingRequest(_Model):
    """Input to an `EvidenceMapper`: the parsed (redacted) payload of one successful call."""

    tool_call_id: uuid.UUID
    server: str
    tool: str
    effective_input: dict[str, JsonValue]
    payload: JsonValue
    retrieved_at: AwareDatetime
    # This run's trusted Agentic-eligibility check passed (agent/session.py, CLAUDE.md §9).
    account_eligible: bool = False


class EvidenceMapper(Protocol):
    """Map one tool's verified result schema to typed evidence. Raise on any schema mismatch.

    `new_id` issues evidence IDs and candidate refs, so every identity is code-issued.
    """

    def __call__(
        self, request: MappingRequest, new_id: Callable[[], uuid.UUID]
    ) -> MappedEvidence: ...
