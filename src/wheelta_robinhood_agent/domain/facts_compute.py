"""Pure `compute_decision_facts` (OUTPUT_ASSEMBLY.md "Facts available before the decision").

`compute_decision_facts(inputs, rules, as_of)` turns validated evidence into a DecisionFacts
record: sizing/capacity, metrics with formula versions and input evidence, and typed gaps.
It reads no clock, environment, database, or network; the adapter in `agent/` loads
evidence, supplies identity and `as_of`, and persists the result.

Rules applied (each formula id below is stamped at version "1"):
- `dte_calendar_days`: expiration date minus `as_of`'s America/New_York calendar date
  (definitions.annualization: DTE in calendar days).
- `collateral_per_contract`: CSP strike x multiplier; CC live share price x multiplier
  (filters notes).
- `annualized_yield_on_collateral`: live BID x multiplier / collateral x 365 / DTE
  (filters.min_annualized_yield_ratio, ADR-0014, definitions.annualization). Opening only.
- `spread_ratio_of_mid`: (ask - bid) / mid, mid = (bid + ask) / 2 (filters notes).
- `csp_capacity`: floor of the smallest applicable capacity (selection.sizing): order cap;
  per-underlying USD cap and ratio x account value, each minus existing CSP collateral on the
  underlying; (C - reserve); (total_ratio x B - R), with C = available settled cash, R = CSP
  reserved cash, B = C + R (definitions.cash_accounting; VALIDATION.md "CSP accounting").
- `cc_capacity`: floor((owned shares x coverage ratio - shares already covered by short
  calls and working sell-to-open calls) / multiplier), plus the order cap (selection.sizing).
- `close_capacity`: short quantity minus working buy-to-close quantity on the contract, and,
  when broker-reported debit funding is supplied, floor(funding / (price x multiplier)).
- `captured_ratio`: (credit received - current buyback ask) / credit received.
- `remaining_yield`: buyback ASK x multiplier / collateral x 365 / remaining DTE (ADR-0015).
- `tested`: put: live underlying <= strike; call: live underlying >= strike (ADR-0015).
- `roll_dte_window`: DTE <= management.roll_dte_window.
- `decision_quantity`: initial = rule-derived capacity (OPEN), the discretionary close
  quantity (CLOSE, ADR-0015), or the whole current short quantity (MANAGE, the ROLL close
  leg); an established target from an earlier fact set is kept. remaining = min(initial -
  confirmed fills, current capacity).

"none" rules do not constrain; "TBD" or "agent_discretion" numeric rules make the dependent
fact unavailable with a gap. Zero is a real zero. A missing quantity is never rounded to one.
"""

from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, DivisionByZero, InvalidOperation, Overflow
from functools import cached_property
from typing import Self, TypeVar
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from pydantic import model_validator

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import (
    Count,
    Dec,
    DomainModel,
    NonEmptyStr,
    PosCount,
    PosDec,
    Ref,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import (
    CandidateOrigin,
    DataQuality,
    OptionRight,
    OrderSide,
    StrategyKind,
)
from wheelta_robinhood_agent.domain.evidence import Derivation, Gap, SourcedValue
from wheelta_robinhood_agent.domain.facts import (
    DecisionFacts,
    DerivedMetric,
    FactsPurpose,
    FormulaVersion,
)
from wheelta_robinhood_agent.domain.facts_rules import FactsRuleMarker, FactsRules
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.positions import PositionBookEntry
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.domain.sanity import is_fresh, require_aware

FORMULA_VERSION = "1"
F_DTE = "dte_calendar_days"
F_COLLATERAL = "collateral_per_contract"
F_YIELD = "annualized_yield_on_collateral"
F_SPREAD = "spread_ratio_of_mid"
F_CSP = "csp_capacity"
F_CC = "cc_capacity"
F_CLOSE = "close_capacity"
F_CAPTURED = "captured_ratio"
F_REMAINING_YIELD = "remaining_yield"
F_TESTED = "tested"
F_ROLL_WINDOW = "roll_dte_window"
F_QUANTITY = "decision_quantity"

_SOURCE = "code:compute_decision_facts"
_MARKET_TZ = "America/New_York"
_ZERO = Decimal(0)
_TWO = Decimal(2)
_DAYS_PER_YEAR = Decimal(365)
# Explicit local context: results never depend on the process-wide decimal context.
_CTX = Context(
    prec=28, rounding=ROUND_HALF_EVEN, traps=[InvalidOperation, DivisionByZero, Overflow]
)
_QUALITY_ORDER = (
    DataQuality.CONTRADICTORY,
    DataQuality.STALE,
    DataQuality.LOW_CONFIDENCE,
    DataQuality.MISSING,
)
_UNUSABLE_SNAPSHOT = (DataQuality.STALE, DataQuality.CONTRADICTORY, DataQuality.LOW_CONFIDENCE)
_R_OPTION_AGE = "data_quality.freshness.option_quote_max_age_seconds"
_R_EQUITY_AGE = "data_quality.freshness.equity_quote_max_age_seconds"
_R_ACCOUNT_AGE = "data_quality.freshness.account_state_max_age_seconds"

T = TypeVar("T", int, Decimal)


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


class _Observation(DomainModel):
    """A validated observation: evidence identity, source time, and source tool calls."""

    evidence_id: UUID
    as_of: UtcDatetime
    source_tool_call_ids: tuple[UUID, ...]

    @model_validator(mode="after")
    def _check_observation(self) -> Self:
        if not self.source_tool_call_ids:
            raise ValueError("an observation needs a source tool call")
        return self


class OptionInstrument(_Observation):
    """Validated instrument identity. `multiplier` is None until verified (VALIDATION.md)."""

    occ_symbol: OccSymbol
    broker_instrument_id: NonEmptyStr
    underlying: NonEmptyStr
    multiplier: PosCount | None


class UnderlyingQuote(_Observation):
    """The underlying's live price from Robinhood (broker field mapping unverified)."""

    symbol: NonEmptyStr
    price: PosDec


class ShareHolding(DomainModel):
    symbol: NonEmptyStr
    quantity: Count


class ShortOptionHolding(DomainModel):
    """A short option position in the Agentic account (multiplier None if unverified)."""

    underlying: NonEmptyStr
    occ_symbol: OccSymbol
    broker_instrument_id: NonEmptyStr
    short_quantity: Count
    multiplier: PosCount | None


class PositionsRead(_Observation):
    """One complete, account-scoped positions read: absence of a holding means zero."""

    share_holdings: tuple[ShareHolding, ...] = ()
    short_options: tuple[ShortOptionHolding, ...] = ()


class WorkingOrder(DomainModel):
    """An unfilled (portion of a) working option order in the account, from any actor."""

    broker_order_ref: NonEmptyStr
    underlying: NonEmptyStr
    occ_symbol: OccSymbol
    broker_instrument_id: NonEmptyStr
    side: OrderSide
    unfilled_quantity: Count
    multiplier: PosCount | None


class OpenOrdersRead(_Observation):
    """One complete, account-scoped open-orders read."""

    orders: tuple[WorkingOrder, ...] = ()


class CandidateProvenance(DomainModel):
    """A code-issued candidate reference with its discovery path."""

    candidate_ref: Ref
    origin: CandidateOrigin
    underlying: NonEmptyStr


class ConfirmedFill(DomainModel):
    """A confirmed fill against this decision's established target (deduplicated upstream)."""

    evidence_id: UUID
    quantity: PosCount


class FactInputs(DomainModel):
    """Validated inputs for one subject and purpose (OUTPUT_ASSEMBLY.md FactInputs).

    Identity (`facts_id`, `facts_ref`, `run_id`) comes from the adapter. The only
    discretionary inputs are `limit_price` and, for CLOSE (ADR-0015), `close_quantity`. No
    balances, rule overrides, or formulas are accepted from the model.

    OPEN targets a candidate; CLOSE and MANAGE target a PositionBook lineage holding
    `instrument`. `established_target_quantity` is the target from an earlier fact set for
    the same decision (OPEN/MANAGE); confirmed fills require it, so a fresh capacity that
    already reflects those fills is never reduced by them a second time.
    """

    facts_id: UUID
    facts_ref: Ref
    run_id: UUID
    subject_ref: Ref
    purpose: FactsPurpose
    instrument: OptionInstrument
    option_quote: Quote | None = None
    underlying_quote: UnderlyingQuote | None = None
    account: AccountSnapshot | None = None
    positions: PositionsRead | None = None
    open_orders: OpenOrdersRead | None = None
    candidate: CandidateProvenance | None = None
    position: PositionBookEntry | None = None
    debit_funding: SourcedValue[Dec] | None = None
    limit_price: PosDec | None = None
    close_quantity: PosCount | None = None
    established_target_quantity: Count | None = None
    confirmed_fills: tuple[ConfirmedFill, ...] = ()

    @model_validator(mode="after")
    def _check_inputs(self) -> Self:
        inst = self.instrument
        if self.option_quote and self.option_quote.broker_instrument_id != (
            inst.broker_instrument_id
        ):
            raise ValueError("option quote does not belong to the instrument")
        if self.underlying_quote and self.underlying_quote.symbol != inst.underlying:
            raise ValueError("underlying quote does not belong to the instrument's underlying")
        require_unique(tuple(f.evidence_id for f in self.confirmed_fills), "fill evidence id")
        if self.close_quantity is not None and self.purpose is not FactsPurpose.CLOSE:
            raise ValueError("close_quantity is a CLOSE-purpose input only")
        if self.purpose is FactsPurpose.OPEN:
            self._check_open()
        else:
            self._check_position()
        if self.confirmed_fills and self.purpose is not FactsPurpose.CLOSE:
            if self.established_target_quantity is None:
                raise ValueError("confirmed fills require an established target quantity")
        return self

    def _check_open(self) -> None:
        if self.candidate is None or self.position is not None:
            raise ValueError("an OPEN fact set targets a candidate, not a position")
        if self.subject_ref != self.candidate.candidate_ref:
            raise ValueError("subject_ref must be the candidate reference")
        if self.candidate.underlying != self.instrument.underlying:
            raise ValueError("candidate underlying does not match the instrument")
        if self.debit_funding is not None:
            raise ValueError("debit funding applies to closing legs only")

    def _check_position(self) -> None:
        position = self.position
        if position is None or self.candidate is not None:
            raise ValueError("a CLOSE/MANAGE fact set targets a position, not a candidate")
        if self.subject_ref != position.position_ref:
            raise ValueError("subject_ref must be the position reference")
        if position.underlying != self.instrument.underlying:
            raise ValueError("position underlying does not match the instrument")
        ids = {ci.broker_instrument_id for ci in position.current_instruments}
        if self.instrument.broker_instrument_id not in ids:
            raise ValueError("the instrument is not a current contract of the position")
        expected = (
            StrategyKind.CASH_SECURED_PUT
            if self.instrument.occ_symbol.right is OptionRight.PUT
            else StrategyKind.COVERED_CALL
        )
        if position.strategy is not expected:
            raise ValueError("position strategy does not match the contract right")
        if self.purpose is FactsPurpose.CLOSE and self.established_target_quantity is not None:
            raise ValueError("a CLOSE target is its discretionary close_quantity")

    @property
    def fills_total(self) -> int:
        return sum(f.quantity for f in self.confirmed_fills)

    def evidence_ids(self) -> tuple[UUID, ...]:
        """Every input evidence id, in a stable order, without duplicates."""
        ids: list[UUID] = [self.instrument.evidence_id]
        ids += [self.option_quote.quote_id] if self.option_quote else []
        ids += [self.underlying_quote.evidence_id] if self.underlying_quote else []
        ids += [self.account.snapshot_id] if self.account else []
        ids += [self.positions.evidence_id] if self.positions else []
        ids += [self.open_orders.evidence_id] if self.open_orders else []
        ids += [self.debit_funding.evidence_id] if self.debit_funding else []
        ids += list(self.position.entry_fill_ids) if self.position else []
        ids += [f.evidence_id for f in self.confirmed_fills]
        return tuple(dict.fromkeys(ids))


# --------------------------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------------------------


class _Computation:
    """Collects gaps, metrics, and formulas for one call. Local to a single call; no state
    survives it."""

    def __init__(self, inputs: FactInputs, rules: FactsRules, as_of: datetime) -> None:
        self.i = inputs
        self.r = rules
        self.as_of = as_of
        self.inst = inputs.instrument
        self.gaps: dict[str, Gap] = {}
        self.metrics: list[DerivedMetric] = []
        self.formulas: dict[str, str] = {F_QUANTITY: FORMULA_VERSION}
        self.times: dict[UUID, datetime] = {self.inst.evidence_id: self.inst.as_of}
        for obs in (inputs.underlying_quote, inputs.positions, inputs.open_orders):
            if obs is not None:
                self.times[obs.evidence_id] = obs.as_of
        if inputs.option_quote is not None:
            self.times[inputs.option_quote.quote_id] = inputs.option_quote.as_of
        if inputs.account is not None:
            self.times[inputs.account.snapshot_id] = inputs.account.as_of

    # -- recording -----------------------------------------------------------------------

    def gap(self, field: str, kind: DataQuality, detail: str) -> None:
        """Record the first gap for `field`; later gaps for the same field are redundant."""
        self.gaps.setdefault(field, Gap(field=field, kind=kind, detail=detail))

    def metric(
        self, name: str, unit: str, value: Decimal, formula: str, ids: tuple[UUID, ...]
    ) -> None:
        self.formulas[formula] = FORMULA_VERSION
        inputs = tuple(dict.fromkeys(ids))
        observed = min(self.times.get(e, self.as_of) for e in inputs)
        self.metrics.append(
            DerivedMetric(
                name=name,
                unit=unit,
                value=SourcedValue[Dec](
                    evidence_id=uuid5(self.i.facts_id, name),
                    value=value,
                    source=_SOURCE,
                    as_of=observed,
                    retrieved_at=self.as_of,
                    tool_call_id=None,
                    quality=DataQuality.OK,
                    derivation=Derivation(
                        formula=formula,
                        formula_version=FORMULA_VERSION,
                        input_evidence_ids=inputs,
                    ),
                ),
            )
        )

    def contracts(
        self,
        name: str,
        available: Decimal,
        per_contract: Decimal,
        formula: str,
        ids: tuple[UUID, ...],
    ) -> int:
        """floor(max(available, 0) / per_contract); recorded as a capacity metric."""
        count = int(_CTX.divide_int(max(available, _ZERO), per_contract))
        self.metric(f"capacity_{name}_contracts", "contracts", Decimal(count), formula, ids)
        return count

    # -- rules and freshness -------------------------------------------------------------

    def rule(self, setting: T | FactsRuleMarker, key: str) -> T | FactsRuleMarker | None:
        """The rule value, `NONE` for "no limit", or None (gap) for TBD/agent_discretion."""
        if setting is FactsRuleMarker.NONE:
            return FactsRuleMarker.NONE
        if isinstance(setting, FactsRuleMarker):
            self.gap(key, DataQuality.MISSING, f"{key} is {setting.value}; not computable")
            return None
        return setting

    def fresh(
        self, field: str, observed: datetime, setting: int | FactsRuleMarker, key: str
    ) -> bool:
        """Fail-closed freshness (sanity.is_fresh); "none" max age only rejects clock skew."""
        max_age = self.rule(setting, key)
        if max_age is None:
            self.gap(field, DataQuality.STALE, f"freshness unknown: {key} is not set")
            return False
        if max_age is FactsRuleMarker.NONE:
            ok = observed <= self.as_of
        else:
            ok = is_fresh(observed, self.as_of, int(max_age))
        if not ok:
            self.gap(field, DataQuality.STALE, f"{field} is older than {key} or in the future")
        return ok

    def missing(self, field: str) -> None:
        self.gap(field, DataQuality.MISSING, f"{field} was not supplied")

    # -- validated inputs (each records its gap on first use) -----------------------------

    @cached_property
    def quote(self) -> Quote | None:
        q = self.i.option_quote
        if q is None:
            self.missing("option_quote")
            return None
        ok = self.fresh("option_quote", q.as_of, self.r.option_quote_max_age_seconds, _R_OPTION_AGE)
        return q if ok else None

    @cached_property
    def underlying(self) -> tuple[Decimal, UUID] | None:
        uq = self.i.underlying_quote
        if uq is None:
            self.missing("underlying_quote")
            return None
        age = self.r.equity_quote_max_age_seconds
        ok = self.fresh("underlying_quote", uq.as_of, age, _R_EQUITY_AGE)
        return (uq.price, uq.evidence_id) if ok else None

    @cached_property
    def account(self) -> AccountSnapshot | None:
        acct = self.i.account
        if acct is None:
            self.missing("account_snapshot")
            return None
        if not acct.agentic_verified:
            self.gap("account_snapshot", DataQuality.MISSING, "not the verified Agentic account")
            return None
        if acct.quality in _UNUSABLE_SNAPSHOT:
            self.gap("account_snapshot", acct.quality, "account snapshot is not usable")
            return None
        age = self.r.account_state_max_age_seconds
        return acct if self.fresh("account_snapshot", acct.as_of, age, _R_ACCOUNT_AGE) else None

    def cash(self, field: str) -> Decimal | None:
        acct = self.account
        if acct is None:
            return None
        values: dict[str, Decimal | None] = {
            "available_settled_cash_usd": acct.available_settled_cash_usd,
            "csp_reserved_cash_usd": acct.csp_reserved_cash_usd,
            "account_value_usd": acct.account_value_usd,
        }
        value = values[field]
        if value is None:
            self.gap(field, DataQuality.MISSING, "the account snapshot does not establish it")
        return value

    @cached_property
    def positions(self) -> PositionsRead | None:
        read = self.i.positions
        if read is None:
            self.missing("positions")
            return None
        age = self.r.account_state_max_age_seconds
        return read if self.fresh("positions", read.as_of, age, _R_ACCOUNT_AGE) else None

    @cached_property
    def orders(self) -> OpenOrdersRead | None:
        read = self.i.open_orders
        if read is None:
            self.missing("open_orders")
            return None
        age = self.r.account_state_max_age_seconds
        return read if self.fresh("open_orders", read.as_of, age, _R_ACCOUNT_AGE) else None

    @cached_property
    def multiplier(self) -> int | None:
        if self.inst.multiplier is None:
            self.gap("multiplier", DataQuality.MISSING, "contract multiplier is not verified")
        return self.inst.multiplier

    @cached_property
    def dte(self) -> int | None:
        today = self.as_of.astimezone(ZoneInfo(_MARKET_TZ)).date()
        days = (self.inst.occ_symbol.expiration - today).days
        if days < 0:
            self.gap("dte", DataQuality.MISSING, "the contract expired before as_of")
            return None
        self.metric("dte", "calendar_days", Decimal(days), F_DTE, (self.inst.evidence_id,))
        return days

    @cached_property
    def collateral(self) -> tuple[Decimal, tuple[UUID, ...]] | None:
        """Per-contract collateral: CSP strike x m; CC live share price x m (filters notes)."""
        m = self.multiplier
        if m is None:
            return None
        ids: tuple[UUID, ...]
        if self.inst.occ_symbol.right is OptionRight.PUT:
            value, ids = self.inst.occ_symbol.strike * m, (self.inst.evidence_id,)
        else:
            underlying = self.underlying
            if underlying is None:
                return None
            value, ids = underlying[0] * m, (self.inst.evidence_id, underlying[1])
        self.metric("collateral_per_contract_usd", "USD", value, F_COLLATERAL, ids)
        return value, ids

    # -- metrics -------------------------------------------------------------------------

    def annualized(self, name: str, price: Decimal, formula: str, ids: tuple[UUID, ...]) -> None:
        """price x m / collateral x 365 / DTE (definitions.annualization)."""
        collateral, dte, m = self.collateral, self.dte, self.multiplier
        if collateral is None or dte is None or m is None:
            return
        if dte == 0:
            self.gap(name, DataQuality.MISSING, "DTE is 0; annualization is undefined")
            return
        value = _CTX.divide(price * m * _DAYS_PER_YEAR, collateral[0] * dte)
        self.metric(name, "ratio", value, formula, (*ids, *collateral[1]))

    def spread(self) -> None:
        q = self.quote
        if q is not None:
            mid = _CTX.divide(q.bid + q.ask, _TWO)
            value = _CTX.divide(q.ask - q.bid, mid)
            self.metric("spread_ratio_of_mid", "ratio", value, F_SPREAD, (q.quote_id,))

    def opening_metrics(self) -> None:
        self.spread()
        q = self.quote
        if q is not None:
            name = "annualized_yield_on_collateral_ratio"
            self.annualized(name, q.bid, F_YIELD, (q.quote_id,))

    def position_metrics(self, position: PositionBookEntry) -> None:
        self.spread()
        ids: tuple[UUID, ...]
        strike = self.inst.occ_symbol.strike
        underlying = self.underlying
        if underlying is not None:
            price, uid = underlying
            put = self.inst.occ_symbol.right is OptionRight.PUT
            tested = price <= strike if put else price >= strike
            ids = (self.inst.evidence_id, uid)
            self.metric("tested_flag", "flag", Decimal(int(tested)), F_TESTED, ids)
        window = self.rule(self.r.roll_dte_window, "management.roll_dte_window")
        dte = self.dte
        if window is not None and dte is not None:
            inside = True if window is FactsRuleMarker.NONE else dte <= int(window)
            ids = (self.inst.evidence_id,)
            self.metric("in_roll_dte_window_flag", "flag", Decimal(int(inside)), F_ROLL_WINDOW, ids)
        q = self.quote
        credit = position.entry_weighted_credit
        if credit is None:
            self.gap("entry_weighted_credit", DataQuality.MISSING, "no entry fill history")
        elif position.roll_count != 0:
            detail = "credit received across rolls is not defined (definitions.captured_ratio)"
            self.gap("captured_ratio", DataQuality.MISSING, detail)
        elif q is not None:
            value = _CTX.divide(credit - q.ask, credit)
            ids = (q.quote_id, *position.entry_fill_ids)
            self.metric("captured_ratio", "ratio", value, F_CAPTURED, ids)
        if q is not None:
            self.annualized("remaining_yield_ratio", q.ask, F_REMAINING_YIELD, (q.quote_id,))

    # -- capacity ------------------------------------------------------------------------

    def order_cap(self) -> int | FactsRuleMarker | None:
        cap = self.rule(self.r.max_contracts_per_order, "limits.max_contracts_per_order")
        if isinstance(cap, int):
            ids = (self.inst.evidence_id,)
            self.metric("capacity_order_cap_contracts", "contracts", Decimal(cap), F_QUANTITY, ids)
        return cap

    def covering(self, right: OptionRight, field: str) -> tuple[Decimal, int] | None:
        """Existing short contracts plus working sell-to-open orders on this underlying and
        right, each counted once: (sum of strike x m x qty, sum of m x qty). None if a
        multiplier is unverified or a read is unavailable."""
        pos, orders = self.positions, self.orders
        if pos is None or orders is None:
            return None
        underlying = self.inst.underlying
        items = [
            (h.occ_symbol.strike, h.multiplier, h.short_quantity)
            for h in pos.short_options
            if h.underlying == underlying and h.occ_symbol.right is right
        ] + [
            (o.occ_symbol.strike, o.multiplier, o.unfilled_quantity)
            for o in orders.orders
            if o.underlying == underlying
            and o.occ_symbol.right is right
            and o.side is OrderSide.SELL_TO_OPEN
        ]
        collateral, shares = _ZERO, 0
        for strike, mult, qty in items:
            if mult is None:
                self.gap(field, DataQuality.MISSING, "an existing contract's multiplier is unknown")
                return None
            collateral += strike * mult * qty
            shares += mult * qty
        return collateral, shares

    def csp_capacity(self) -> int | None:
        collateral = self.collateral
        if collateral is None:
            return None
        per = collateral[0]
        acct = (self.i.account.snapshot_id,) if self.i.account else ()
        reads = tuple(
            r.evidence_id for r in (self.i.positions, self.i.open_orders) if r is not None
        )
        inst = self.inst.evidence_id
        caps: list[int | FactsRuleMarker | None] = [self.order_cap()]

        usd = self.rule(
            self.r.max_collateral_per_underlying_usd, "limits.max_collateral_per_underlying_usd"
        )
        ratio = self.rule(
            self.r.max_collateral_per_underlying_ratio,
            "limits.max_collateral_per_underlying_ratio",
        )
        existing: Decimal | None = None
        if isinstance(usd, Decimal) or isinstance(ratio, Decimal):
            cover = self.covering(OptionRight.PUT, "existing_csp_collateral_usd")
            if cover is not None:
                existing = cover[0]
                name = "existing_underlying_csp_collateral_usd"
                self.metric(name, "USD", existing, F_CSP, (inst, *reads))
        caps.append(self.underlying_cap(usd, None, existing, per, "underlying_usd", reads))
        value = self.cash("account_value_usd") if isinstance(ratio, Decimal) else None
        ratio_ids = (*reads, *acct)
        caps.append(self.underlying_cap(ratio, value, existing, per, "underlying_ratio", ratio_ids))

        reserve = self.rule(self.r.min_cash_reserve_usd, "limits.min_cash_reserve_usd")
        c = self.cash("available_settled_cash_usd")
        if reserve is None or c is None:
            caps.append(None)
        else:
            reserve_usd = _ZERO if reserve is FactsRuleMarker.NONE else Decimal(reserve)
            caps.append(self.contracts("cash", c - reserve_usd, per, F_CSP, (inst, *acct)))

        total = self.rule(
            self.r.max_total_csp_collateral_ratio, "limits.max_total_csp_collateral_ratio"
        )
        if total is FactsRuleMarker.NONE:
            caps.append(total)
        else:
            r = self.cash("csp_reserved_cash_usd")
            if total is None or c is None or r is None:
                caps.append(None)
            else:
                base = c + r  # B = C + R, pre-order (definitions.cash_accounting)
                room = Decimal(total) * base - r
                caps.append(self.contracts("total_csp", room, per, F_CSP, (inst, *acct)))
        return _smallest(caps)

    def underlying_cap(
        self,
        setting: Decimal | FactsRuleMarker | None,
        account_value: Decimal | None,
        existing: Decimal | None,
        per: Decimal,
        name: str,
        reads: tuple[UUID, ...],
    ) -> int | FactsRuleMarker | None:
        """Per-underlying CSP room: USD cap (or ratio x account value) minus existing."""
        if setting is FactsRuleMarker.NONE:
            return setting
        if setting is None or existing is None:
            return None
        ids = (self.inst.evidence_id, *reads)
        if name == "underlying_usd":
            return self.contracts(name, Decimal(setting) - existing, per, F_CSP, ids)
        if account_value is None:
            return None
        room = Decimal(setting) * account_value - existing
        return self.contracts(name, room, per, F_CSP, ids)

    def cc_capacity(self) -> int | None:
        m = self.multiplier
        ratio = self.rule(self.r.cc_max_lot_coverage_ratio, "limits.cc_max_lot_coverage_ratio")
        cover = self.covering(OptionRight.CALL, "covered_shares")
        caps: list[int | FactsRuleMarker | None] = [self.order_cap()]
        pos = self.positions
        if m is None or ratio is None or cover is None or pos is None or self.i.open_orders is None:
            caps.append(None)
        else:
            owned = sum(h.quantity for h in pos.share_holdings if h.symbol == self.inst.underlying)
            coverage = Decimal(1) if ratio is FactsRuleMarker.NONE else Decimal(ratio)
            ids = (self.inst.evidence_id, pos.evidence_id, self.i.open_orders.evidence_id)
            uncovered = Decimal(owned - cover[1])
            self.metric("uncovered_shares", "shares", uncovered, F_CC, ids)
            room = Decimal(owned) * coverage - cover[1]
            caps.append(self.contracts("coverage", room, Decimal(m), F_CC, ids))
        return _smallest(caps)

    def close_capacity(self, book_short: int) -> int | None:
        pos, orders = self.positions, self.orders
        if pos is None or orders is None:
            return None
        inst_id = self.inst.broker_instrument_id
        broker_short = sum(
            h.short_quantity for h in pos.short_options if h.broker_instrument_id == inst_id
        )
        expected = book_short - self.i.fills_total
        if broker_short != expected:
            detail = f"broker short {broker_short} != position book {book_short} - fills"
            self.gap("short_quantity", DataQuality.CONTRADICTORY, detail)
            return None
        working = sum(
            o.unfilled_quantity
            for o in orders.orders
            if o.broker_instrument_id == inst_id and o.side is OrderSide.BUY_TO_CLOSE
        )
        ids = (self.inst.evidence_id, pos.evidence_id, orders.evidence_id)
        available = max(broker_short - working, 0)
        self.metric("available_short_quantity", "contracts", Decimal(available), F_CLOSE, ids)
        funding = self.funding_cap(ids)
        capacity = available if isinstance(funding, FactsRuleMarker) else min(available, funding)
        self.metric("close_capacity_contracts", "contracts", Decimal(capacity), F_CLOSE, ids)
        return capacity

    def funding_cap(self, ids: tuple[UUID, ...]) -> int | FactsRuleMarker:
        """Broker-reported debit funding, if supplied, limits the close. Without it the
        capacity reflects short quantity only, and the gap is recorded."""
        funding = self.i.debit_funding
        if funding is None:
            self.missing("debit_funding_usd")
            return FactsRuleMarker.NONE
        if not funding.usable or funding.value is None or funding.as_of is None:
            self.gap("debit_funding_usd", funding.quality, funding.reason or "not usable")
            return FactsRuleMarker.NONE
        age = self.r.account_state_max_age_seconds
        if not self.fresh("debit_funding_usd", funding.as_of, age, _R_ACCOUNT_AGE):
            return FactsRuleMarker.NONE
        q = self.i.limit_price
        price_ids: tuple[UUID, ...] = ()
        if q is None and self.quote is not None:
            q, price_ids = self.quote.ask, (self.quote.quote_id,)
        m = self.multiplier
        if q is None or m is None:
            self.gap("debit_funding_capacity", DataQuality.MISSING, "no debit price or multiplier")
            return FactsRuleMarker.NONE
        self.times[funding.evidence_id] = funding.as_of
        all_ids = (*ids, funding.evidence_id, *price_ids)
        return self.contracts("debit_funding", funding.value, q * m, F_CLOSE, all_ids)


def _smallest(caps: list[int | FactsRuleMarker | None]) -> int | None:
    """Floor of the smallest applicable capacity; any unavailable cap makes it unavailable.

    `NONE` entries do not constrain. At least one numeric cap is always present when no cap
    is unavailable (cash/coverage/short quantity always apply).
    """
    if any(cap is None for cap in caps):
        return None
    return min(cap for cap in caps if isinstance(cap, int))


def compute_decision_facts(inputs: FactInputs, rules: FactsRules, as_of: datetime) -> DecisionFacts:
    """Compute DecisionFacts for one subject and purpose (OUTPUT_ASSEMBLY.md; ADR-0014/0015).

    Pure: `as_of` is the observation time supplied by the caller; no clock, env, DB, or
    network. Every value comes from `inputs` via a formula listed in the module docstring;
    anything missing, stale, or contradictory is None with a typed gap.
    """
    require_aware(as_of, "as_of")
    comp = _Computation(inputs, rules, as_of)
    capacity: int | None
    target: int | None
    if inputs.purpose is FactsPurpose.OPEN:
        comp.opening_metrics()
        put = inputs.instrument.occ_symbol.right is OptionRight.PUT
        capacity = comp.csp_capacity() if put else comp.cc_capacity()
        established = inputs.established_target_quantity
        target = capacity if established is None else established
    else:
        position = inputs.position
        if position is None:  # pragma: no cover - FactInputs guarantees a position
            raise ValueError("a CLOSE/MANAGE fact set needs a position")
        comp.position_metrics(position)
        book_short = next(
            ci.short_quantity
            for ci in position.current_instruments
            if ci.broker_instrument_id == inputs.instrument.broker_instrument_id
        )
        capacity = comp.close_capacity(book_short)
        if inputs.purpose is FactsPurpose.CLOSE:
            target = _close_target(comp, inputs.close_quantity, book_short)
        else:
            established = inputs.established_target_quantity
            target = book_short if established is None else established

    remaining: int | None = None
    fills = inputs.fills_total
    if target is not None and capacity is not None:
        if fills > target:
            detail = "confirmed fills exceed the established target"
            comp.gap("remaining_quantity", DataQuality.CONTRADICTORY, detail)
        else:
            remaining = min(target - fills, capacity)
    others = ", ".join(sorted(comp.gaps)) or "none"
    if target is None:
        comp.gap("initial_quantity", DataQuality.MISSING, f"unavailable; see gaps: {others}")
    if remaining is None:
        comp.gap("remaining_quantity", DataQuality.MISSING, f"unavailable; see gaps: {others}")

    gaps = tuple(comp.gaps.values())
    kinds = {g.kind for g in gaps}
    quality = next((k for k in _QUALITY_ORDER if k in kinds), DataQuality.OK)
    return DecisionFacts(
        facts_id=inputs.facts_id,
        facts_ref=inputs.facts_ref,
        run_id=inputs.run_id,
        subject_ref=inputs.subject_ref,
        purpose=inputs.purpose,
        observed_at=as_of,
        rules_version=rules.rules_version,
        rules_hash=rules.rules_hash,
        input_evidence_ids=inputs.evidence_ids(),
        snapshot_ref=inputs.account.snapshot_id if inputs.account else None,
        quote_refs=(inputs.option_quote.quote_id,) if inputs.option_quote else (),
        candidate_ref=inputs.candidate.candidate_ref if inputs.candidate else None,
        position_id=inputs.position.position_id if inputs.position else None,
        limit_price=inputs.limit_price,
        close_quantity=inputs.close_quantity,
        formula_versions=tuple(
            FormulaVersion(formula=f, version=v) for f, v in sorted(comp.formulas.items())
        ),
        metrics=tuple(comp.metrics),
        initial_quantity=target,
        remaining_quantity=remaining,
        quality=quality,
        gaps=gaps,
    )


def _close_target(comp: _Computation, close_quantity: int | None, book_short: int) -> int | None:
    """CLOSE target is the agent's discretionary quantity, 1..current short (ADR-0015)."""
    if close_quantity is None:
        detail = "management.close_quantity is agent_discretion; pass a close quantity"
        comp.gap("close_quantity", DataQuality.MISSING, detail)
        return None
    if close_quantity > book_short:
        detail = f"close quantity {close_quantity} exceeds current short quantity {book_short}"
        comp.gap("close_quantity", DataQuality.CONTRADICTORY, detail)
        return None
    return close_quantity
