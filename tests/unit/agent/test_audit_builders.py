"""Scenario builders for the post-run audit tests (no network, fixed clock).

`Scenario` records tool calls, broker evidence, and the assembled record, then builds an
immutable `AuditContext`. Times are seconds after T0 on a fixed NYSE session date.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from wheelta_robinhood_agent.agent.audit import (
    AuditContext,
    BrokerState,
    DayHistory,
    InstrumentFact,
    ReviewObservation,
    ShareHolding,
    ShortOptionHolding,
    WorkingOrderObservation,
)
from wheelta_robinhood_agent.config.rules import TradingRules, load_rules
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    Decision,
    ProposedLeg,
)
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    AuditCheck,
    AuditOutcome,
    CancellationStatus,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OrderSide,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts import DecisionFacts, FactsPurpose
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import (
    Attempt,
    BrokerOrder,
    Cancellation,
    FillObservationKind,
    FillRecord,
    OrderIntent,
    OrderRecord,
    ReasonCode,
    StatusObservation,
)
from wheelta_robinhood_agent.domain.positions import (
    PositionBook,
    PositionBookEntry,
    PositionInstrument,
)
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import (
    AssemblyFinding,
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    Quote,
    RunRecord,
    UnassociatedAction,
    UnassociatedActionKind,
)
from wheelta_robinhood_agent.domain.tool_calls import (
    ToolCallDecision,
    ToolCallIdentity,
    ToolCallRecord,
)

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
RULES: TradingRules = load_rules().rules
RUN_ID = UUID("00000000-0000-4000-8000-000000000001")

PUT = OccSymbol.parse("AAPL  261016P00050000")  # strike 50 -> 5000 collateral per contract
PUT2 = OccSymbol.parse("MSFT  261016P00070000")  # strike 70 -> 7000
CALL = OccSymbol.parse("AAPL  261016C00060000")
NEW_PUT = OccSymbol.parse("AAPL  261120P00045000")

STO = OrderSide.SELL_TO_OPEN.value
BTC = OrderSide.BUY_TO_CLOSE.value


def t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def rules_with(**sections: dict[str, object]) -> TradingRules:
    """Copy of the real rules with section values replaced (e.g. limits={...})."""
    update = {}
    for name, values in sections.items():
        if name == "freshness":
            dq = RULES.data_quality
            update["data_quality"] = dq.model_copy(
                update={"freshness": dq.freshness.model_copy(update=values)}
            )
        else:
            update[name] = getattr(RULES, name).model_copy(update=values)
    return RULES.model_copy(update=update)


@dataclass
class _Order:
    intent: OrderIntent
    broker: BrokerOrder | None
    statuses: list[StatusObservation] = field(default_factory=list)
    fills: list[FillRecord] = field(default_factory=list)
    cancellations: list[Cancellation] = field(default_factory=list)
    quote_refs: tuple[UUID, ...] = ()
    snapshot_ref: UUID | None = None


@dataclass
class LegSpec:
    side: OrderSide
    occ: OccSymbol | None
    iid: str | None
    target: int | None
    places: list[ToolCallRecord] = field(default_factory=list)
    proposal: tuple[int, Decimal, UUID | None, tuple[UUID, ...]] | None = None
    conditional: bool = False
    depends_on: str | None = None
    facts_ref: str | None = None
    leg_ref: str = ""
    record_filled: dict[int, int] = field(default_factory=dict)
    record_overrides: dict[int, dict[str, object]] = field(default_factory=dict)


@dataclass
class DecisionSpec:
    action: DecisionAction
    legs: list[LegSpec]
    target_ref: str
    priority: int | None
    position_id: UUID | None = None
    depends_on: tuple[str, ...] = ()
    decision_ref: str = ""


class Scenario:
    """Mutable builder; `ctx()` returns the frozen AuditContext."""

    def __init__(self, mode: ExecutionMode = ExecutionMode.LIVE) -> None:
        self.mode = mode
        self.rules = RULES
        self.calls: list[ToolCallRecord] = []
        self.orders: dict[UUID, _Order] = {}
        self.extra_orders: list[OrderRecord] = []
        self.quotes: list[Quote] = []
        self.instruments: list[InstrumentFact] = []
        self.reviews: list[ReviewObservation] = []
        self.states: list[BrokerState] = []
        self.fact_sets: list[DecisionFacts] = []
        self.decisions: list[DecisionSpec] = []
        self.unassociated: list[UnassociatedAction] = []
        self.standalone_cancels: list[Cancellation] = []
        self.findings: list[AssemblyFinding] = []
        self.output_status = DecisionOutputStatus.PARSED
        self.output: AgentDecisionOutput | None = None
        self.auto_output = True
        self.record = True
        self.book: PositionBook | None = PositionBook(as_of=T0, entries=())
        self.day: DayHistory | None = DayHistory(trading_date=T0.date(), prior_new_lineage_ids=())
        self.place_unrecorded: set[UUID] = set()
        self.drop_calls: set[UUID] = set()

    # -- tool calls ---------------------------------------------------------------------------

    def call(
        self,
        tool: str,
        at: float,
        *,
        status: ToolCallStatus = ToolCallStatus.SUCCEEDED,
        duration: float = 1,
        denied: bool = False,
        args: dict[str, object] | None = None,
    ) -> ToolCallRecord:
        call_id = uuid4()
        identity = ToolCallIdentity(
            tool_call_id=call_id,
            sdk_tool_use_id=f"toolu_{call_id.hex[:12]}",
            run_id=RUN_ID,
            stage="agent",
            server="robinhood",
            tool=f"mcp__robinhood__{tool}",
            tier=ToolTier.X
            if tool.endswith("_order") and not tool.startswith("get")
            else ToolTier.R,
            requested_at=t(at),
            arguments_redacted=args or {},
        )
        if denied:
            record = ToolCallRecord(
                identity=identity,
                effective_arguments_redacted=None,
                decision=ToolCallDecision.DENIED,
                deny_reason="kill switch",
                status=ToolCallStatus.DENIED,
                dispatched_at=None,
                completed_at=None,
            )
        else:
            completed = (
                None
                if status in (ToolCallStatus.UNKNOWN, ToolCallStatus.REQUESTED)
                else t(at + duration)
            )
            record = ToolCallRecord(
                identity=identity,
                effective_arguments_redacted=args or {},
                decision=ToolCallDecision.ALLOWED,
                status=status,
                dispatched_at=t(at),
                completed_at=completed,
                result_ref=uuid4() if status is ToolCallStatus.SUCCEEDED else None,
            )
        self.calls.append(record)
        return record

    # -- evidence -----------------------------------------------------------------------------

    def inst(
        self,
        iid: str,
        occ: OccSymbol,
        *,
        multiplier: int | None = 100,
        tick: Decimal | None = Decimal("0.01"),
    ) -> InstrumentFact:
        call = self.call("get_option_instruments", -100)
        fact = InstrumentFact(
            broker_instrument_id=iid,
            occ_symbol=occ,
            multiplier=multiplier,
            tick_increment=tick,
            source_tool_call_id=call.identity.tool_call_id,
        )
        self.instruments.append(fact)
        return fact

    def quote(
        self,
        iid: str,
        at: float,
        bid: str = "1.00",
        ask: str = "1.20",
        *,
        tool: str = "get_option_quotes",
        status: ToolCallStatus = ToolCallStatus.SUCCEEDED,
    ) -> Quote:
        call = self.call(tool, at - 1, status=status)
        q = Quote(
            quote_id=uuid4(),
            broker_instrument_id=iid,
            bid=Decimal(bid),
            ask=Decimal(ask),
            as_of=t(at - 1),
            source_tool_call_ids=(call.identity.tool_call_id,),
        )
        self.quotes.append(q)
        return q

    def state(
        self,
        at: float,
        *,
        cash: str | None = "10000",
        reserved: str | None = "0",
        value: str | None = "20000",
        shorts: tuple[tuple[str, int], ...] | None = (),
        shares: tuple[tuple[str, int, int | None], ...] | None = (),
        working: tuple[WorkingOrderObservation, ...] | None = (),
        reflects: tuple[ToolCallRecord, ...] = (),
        verified: bool = True,
        as_of: float | None = None,
    ) -> BrokerState:
        call = self.call("get_accounts", at - 1)
        c = Decimal(cash) if cash is not None else None
        r = Decimal(reserved) if reserved is not None else None
        base = c + r if c is not None and r is not None else None
        gaps = tuple(
            Gap(field=name, kind=DataQuality.MISSING, detail="unmapped")
            for name, v in (
                ("available_settled_cash_usd", c),
                ("csp_reserved_cash_usd", r),
                ("csp_cash_base_usd", base),
                ("account_value_usd", value),
            )
            if v is None
        )
        snap = AccountSnapshot(
            snapshot_id=uuid4(),
            as_of=t(as_of if as_of is not None else at - 1),
            retrieved_at=t(at - 1),
            tool_call_ids=(call.identity.tool_call_id,),
            account_ref="****1234",
            agentic_verified=verified,
            account_value_usd=Decimal(value) if value is not None else None,
            available_settled_cash_usd=c,
            csp_reserved_cash_usd=r,
            csp_cash_base_usd=base,
            csp_cash_base_evidence_ids=(uuid4(),) if base is not None else (),
            positions_ref=None,
            open_orders_ref=None,
            tax_lots_ref=None,
            quality=DataQuality.MISSING if gaps else DataQuality.OK,
            gaps=gaps,
        )
        state = BrokerState(
            snapshot=snap,
            completed_at=t(at),
            short_options=None
            if shorts is None
            else tuple(
                ShortOptionHolding(broker_instrument_id=i, short_quantity=q) for i, q in shorts
            ),
            share_holdings=None
            if shares is None
            else tuple(
                ShareHolding(underlying=u, quantity=q, other_reserved_shares=o)
                for u, q, o in shares
            ),
            working_orders=working,
            reflects_mutation_tool_call_ids=tuple(c.identity.tool_call_id for c in reflects),
        )
        self.states.append(state)
        return state

    def review(
        self,
        at: float,
        iid: str,
        side: str,
        qty: int,
        price: str,
        *,
        succeeded: bool = True,
        warnings: tuple[str, ...] = (),
        tif: str | None = "day",
        record: bool = True,
    ) -> ToolCallRecord:
        call = self.call("review_option_order", at - 1)
        if record:
            self.reviews.append(
                ReviewObservation(
                    review_tool_call_id=call.identity.tool_call_id,
                    completed_at=t(at),
                    succeeded=succeeded,
                    warnings=warnings,
                    broker_instrument_id=iid,
                    side_raw=side,
                    quantity=qty,
                    order_type_raw="limit",
                    time_in_force_raw=tif,
                    limit_price=Decimal(price),
                )
            )
        return call

    def place(
        self,
        at: float,
        iid: str,
        occ: OccSymbol,
        side: str,
        qty: int,
        price: str,
        *,
        status: ToolCallStatus = ToolCallStatus.SUCCEEDED,
        denied: bool = False,
        order_type: str = "limit",
        tif: str = "day",
        broker_id: str | None = None,
        quotes: tuple[Quote, ...] = (),
        snapshot: BrokerState | None = None,
        review: bool = True,
    ) -> ToolCallRecord:
        if review and not denied:
            self.review(at - 0.5, iid, side, qty, price, tif=tif)
        call = self.call("place_option_order", at, status=status, denied=denied)
        cid = call.identity.tool_call_id
        intent = OrderIntent(
            intent_id=uuid4(),
            run_id=RUN_ID,
            place_tool_call_id=cid,
            account_scope_id="acct",
            occ_symbol=occ,
            broker_instrument_id=iid,
            side_raw=side,
            quantity=qty,
            order_type_raw=order_type,
            time_in_force_raw=tif,
            limit_price=Decimal(price),
            requested_at=t(at),
        )
        broker = None
        if status is ToolCallStatus.SUCCEEDED and not denied:
            broker = BrokerOrder(
                order_id=uuid4(),
                account_scope_id="acct",
                broker_order_id=broker_id or f"ord-{cid.hex[:8]}",
                intent_id=intent.intent_id,
                first_observed_at=t(at + 1),
            )
        self.orders[cid] = _Order(
            intent=intent,
            broker=broker,
            quote_refs=tuple(q.quote_id for q in quotes),
            snapshot_ref=snapshot.snapshot.snapshot_id if snapshot else None,
        )
        return call

    def status(
        self,
        place: ToolCallRecord,
        at: float,
        status: AttemptStatus,
        *,
        filled: int | None = None,
        duration: float = 1,
    ) -> ToolCallRecord:
        """An order read at `at` reporting `status` (and a cumulative filled total)."""
        call = self.call("get_option_orders", at - duration, duration=duration)
        order = self.orders[place.identity.tool_call_id]
        order.statuses.append(
            StatusObservation(
                status=status,
                broker_status_raw=status.value,
                observed_at=t(at),
                tool_call_id=call.identity.tool_call_id,
            )
        )
        if filled is not None:
            self.fill(place, at, filled, source=call, kind=FillObservationKind.CUMULATIVE)
        return call

    def fill(
        self,
        place: ToolCallRecord,
        at: float,
        qty: int,
        *,
        source: ToolCallRecord | None = None,
        kind: FillObservationKind = FillObservationKind.EXECUTION,
        execution_id: str | None = None,
    ) -> FillRecord:
        order = self.orders[place.identity.tool_call_id]
        assert order.broker is not None
        src = source or self.call("get_option_orders", at - 1)
        record = FillRecord(
            fill_id=uuid4(),
            order_id=order.broker.order_id,
            kind=kind,
            broker_execution_id=(execution_id or f"x-{uuid4().hex[:6]}")
            if kind is FillObservationKind.EXECUTION
            else None,
            quantity=qty,
            price=Decimal("1.00"),
            executed_at=t(at),
            observed_at=t(at),
            source_tool_call_id=src.identity.tool_call_id,
        )
        order.fills.append(record)
        return record

    def cancel(
        self,
        place: ToolCallRecord,
        at: float,
        *,
        status: ToolCallStatus = ToolCallStatus.SUCCEEDED,
        outcome: CancellationStatus = CancellationStatus.CONFIRMED,
        confirm: ToolCallRecord | None = None,
        standalone: bool = False,
    ) -> ToolCallRecord:
        call = self.call("cancel_option_order", at, status=status)
        order = self.orders[place.identity.tool_call_id]
        c = Cancellation(
            cancel_tool_call_id=call.identity.tool_call_id,
            broker_order_id=order.broker.broker_order_id if order.broker else None,
            confirmation_tool_call_ids=(confirm.identity.tool_call_id,) if confirm else (),
            status=outcome
            if confirm or outcome is not CancellationStatus.CONFIRMED
            else CancellationStatus.PENDING,
        )
        order.cancellations.append(c)
        if standalone:
            self.standalone_cancels.append(c)
        return call

    def facts(
        self,
        ref: str,
        at: float,
        *,
        snapshot: BrokerState | None = None,
        quotes: tuple[Quote, ...] = (),
        purpose: FactsPurpose = FactsPurpose.OPEN,
        qty: int | None = 1,
        limit_price: str | None = None,
    ) -> DecisionFacts:
        gaps = (
            ()
            if qty is not None
            else tuple(
                Gap(field=f, kind=DataQuality.MISSING, detail="x")
                for f in ("initial_quantity", "remaining_quantity")
            )
        )
        f = DecisionFacts(
            facts_id=uuid4(),
            facts_ref=ref,
            run_id=RUN_ID,
            subject_ref=f"subject:{ref}",
            purpose=purpose,
            observed_at=t(at),
            rules_version="5",
            rules_hash="h",
            input_evidence_ids=(uuid4(),),
            snapshot_ref=snapshot.snapshot.snapshot_id if snapshot else None,
            quote_refs=tuple(q.quote_id for q in quotes),
            limit_price=Decimal(limit_price) if limit_price else None,
            initial_quantity=qty,
            remaining_quantity=qty,
            quality=DataQuality.OK if qty is not None else DataQuality.MISSING,
            gaps=gaps,
        )
        self.fact_sets.append(f)
        return f

    def decide(
        self,
        action: DecisionAction,
        *legs: LegSpec,
        target_ref: str | None = None,
        priority: int | None = None,
        position_id: UUID | None = None,
        depends_on: tuple[str, ...] = (),
    ) -> DecisionSpec:
        n = len(self.decisions)
        spec = DecisionSpec(
            action=action,
            legs=list(legs),
            target_ref=target_ref or f"candidate:{n}",
            priority=n if priority is None else priority,
            position_id=position_id,
            depends_on=depends_on,
            decision_ref=f"decision:{n}",
        )
        for i, leg in enumerate(legs):
            leg.leg_ref = leg.leg_ref or f"leg:{n}:{i}"
        self.decisions.append(spec)
        return spec

    def position(self, ref: str, iid: str, occ: OccSymbol, qty: int = 1) -> PositionBookEntry:
        entry = PositionBookEntry(
            position_id=uuid4(),
            position_ref=ref,
            underlying=occ.root,
            strategy="csp" if occ.right.value == "put" else "cc",
            current_instruments=(
                PositionInstrument(occ_symbol=occ, broker_instrument_id=iid, short_quantity=qty),
            ),
            entry_date=None,
            entry_weighted_credit=None,
            thesis=None,
            roll_count=None,
            history_quality=DataQuality.MISSING,
            gaps=(Gap(field="entry", kind=DataQuality.MISSING, detail="imported"),),
        )
        book = self.book or PositionBook(as_of=T0, entries=())
        self.book = PositionBook(as_of=T0, entries=(*book.entries, entry))
        return entry

    # -- build --------------------------------------------------------------------------------

    def order_record(self, cid: UUID) -> OrderRecord:
        o = self.orders[cid]
        return OrderRecord(
            intent=o.intent,
            broker_order=o.broker,
            status_history=tuple(o.statuses),
            cancellations=tuple(o.cancellations),
            fills=tuple(o.fills),
        )

    def attempt_for(self, index: int, place: ToolCallRecord, leg: LegSpec | None = None) -> Attempt:
        cid = place.identity.tool_call_id
        o = self.orders[cid]
        rec = self.order_record(cid)
        if place.status is ToolCallStatus.DENIED:
            status, broker_id, filled = AttemptStatus.NOT_PLACED, None, None
        elif o.broker is None:
            status, broker_id, filled = AttemptStatus.UNKNOWN, None, None
        else:
            status = rec.status if o.statuses else AttemptStatus.PLACED
            broker_id = o.broker.broker_order_id
            filled = rec.filled_quantity
        if leg is not None and index in leg.record_filled:
            filled = leg.record_filled[index]
        data: dict[str, object] = dict(
            index=index,
            place_tool_call_id=cid,
            proposal_ref=None,
            requested_quantity=o.intent.quantity,
            order_type_raw=o.intent.order_type_raw,
            time_in_force_raw=o.intent.time_in_force_raw,
            limit_price=o.intent.limit_price,
            snapshot_ref=o.snapshot_ref,
            quote_refs=o.quote_refs,
            status=status,
            broker_order_id=broker_id,
            fill_ids=tuple(f.fill_id for f in o.fills)
            if status is not AttemptStatus.NOT_PLACED
            else (),
            filled_quantity=filled,
            cancel_tool_call_ids=tuple(c.cancel_tool_call_id for c in o.cancellations),
        )
        if leg is not None:
            data.update(leg.record_overrides.get(index, {}))
        return Attempt.model_validate(data)

    def _leg(self, spec: LegSpec) -> LegRecord:
        attempts = [self.attempt_for(i, p, spec) for i, p in enumerate(spec.places)]
        if spec.proposal is not None:
            qty, price, snap_ref, quote_refs = spec.proposal
            attempts.append(
                Attempt(
                    index=len(attempts),
                    place_tool_call_id=None,
                    proposal_ref=f"proposal:{spec.leg_ref}",
                    requested_quantity=qty,
                    order_type_raw="limit",
                    time_in_force_raw=self.rules.orders.time_in_force,
                    limit_price=price,
                    snapshot_ref=snap_ref,
                    quote_refs=quote_refs,
                    status=AttemptStatus.NOT_PLACED,
                    broker_order_id=None,
                    filled_quantity=None,
                    reason_codes=(ReasonCode.DRY_RUN,),
                )
            )
        quotes = tuple(q for q in self.quotes if any(q.quote_id in a.quote_refs for a in attempts))
        return LegRecord(
            leg_ref=spec.leg_ref,
            depends_on_leg_ref=spec.depends_on,
            side=spec.side,
            occ_symbol=spec.occ,
            broker_instrument_id=spec.iid,
            right=spec.occ.right if spec.occ else None,
            strike=spec.occ.strike if spec.occ else None,
            expiration=spec.occ.expiration if spec.occ else None,
            target_quantity=spec.target,
            facts_ref=spec.facts_ref,
            quotes=quotes,
            conditional=spec.conditional,
            attempts=tuple(attempts),
            gaps=()
            if spec.target is not None
            else (Gap(field="target_quantity", kind=DataQuality.MISSING, detail="x"),),
        )

    def run_record(self) -> RunRecord | None:
        if not self.record:
            return None
        parsed = self.output_status is DecisionOutputStatus.PARSED
        decisions = (
            tuple(
                DecisionRecord(
                    decision_ref=d.decision_ref,
                    action=d.action,
                    priority=d.priority,
                    target_ref=d.target_ref,
                    replacement_ref=None,
                    underlying=None,
                    position_id=d.position_id,
                    depends_on_decision_refs=d.depends_on,
                    rationale="because",
                    thesis=None,
                    legs=tuple(self._leg(leg) for leg in d.legs),
                )
                for d in self.decisions
            )
            if parsed
            else ()
        )
        associated = (
            {p.identity.tool_call_id for d in self.decisions for leg in d.legs for p in leg.places}
            if parsed
            else set()
        )
        unassoc = list(self.unassociated)
        for cid in self.orders:
            call = next(c for c in self.calls if c.identity.tool_call_id == cid)
            if cid not in associated and cid not in self.place_unrecorded:
                unassoc.append(
                    UnassociatedAction(
                        kind=UnassociatedActionKind.PLACE,
                        attempt=self.attempt_for(0, call),
                    )
                )
        cancels = [c for o in self.orders.values() for c in o.cancellations]
        return RunRecord(
            schema_version=4,
            assembler_version="asm-1",
            input_hash="ih",
            run_id=RUN_ID,
            environment=AppEnv.STAGING,
            slot=T0,
            terminated_at=t(3000),
            requested_execution_mode=self.mode,
            effective_execution_mode=self.mode,
            rules_version="5",
            rules_hash="rh",
            prompt_id="wheel_agent",
            prompt_hash="ph",
            model_id="m",
            decision_output_status=self.output_status,
            decisions=decisions,
            cancellations=tuple(cancels),
            unassociated_actions=tuple(unassoc),
            findings=tuple(self.findings),
            summary="summary",
        )

    def decision_output(self) -> AgentDecisionOutput | None:
        if not self.auto_output:
            return self.output
        if self.output_status is not DecisionOutputStatus.PARSED:
            return None
        decisions = []
        for d in self.decisions:
            proposed = tuple(
                ProposedLeg(facts_ref=leg.facts_ref, limit_price=str(leg.proposal[1]))
                for leg in d.legs
                if leg.proposal is not None and leg.facts_ref is not None
            )
            decisions.append(
                Decision(
                    action=d.action,
                    target_ref=d.target_ref,
                    replacement_ref=None,
                    funding_close_refs=(),
                    proposed_legs=proposed,
                    execution_refs=(),
                    rationale="because",
                    thesis=None,
                    invalidation_conditions=(),
                    evidence_refs=(),
                )
            )
        return AgentDecisionOutput(
            decisions=tuple(decisions), cancellation_rationales=(), unresolved_questions=()
        )

    def ctx(self) -> AuditContext:
        return AuditContext(
            run_id=RUN_ID,
            effective_execution_mode=self.mode,
            rules=self.rules,
            rules_version="5",
            rules_hash="rh",
            tool_calls=tuple(
                c for c in self.calls if c.identity.tool_call_id not in self.drop_calls
            ),
            order_records=tuple(self.order_record(cid) for cid in self.orders)
            + tuple(self.extra_orders),
            quotes=tuple(self.quotes),
            instruments=tuple(self.instruments),
            reviews=tuple(self.reviews),
            broker_states=tuple(self.states),
            decision_facts=tuple(self.fact_sets),
            decision_output=self.decision_output(),
            run_record=self.run_record(),
            position_book=self.book,
            day_history=self.day,
        )


def outcomes(findings: tuple[AuditFinding, ...], sub: str | None = None) -> list[AuditOutcome]:
    return [f.outcome for f in findings if sub is None or f.sub_item == sub]


def only(findings: tuple[AuditFinding, ...], sub: str) -> AuditFinding:
    hits = [f for f in findings if f.sub_item == sub]
    assert len(hits) == 1, hits
    return hits[0]


def reasons(findings: tuple[AuditFinding, ...], sub: str | None = None) -> list[str]:
    return [
        f.detail.split(":", 1)[0]
        for f in findings
        if f.outcome is AuditOutcome.UNVERIFIABLE and (sub is None or f.sub_item == sub)
    ]


def live_csp(
    s: Scenario,
    *,
    qty: int = 1,
    price: str = "1.10",
    cash: str = "10000",
    reserved: str = "0",
    at: float = 100,
) -> tuple[ToolCallRecord, LegSpec]:
    """A standard live OPEN_CSP: instrument, fresh state, quote, review, place, fill."""
    if not any(i.broker_instrument_id == "put-1" for i in s.instruments):
        s.inst("put-1", PUT)
    state = s.state(at - 10, cash=cash, reserved=reserved)
    q = s.quote("put-1", at - 5)
    place = s.place(at, "put-1", PUT, STO, qty, price, quotes=(q,), snapshot=state)
    s.status(place, at + 5, AttemptStatus.FILLED, filled=qty)
    leg = LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", qty, places=[place], facts_ref="facts:0")
    s.facts("facts:0", at - 4, snapshot=state, quotes=(q,), qty=qty)
    s.decide(DecisionAction.OPEN_CSP, leg)
    return place, leg


def off_csp(
    s: Scenario,
    *,
    iid: str = "put-1",
    occ: OccSymbol = PUT,
    qty: int = 1,
    price: str = "1.10",
    state: BrokerState | None = None,
    priority: int | None = None,
    conditional: bool = False,
    depends_on: str | None = None,
    ref: str | None = None,
) -> LegSpec:
    if not any(i.broker_instrument_id == iid for i in s.instruments):
        s.inst(iid, occ)
    state = state or (s.states[0] if s.states else s.state(10))
    q = s.quote(iid, 50)
    ref = ref or f"facts:{len(s.fact_sets)}"
    s.facts(ref, 50, snapshot=state, quotes=(q,), qty=qty)
    leg = LegSpec(
        OrderSide.SELL_TO_OPEN,
        occ,
        iid,
        qty,
        proposal=(qty, Decimal(price), state.snapshot.snapshot_id, (q.quote_id,)),
        facts_ref=ref,
        conditional=conditional,
        depends_on=depends_on,
    )
    return leg


def check_ids(findings: tuple[AuditFinding, ...]) -> set[AuditCheck]:
    return {f.check_id for f in findings}


def test_builders_produce_valid_contexts() -> None:
    s = Scenario()
    live_csp(s)
    ctx = s.ctx()
    assert ctx.run_record is not None
    assert len(ctx.context_hash) == 64
    assert ctx.context_hash == ctx.context_hash
    off = Scenario(ExecutionMode.OFF)
    off.decide(DecisionAction.OPEN_CSP, off_csp(off))
    assert off.ctx().run_record is not None
