"""Shared, pure helpers for the V1–V7 checks: findings, attempts, pre-order state.

Nothing here reads the clock, the network, or an LLM. Every helper returns None (or a named
reason) when evidence is missing; callers turn that into `unverifiable`, never `pass`.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import TypedDict, Unpack
from uuid import UUID, uuid5

from wheelta_robinhood_agent.agent.audit.context import (
    AuditContext,
    BrokerState,
    InstrumentFact,
)
from wheelta_robinhood_agent.config.rules import RuleMarker
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditCheck,
    AuditOutcome,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.facts import DecisionFacts
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import (
    Attempt,
    FillRecord,
    OrderIntent,
    OrderRecord,
    StatusObservation,
    filled_quantity,
)
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import (
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    Quote,
)
from wheelta_robinhood_agent.domain.sanity import is_fresh
from wheelta_robinhood_agent.domain.tool_calls import ToolCallRecord

_FINDING_NS = UUID("6f1b0c52-8f7e-4f2a-9a57-3a3b0f6f1a11")

PLACE_TOOL = "place_option_order"
CANCEL_TOOL = "cancel_option_order"
REVIEW_TOOL = "review_option_order"
ORDER_READ_TOOL = "get_option_orders"
QUOTE_TOOLS = frozenset({"get_option_quotes", "get_option_chains"})
MUTATION_TOOLS = frozenset({PLACE_TOOL, CANCEL_TOOL})

TERMINAL_STATUSES = frozenset(
    {
        AttemptStatus.FILLED,
        AttemptStatus.CANCELLED,
        AttemptStatus.REJECTED,
        AttemptStatus.EXPIRED,
    }
)
_UNCERTAIN_TOOL_STATUSES = frozenset(
    {ToolCallStatus.FAILED, ToolCallStatus.TIMED_OUT, ToolCallStatus.UNKNOWN}
)


class Reason(StrEnum):
    """Named reasons carried at the start of every `unverifiable` finding's detail."""

    DRY_RUN_NO_EXECUTION = "dry_run_no_execution"
    MISSING_FINAL_OUTPUT = "missing_final_output"
    MISSING_RUN_RECORD = "missing_run_record"
    MISSING_EVIDENCE = "missing_evidence"
    NO_PRE_ORDER_STATE = "no_pre_order_state"
    STALE_STATE = "stale_state"
    STATE_MISSING_MUTATION = "state_missing_mutation"
    UNVERIFIED_ACCOUNT = "unverified_account"
    UNMAPPED_BROKER_FIELD = "unmapped_broker_field"
    RULE_TBD = "rule_tbd"
    OPEN_DEFINITION = "open_definition"
    AMBIGUOUS_ORDER = "ambiguous_order"
    BASELINE_CHANGED = "baseline_changed"
    UPSTREAM_UNKNOWN = "upstream_reservation_unknown"
    UNCERTAIN_ORDERING = "uncertain_ordering"
    TERMINAL_UNKNOWN = "terminal_state_unknown"
    UNASSOCIATED = "unassociated_action"
    TARGET_UNKNOWN = "target_unknown"
    LINEAGE_UNKNOWN = "lineage_unknown"
    RULE_ATTRIBUTION_UNKNOWN = "management_rule_attribution_unknown"


@dataclass(frozen=True)
class Unknown:
    """Why a value needed by a check cannot be established (becomes `unverifiable`)."""

    reason: Reason
    detail: str


class FindingKw(TypedDict, total=False):
    attempt: "AuditAttempt | None"
    decision_ref: str | None
    leg_ref: str | None
    rule_key: str | None
    rule_value: object
    observed: object
    tool_call_ids: Iterable[UUID]


class Findings:
    """Collects findings for one check; ids are deterministic from the context hash."""

    def __init__(self, ctx: AuditContext, check: AuditCheck) -> None:
        self._ctx = ctx
        self._check = check
        self._hash = ctx.context_hash
        self._items: list[AuditFinding] = []

    def add(
        self,
        sub_item: str,
        outcome: AuditOutcome,
        detail: str,
        *,
        attempt: "AuditAttempt | None" = None,
        decision_ref: str | None = None,
        leg_ref: str | None = None,
        rule_key: str | None = None,
        rule_value: object = None,
        observed: object = None,
        tool_call_ids: Iterable[UUID] = (),
    ) -> None:
        if attempt is not None:
            decision_ref = decision_ref or attempt.decision_ref
            leg_ref = leg_ref or attempt.leg_ref
            tool_call_ids = (*tool_call_ids, *attempt.tool_call_ids)
        index = attempt.index if attempt is not None and leg_ref is not None else None
        ids = tuple(dict.fromkeys(tool_call_ids))
        seq = len(self._items)
        self._items.append(
            AuditFinding(
                finding_id=uuid5(_FINDING_NS, f"{self._hash}:{self._check.value}:{seq}"),
                run_id=self._ctx.run_id,
                check_id=self._check,
                sub_item=sub_item,
                outcome=outcome,
                effective_execution_mode=self._ctx.effective_execution_mode,
                decision_ref=decision_ref,
                leg_ref=leg_ref,
                attempt_index=index,
                rule_key=rule_key,
                rule_value=None if rule_value is None else str(rule_value),
                observed_value=None if observed is None or str(observed) == "" else str(observed),
                tool_call_ids=ids,
                detail=detail,
                audit_version=self._ctx.audit_version,
                context_hash=self._hash,
            )
        )

    def ok(self, sub_item: str, detail: str, **kw: Unpack[FindingKw]) -> None:
        self.add(sub_item, AuditOutcome.PASS, detail, **kw)

    def bad(self, sub_item: str, detail: str, **kw: Unpack[FindingKw]) -> None:
        self.add(sub_item, AuditOutcome.VIOLATION, detail, **kw)

    def unknown(self, sub_item: str, reason: Reason, detail: str, **kw: Unpack[FindingKw]) -> None:
        self.add(sub_item, AuditOutcome.UNVERIFIABLE, f"{reason.value}: {detail}", **kw)

    def result(self) -> tuple[AuditFinding, ...]:
        return tuple(self._items)


# --------------------------------------------------------------------------------------------
# Tool calls
# --------------------------------------------------------------------------------------------


def tool_name(call: ToolCallRecord) -> str:
    """Bare tool name: `mcp__robinhood__place_option_order` -> `place_option_order`."""
    return call.identity.tool.rsplit("__", 1)[-1]


def calls_named(ctx: AuditContext, *names: str) -> tuple[ToolCallRecord, ...]:
    wanted = set(names)
    return tuple(c for c in ctx.tool_calls if tool_name(c) in wanted)


def call_by_id(ctx: AuditContext, tool_call_id: UUID) -> ToolCallRecord | None:
    return next((c for c in ctx.tool_calls if c.identity.tool_call_id == tool_call_id), None)


def is_uncertain(call: ToolCallRecord) -> bool:
    """A dispatched call whose outcome is an error, timeout, or unknown (V6.3)."""
    if call.dispatched_at is None:
        return False
    return call.status in _UNCERTAIN_TOOL_STATUSES or call.outcome_unresolved


def event_time(call: ToolCallRecord) -> datetime:
    return call.dispatched_at or call.identity.requested_at


# --------------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------------


def is_tbd(value: object) -> bool:
    return value is RuleMarker.TBD or value == RuleMarker.TBD.value


# --------------------------------------------------------------------------------------------
# Attempts
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditAttempt:
    """One placed (live) or intended (off) attempt, normalized from original evidence.

    Live values come from the recorded place arguments (OrderIntent); the RunRecord attempt
    is kept only for comparison. Off values come from the assembled proposal.
    """

    key: str
    live: bool
    place_call: ToolCallRecord | None
    intent: OrderIntent | None
    order: OrderRecord | None
    decision: DecisionRecord | None
    leg: LegRecord | None
    record_attempt: Attempt | None
    record_occurrences: int
    side_raw: str | None
    quantity: int | None
    order_type_raw: str | None
    tif_raw: str | None
    limit_price: Decimal | None
    instrument_id: str | None
    occ: OccSymbol | None
    at: datetime | None
    conditional: bool

    @property
    def side(self) -> OrderSide | None:
        if self.side_raw is None:
            return None
        try:
            return OrderSide(self.side_raw)
        except ValueError:
            return None

    @property
    def dispatched(self) -> bool:
        return self.place_call is not None and self.place_call.dispatched_at is not None

    @property
    def decision_ref(self) -> str | None:
        return self.decision.decision_ref if self.decision is not None else None

    @property
    def leg_ref(self) -> str | None:
        return self.leg.leg_ref if self.leg is not None else None

    @property
    def index(self) -> int | None:
        return self.record_attempt.index if self.record_attempt is not None else None

    @property
    def action(self) -> DecisionAction | None:
        return self.decision.action if self.decision is not None else None

    @property
    def tool_call_ids(self) -> tuple[UUID, ...]:
        if self.place_call is None:
            return ()
        return (self.place_call.identity.tool_call_id,)

    @property
    def is_roll_replacement(self) -> bool:
        return self.action is DecisionAction.ROLL and self.side is OrderSide.SELL_TO_OPEN

    @property
    def is_new_open(self) -> bool:
        return self.action in (DecisionAction.OPEN_CSP, DecisionAction.OPEN_CC)


def order_for_place(ctx: AuditContext, place_call_id: UUID) -> OrderRecord | None:
    return next(
        (
            o
            for o in ctx.order_records
            if o.intent is not None and o.intent.place_tool_call_id == place_call_id
        ),
        None,
    )


def order_for_broker_id(ctx: AuditContext, broker_order_id: str) -> OrderRecord | None:
    return next(
        (
            o
            for o in ctx.order_records
            if o.broker_order is not None and o.broker_order.broker_order_id == broker_order_id
        ),
        None,
    )


def _record_location(
    ctx: AuditContext, place_call_id: UUID
) -> tuple[DecisionRecord | None, LegRecord | None, Attempt | None, int]:
    found: list[tuple[DecisionRecord | None, LegRecord | None, Attempt]] = []
    decision: DecisionRecord | None
    leg: LegRecord | None
    record = ctx.run_record
    if record is not None:
        for decision in record.decisions:
            for leg in decision.legs:
                for attempt in leg.attempts:
                    if attempt.place_tool_call_id == place_call_id:
                        found.append((decision, leg, attempt))
        for action in record.unassociated_actions:
            if action.attempt is not None and action.attempt.place_tool_call_id == place_call_id:
                found.append((None, None, action.attempt))
    if not found:
        return None, None, None, 0
    decision, leg, attempt = found[0]
    return decision, leg, attempt, len(found)


def facts_by_ref(ctx: AuditContext, ref: str | None) -> DecisionFacts | None:
    if ref is None:
        return None
    return next((f for f in ctx.decision_facts if f.facts_ref == ref), None)


def is_conditional(decision: DecisionRecord, leg: LegRecord) -> bool:
    """A leg that depends on another leg/decision completing (roll replacement, funding)."""
    return leg.conditional or leg.depends_on_leg_ref is not None


def live_attempts(ctx: AuditContext) -> tuple[AuditAttempt, ...]:
    """One attempt per recorded place call (dispatched or denied), in event order."""
    places = sorted(calls_named(ctx, PLACE_TOOL), key=lambda c: c.identity.requested_at)
    out: list[AuditAttempt] = []
    for call in places:
        call_id = call.identity.tool_call_id
        order = order_for_place(ctx, call_id)
        intent = order.intent if order is not None else None
        decision, leg, rec, occurrences = _record_location(ctx, call_id)
        out.append(
            AuditAttempt(
                key=str(call_id),
                live=True,
                place_call=call,
                intent=intent,
                order=order,
                decision=decision,
                leg=leg,
                record_attempt=rec,
                record_occurrences=occurrences,
                side_raw=intent.side_raw if intent else None,
                quantity=intent.quantity if intent else None,
                order_type_raw=intent.order_type_raw if intent else None,
                tif_raw=intent.time_in_force_raw if intent else None,
                limit_price=intent.limit_price if intent else None,
                instrument_id=intent.broker_instrument_id if intent else None,
                occ=intent.occ_symbol if intent else None,
                at=event_time(call),
                conditional=False,
            )
        )
    return tuple(out)


def output_available(ctx: AuditContext) -> bool:
    record = ctx.run_record
    return record is not None and record.decision_output_status is DecisionOutputStatus.PARSED


def dry_run_attempts(ctx: AuditContext) -> tuple[AuditAttempt, ...] | None:
    """Unsubmitted proposal attempts in code-derived priority order, or None if unknown.

    None means the intents cannot be known (no RunRecord or no parsed model output).
    """
    record = ctx.run_record
    if record is None or not output_available(ctx):
        return None
    items: list[tuple[tuple[int, int, str, int], AuditAttempt]] = []
    for decision in record.decisions:
        for leg_pos, leg in enumerate(decision.legs):
            for attempt in leg.attempts:
                if attempt.proposal_ref is None:
                    continue
                facts = facts_by_ref(ctx, leg.facts_ref)
                priority = decision.priority
                sort_key = (
                    0 if priority is not None else 1,
                    priority if priority is not None else 0,
                    decision.decision_ref,
                    leg_pos,
                )
                items.append(
                    (
                        sort_key,
                        AuditAttempt(
                            key=attempt.proposal_ref,
                            live=False,
                            place_call=None,
                            intent=None,
                            order=None,
                            decision=decision,
                            leg=leg,
                            record_attempt=attempt,
                            record_occurrences=1,
                            side_raw=leg.side.value,
                            quantity=attempt.requested_quantity,
                            order_type_raw=attempt.order_type_raw,
                            tif_raw=attempt.time_in_force_raw,
                            limit_price=attempt.limit_price,
                            instrument_id=leg.broker_instrument_id,
                            occ=leg.occ_symbol,
                            at=facts.observed_at if facts is not None else None,
                            conditional=is_conditional(decision, leg),
                        ),
                    )
                )
    items.sort(key=lambda pair: pair[0])
    return tuple(a for _, a in items)


def priorities_known(attempts: tuple[AuditAttempt, ...]) -> bool:
    return len(attempts) <= 1 or all(
        a.decision is not None and a.decision.priority is not None for a in attempts
    )


# --------------------------------------------------------------------------------------------
# Instruments and quotes
# --------------------------------------------------------------------------------------------


def instrument(ctx: AuditContext, instrument_id: str | None) -> InstrumentFact | None:
    """The validated instrument result, or None if absent or contradictory."""
    if instrument_id is None:
        return None
    facts = [i for i in ctx.instruments if i.broker_instrument_id == instrument_id]
    if not facts:
        return None
    first = facts[0]
    same = all(
        (f.occ_symbol, f.multiplier, f.tick_increment)
        == (first.occ_symbol, first.multiplier, first.tick_increment)
        for f in facts
    )
    return first if same else None


def quote_source_completed(ctx: AuditContext, quote: Quote) -> datetime | None:
    """Latest completion time of the quote's source calls, or None if any is unknown."""
    times: list[datetime] = []
    for call_id in quote.source_tool_call_ids:
        call = call_by_id(ctx, call_id)
        if call is None or call.completed_at is None:
            return None
        times.append(call.completed_at)
    return max(times)


def linked_quotes(ctx: AuditContext, attempt: AuditAttempt) -> tuple[Quote, ...] | None:
    """Original quotes the attempt references; None if a referenced quote is absent."""
    if attempt.record_attempt is None:
        return ()
    by_id = {q.quote_id: q for q in ctx.quotes}
    out: list[Quote] = []
    for ref in attempt.record_attempt.quote_refs:
        quote = by_id.get(ref)
        if quote is None:
            return None
        out.append(quote)
    return tuple(out)


def quote_before(ctx: AuditContext, quotes: Iterable[Quote], at: datetime | None) -> Quote | None:
    """The latest (by as_of) quote whose source completed at or before `at`."""
    if at is None:
        return None
    usable = []
    for quote in quotes:
        done = quote_source_completed(ctx, quote)
        if done is not None and done <= at:
            usable.append(quote)
    return max(usable, key=lambda q: q.as_of) if usable else None


# --------------------------------------------------------------------------------------------
# Broker state before an attempt (live) and dry-run reservation simulation (off)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreState:
    """Account/position/order state before one attempt, plus dry-run reservations.

    The extras are audit-only projections for dry run (VALIDATION.md "Dry-run
    reservations"); they are empty in live mode, where the broker snapshot is used as-is.
    """

    state: BrokerState
    cash: Decimal | None
    reserved: Decimal | None
    extra_csp_by_underlying: dict[str, Decimal] = field(default_factory=dict)
    extra_shares_reserved: dict[str, int] = field(default_factory=dict)
    extra_btc_reserved: dict[str, int] = field(default_factory=dict)

    @property
    def snapshot(self) -> AccountSnapshot:
        return self.state.snapshot

    @property
    def base(self) -> Decimal | None:
        """B = C + R before any dry-run adjustment (stays constant in the simulation)."""
        return self.state.snapshot.csp_cash_base_usd

    @property
    def tool_call_ids(self) -> tuple[UUID, ...]:
        return self.state.snapshot.tool_call_ids


def pre_order_state(ctx: AuditContext, attempt: AuditAttempt) -> PreState | Unknown:
    """The latest state completed at/before dispatch that is fresh and includes prior mutations."""
    at = attempt.at
    if at is None:
        return Unknown(Reason.MISSING_EVIDENCE, "attempt time unknown")
    candidates = [s for s in ctx.broker_states if s.completed_at <= at]
    if not candidates:
        return Unknown(Reason.NO_PRE_ORDER_STATE, "no account state completed before dispatch")
    state = max(candidates, key=lambda s: s.completed_at)
    if not state.snapshot.agentic_verified:
        return Unknown(
            Reason.UNVERIFIED_ACCOUNT, "snapshot account is not verified as the Agentic account"
        )
    max_age = ctx.rules.data_quality.freshness.account_state_max_age_seconds
    if is_tbd(max_age):
        return Unknown(
            Reason.RULE_TBD, "data_quality.freshness.account_state_max_age_seconds is TBD"
        )
    if isinstance(max_age, int) and not is_fresh(state.snapshot.as_of, at, max_age):
        return Unknown(Reason.STALE_STATE, "account state is not fresh at dispatch")
    own = attempt.place_call.identity.tool_call_id if attempt.place_call else None
    reflected = set(state.reflects_mutation_tool_call_ids)
    for call in calls_named(ctx, *MUTATION_TOOLS):
        call_id = call.identity.tool_call_id
        if call_id == own or call.dispatched_at is None or call.dispatched_at >= at:
            continue
        if call_id not in reflected:
            return Unknown(
                Reason.STATE_MISSING_MUTATION,
                f"state is not known to include earlier mutation {call_id}",
            )
    snap = state.snapshot
    return PreState(
        state=state, cash=snap.available_settled_cash_usd, reserved=snap.csp_reserved_cash_usd
    )


def _unfilled(order_quantity: int, filled: int | None) -> int | None:
    return None if filled is None else order_quantity - filled


def available_to_close(ctx: AuditContext, pre: PreState, instrument_id: str) -> int | None:
    """Short quantity on the contract minus other working BTC orders (unfilled portions)."""
    state = pre.state
    if state.short_options is None or state.working_orders is None:
        return None
    short = sum(
        h.short_quantity for h in state.short_options if h.broker_instrument_id == instrument_id
    )
    working = 0
    for order in state.working_orders:
        if order.broker_instrument_id != instrument_id:
            continue
        if order.side_raw is None:
            return None
        if order.side_raw != OrderSide.BUY_TO_CLOSE.value:
            continue
        unfilled = _unfilled(order.quantity, order.filled_quantity)
        if unfilled is None:
            return None
        working += unfilled
    return short - working - pre.extra_btc_reserved.get(instrument_id, 0)


def _right_and_root(
    ctx: AuditContext, instrument_id: str
) -> tuple[OptionRight, str, InstrumentFact] | None:
    fact = instrument(ctx, instrument_id)
    if fact is None:
        return None
    return fact.occ_symbol.right, fact.occ_symbol.root, fact


def available_shares(ctx: AuditContext, pre: PreState, underlying: str) -> int | None:
    """Owned shares not reserved by short calls, working STO calls, or other reservations."""
    state = pre.state
    if state.share_holdings is None or state.short_options is None or state.working_orders is None:
        return None
    owned = 0
    for holding in state.share_holdings:
        if holding.underlying != underlying:
            continue
        if holding.other_reserved_shares is None:
            return None
        owned += holding.quantity - holding.other_reserved_shares
    reserved = 0
    for short in state.short_options:
        info = _right_and_root(ctx, short.broker_instrument_id)
        if info is None or info[2].multiplier is None:
            return None
        right, root, fact = info
        if right is OptionRight.CALL and root == underlying:
            reserved += short.short_quantity * fact.multiplier  # type: ignore[operator]
    for order in state.working_orders:
        info = _right_and_root(ctx, order.broker_instrument_id)
        if info is None or info[2].multiplier is None or order.side_raw is None:
            return None
        right, root, fact = info
        if (
            right is OptionRight.CALL
            and root == underlying
            and order.side_raw == OrderSide.SELL_TO_OPEN.value
        ):
            unfilled = _unfilled(order.quantity, order.filled_quantity)
            if unfilled is None:
                return None
            reserved += unfilled * fact.multiplier  # type: ignore[operator]
    return owned - reserved - pre.extra_shares_reserved.get(underlying, 0)


def csp_exposure(ctx: AuditContext, pre: PreState, underlying: str) -> Decimal | None:
    """Held short puts plus unfilled working STO puts on `underlying`, in USD collateral."""
    state = pre.state
    if state.short_options is None or state.working_orders is None:
        return None
    total = Decimal(0)
    for short in state.short_options:
        info = _right_and_root(ctx, short.broker_instrument_id)
        if info is None or info[2].multiplier is None:
            return None
        right, root, fact = info
        if right is OptionRight.PUT and root == underlying:
            total += fact.occ_symbol.strike * fact.multiplier * short.short_quantity  # type: ignore[operator]
    for order in state.working_orders:
        info = _right_and_root(ctx, order.broker_instrument_id)
        if info is None or info[2].multiplier is None or order.side_raw is None:
            return None
        right, root, fact = info
        if (
            right is OptionRight.PUT
            and root == underlying
            and order.side_raw == OrderSide.SELL_TO_OPEN.value
        ):
            unfilled = _unfilled(order.quantity, order.filled_quantity)
            if unfilled is None:
                return None
            total += fact.occ_symbol.strike * fact.multiplier * unfilled  # type: ignore[operator]
    return total + pre.extra_csp_by_underlying.get(underlying, Decimal(0))


def collateral(ctx: AuditContext, attempt: AuditAttempt) -> Decimal | None:
    """N = strike x multiplier x attempt quantity, from the validated instrument result."""
    fact = instrument(ctx, attempt.instrument_id)
    if fact is None or fact.multiplier is None or attempt.quantity is None:
        return None
    return fact.occ_symbol.strike * fact.multiplier * attempt.quantity


def _state_signature(state: BrokerState) -> tuple[object, ...]:
    snap = state.snapshot
    return (
        snap.available_settled_cash_usd,
        snap.csp_reserved_cash_usd,
        snap.account_value_usd,
        state.short_options,
        state.share_holdings,
        state.working_orders,
    )


def simulate_dry_run(
    ctx: AuditContext, attempts: tuple[AuditAttempt, ...]
) -> dict[str, PreState | Unknown]:
    """Pre-states for dry-run intents from one baseline, reserving in priority order.

    Conditional openings reserve nothing and are unverifiable. If the order is ambiguous, the
    baseline is missing, actual broker state changed, or an earlier reservation cannot be
    computed, dependent intents are unverifiable (VALIDATION.md "Dry-run reservations").
    """
    out: dict[str, PreState | Unknown] = {}
    if not attempts:
        return out
    if not priorities_known(attempts):
        for a in attempts:
            out[a.key] = Unknown(
                Reason.AMBIGUOUS_ORDER, "execution priority unknown for competing intents"
            )
        return out
    first = attempts[0].record_attempt
    baseline_id = first.snapshot_ref if first is not None else None
    states = {s.snapshot.snapshot_id: s for s in ctx.broker_states}
    baseline = states.get(baseline_id) if baseline_id is not None else None
    if baseline is None:
        for a in attempts:
            out[a.key] = Unknown(
                Reason.NO_PRE_ORDER_STATE, "dry-run baseline snapshot not recorded"
            )
        return out
    if not baseline.snapshot.agentic_verified:
        for a in attempts:
            out[a.key] = Unknown(Reason.UNVERIFIED_ACCOUNT, "baseline account is not verified")
        return out
    snap = baseline.snapshot
    current = PreState(
        state=baseline, cash=snap.available_settled_cash_usd, reserved=snap.csp_reserved_cash_usd
    )
    blocked: Unknown | None = None
    for a in attempts:
        if blocked is not None:
            out[a.key] = blocked
            continue
        if a.conditional:
            out[a.key] = Unknown(
                Reason.DRY_RUN_NO_EXECUTION,
                "conditional opening: funding depends on a hypothetical close",
            )
            continue
        ref = a.record_attempt.snapshot_ref if a.record_attempt is not None else None
        if ref is not None and ref != baseline_id:
            other = states.get(ref)
            if other is None or _state_signature(other) != _state_signature(baseline):
                blocked = Unknown(
                    Reason.BASELINE_CHANGED, "actual broker state changed during the run"
                )
                out[a.key] = blocked
                continue
        out[a.key] = current
        next_state = _reserve(ctx, current, a)
        if next_state is None:
            blocked = Unknown(
                Reason.UPSTREAM_UNKNOWN, f"reservation of earlier intent {a.key} unknown"
            )
        else:
            current = next_state
    return out


def _reserve(ctx: AuditContext, pre: PreState, attempt: AuditAttempt) -> PreState | None:
    fact = instrument(ctx, attempt.instrument_id)
    side = attempt.side
    if fact is None or fact.multiplier is None or attempt.quantity is None or side is None:
        return None
    root = fact.occ_symbol.root
    if side is OrderSide.BUY_TO_CLOSE:
        btc = dict(pre.extra_btc_reserved)
        btc[fact.broker_instrument_id] = btc.get(fact.broker_instrument_id, 0) + attempt.quantity
        return PreState(
            state=pre.state,
            cash=pre.cash,
            reserved=pre.reserved,
            extra_csp_by_underlying=pre.extra_csp_by_underlying,
            extra_shares_reserved=pre.extra_shares_reserved,
            extra_btc_reserved=btc,
        )
    if fact.occ_symbol.right is OptionRight.CALL:
        shares = dict(pre.extra_shares_reserved)
        shares[root] = shares.get(root, 0) + attempt.quantity * fact.multiplier
        return PreState(
            state=pre.state,
            cash=pre.cash,
            reserved=pre.reserved,
            extra_csp_by_underlying=pre.extra_csp_by_underlying,
            extra_shares_reserved=shares,
            extra_btc_reserved=pre.extra_btc_reserved,
        )
    n = fact.occ_symbol.strike * fact.multiplier * attempt.quantity
    csp = dict(pre.extra_csp_by_underlying)
    csp[root] = csp.get(root, Decimal(0)) + n
    return PreState(
        state=pre.state,
        cash=None if pre.cash is None else pre.cash - n,
        reserved=None if pre.reserved is None else pre.reserved + n,
        extra_csp_by_underlying=csp,
        extra_shares_reserved=pre.extra_shares_reserved,
        extra_btc_reserved=pre.extra_btc_reserved,
    )


def pre_states(
    ctx: AuditContext, attempts: tuple[AuditAttempt, ...]
) -> dict[str, PreState | Unknown]:
    """Pre-state per attempt: pre-order broker state (live) or the dry-run simulation (off)."""
    if ctx.effective_execution_mode is ExecutionMode.LIVE:
        return {a.key: pre_order_state(ctx, a) for a in attempts}
    return simulate_dry_run(ctx, attempts)


# --------------------------------------------------------------------------------------------
# Order status and fills at a point in time
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusAt:
    status: StatusObservation | None
    overlap: bool


def status_before(ctx: AuditContext, order: OrderRecord, at: datetime) -> StatusAt:
    """Latest status whose source read completed at/before `at`; flags overlapping reads."""
    known: list[StatusObservation] = []
    overlap = False
    for obs in order.status_history:
        call = call_by_id(ctx, obs.tool_call_id)
        if call is None or call.completed_at is None:
            overlap = overlap or (call is not None and event_time(call) <= at)
            continue
        if call.completed_at <= at:
            known.append(obs)
        elif event_time(call) <= at:
            overlap = True
    latest = max(known, key=lambda s: s.observed_at) if known else None
    return StatusAt(status=latest, overlap=overlap)


def fills_before(ctx: AuditContext, order: OrderRecord, at: datetime) -> tuple[int | None, bool]:
    """Filled quantity from fill observations whose source completed at/before `at`.

    Returns (quantity or None if inconsistent, whether any observation existed).
    """
    fills: list[FillRecord] = []
    for fill in order.fills:
        call = call_by_id(ctx, fill.source_tool_call_id)
        done = call.completed_at if call is not None else None
        if done is not None and done <= at:
            fills.append(fill)
    return filled_quantity(fills), bool(fills)


__all__ = [
    "AuditAttempt",
    "Findings",
    "PreState",
    "Reason",
]


def audited_attempts(ctx: AuditContext) -> tuple[AuditAttempt, ...] | Unknown:
    """Live: every recorded place call. Off: the assembled dry-run intents, if knowable."""
    if ctx.effective_execution_mode is ExecutionMode.LIVE:
        return live_attempts(ctx)
    attempts = dry_run_attempts(ctx)
    if attempts is not None:
        return attempts
    if ctx.run_record is None:
        return Unknown(
            Reason.MISSING_RUN_RECORD, "no assembled run record; dry-run intents unknown"
        )
    return Unknown(Reason.MISSING_FINAL_OUTPUT, "no parsed model output; dry-run intents unknown")
