"""V6 No retry, stop on unknown (docs/VALIDATION.md "V6"). Live tool events only.

1. A later step on the same contract and side follows a read proving the previous order's
   terminal status and its final filled quantity; no step once fills completed the target.
2. Never more than one owned working order on the same contract; a roll replacement follows
   the full target close.
3. After any place/cancel error, timeout, or unknown outcome, no later placement is dispatched.
4. Missing events, overlapping calls, or unknown terminal state are unverifiable.
Off mode: the whole check is unverifiable (`dry_run_no_execution`).
"""

from datetime import datetime

from wheelta_robinhood_agent.agent.audit._common import (
    MUTATION_TOOLS,
    TERMINAL_STATUSES,
    AuditAttempt,
    Findings,
    Reason,
    calls_named,
    fills_before,
    is_uncertain,
    live_attempts,
    order_for_broker_id,
    order_for_place,
    status_before,
)
from wheelta_robinhood_agent.agent.audit.context import AuditContext
from wheelta_robinhood_agent.domain.enums import (
    AuditCheck,
    DecisionAction,
    ExecutionMode,
    OrderSide,
)
from wheelta_robinhood_agent.domain.orders import OrderRecord
from wheelta_robinhood_agent.domain.run import AuditFinding


def check_v6(ctx: AuditContext) -> tuple[AuditFinding, ...]:
    """Run V6 over the recorded live placement sequence."""
    out = Findings(ctx, AuditCheck.V6)
    if ctx.effective_execution_mode is not ExecutionMode.LIVE:
        out.unknown("all", Reason.DRY_RUN_NO_EXECUTION, "no order sequence exists in dry run")
        return out.result()
    _missing_events(ctx, out)
    attempts = [a for a in live_attempts(ctx)]
    if not attempts:
        out.ok("all", "no placements")
        return out.result()
    for attempt in attempts:
        if not attempt.dispatched:
            out.ok("3", "place call denied before dispatch", attempt=attempt)
            continue
        _replacement(ctx, out, attempt, attempts)
        _single_working(ctx, out, attempt, attempts)
        _roll_sequence(ctx, out, attempt)
        _stop_on_error(ctx, out, attempt)
    return out.result()


def _missing_events(ctx: AuditContext, out: Findings) -> None:
    known = {c.identity.tool_call_id for c in ctx.tool_calls}
    for order in ctx.order_records:
        if order.intent is not None and order.intent.place_tool_call_id not in known:
            out.unknown(
                "4",
                Reason.MISSING_EVIDENCE,
                "order intent without its recorded place call",
                tool_call_ids=(order.intent.place_tool_call_id,),
            )


def _at(attempt: AuditAttempt) -> datetime:
    call = attempt.place_call
    assert call is not None and call.dispatched_at is not None  # noqa: S101 - dispatched only
    return call.dispatched_at


def _earlier_same(
    attempt: AuditAttempt, attempts: list[AuditAttempt], *, same_side: bool
) -> list[tuple[AuditAttempt, OrderRecord]]:
    """Earlier dispatched steps on the same contract (known instrument implies an order)."""
    at = _at(attempt)
    return [
        (a, a.order)
        for a in attempts
        if a.dispatched
        and a.order is not None
        and a.key != attempt.key
        and _at(a) < at
        and a.instrument_id is not None
        and a.instrument_id == attempt.instrument_id
        and (not same_side or a.side_raw == attempt.side_raw)
    ]


def _replacement(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt, attempts: list[AuditAttempt]
) -> None:
    """V6.1: a later step needs a proven terminal predecessor and an unfilled remainder."""
    if attempt.instrument_id is None:
        out.unknown("1", Reason.MISSING_EVIDENCE, "instrument not recorded", attempt=attempt)
        return
    prior = _earlier_same(attempt, attempts, same_side=True)
    if not prior:
        out.ok("1", "first step on this contract and side", attempt=attempt)
        return
    at = _at(attempt)
    _, order = max(prior, key=lambda pair: _at(pair[0]))
    state = status_before(ctx, order, at)
    if state.overlap:
        out.unknown(
            "1", Reason.UNCERTAIN_ORDERING, "status read overlaps the placement", attempt=attempt
        )
        return
    if state.status is None:
        out.unknown(
            "1", Reason.TERMINAL_UNKNOWN, "previous step's status never read", attempt=attempt
        )
        return
    if state.status.status not in TERMINAL_STATUSES:
        out.bad(
            "1",
            "replacement placed while the previous step was still working",
            attempt=attempt,
            observed=state.status.status.value,
            tool_call_ids=(state.status.tool_call_id,),
        )
        return
    total = 0
    for _, prior_order in prior:
        filled, observed = fills_before(ctx, prior_order, at)
        if filled is None or not observed:
            out.unknown(
                "1",
                Reason.TERMINAL_UNKNOWN,
                "final filled quantity of an earlier step not observed before the replacement",
                attempt=attempt,
            )
            return
        total += filled
    target = attempt.leg.target_quantity if attempt.leg is not None else None
    if target is None:
        out.unknown("1", Reason.TARGET_UNKNOWN, "leg target quantity unknown", attempt=attempt)
    elif total >= target:
        out.bad(
            "1",
            "replacement placed after fills completed the target",
            attempt=attempt,
            observed=f"filled={total} target={target}",
        )
    else:
        out.ok(
            "1",
            "replacement follows a confirmed terminal step with remaining quantity",
            attempt=attempt,
            observed=f"filled={total} target={target}",
        )


def _working(ctx: AuditContext, order: OrderRecord, at: datetime) -> bool | None:
    """True if known working, False if proven terminal, None if unknown/overlapping."""
    state = status_before(ctx, order, at)
    if state.overlap or state.status is None:
        return None
    return state.status.status not in TERMINAL_STATUSES


def _single_working(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt, attempts: list[AuditAttempt]
) -> None:
    """V6.2: no other owned working order on the same contract at dispatch."""
    at = _at(attempt)
    if attempt.instrument_id is None:
        out.unknown("2", Reason.MISSING_EVIDENCE, "instrument not recorded", attempt=attempt)
        return
    unknown = False
    this_run_broker_ids: set[str] = set()
    for prior, order in _earlier_same(attempt, attempts, same_side=False):
        if order.broker_order is not None:
            this_run_broker_ids.add(order.broker_order.broker_order_id)
        working = _working(ctx, order, at)
        if working:
            out.bad(
                "2",
                "another owned order on the contract was still working",
                attempt=attempt,
                observed=prior.key,
            )
            return
        unknown = unknown or working is None
    states = [s for s in ctx.broker_states if s.completed_at <= at]
    if not states:
        out.unknown(
            "2", Reason.NO_PRE_ORDER_STATE, "pre-existing working orders unknown", attempt=attempt
        )
        return
    state = max(states, key=lambda s: s.completed_at)
    if state.working_orders is None:
        out.unknown("2", Reason.UNMAPPED_BROKER_FIELD, "open orders not read", attempt=attempt)
        return
    for existing in state.working_orders:
        if (
            not existing.owned
            or existing.broker_instrument_id != attempt.instrument_id
            or existing.broker_order_id in this_run_broker_ids
        ):
            continue
        record = order_for_broker_id(ctx, existing.broker_order_id)
        if record is None or _working(ctx, record, at) is not False:
            out.bad(
                "2",
                "a pre-existing owned working order on the contract was not proven terminal",
                attempt=attempt,
                observed=existing.broker_order_id,
            )
            return
    if unknown:
        out.unknown(
            "2", Reason.TERMINAL_UNKNOWN, "an earlier order's state is unknown", attempt=attempt
        )
    else:
        out.ok("2", "no other owned working order on the contract", attempt=attempt)


def _roll_sequence(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    """V6.2 (roll): the replacement's first step follows the full target close fill."""
    decision = attempt.decision
    if (
        decision is None
        or decision.action is not DecisionAction.ROLL
        or attempt.side is not OrderSide.SELL_TO_OPEN
        or attempt.index not in (0, None)
    ):
        return
    close = next((leg for leg in decision.legs if leg.side is OrderSide.BUY_TO_CLOSE), None)
    at = _at(attempt)
    if close is None or close.target_quantity is None:
        out.unknown(
            "2", Reason.TARGET_UNKNOWN, "roll close leg or its target unknown", attempt=attempt
        )
        return
    total = 0
    for step in close.attempts:
        order = order_for_place(ctx, step.place_tool_call_id) if step.place_tool_call_id else None
        filled, observed = fills_before(ctx, order, at) if order is not None else (None, False)
        if filled is None or not observed:
            out.unknown(
                "2",
                Reason.TERMINAL_UNKNOWN,
                "roll close fills not observed before the replacement",
                attempt=attempt,
            )
            return
        total += filled
    if total < close.target_quantity:
        out.bad(
            "2",
            "roll replacement placed after a partial or unfilled close",
            attempt=attempt,
            observed=f"closed={total} target={close.target_quantity}",
        )
    else:
        out.ok(
            "2",
            "roll replacement follows the full target close",
            attempt=attempt,
            observed=f"closed={total} target={close.target_quantity}",
        )


def _stop_on_error(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    """V6.3: no placement dispatched after a place/cancel error, timeout, or unknown outcome."""
    at = _at(attempt)
    own = attempt.place_call.identity.tool_call_id if attempt.place_call else None
    uncertain = False
    for call in calls_named(ctx, *MUTATION_TOOLS):
        if call.identity.tool_call_id == own or not is_uncertain(call):
            continue
        dispatched = call.dispatched_at
        assert dispatched is not None  # noqa: S101 - is_uncertain implies dispatch
        if dispatched >= at:
            continue
        if call.completed_at is not None and call.completed_at <= at:
            out.bad(
                "3",
                "placement dispatched after an order-action error, timeout, or unknown outcome",
                attempt=attempt,
                observed=call.status.value,
                tool_call_ids=(call.identity.tool_call_id,),
            )
            return
        uncertain = True
    if uncertain:
        out.unknown(
            "3",
            Reason.UNCERTAIN_ORDERING,
            "placement overlaps an uncertain order action",
            attempt=attempt,
        )
    else:
        out.ok("3", "no earlier uncertain order action", attempt=attempt)
