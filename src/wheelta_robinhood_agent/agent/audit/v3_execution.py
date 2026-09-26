"""V3 Reviewed = placed = assembled (docs/VALIDATION.md "V3 Reviewed = placed = assembled").

1. Live: a successful, warning-free review of the same contract/side/quantity/limit/type/TIF
   completed before dispatch (after any earlier step on the contract).
2. Live: the assembled attempt's requested parameters equal the recorded place arguments.
   A submitted quantity that differs from the decision-facts quantity is noted in the detail;
   the execution record is never rewritten.
3. Limit price within the attempt's quote bid/ask and on the verified tick increment.
4. Time in force equals `orders.time_in_force`.
Off: sub-items 1-2 are unverifiable (`dry_run_no_execution`); 3-4 use the proposal.
"""

from wheelta_robinhood_agent.agent.audit._common import (
    PLACE_TOOL,
    REVIEW_TOOL,
    AuditAttempt,
    Findings,
    Reason,
    Unknown,
    audited_attempts,
    calls_named,
    facts_by_ref,
    instrument,
    is_tbd,
    linked_quotes,
    quote_before,
)
from wheelta_robinhood_agent.agent.audit.context import AuditContext, ReviewObservation
from wheelta_robinhood_agent.domain.enums import AuditCheck
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.sanity import is_on_tick


def check_v3(ctx: AuditContext) -> tuple[AuditFinding, ...]:
    """Run V3 over every audited attempt."""
    out = Findings(ctx, AuditCheck.V3)
    attempts = audited_attempts(ctx)
    if isinstance(attempts, Unknown):
        out.unknown("all", attempts.reason, attempts.detail)
        return out.result()
    if not attempts:
        out.ok("all", "no placed or intended attempts")
        return out.result()
    for attempt in attempts:
        if attempt.live:
            _review(ctx, out, attempt)
            _assembled(ctx, out, attempt)
        else:
            detail = "no review or place call exists in dry run"
            out.unknown("1", Reason.DRY_RUN_NO_EXECUTION, detail, attempt=attempt)
            out.unknown("2", Reason.DRY_RUN_NO_EXECUTION, detail, attempt=attempt)
        _limit_bounds(ctx, out, attempt)
        _time_in_force(ctx, out, attempt)
    return out.result()


def _params(attempt: AuditAttempt) -> tuple[object, ...]:
    return (
        attempt.instrument_id,
        attempt.side_raw,
        attempt.quantity,
        attempt.order_type_raw,
        attempt.tif_raw,
        attempt.limit_price,
    )


def _reviewed(review: ReviewObservation) -> tuple[object, ...]:
    return (
        review.broker_instrument_id,
        review.side_raw,
        review.quantity,
        review.order_type_raw,
        review.time_in_force_raw,
        review.limit_price,
    )


def _review(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    """V3.1: a successful matching review completed before this step's dispatch."""
    call = attempt.place_call
    if call is None or call.dispatched_at is None:
        out.ok(
            "1", "place call denied before dispatch; nothing reached the broker", attempt=attempt
        )
        return
    at = call.dispatched_at
    if attempt.intent is None or None in _params(attempt):
        out.unknown("1", Reason.MISSING_EVIDENCE, "place arguments not recorded", attempt=attempt)
        return
    earlier_steps = [
        c.dispatched_at
        for c in calls_named(ctx, PLACE_TOOL)
        if c.dispatched_at is not None
        and c.dispatched_at < at
        and c.identity.tool_call_id != call.identity.tool_call_id
        and _place_instrument(ctx, c.identity.tool_call_id) == attempt.instrument_id
    ]
    after = max(earlier_steps) if earlier_steps else None
    candidates = [
        r
        for r in ctx.reviews
        if r.broker_instrument_id == attempt.instrument_id
        and r.completed_at <= at
        and (after is None or r.completed_at > after)
    ]
    matching = [
        r for r in candidates if r.succeeded and not r.warnings and _reviewed(r) == _params(attempt)
    ]
    if matching:
        ids = tuple(r.review_tool_call_id for r in matching)
        out.ok(
            "1", "matching successful review before placement", attempt=attempt, tool_call_ids=ids
        )
        return
    unmapped = [r for r in candidates if r.succeeded and not r.warnings and None in _reviewed(r)]
    known_reviews = {r.review_tool_call_id for r in ctx.reviews}
    unrecorded = [
        c
        for c in calls_named(ctx, REVIEW_TOOL)
        if c.identity.tool_call_id not in known_reviews
        and c.dispatched_at is not None
        and c.dispatched_at < at
    ]
    if unmapped or unrecorded:
        out.unknown(
            "1",
            Reason.MISSING_EVIDENCE,
            "review outcome or reviewed parameters not recorded",
            attempt=attempt,
        )
    elif candidates:
        out.bad(
            "1",
            "review failed, warned, or does not match the placed order",
            attempt=attempt,
            tool_call_ids=tuple(r.review_tool_call_id for r in candidates),
        )
    else:
        out.bad("1", "order placed without a review of this step", attempt=attempt)


def _place_instrument(ctx: AuditContext, place_call_id: object) -> str | None:
    for order in ctx.order_records:
        if order.intent is not None and order.intent.place_tool_call_id == place_call_id:
            return order.intent.broker_instrument_id
    return None


def _assembled(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    """V3.2: assembled requested parameters equal the recorded place arguments."""
    rec = attempt.record_attempt
    if rec is None:
        reason = Reason.MISSING_RUN_RECORD if ctx.run_record is None else Reason.MISSING_EVIDENCE
        out.unknown("2", reason, "place call not in the assembled record", attempt=attempt)
        return
    if attempt.intent is None:
        out.unknown("2", Reason.MISSING_EVIDENCE, "place arguments not recorded", attempt=attempt)
        return
    assembled = (rec.requested_quantity, rec.order_type_raw, rec.time_in_force_raw, rec.limit_price)
    placed = (attempt.quantity, attempt.order_type_raw, attempt.tif_raw, attempt.limit_price)
    if assembled != placed:
        out.bad(
            "2",
            "assembled attempt differs from the recorded place arguments",
            attempt=attempt,
            observed=f"assembled={assembled} placed={placed}",
        )
        return
    note = ""
    facts = facts_by_ref(ctx, attempt.leg.facts_ref) if attempt.leg is not None else None
    if facts is not None and facts.remaining_quantity is not None:
        if attempt.quantity != facts.remaining_quantity:
            note = (
                f"; submitted quantity {attempt.quantity} differs from decision-facts "
                f"remaining quantity {facts.remaining_quantity}"
            )
    out.ok("2", f"assembled attempt equals the place arguments{note}", attempt=attempt)


def _limit_bounds(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    """V3.3: limit price within the attempt's quoted [bid, ask] and on the tick increment."""
    quotes = linked_quotes(ctx, attempt)
    quote = quote_before(ctx, quotes or (), attempt.at)
    fact = instrument(ctx, attempt.instrument_id)
    price = attempt.limit_price
    if price is None or quote is None or fact is None or fact.tick_increment is None:
        out.unknown(
            "3",
            Reason.MISSING_EVIDENCE,
            "limit price, linked quote, or verified tick increment unknown",
            attempt=attempt,
        )
        return
    observed = f"limit={price} bid={quote.bid} ask={quote.ask} tick={fact.tick_increment}"
    ids = (*quote.source_tool_call_ids, fact.source_tool_call_id)
    if not quote.bid <= price <= quote.ask or not is_on_tick(price, fact.tick_increment):
        out.bad(
            "3",
            "limit price outside the quoted bid/ask or off the tick increment",
            attempt=attempt,
            observed=observed,
            tool_call_ids=ids,
        )
    else:
        out.ok(
            "3",
            "limit price within bid/ask and on tick",
            attempt=attempt,
            observed=observed,
            tool_call_ids=ids,
        )


def _time_in_force(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    """V3.4: time in force equals orders.time_in_force."""
    key = "orders.time_in_force"
    rule = ctx.rules.orders.time_in_force
    if is_tbd(rule):
        out.unknown("4", Reason.RULE_TBD, f"{key} is TBD", attempt=attempt, rule_key=key)
    elif attempt.tif_raw is None:
        out.unknown(
            "4",
            Reason.MISSING_EVIDENCE,
            "time in force not recorded",
            attempt=attempt,
            rule_key=key,
        )
    elif attempt.tif_raw != rule:
        out.bad(
            "4",
            "time in force differs from the rule",
            attempt=attempt,
            rule_key=key,
            rule_value=rule,
            observed=attempt.tif_raw,
        )
    else:
        out.ok("4", "time in force matches", attempt=attempt, rule_key=key, rule_value=rule)
