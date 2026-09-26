"""V4 Decision and assembled-record invariants (docs/VALIDATION.md "V4").

1. Each initial short lineage has exactly one management decision (CLOSE/ROLL/HOLD).
2. Legs match the decision's action (exact shape when unsubmitted; compatible prefix live).
3. Live: every recorded place/cancel call appears exactly once in the record; known placed or
   filled statuses carry the broker ID; unassociated actions make attribution unverifiable.
4. Live: original fills (deduplicated) sum to no more than the leg target, assembled filled
   quantities equal the original ones, and later steps use only the confirmed remainder.
5. Off: no actual actions, every attempt not_placed, conditional openings name a dependency.
6. Model boundary: invalid output and assembly/association findings are violations;
   proposal fact refs resolve to this run's DecisionFacts with a matching discretionary price.
"""

from wheelta_robinhood_agent.agent.audit._common import (
    CANCEL_TOOL,
    MUTATION_TOOLS,
    PLACE_TOOL,
    Findings,
    Reason,
    calls_named,
    order_for_place,
    output_available,
)
from wheelta_robinhood_agent.agent.audit.context import AuditContext
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditCheck,
    DecisionAction,
    ExecutionMode,
    OptionRight,
    OrderSide,
)
from wheelta_robinhood_agent.domain.orders import Attempt
from wheelta_robinhood_agent.domain.positions import PositionBookEntry
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import (
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    RunRecord,
)

MANAGEMENT_ACTIONS = frozenset({DecisionAction.CLOSE, DecisionAction.ROLL, DecisionAction.HOLD})
_KNOWN_BROKER_STATUSES = frozenset(
    {AttemptStatus.PLACED, AttemptStatus.FILLED, AttemptStatus.PARTIALLY_FILLED}
)


def check_v4(ctx: AuditContext) -> tuple[AuditFinding, ...]:
    """Run V4 over the assembled record and the original events."""
    out = Findings(ctx, AuditCheck.V4)
    record = ctx.run_record
    if record is None:
        out.unknown("all", Reason.MISSING_RUN_RECORD, "no assembled run record")
        return out.result()
    _coverage(ctx, out, record)
    _shapes(ctx, out, record)
    if ctx.effective_execution_mode is ExecutionMode.LIVE:
        _actions(ctx, out, record)
        _fills(ctx, out, record)
    else:
        _off_invariants(ctx, out, record)
    _boundary(ctx, out, record)
    return out.result()


# V4.1 ----------------------------------------------------------------------------------------


def _coverage(ctx: AuditContext, out: Findings, record: RunRecord) -> None:
    book = ctx.position_book
    if not output_available(ctx):
        out.unknown("1", Reason.MISSING_FINAL_OUTPUT, "decision coverage unknown without output")
        return
    if book is None or book.gaps:
        out.unknown("1", Reason.MISSING_EVIDENCE, "initial position state unknown or ambiguous")
        return
    refs = {e.position_ref: e.position_id for e in book.entries}
    ids = set(refs.values())
    counts: dict[object, int] = dict.fromkeys(ids, 0)
    for d in record.decisions:
        if d.action not in MANAGEMENT_ACTIONS:
            continue
        pid = d.position_id if d.position_id is not None else refs.get(d.target_ref)
        if pid not in ids:
            out.bad(
                "1",
                "management decision targets a lineage not in the initial position book",
                decision_ref=d.decision_ref,
                observed=d.target_ref,
            )
            continue
        counts[pid] += 1
    for entry in book.entries:
        n = counts[entry.position_id]
        if n == 1:
            out.ok("1", "exactly one management decision", observed=entry.position_ref)
        else:
            out.bad(
                "1",
                f"{n} management decisions for one initial lineage (expected exactly one)",
                observed=entry.position_ref,
            )


# V4.2 ----------------------------------------------------------------------------------------


def _leg_right(leg: LegRecord) -> OptionRight | None:
    if leg.right is not None:
        return leg.right
    return leg.occ_symbol.right if leg.occ_symbol is not None else None


def _expected(decision: DecisionRecord) -> list[tuple[OrderSide, OptionRight | None]]:
    action = decision.action
    if action is DecisionAction.OPEN_CSP:
        return [(OrderSide.SELL_TO_OPEN, OptionRight.PUT)]
    if action is DecisionAction.OPEN_CC:
        return [(OrderSide.SELL_TO_OPEN, OptionRight.CALL)]
    if action is DecisionAction.CLOSE:
        return [(OrderSide.BUY_TO_CLOSE, None)]
    if action is DecisionAction.ROLL:
        return [(OrderSide.BUY_TO_CLOSE, None), (OrderSide.SELL_TO_OPEN, None)]
    return []


def _submitted(decision: DecisionRecord) -> bool:
    return any(a.place_tool_call_id is not None for leg in decision.legs for a in leg.attempts)


def _shapes(ctx: AuditContext, out: Findings, record: RunRecord) -> None:
    if not output_available(ctx):
        out.unknown("2", Reason.MISSING_FINAL_OUTPUT, "no decisions to compare")
        return
    if not record.decisions:
        out.ok("2", "no decisions")
        return
    for decision in record.decisions:
        _shape(ctx, out, decision)


def _shape(ctx: AuditContext, out: Findings, decision: DecisionRecord) -> None:
    ref = decision.decision_ref
    expected = _expected(decision)
    legs = decision.legs
    live_partial = ctx.effective_execution_mode is ExecutionMode.LIVE and _submitted(decision)
    count_ok = len(legs) <= len(expected) if live_partial else len(legs) == len(expected)
    sides_ok = count_ok and all(
        leg.side is side for leg, (side, _) in zip(legs, expected, strict=False)
    )
    if not sides_ok:
        out.bad(
            "2",
            f"legs do not match action {decision.action.value}",
            decision_ref=ref,
            observed=",".join(leg.side.value for leg in legs),
        )
        return
    rights = [_leg_right(leg) for leg in legs]
    if any(r is None for r in rights):
        out.unknown("2", Reason.MISSING_EVIDENCE, "leg right unknown", decision_ref=ref)
        return
    wrong_right = any(
        want is not None and got is not want
        for got, (_, want) in zip(rights, expected, strict=False)
    )
    if decision.action is DecisionAction.ROLL and len(legs) == 2:
        close, open_ = legs
        roots = {leg.occ_symbol.root for leg in legs if leg.occ_symbol is not None}
        wrong_right = (
            wrong_right
            or rights[0] is not rights[1]
            or len(roots) > 1
            or open_.depends_on_leg_ref != close.leg_ref
        )
    if wrong_right:
        out.bad("2", "leg instrument incompatible with the action", decision_ref=ref)
        return
    if decision.action in (DecisionAction.CLOSE, DecisionAction.ROLL) and legs:
        entry = _book_entry(ctx, decision)
        close_leg = legs[0]
        if entry is None or close_leg.broker_instrument_id is None:
            out.unknown(
                "2",
                Reason.MISSING_EVIDENCE,
                "closed lineage or instrument unknown",
                decision_ref=ref,
            )
            return
        held = {i.broker_instrument_id for i in entry.current_instruments}
        if close_leg.broker_instrument_id not in held:
            out.bad("2", "close leg is not a contract held in the lineage", decision_ref=ref)
            return
    out.ok("2", f"legs compatible with {decision.action.value}", decision_ref=ref)


def _book_entry(ctx: AuditContext, decision: DecisionRecord) -> PositionBookEntry | None:
    book = ctx.position_book
    if book is None:
        return None
    for entry in book.entries:
        if entry.position_id == decision.position_id or entry.position_ref == decision.target_ref:
            return entry
    return None


# V4.3 ----------------------------------------------------------------------------------------


def _record_attempts(record: RunRecord) -> list[Attempt]:
    attempts = [a for d in record.decisions for leg in d.legs for a in leg.attempts]
    attempts += [u.attempt for u in record.unassociated_actions if u.attempt is not None]
    return attempts


def _actions(ctx: AuditContext, out: Findings, record: RunRecord) -> None:
    attempts = _record_attempts(record)
    by_place = {a.place_tool_call_id: a for a in attempts if a.place_tool_call_id is not None}
    for call in calls_named(ctx, PLACE_TOOL):
        call_id = call.identity.tool_call_id
        attempt = by_place.get(call_id)
        if attempt is None:
            out.bad("3", "place call missing from the assembled record", tool_call_ids=(call_id,))
            continue
        order = order_for_place(ctx, call_id)
        broker_id = (
            order.broker_order.broker_order_id
            if order is not None and order.broker_order is not None
            else None
        )
        if attempt.status in _KNOWN_BROKER_STATUSES and (
            attempt.broker_order_id is None or attempt.broker_order_id != broker_id
        ):
            out.bad(
                "3",
                "known placed/filled attempt lacks the matching broker order id",
                tool_call_ids=(call_id,),
                observed=f"record={attempt.broker_order_id} broker={broker_id}",
            )
        else:
            out.ok("3", "place call recorded exactly once", tool_call_ids=(call_id,))
    cancels = list(record.cancellations) + [
        u.cancellation for u in record.unassociated_actions if u.cancellation is not None
    ]
    cancel_ids = {c.cancel_tool_call_id for c in cancels}
    for call in calls_named(ctx, CANCEL_TOOL):
        call_id = call.identity.tool_call_id
        if call_id in cancel_ids:
            out.ok("3", "cancel call recorded", tool_call_ids=(call_id,))
        else:
            out.bad("3", "cancel call missing from the assembled record", tool_call_ids=(call_id,))
    for action in record.unassociated_actions:
        ids = (
            (action.attempt.place_tool_call_id,)
            if action.attempt is not None and action.attempt.place_tool_call_id is not None
            else (action.cancellation.cancel_tool_call_id,)
            if action.cancellation is not None
            else ()
        )
        out.unknown(
            "3",
            Reason.UNASSOCIATED,
            f"unassociated {action.kind.value}: decision attribution unknown",
            tool_call_ids=ids,
        )


# V4.4 ----------------------------------------------------------------------------------------


def _fills(ctx: AuditContext, out: Findings, record: RunRecord) -> None:
    for decision in record.decisions:
        for leg in decision.legs:
            placed = [a for a in leg.attempts if a.place_tool_call_id is not None]
            if placed:
                _leg_fills(ctx, out, decision, leg, placed)


def _leg_fills(
    ctx: AuditContext,
    out: Findings,
    decision: DecisionRecord,
    leg: LegRecord,
    attempts: list[Attempt],
) -> None:
    dref, lref = decision.decision_ref, leg.leg_ref
    total: int | None = 0
    target = leg.target_quantity
    for attempt in attempts:
        assert attempt.place_tool_call_id is not None  # noqa: S101 - filtered by caller
        order = order_for_place(ctx, attempt.place_tool_call_id)
        if order is None:
            original: int | None = 0 if attempt.status is AttemptStatus.NOT_PLACED else None
        else:
            original = order.filled_quantity
        if attempt.filled_quantity is not None and original is not None:
            if attempt.filled_quantity != original:
                out.bad(
                    "4",
                    "assembled filled quantity differs from the deduplicated broker fills",
                    observed=f"record={attempt.filled_quantity} broker={original}",
                    tool_call_ids=(attempt.place_tool_call_id,),
                    decision_ref=dref,
                    leg_ref=lref,
                )
        if (
            attempt.index > 0
            and target is not None
            and total is not None
            and attempt.requested_quantity is not None
            and attempt.requested_quantity > target - total
        ):
            out.bad(
                "4",
                "later step exceeds the confirmed remaining quantity",
                observed=f"requested={attempt.requested_quantity} remaining={target - total}",
                tool_call_ids=(attempt.place_tool_call_id,),
                decision_ref=dref,
                leg_ref=lref,
            )
        total = None if total is None or original is None else total + original
    if total is None:
        out.unknown(
            "4",
            Reason.MISSING_EVIDENCE,
            "fills inconsistent or not observed",
            decision_ref=dref,
            leg_ref=lref,
        )
    elif target is None:
        out.unknown(
            "4",
            Reason.TARGET_UNKNOWN,
            "leg target quantity unknown",
            observed=total,
            decision_ref=dref,
            leg_ref=lref,
        )
    elif total > target:
        out.bad(
            "4",
            "fills exceed the leg target",
            observed=f"{total}>{target}",
            decision_ref=dref,
            leg_ref=lref,
        )
    else:
        out.ok(
            "4",
            "fills within the leg target",
            observed=f"{total}<={target}",
            decision_ref=dref,
            leg_ref=lref,
        )


# V4.5 ----------------------------------------------------------------------------------------


def _off_invariants(ctx: AuditContext, out: Findings, record: RunRecord) -> None:
    actions = [c for c in calls_named(ctx, *MUTATION_TOOLS) if c.dispatched_at is not None]
    if actions:
        out.bad(
            "5",
            "order action dispatched in effective off mode",
            tool_call_ids=tuple(c.identity.tool_call_id for c in actions),
        )
    placed = [a for a in _record_attempts(record) if a.place_tool_call_id is not None]
    if placed or record.cancellations or record.unassociated_actions:
        out.bad("5", "dry-run record contains actual-action references or cancellations")
    for decision in record.decisions:
        for leg in decision.legs:
            _off_leg(out, decision, leg)
    if not actions and not placed and not record.cancellations and not record.unassociated_actions:
        out.ok("5", "no actual actions in dry run")


def _off_leg(out: Findings, decision: DecisionRecord, leg: LegRecord) -> None:
    dref, lref = decision.decision_ref, leg.leg_ref
    dependent = (decision.action is DecisionAction.ROLL and leg.side is OrderSide.SELL_TO_OPEN) or (
        leg.side is OrderSide.SELL_TO_OPEN and bool(decision.depends_on_decision_refs)
    )
    names_dependency = leg.depends_on_leg_ref is not None or bool(decision.depends_on_decision_refs)
    if dependent and not leg.conditional:
        out.bad(
            "5", "dependent opening is not labeled conditional", decision_ref=dref, leg_ref=lref
        )
    elif leg.conditional and not names_dependency:
        out.bad(
            "5", "conditional opening does not name its dependency", decision_ref=dref, leg_ref=lref
        )
    elif leg.conditional:
        out.ok("5", "conditional opening names its dependency", decision_ref=dref, leg_ref=lref)


# V4.6 ----------------------------------------------------------------------------------------


def _boundary(ctx: AuditContext, out: Findings, record: RunRecord) -> None:
    status = record.decision_output_status
    if status is DecisionOutputStatus.INVALID:
        out.bad("6", "model output rejected at the schema boundary")
    elif status is DecisionOutputStatus.MISSING or ctx.decision_output is None:
        out.unknown("6", Reason.MISSING_FINAL_OUTPUT, "no parsed model output")
    else:
        _refs(ctx, out)
    for finding in record.findings:
        out.bad(
            "6",
            f"assembly finding {finding.code}: {finding.detail}",
            decision_ref=finding.decision_ref,
            leg_ref=finding.leg_ref,
            tool_call_ids=finding.tool_call_ids,
        )


def _refs(ctx: AuditContext, out: Findings) -> None:
    output = ctx.decision_output
    assert output is not None  # noqa: S101 - checked by caller
    facts = {f.facts_ref: f for f in ctx.decision_facts if f.run_id == ctx.run_id}
    problems: list[str] = []
    for decision in output.decisions:
        for leg in decision.proposed_legs:
            fact = facts.get(leg.facts_ref)
            if fact is None:
                problems.append(f"unresolved facts_ref {leg.facts_ref}")
            elif fact.limit_price is not None and fact.limit_price != leg.limit_price:
                problems.append(f"facts_ref {leg.facts_ref} price differs from proposal")
        if ctx.effective_execution_mode is ExecutionMode.OFF and decision.execution_refs:
            problems.append("execution_refs in dry run")
    if ctx.effective_execution_mode is ExecutionMode.OFF and output.cancellation_rationales:
        problems.append("cancellation_rationales in dry run")
    if problems:
        out.bad("6", "; ".join(problems))
    else:
        out.ok("6", "model choices resolve to references supplied in this run")
