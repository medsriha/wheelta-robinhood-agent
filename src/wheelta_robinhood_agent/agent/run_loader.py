"""Load the immutable input bundles for assembly and audit from the ledger (OUTPUT_ASSEMBLY.md).

`assemble_run_record` and `run_audit` are pure; this module does the reading. Everything comes
from recorded rows of this run: tool-call projections, delivered result envelopes (which
define the code-issued references the model actually saw), persisted DecisionFacts, the
stored agent output/parse, and the PositionBook rendered into the prompt.

Loader decisions (documented, not guessed values):
- ADR-0052: a review/place/cancel call's `order_call:` ref is registered (kind `tool_call`)
  only when a delivered envelope of that call carries exactly `order_call_ref_for(call id)`.
  A call the model never received a result for (e.g. denied before dispatch) has no ref.
- Proposal-only dry runs (order venue `none`) place no order, so `orders` is empty. Runs
  that execute orders (venue `broker`, ADR-0034, or `simulated`, ADR-0038) load every order
  the run placed or observed (`ledger.orders.run_order_records`), for assembly and the audit.
- `attempt_evidence` links each dispatched place call to the latest pre-dispatch evidence of
  this run, by a fixed rule: the delivered quote of the placed instrument (`legs[0].option_id`)
  and the delivered account snapshot whose source calls all completed at or before the
  dispatch, latest by completion then `as_of`. None found leaves the link empty (V2/V3
  unverifiable), never a later or another instrument's quote.
- The audit's `broker_states` are rebuilt per dispatched place call from this run's validated
  reads completed at or before its dispatch (the facts service's own selection: latest account
  snapshot, positions read, and open-orders read). A state reflects an earlier place when its
  orders read lists that order, and an earlier cancel when it lists the order as terminal.
  A missing half stays None (unverifiable), never zero. Working orders carry their unfilled
  remainder as the quantity; `owned` means placed through a recorded intent.
- The audit's `reviews` are the delivered `review_option_order` results; a review succeeds
  only when its call succeeded, and a pre-trade alert is its warning.
- `reservation_baseline` and `reservation_requirements` are not built: the cash/share
  headroom definitions depend on unverified broker semantics. The assembler then sizes the
  first dry-run proposal from its fact set and leaves later proposals unavailable with a gap.
- `ranking_keys` are empty: `selection.ranking` names "lower absolute delta", which no
  DecisionFacts metric computes yet, and a partial key list would misstate the ranking.
- `day_history` is None (earlier same-day lineages are not yet established).
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

from wheelta_robinhood_agent.agent.audit.context import (
    AuditContext,
    BrokerState,
    InstrumentFact,
    ReviewObservation,
    ShareHolding,
    ShortOptionHolding,
    WorkingOrderObservation,
)
from wheelta_robinhood_agent.agent.facts_tool import FACTS_TOOL_NAME, RunEvidence
from wheelta_robinhood_agent.agent.mapped_evidence import OrderReviewObservation
from wheelta_robinhood_agent.agent.model_view import (
    ORDER_CALL_REF_KEY,
    is_model_view,
    model_view,
)
from wheelta_robinhood_agent.agent.result_boundary import (
    evidence_ref_for,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.config.rules import LoadedRules
from wheelta_robinhood_agent.domain.assembly import DecisionsInput, assemble_run_record
from wheelta_robinhood_agent.domain.assembly_context import (
    ASSEMBLER_VERSION,
    AssemblyContext,
    AttemptEvidence,
    DeliveredRef,
    RefKind,
    order_call_ref_for,
)
from wheelta_robinhood_agent.domain.assembly_events import CANCEL_TOOL, PLACE_TOOL, REVIEW_TOOL
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    DecisionOutputParsed,
    DecisionOutputParseFailure,
)
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    ExecutionMode,
    OrderVenue,
    PositionsCoverage,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.facts import DecisionFacts
from wheelta_robinhood_agent.domain.gating import executes_orders
from wheelta_robinhood_agent.domain.orders import OrderRecord
from wheelta_robinhood_agent.domain.positions import PositionBook
from wheelta_robinhood_agent.domain.reference_check import reference_issues
from wheelta_robinhood_agent.domain.run_record import Quote, RunRecord
from wheelta_robinhood_agent.domain.tool_calls import ToolCallRecord
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger.orders import owned_unresolved_orders, run_order_records
from wheelta_robinhood_agent.ledger.tool_calls import tool_call_records

Conn = psycopg.Connection[tuple[object, ...]]


class LoaderError(RuntimeError):
    """The recorded run cannot be loaded consistently (e.g. an unsupported mode)."""


@dataclass(frozen=True)
class RunMeta:
    """Trusted run metadata (from Settings, rules, prompt, and the run identity)."""

    run_id: uuid.UUID
    environment: AppEnv
    slot: datetime
    requested_mode: ExecutionMode
    effective_mode: ExecutionMode
    order_venue: OrderVenue
    account_scope_id: str
    rules: LoadedRules
    prompt_id: str | None
    prompt_hash: str | None
    model_id: str | None


@dataclass(frozen=True)
class DeliveredEvidence:
    """What the model was shown, reconstructed from DELIVERED result rows."""

    refs: tuple[DeliveredRef, ...]
    quotes: tuple[Quote, ...]
    instruments: tuple[InstrumentFact, ...]
    # (snapshot_id, source tool call ids) of delivered account snapshots.
    snapshots: tuple[tuple[uuid.UUID, tuple[uuid.UUID, ...]], ...] = ()
    reviews: tuple[OrderReviewObservation, ...] = ()


def _delivered_envelopes(conn: Conn, run_id: uuid.UUID) -> list[Mapping[str, Any]]:
    """Delivered envelopes, each model view (ADR-0037) replaced by the validated envelope it
    was built from. A view is kept only if rebuilding it from that envelope reproduces it
    exactly; otherwise nothing of it counts as delivered."""
    results = ledger_evidence.effective(ledger_evidence.results_for_run(conn, run_id))
    validated: dict[uuid.UUID, object] = {
        r.tool_call_id: r.payload
        for r in results
        if r.kind is ledger_evidence.ResultKind.VALIDATED and r.tool_call_id is not None
    }
    envelopes: list[Mapping[str, Any]] = []
    for stored in results:
        if stored.kind is not ledger_evidence.ResultKind.DELIVERED:
            continue
        payload = stored.payload
        if not isinstance(payload, dict) or payload.get("replaced") is not True:
            continue
        envelope = payload.get("tool_output")
        if not isinstance(envelope, dict):
            continue
        if is_model_view(envelope):
            source = validated.get(stored.tool_call_id) if stored.tool_call_id else None
            if not isinstance(source, dict) or model_view(source) != envelope:
                continue
            envelope = source
        envelopes.append(envelope)
    return envelopes


_ORDER_TOOLS = frozenset({REVIEW_TOOL, PLACE_TOOL, CANCEL_TOOL})


def _order_call_ref(
    envelope: Mapping[str, Any], run_id: uuid.UUID, scope: str
) -> DeliveredRef | None:
    """The order-call ref a delivered order-tool envelope carried (ADR-0052), or None."""
    call_id = envelope.get("tool_call_id")
    ref = envelope.get(ORDER_CALL_REF_KEY)
    if envelope.get("tool") not in _ORDER_TOOLS or not isinstance(call_id, str):
        return None
    try:
        tool_call_id = uuid.UUID(call_id)
    except ValueError:
        return None
    if ref != order_call_ref_for(tool_call_id):
        return None
    return DeliveredRef(
        ref=ref,
        kind=RefKind.TOOL_CALL,
        run_id=run_id,
        account_scope_id=scope,
        delivered=True,
        tool_call_id=tool_call_id,
    )


def load_delivered(conn: Conn, meta: RunMeta, book: PositionBook | None) -> DeliveredEvidence:
    """Code-issued references delivered in this run (tool results and the rendered book)."""
    refs: dict[str, DeliveredRef] = {}
    quotes: dict[uuid.UUID, Quote] = {}
    instruments: dict[str, InstrumentFact] = {}
    snapshots: dict[uuid.UUID, tuple[uuid.UUID, ...]] = {}
    reviews: dict[uuid.UUID, OrderReviewObservation] = {}
    scope = meta.account_scope_id
    for envelope in _delivered_envelopes(conn, meta.run_id):
        call_id = envelope.get("tool_call_id")
        data = envelope.get("data")
        order_ref = _order_call_ref(envelope, meta.run_id, scope)
        if order_ref is not None:
            refs[order_ref.ref] = order_ref
        if envelope.get("server") == LOCAL_SERVER_NAME and envelope.get("tool") == (
            FACTS_TOOL_NAME
        ):
            if not isinstance(data, dict):
                continue
            facts_ref = data.get("facts_ref")
            if isinstance(facts_ref, str) and data.get("status") == "ok":
                refs[facts_ref] = DeliveredRef(
                    ref=facts_ref,
                    kind=RefKind.FACTS,
                    run_id=meta.run_id,
                    account_scope_id=scope,
                    delivered=True,
                )
            continue
        mapped = mapped_evidence_of(envelope)
        if mapped is None or not isinstance(call_id, str):
            continue
        evidence_ref = evidence_ref_for(uuid.UUID(call_id))
        refs[evidence_ref] = DeliveredRef(
            ref=evidence_ref,
            kind=RefKind.EVIDENCE,
            run_id=meta.run_id,
            account_scope_id=scope,
            delivered=True,
            source_evidence_ids=tuple(dict.fromkeys(mapped.evidence_ids())),
        )
        for quote in mapped.option_quotes:
            quotes[quote.quote_id] = quote
        for inst in mapped.instruments:
            instruments[inst.broker_instrument_id] = InstrumentFact(
                broker_instrument_id=inst.broker_instrument_id,
                occ_symbol=inst.occ_symbol,
                multiplier=inst.multiplier,
                tick_increment=inst.tick_increment,
                source_tool_call_id=inst.source_tool_call_ids[0],
            )
        for snap in mapped.account_snapshots:
            snapshots[snap.snapshot_id] = snap.tool_call_ids
        for review in mapped.order_reviews:
            reviews[review.evidence_id] = review
        for candidate in mapped.candidates:
            refs[candidate.candidate_ref] = DeliveredRef(
                ref=candidate.candidate_ref,
                kind=RefKind.CANDIDATE,
                run_id=meta.run_id,
                account_scope_id=scope,
                delivered=True,
                underlying=candidate.underlying,
                occ_symbol=candidate.occ_symbol,
                broker_instrument_id=candidate.broker_instrument_id,
                candidate_origin=candidate.origin,
                source_evidence_ids=(candidate.instrument_evidence_id,),
            )
    for entry in book.entries if book else ():
        current = entry.current_instruments[0] if len(entry.current_instruments) == 1 else None
        refs[entry.position_ref] = DeliveredRef(
            ref=entry.position_ref,
            kind=RefKind.POSITION,
            run_id=None,
            account_scope_id=scope,
            delivered=True,
            underlying=entry.underlying,
            occ_symbol=current.occ_symbol if current else None,
            broker_instrument_id=current.broker_instrument_id if current else None,
            position_id=entry.position_id,
            source_evidence_ids=entry.entry_fill_ids,
        )
    return DeliveredEvidence(
        refs=tuple(refs.values()),
        quotes=tuple(quotes.values()),
        instruments=tuple(instruments.values()),
        snapshots=tuple(snapshots.items()),
        reviews=tuple(reviews.values()),
    )


def _completed(
    calls: Mapping[uuid.UUID, ToolCallRecord], ids: tuple[uuid.UUID, ...]
) -> datetime | None:
    """Latest completion time of the source calls, or None if any is unknown."""
    times = [calls[i].completed_at if i in calls else None for i in ids]
    if not times or any(t is None for t in times):
        return None
    return max(t for t in times if t is not None)


def _placed_instrument(call: ToolCallRecord) -> str | None:
    args = call.effective_arguments_redacted or call.identity.arguments_redacted
    legs = args.get("legs")
    if isinstance(legs, list) and len(legs) == 1 and isinstance(legs[0], dict):
        option_id = legs[0].get("option_id")
        return option_id if isinstance(option_id, str) and option_id else None
    return None


def attempt_evidence(
    tool_calls: tuple[ToolCallRecord, ...], delivered: DeliveredEvidence
) -> tuple[AttemptEvidence, ...]:
    """Each dispatched place call's latest pre-dispatch quote and snapshot (module docstring)."""
    calls = {c.identity.tool_call_id: c for c in tool_calls}
    out: list[AttemptEvidence] = []
    for call in tool_calls:
        at = call.dispatched_at
        if call.identity.tool != "place_option_order" or at is None:
            continue
        iid = _placed_instrument(call)
        quotes = [
            (done, q.as_of, q.quote_id)
            for q in delivered.quotes
            if q.broker_instrument_id == iid
            and (done := _completed(calls, q.source_tool_call_ids)) is not None
            and done <= at
        ]
        snaps = [
            (done, str(sid), sid)
            for sid, ids in delivered.snapshots
            if (done := _completed(calls, ids)) is not None and done <= at
        ]
        out.append(
            AttemptEvidence(
                place_tool_call_id=call.identity.tool_call_id,
                snapshot_ref=max(snaps)[2] if snaps else None,
                quote_refs=(max(quotes)[2],) if quotes else (),
            )
        )
    return tuple(out)


_TERMINAL = frozenset(
    {AttemptStatus.FILLED, AttemptStatus.CANCELLED, AttemptStatus.REJECTED, AttemptStatus.EXPIRED}
)


def _validated_items(conn: Conn, run_id: uuid.UUID) -> list[Any]:
    items = []
    for stored in ledger_evidence.effective(ledger_evidence.results_for_run(conn, run_id)):
        if stored.kind is ledger_evidence.ResultKind.VALIDATED and isinstance(stored.payload, dict):
            mapped = mapped_evidence_of(stored.payload)
            if mapped is not None:
                items.append(mapped)
    return items


def broker_states(
    conn: Conn,
    meta: RunMeta,
    tool_calls: tuple[ToolCallRecord, ...],
    orders: tuple[OrderRecord, ...],
) -> tuple[BrokerState, ...]:
    """Pre-order broker state per dispatched place call (module docstring)."""
    calls = {c.identity.tool_call_id: c for c in tool_calls}
    items = _validated_items(conn, meta.run_id)
    owned = {
        r.broker_order.broker_order_id
        for r in (*orders, *owned_unresolved_orders(conn, meta.account_scope_id))
        if r.intent is not None and r.broker_order is not None
    }
    order_of_place = {
        r.intent.place_tool_call_id: r.broker_order.broker_order_id
        for r in orders
        if r.intent is not None and r.broker_order is not None
    }
    mutations = [
        c
        for c in tool_calls
        if c.identity.tool in ("place_option_order", "cancel_option_order")
        and c.dispatched_at is not None
    ]
    states: dict[tuple[uuid.UUID, datetime], BrokerState] = {}
    for place in mutations:
        at = place.dispatched_at
        if place.identity.tool != "place_option_order" or at is None:
            continue
        before = [
            e
            for e in items
            if (done := _completed(calls, tuple(e.source_tool_call_ids()))) is not None
            and done <= at
        ]
        view = RunEvidence(tuple(before))
        snapshot, positions, open_orders = view.account(), view.positions(), view.open_orders()
        if snapshot is None:
            continue
        sources = [*snapshot.tool_call_ids]
        if positions is not None:
            sources += positions.source_tool_call_ids
        if open_orders is not None:
            sources += open_orders.source_tool_call_ids
        completed = _completed(calls, tuple(sources))
        if completed is None:
            continue
        listed: dict[str, AttemptStatus] = {}
        if open_orders is not None:
            read = next((e for e in before if any(o is open_orders for o in e.open_orders)), None)
            for obs in read.broker_orders if read is not None else ():
                listed[obs.broker_order_id] = obs.status
        reflects: list[uuid.UUID] = []
        for m in mutations:
            if m.dispatched_at is None or m.dispatched_at >= at:
                continue
            if m.identity.tool == "place_option_order":
                bid = order_of_place.get(m.identity.tool_call_id)
                if bid is not None and bid in listed:
                    reflects.append(m.identity.tool_call_id)
            else:
                args = m.effective_arguments_redacted or m.identity.arguments_redacted
                oid = args.get("order_id")
                if isinstance(oid, str) and listed.get(oid) in _TERMINAL:
                    reflects.append(m.identity.tool_call_id)
        covers = positions.covers if positions is not None else frozenset()
        state = BrokerState(
            snapshot=snapshot,
            completed_at=max(completed, snapshot.retrieved_at),
            short_options=(
                tuple(
                    ShortOptionHolding(
                        broker_instrument_id=h.broker_instrument_id,
                        short_quantity=h.short_quantity,
                    )
                    for h in positions.short_options
                )
                if positions is not None and PositionsCoverage.OPTIONS in covers
                else None
            ),
            share_holdings=(
                tuple(
                    ShareHolding(
                        underlying=h.symbol, quantity=h.quantity, other_reserved_shares=None
                    )
                    for h in positions.share_holdings
                )
                if positions is not None and PositionsCoverage.SHARES in covers
                else None
            ),
            working_orders=(
                tuple(
                    WorkingOrderObservation(
                        broker_order_id=o.broker_order_ref,
                        broker_instrument_id=o.broker_instrument_id,
                        side_raw=o.side.value,
                        quantity=o.unfilled_quantity,
                        filled_quantity=0,
                        owned=o.broker_order_ref in owned,
                    )
                    for o in open_orders.orders
                    if o.unfilled_quantity > 0
                )
                if open_orders is not None
                else None
            ),
            reflects_mutation_tool_call_ids=tuple(dict.fromkeys(reflects)),
        )
        states.setdefault((snapshot.snapshot_id, state.completed_at), state)
    return tuple(states.values())


def _review_observations(
    tool_calls: tuple[ToolCallRecord, ...], delivered: DeliveredEvidence
) -> tuple[ReviewObservation, ...]:
    calls = {c.identity.tool_call_id: c for c in tool_calls}
    out: dict[uuid.UUID, ReviewObservation] = {}
    for review in delivered.reviews:
        (call_id,) = review.source_tool_call_ids[:1] or (None,)
        call = calls.get(call_id) if call_id is not None else None
        if call is None or call.completed_at is None or call_id in out:
            continue
        leg = review.legs[0] if len(review.legs) == 1 else None
        out[call.identity.tool_call_id] = ReviewObservation(
            review_tool_call_id=call.identity.tool_call_id,
            completed_at=call.completed_at,
            succeeded=call.status is ToolCallStatus.SUCCEEDED,
            warnings=() if review.clean else (review.alert_type or "pre-trade alert",),
            broker_instrument_id=leg.broker_instrument_id if leg else None,
            side_raw=leg.side_raw if leg else None,
            quantity=review.quantity,
            order_type_raw=review.order_type_raw,
            time_in_force_raw=review.time_in_force_raw,
            limit_price=review.limit_price,
        )
    return tuple(out.values())


def load_decisions(conn: Conn, run_id: uuid.UUID) -> tuple[DecisionsInput, uuid.UUID | None]:
    """The effective parse of the latest stored output, and that output's id.

    No output row, or an output without text, means no model output (None).
    """
    outputs = ledger_evidence.effective(ledger_evidence.agent_outputs_for_run(conn, run_id))
    if not outputs:
        return None, None
    output = outputs[-1]
    if output.raw_redacted is None:
        return None, output.record_id
    decisions = [
        d
        for d in ledger_evidence.effective(ledger_evidence.agent_decisions_for_run(conn, run_id))
        if d.output_id == output.record_id
    ]
    if not decisions:
        return None, output.record_id
    decision = decisions[-1]
    if decision.parse_status is ledger_evidence.ParseStatus.VALID and decision.output is not None:
        return DecisionOutputParsed(ok=True, output=decision.output), output.record_id
    return (
        DecisionOutputParseFailure(ok=False, raw_text=output.raw_redacted, issues=decision.issues),
        output.record_id,
    )


def _facts(conn: Conn, run_id: uuid.UUID) -> tuple[DecisionFacts, ...]:
    stored = ledger_evidence.effective(ledger_evidence.decision_facts_for_run(conn, run_id))
    return tuple(f.facts for f in stored)


def _require_off(meta: RunMeta) -> None:
    """Live runs load too (ADR-0034); their order projections come from `_orders`."""


def load_assembly_context(
    conn: Conn,
    meta: RunMeta,
    *,
    terminated_at: datetime,
    book: PositionBook | None,
    output_id: uuid.UUID | None,
) -> AssemblyContext:
    _require_off(meta)
    delivered = load_delivered(conn, meta, book)
    facts = _facts(conn, meta.run_id)
    calls = tool_call_records(conn, meta.run_id)
    return AssemblyContext(
        run_id=meta.run_id,
        environment=meta.environment,
        slot=meta.slot,
        terminated_at=terminated_at,
        requested_execution_mode=meta.requested_mode,
        effective_execution_mode=meta.effective_mode,
        order_venue=meta.order_venue,
        account_scope_id=meta.account_scope_id,
        rules_version=str(meta.rules.version),
        rules_hash=meta.rules.sha256,
        prompt_id=meta.prompt_id,
        prompt_hash=meta.prompt_hash,
        model_id=meta.model_id,
        output_record_id=output_id,
        time_in_force=meta.rules.rules.orders.time_in_force,
        tool_calls=calls,
        orders=run_order_records(conn, meta.run_id) if executes_orders(meta.order_venue) else (),
        facts=facts,
        refs=delivered.refs,
        quotes=delivered.quotes,
        attempt_evidence=attempt_evidence(calls, delivered)
        if executes_orders(meta.order_venue)
        else (),
    )


def check_references(
    conn: Conn,
    meta: RunMeta,
    *,
    as_of: datetime,
    book: PositionBook | None,
    decisions: DecisionOutputParsed,
) -> tuple[str, ...]:
    """ADR-0052: the reference issues of a parsed output, assembled now from this run's
    recorded calls with the same assembler the run record uses (`reference_check`)."""
    context = load_assembly_context(conn, meta, terminated_at=as_of, book=book, output_id=None)
    record = assemble_run_record(context, decisions)
    citable = frozenset(r.ref for r in context.refs if r.kind is RefKind.TOOL_CALL)
    return reference_issues(record, citable)


def load_audit_context(
    conn: Conn,
    meta: RunMeta,
    *,
    book: PositionBook | None,
    decision_output: AgentDecisionOutput | None,
    run_record: RunRecord | None,
) -> AuditContext:
    """Independent evidence for V1–V7: recorded calls, delivered quotes, facts, and output."""
    _require_off(meta)
    delivered = load_delivered(conn, meta, book)
    facts = _facts(conn, meta.run_id)
    calls = tool_call_records(conn, meta.run_id)
    executes = executes_orders(meta.order_venue)
    orders = run_order_records(conn, meta.run_id) if executes else ()
    return AuditContext(
        run_id=meta.run_id,
        effective_execution_mode=meta.effective_mode,
        order_venue=meta.order_venue,
        rules=meta.rules.rules,
        rules_version=str(meta.rules.version),
        rules_hash=meta.rules.sha256,
        assembler_version=ASSEMBLER_VERSION if run_record is not None else None,
        tool_calls=calls,
        order_records=orders,
        broker_states=broker_states(conn, meta, calls, orders) if executes else (),
        quotes=delivered.quotes,
        instruments=delivered.instruments,
        reviews=_review_observations(calls, delivered) if executes else (),
        decision_facts=facts,
        decision_output=decision_output,
        run_record=run_record,
        position_book=book,
    )
