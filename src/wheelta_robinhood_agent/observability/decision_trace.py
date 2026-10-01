"""Decision trace: every decision of one run, traced to what it relied on and what it did.

Pure: built from recorded facts only (the assembled RunRecord, tool-call projections,
persisted DecisionFacts, and audit findings), never from the agent's free text alone. It is
the operator's view of CLAUDE.md prime directives 5 and 8: for each decision, the agent's
rationale and thesis, every cited reference resolved to the recorded tool call or fact set
behind it, the code-computed facts and quotes per leg, each order attempt with the
review/place/cancel calls that make it up, and the audit findings attached to it. A run-level
timeline lists every tool call with its caller (orchestrator or Mignon), tier, and outcome.

Nothing here decides or gates anything. `build_decision_trace` is deterministic for the same
inputs; `render_trace_markdown` and `decision_log_events` are views of the result. Tool
arguments come from the ledger, which stores them redacted; agent text is copied verbatim and
is data, never instructions.
"""

import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AuditOutcome,
    ExecutionMode,
    OrderVenue,
    RunStatus,
)
from wheelta_robinhood_agent.domain.facts import DecisionFacts, DerivedMetric
from wheelta_robinhood_agent.domain.orders import Attempt
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import DecisionRecord, LegRecord, RunRecord
from wheelta_robinhood_agent.domain.tool_calls import ToolCallRecord

TRACE_VERSION: Final = "1"
EVIDENCE_PREFIX: Final = "evidence:"
FACTS_PREFIX: Final = "facts:"
ORCHESTRATOR: Final = "orchestrator"

__all__ = [
    "TRACE_VERSION",
    "DecisionTrace",
    "TraceInput",
    "build_decision_trace",
    "decision_log_events",
    "render_trace_markdown",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TraceCall(_Model):
    """One recorded tool call, as the ledger projects it."""

    tool_call_id: uuid.UUID
    requested_at: datetime
    caller: str  # "orchestrator", the Mignon's agent_type, or "executor (order_work:<id>)"
    server: str
    tool: str
    tier: str | None
    status: str
    deny_reason: str | None
    latency_ms: int | None
    arguments: dict[str, JsonValue]


class TraceRef(_Model):
    """A reference the agent cited, resolved to what code recorded for it."""

    ref: str
    kind: str  # evidence | facts | other
    call: TraceCall | None = None
    facts_ref: str | None = None
    resolved: bool


class TraceFindings(_Model):
    passed: int = 0
    violations: tuple[str, ...] = ()
    unverifiable: tuple[str, ...] = ()


class TraceFacts(_Model):
    facts_ref: str
    purpose: str
    subject_ref: str
    quality: str
    limit_price: Decimal | None
    initial_quantity: int | None
    remaining_quantity: int | None
    metrics: tuple[tuple[str, str, Decimal | None], ...]  # name, unit, value
    gaps: tuple[str, ...]
    input_evidence_ids: tuple[uuid.UUID, ...]


class TraceQuote(_Model):
    bid: Decimal
    ask: Decimal
    as_of: datetime
    source_tool_call_ids: tuple[uuid.UUID, ...]


class TraceAttempt(_Model):
    index: int
    status: str
    requested_quantity: int | None
    limit_price: Decimal | None
    filled_quantity: int | None
    broker_order_id: str | None
    reason_codes: tuple[str, ...]
    proposal_ref: str | None
    place: TraceCall | None
    reviews: tuple[TraceCall, ...]
    cancels: tuple[TraceCall, ...]
    findings: TraceFindings


class TraceLeg(_Model):
    leg_ref: str
    side: str
    occ_symbol: str | None
    target_quantity: int | None
    conditional: bool
    depends_on_leg_ref: str | None
    facts: TraceFacts | None
    quotes: tuple[TraceQuote, ...]
    attempts: tuple[TraceAttempt, ...]
    reason_codes: tuple[str, ...]
    gaps: tuple[str, ...]
    findings: TraceFindings


class TraceDecision(_Model):
    decision_ref: str
    action: str
    underlying: str | None
    priority: int | None
    target_ref: str
    replacement_ref: str | None
    rationale: str
    thesis: str | None
    invalidation_conditions: tuple[str, ...]
    evidence: tuple[TraceRef, ...]
    metrics: tuple[tuple[str, str, Decimal | None], ...]
    gaps: tuple[str, ...]
    legs: tuple[TraceLeg, ...]
    findings: TraceFindings


class TraceUnassociated(_Model):
    kind: str
    occ_symbol: str | None
    side: str | None
    broker_order_id: str | None
    status: str | None
    calls: tuple[TraceCall, ...]
    findings: TraceFindings


class DecisionTrace(_Model):
    trace_version: str = TRACE_VERSION
    run_id: uuid.UUID
    environment: AppEnv
    slot: datetime
    status: RunStatus | None
    reason: str | None
    effective_execution_mode: ExecutionMode | None
    order_venue: OrderVenue | None
    prompt_execution_mode: str | None
    model_id: str | None
    rules_version: str | None
    prompt_hash: str | None
    decision_output_status: str | None
    summary: str | None
    decisions: tuple[TraceDecision, ...]
    unassociated: tuple[TraceUnassociated, ...]
    unresolved_questions: tuple[str, ...]
    cancellation_rationales: tuple[str, ...]
    run_findings: TraceFindings
    audit_counts: dict[str, int]
    tool_call_counts: dict[str, int]
    mignon_spawns: int
    next_run: dict[str, JsonValue] | None
    timeline: tuple[TraceCall, ...]


@dataclass(frozen=True)
class TraceInput:
    """Recorded inputs for one run's trace (loaded by `agent/trace_loader.py`)."""

    run_id: uuid.UUID
    environment: AppEnv
    slot: datetime
    status: RunStatus | None = None
    reason: str | None = None
    effective_execution_mode: ExecutionMode | None = None
    order_venue: OrderVenue | None = None
    prompt_execution_mode: str | None = None
    model_id: str | None = None
    rules_version: str | None = None
    prompt_hash: str | None = None
    record: RunRecord | None = None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    facts: tuple[DecisionFacts, ...] = ()
    findings: tuple[AuditFinding, ...] = ()
    next_run: Mapping[str, JsonValue] | None = None


def _call(record: ToolCallRecord) -> TraceCall:
    identity = record.identity
    arguments = record.effective_arguments_redacted or identity.arguments_redacted
    return TraceCall(
        tool_call_id=identity.tool_call_id,
        requested_at=identity.requested_at,
        caller=(
            f"executor (order_work:{identity.parent_tool_call_id})"
            if identity.parent_tool_call_id is not None
            else identity.agent_type or ORCHESTRATOR
        ),
        server=identity.server,
        tool=identity.tool,
        tier=identity.tier.value if identity.tier is not None else None,
        status=record.status.value,
        deny_reason=record.deny_reason,
        latency_ms=record.latency_ms,
        arguments=dict(arguments),
    )


def _findings(items: Iterable[AuditFinding]) -> TraceFindings:
    passed, violations, unverifiable = 0, [], []
    for f in items:
        label = f"{f.check_id.value}.{f.sub_item or 'all'}: {f.detail}"
        if f.outcome is AuditOutcome.PASS:
            passed += 1
        elif f.outcome is AuditOutcome.VIOLATION:
            violations.append(label)
        else:
            unverifiable.append(label)
    return TraceFindings(
        passed=passed, violations=tuple(violations), unverifiable=tuple(unverifiable)
    )


def _metrics(items: Iterable[DerivedMetric]) -> tuple[tuple[str, str, Decimal | None], ...]:
    return tuple((m.name, m.unit, m.value.value) for m in items)


@dataclass
class _Builder:
    inp: TraceInput
    calls: dict[uuid.UUID, TraceCall] = field(default_factory=dict)
    facts: dict[str, DecisionFacts] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.calls = {r.identity.tool_call_id: _call(r) for r in self.inp.tool_calls}
        self.facts = {f.facts_ref: f for f in self.inp.facts}

    def _pick(self, ids: Iterable[uuid.UUID]) -> tuple[TraceCall, ...]:
        return tuple(self.calls[i] for i in ids if i in self.calls)

    def ref(self, ref: str) -> TraceRef:
        if ref.startswith(EVIDENCE_PREFIX):
            try:
                call = self.calls.get(uuid.UUID(ref.removeprefix(EVIDENCE_PREFIX)))
            except ValueError:
                call = None
            return TraceRef(ref=ref, kind="evidence", call=call, resolved=call is not None)
        if ref in self.facts:
            return TraceRef(ref=ref, kind="facts", facts_ref=ref, resolved=True)
        if ref.startswith(FACTS_PREFIX):
            return TraceRef(ref=ref, kind="facts", resolved=False)
        return TraceRef(ref=ref, kind="other", resolved=True)

    def fact_set(self, ref: str | None) -> TraceFacts | None:
        facts = self.facts.get(ref) if ref else None
        if facts is None:
            return None
        return TraceFacts(
            facts_ref=facts.facts_ref,
            purpose=facts.purpose.value,
            subject_ref=facts.subject_ref,
            quality=facts.quality.value,
            limit_price=facts.limit_price,
            initial_quantity=facts.initial_quantity,
            remaining_quantity=facts.remaining_quantity,
            metrics=_metrics(facts.metrics),
            gaps=tuple(f"{g.field}: {g.detail}" for g in facts.gaps),
            input_evidence_ids=facts.input_evidence_ids,
        )

    def attempt(self, attempt: Attempt, findings: Sequence[AuditFinding]) -> TraceAttempt:
        place = self.calls.get(attempt.place_tool_call_id) if attempt.place_tool_call_id else None
        return TraceAttempt(
            index=attempt.index,
            status=attempt.status.value,
            requested_quantity=attempt.requested_quantity,
            limit_price=attempt.limit_price,
            filled_quantity=attempt.filled_quantity,
            broker_order_id=attempt.broker_order_id,
            reason_codes=tuple(c.value for c in attempt.reason_codes),
            proposal_ref=attempt.proposal_ref,
            place=place,
            reviews=self._pick(attempt.review_tool_call_ids),
            cancels=self._pick(attempt.cancel_tool_call_ids),
            findings=_findings(f for f in findings if f.attempt_index == attempt.index),
        )

    def leg(self, leg: LegRecord, findings: Sequence[AuditFinding]) -> TraceLeg:
        mine = [f for f in findings if f.leg_ref == leg.leg_ref]
        return TraceLeg(
            leg_ref=leg.leg_ref,
            side=leg.side.value,
            occ_symbol=str(leg.occ_symbol) if leg.occ_symbol is not None else None,
            target_quantity=leg.target_quantity,
            conditional=leg.conditional,
            depends_on_leg_ref=leg.depends_on_leg_ref,
            facts=self.fact_set(leg.facts_ref),
            quotes=tuple(
                TraceQuote(
                    bid=q.bid, ask=q.ask, as_of=q.as_of, source_tool_call_ids=q.source_tool_call_ids
                )
                for q in leg.quotes
            ),
            attempts=tuple(self.attempt(a, mine) for a in leg.attempts),
            reason_codes=tuple(c.value for c in leg.reason_codes),
            gaps=tuple(f"{g.field}: {g.detail}" for g in leg.gaps),
            findings=_findings(f for f in mine if f.attempt_index is None),
        )

    def decision(self, d: DecisionRecord) -> TraceDecision:
        mine = [f for f in self.inp.findings if f.decision_ref == d.decision_ref]
        return TraceDecision(
            decision_ref=d.decision_ref,
            action=d.action.value,
            underlying=d.underlying,
            priority=d.priority,
            target_ref=d.target_ref,
            replacement_ref=d.replacement_ref,
            rationale=d.rationale,
            thesis=d.thesis,
            invalidation_conditions=d.invalidation_conditions,
            evidence=tuple(self.ref(r) for r in d.evidence_refs),
            metrics=_metrics(d.metrics),
            gaps=tuple(f"{g.field}: {g.detail}" for g in d.gaps),
            legs=tuple(self.leg(leg, mine) for leg in d.legs),
            findings=_findings(f for f in mine if f.leg_ref is None),
        )

    def unassociated(
        self, record: RunRecord, findings: Sequence[AuditFinding]
    ) -> tuple[TraceUnassociated, ...]:
        """Actions no decision selected, each with the run-level findings about its calls."""
        out = []
        for u in record.unassociated_actions:
            ids: list[uuid.UUID] = []
            status = order_id = None
            if u.attempt is not None:
                status, order_id = u.attempt.status.value, u.attempt.broker_order_id
                if u.attempt.place_tool_call_id:
                    ids.append(u.attempt.place_tool_call_id)
                ids.extend(u.attempt.review_tool_call_ids)
                ids.extend(u.attempt.cancel_tool_call_ids)
            if u.cancellation is not None:
                status = u.cancellation.status.value
                order_id = order_id or u.cancellation.broker_order_id
                ids.append(u.cancellation.cancel_tool_call_id)
            mine = set(ids)
            out.append(
                TraceUnassociated(
                    kind=u.kind.value,
                    occ_symbol=str(u.occ_symbol) if u.occ_symbol is not None else None,
                    side=u.side_raw,
                    broker_order_id=order_id,
                    status=status,
                    calls=tuple(sorted(self._pick(ids), key=lambda c: c.requested_at)),
                    findings=_findings(f for f in findings if mine & set(f.tool_call_ids)),
                )
            )
        return tuple(out)


def build_decision_trace(inp: TraceInput) -> DecisionTrace:
    """Assemble the trace (module docstring). Deterministic for the same inputs."""
    b = _Builder(inp)
    record = inp.record
    run_level = [f for f in inp.findings if f.decision_ref is None]
    unassociated = b.unassociated(record, run_level) if record else ()
    claimed = {c.tool_call_id for u in unassociated for c in u.calls}
    run_level = [f for f in run_level if not claimed & set(f.tool_call_ids)]
    timeline = tuple(sorted(b.calls.values(), key=lambda c: (c.requested_at, str(c.tool_call_id))))
    counts = Counter(c.status for c in timeline)
    return DecisionTrace(
        run_id=inp.run_id,
        environment=inp.environment,
        slot=inp.slot,
        status=inp.status,
        reason=inp.reason,
        effective_execution_mode=inp.effective_execution_mode,
        order_venue=inp.order_venue,
        prompt_execution_mode=inp.prompt_execution_mode,
        model_id=inp.model_id or (record.model_id if record else None),
        rules_version=inp.rules_version or (record.rules_version if record else None),
        prompt_hash=inp.prompt_hash or (record.prompt_hash if record else None),
        decision_output_status=record.decision_output_status.value if record else None,
        summary=record.summary if record else None,
        decisions=tuple(b.decision(d) for d in record.decisions) if record else (),
        unassociated=unassociated,
        unresolved_questions=(
            tuple(q.question for q in record.unresolved_questions) if record else ()
        ),
        cancellation_rationales=(
            tuple(c.rationale for c in record.cancellation_rationales) if record else ()
        ),
        run_findings=_findings(run_level),
        audit_counts=dict(sorted(Counter(f.outcome.value for f in inp.findings).items())),
        tool_call_counts=dict(sorted(counts.items())),
        mignon_spawns=sum(1 for c in timeline if c.tier == "D"),
        next_run=dict(inp.next_run) if inp.next_run is not None else None,
        timeline=timeline,
    )


# ------------------------------------------------------------------------------------ views
def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def _money(value: Decimal | None) -> str:
    return "n/a" if value is None else str(value)


def _call_line(call: TraceCall) -> str:
    extra = f" — denied: {call.deny_reason}" if call.deny_reason else ""
    return (
        f"`{call.requested_at:%H:%M:%S}` {call.caller} → `{call.server}.{call.tool}` "
        f"[{call.tier or '-'}] **{call.status}**{extra} (`{call.tool_call_id}`)"
    )


def _findings_lines(findings: TraceFindings, indent: str) -> list[str]:
    lines = [f"{indent}- audit: {findings.passed} pass"]
    lines += [f"{indent}  - ⚠ violation {v}" for v in findings.violations]
    lines += [f"{indent}  - ? unverifiable {u}" for u in findings.unverifiable]
    return lines


def _attempt_lines(a: TraceAttempt) -> list[str]:
    head = (
        f"    - attempt {a.index}: **{a.status}** qty {a.requested_quantity} @ "
        f"{_money(a.limit_price)}"
    )
    if a.filled_quantity is not None:
        head += f", filled {a.filled_quantity}"
    if a.broker_order_id:
        head += f", order `{a.broker_order_id}`"
    if a.reason_codes:
        head += f", codes {', '.join(a.reason_codes)}"
    lines = [head]
    lines += [f"      - review {_call_line(c)}" for c in a.reviews]
    if a.place is not None:
        lines.append(f"      - place {_call_line(a.place)}")
    lines += [f"      - cancel {_call_line(c)}" for c in a.cancels]
    lines += _findings_lines(a.findings, "      ")
    return lines


def _leg_lines(leg: TraceLeg) -> list[str]:
    lines = [
        f"  - leg `{leg.leg_ref}`: {leg.side} {leg.occ_symbol or '(contract unknown)'}, "
        f"target qty {leg.target_quantity}" + (" (conditional)" if leg.conditional else "")
    ]
    if leg.facts is not None:
        f = leg.facts
        lines.append(
            f"    - facts `{f.facts_ref}` ({f.purpose}, quality {f.quality}): limit "
            f"{_money(f.limit_price)}, quantity {f.initial_quantity}"
        )
        lines += [f"      - {n} = {_money(v)} {u}" for n, u, v in f.metrics]
        lines += [f"      - gap {g}" for g in f.gaps]
    lines += [f"    - quote bid {q.bid} / ask {q.ask} at {q.as_of.isoformat()}" for q in leg.quotes]
    if not leg.attempts:
        lines.append("    - no order attempt")
    for a in leg.attempts:
        lines += _attempt_lines(a)
    lines += [f"    - gap {g}" for g in leg.gaps]
    if leg.reason_codes:
        lines.append(f"    - reason codes: {', '.join(leg.reason_codes)}")
    lines += _findings_lines(leg.findings, "    ")
    return lines


def render_trace_markdown(trace: DecisionTrace) -> str:
    """The trace as Markdown for an operator (no network, no clock)."""
    mode = trace.effective_execution_mode.value if trace.effective_execution_mode else "?"
    venue = trace.order_venue.value if trace.order_venue else "?"
    lines = [
        f"# Run {trace.run_id}",
        "",
        f"- slot {trace.slot.isoformat()} · {trace.environment.value} · status "
        f"**{trace.status.value if trace.status else 'unknown'}**"
        + (f" ({trace.reason})" if trace.reason else ""),
        f"- effective mode **{mode}**, order venue **{venue}**, prompt told "
        f"`{trace.prompt_execution_mode or '?'}`",
        f"- model {trace.model_id or '?'} · rules v{trace.rules_version or '?'} · prompt "
        f"{trace.prompt_hash or '?'}",
        f"- agent output: {trace.decision_output_status or 'none'}",
        f"- tool calls: {_counts(trace.tool_call_counts) or 'none'}"
        f" · Mignon spawns: {trace.mignon_spawns}",
        f"- audit: {_counts(trace.audit_counts) or 'not run'}",
    ]
    if trace.next_run is not None:
        lines.append(f"- next run: {trace.next_run}")
    if trace.summary:
        lines += ["", trace.summary]
    lines += ["", "## Decisions", ""]
    if not trace.decisions:
        lines.append("None.")
    for d in trace.decisions:
        lines += [
            f"### {d.action} {d.underlying or ''} (`{d.decision_ref}`, priority {d.priority})",
            "",
            f"- target `{d.target_ref}`"
            + (f", replacement `{d.replacement_ref}`" if d.replacement_ref else ""),
            f"- rationale: {d.rationale}",
        ]
        if d.thesis:
            lines.append(f"- thesis: {d.thesis}")
        lines += [f"- invalidated if: {c}" for c in d.invalidation_conditions]
        for e in d.evidence:
            if e.call is not None:
                lines.append(f"- cites `{e.ref}`: {_call_line(e.call)}")
            else:
                state = "" if e.resolved else " (**unresolved**)"
                lines.append(f"- cites `{e.ref}` ({e.kind}){state}")
        lines += [f"- metric {n} = {_money(v)} {u}" for n, u, v in d.metrics]
        lines += [f"- gap {g}" for g in d.gaps]
        for leg in d.legs:
            lines += _leg_lines(leg)
        lines += _findings_lines(d.findings, "")
        lines.append("")
    if trace.unassociated:
        lines += ["## Actions not linked to a decision", ""]
        for u in trace.unassociated:
            what = " ".join(p for p in (u.kind, u.occ_symbol, u.side) if p)
            order = f" order `{u.broker_order_id}`" if u.broker_order_id else ""
            lines.append(f"- {what}{order}: **{u.status or 'unknown'}**")
            lines += [f"  - {_call_line(c)}" for c in u.calls]
            lines += _findings_lines(u.findings, "  ")
        lines.append("")
    if trace.cancellation_rationales or trace.unresolved_questions:
        lines += ["## Agent notes", ""]
        lines += [f"- cancellation: {c}" for c in trace.cancellation_rationales]
        lines += [f"- open question: {q}" for q in trace.unresolved_questions]
        lines.append("")
    lines += ["## Run-level audit", "", *_findings_lines(trace.run_findings, ""), ""]
    lines += ["## Timeline", ""]
    lines += [f"- {_call_line(c)}" for c in trace.timeline] or ["No tool calls."]
    return "\n".join(lines) + "\n"


def decision_log_events(trace: DecisionTrace) -> list[dict[str, JsonValue]]:
    """One compact, structured log payload per decision, for stdout (Railway logs)."""
    events: list[dict[str, JsonValue]] = []
    for d in trace.decisions:
        attempts = [a for leg in d.legs for a in leg.attempts]
        violations: list[JsonValue] = [
            *d.findings.violations,
            *(v for leg in d.legs for v in leg.findings.violations),
            *(v for leg in d.legs for a in leg.attempts for v in a.findings.violations),
        ]
        events.append(
            {
                "decision_ref": d.decision_ref,
                "action": d.action,
                "underlying": d.underlying,
                "legs": [
                    {
                        "side": leg.side,
                        "occ_symbol": leg.occ_symbol,
                        "target_quantity": leg.target_quantity,
                        "facts_ref": leg.facts.facts_ref if leg.facts else None,
                    }
                    for leg in d.legs
                ],
                "attempts": [
                    {
                        "status": a.status,
                        "requested_quantity": a.requested_quantity,
                        "limit_price": _money(a.limit_price),
                        "broker_order_id": a.broker_order_id,
                    }
                    for a in attempts
                ],
                "evidence_refs": len(d.evidence),
                "unresolved_refs": sum(1 for e in d.evidence if not e.resolved),
                "audit_violations": violations,
            }
        )
    return events
