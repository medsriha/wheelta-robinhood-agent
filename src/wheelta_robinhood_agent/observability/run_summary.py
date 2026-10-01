"""Run-summary email content (ADR-0029). Pure: no HTTP, no clock, no LLM.

One summary is sent for every run whose agent session started. It is built from what the run
recorded (the assembled RunRecord, the audit status, alerts, the next run), never from the
agent's free text alone. `summary_facts` is the redacted JSON handed to the prose writer and
`render_facts_text` is the deterministic facts block every email carries, so a figure in the
prose can always be checked against the recorded one. Delivery is in
``integrations/notifications``.
"""

import html
from datetime import datetime
from decimal import Decimal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    ExecutionMode,
    OrderVenue,
    RunStatus,
)
from wheelta_robinhood_agent.domain.gating import executes_orders, with_default_venue
from wheelta_robinhood_agent.domain.mignon_report import MignonReport
from wheelta_robinhood_agent.domain.orders import Attempt, ReasonCode
from wheelta_robinhood_agent.domain.run_record import LegRecord, RunRecord, UnassociatedAction
from wheelta_robinhood_agent.observability.redaction import Redactor

SUBJECT_PREFIX = "[Wheelta agent]"
FACTS_HEADING = "Recorded facts (authoritative)"
PROSE_UNAVAILABLE = "The written summary is unavailable for this run; the recorded facts follow."


class ConsideredOption(BaseModel):
    """A candidate delivered during research, with recorded limitations, not inferred motives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_ref: str
    underlying: str
    occ_symbol: str
    gaps: tuple[str, ...] = ()


class RunSummaryInput(BaseModel):
    """Everything one summary email may say, as recorded by the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    environment: AppEnv
    slot: AwareDatetime
    status: RunStatus
    reason: str | None
    requested_execution_mode: ExecutionMode
    effective_execution_mode: ExecutionMode
    # ADR-0038; defaults from the mode (live: broker, off: none) when not given.
    order_venue: OrderVenue
    # None when assembly failed; the email then says so instead of listing decisions.
    record: RunRecord | None
    # None when the audit did not run in this invocation.
    audit_status: str | None = None
    audit_violations: int = Field(default=0, ge=0)
    audit_unverifiable: int = Field(default=0, ge=0)
    alerts: tuple[str, ...] = ()
    next_run_at: AwareDatetime | None = None
    next_run_source: str | None = None
    next_run_rationale: str | None = None
    candidates: tuple[ConsideredOption, ...] = ()
    research_reports: tuple[MignonReport, ...] = ()
    diagnostic_details: tuple[str, ...] = ()
    audit_details: tuple[str, ...] = ()
    research_unavailable: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_venue(cls, data: object) -> object:
        return with_default_venue(data)


def _is_proposal(attempt: Attempt) -> bool:
    return attempt.proposal_ref is not None and ReasonCode.DRY_RUN in attempt.reason_codes


def _attempts(record: RunRecord | None) -> list[Attempt]:
    if record is None:
        return []
    return [a for d in record.decisions for leg in d.legs for a in leg.attempts]


def proposal_count(record: RunRecord | None) -> int:
    """Dry-run proposals: unsubmitted attempts stamped DRY_RUN."""
    return sum(1 for a in _attempts(record) if _is_proposal(a))


def placed_count(record: RunRecord | None) -> int:
    """Recorded place calls that reached a broker status other than not_placed."""
    placed = [a for a in _attempts(record) if a.place_tool_call_id is not None]
    if record is not None:
        placed += [
            u.attempt
            for u in record.unassociated_actions
            if u.attempt is not None and u.attempt.place_tool_call_id is not None
        ]
    return sum(1 for a in placed if a.status is not AttemptStatus.NOT_PLACED)


def _mode_label(mode: ExecutionMode, venue: OrderVenue) -> str:
    if mode is ExecutionMode.LIVE:
        return "live"
    return "dry run (simulated orders)" if venue is OrderVenue.SIMULATED else "dry run"


def build_subject(summary: RunSummaryInput) -> str:
    """`[Wheelta agent] <status> · <mode> · <activity> · <slot UTC>`."""
    record = summary.record
    if executes_orders(summary.order_venue):
        count = placed_count(record)
        simulated = "simulated " if summary.order_venue is OrderVenue.SIMULATED else ""
        plural = "s" if count != 1 else ""
        activity = f"{count} {simulated}order{plural} placed" if count else "no trades"
    else:
        count = proposal_count(record)
        activity = f"{count} proposal{'s' if count != 1 else ''}" if count else "no trades"
    return " · ".join(
        (
            f"{SUBJECT_PREFIX} {summary.status.value}",
            _mode_label(summary.effective_execution_mode, summary.order_venue),
            activity,
            f"{summary.slot:%Y-%m-%d %H:%M} UTC",
        )
    )


def _num(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _attempt_facts(attempt: Attempt) -> dict[str, JsonValue]:
    return {
        "kind": "dry_run_proposal" if _is_proposal(attempt) else "place_call",
        "status": attempt.status.value,
        "requested_quantity": attempt.requested_quantity,
        "limit_price": _num(attempt.limit_price),
        "filled_quantity": attempt.filled_quantity,
        "reason_codes": [c.value for c in attempt.reason_codes],
    }


def _leg_facts(leg: LegRecord) -> dict[str, JsonValue]:
    return {
        "side": leg.side.value,
        "occ_symbol": str(leg.occ_symbol) if leg.occ_symbol is not None else None,
        "right": leg.right.value if leg.right is not None else None,
        "strike": _num(leg.strike),
        "expiration": leg.expiration.isoformat() if leg.expiration is not None else None,
        "target_quantity": leg.target_quantity,
        "conditional": leg.conditional,
        "reason_codes": [c.value for c in leg.reason_codes],
        "gaps": [f"{g.field}: {g.detail}" for g in leg.gaps],
        "attempts": [_attempt_facts(a) for a in leg.attempts],
    }


def _record_facts(record: RunRecord) -> dict[str, JsonValue]:
    return {
        "decision_output_status": record.decision_output_status.value,
        "model_id": record.model_id,
        "rules_version": record.rules_version,
        "prompt_id": record.prompt_id,
        "assembly_summary": record.summary,
        "decisions": [
            {
                "decision_ref": d.decision_ref,
                "action": d.action.value,
                "underlying": d.underlying,
                "target_ref": d.target_ref,
                "replacement_ref": d.replacement_ref,
                "rationale": d.rationale,
                "thesis": d.thesis,
                "invalidation_conditions": list(d.invalidation_conditions),
                "gaps": [f"{g.field}: {g.detail}" for g in d.gaps],
                "metrics": [
                    {"name": m.name, "value": _num(m.value.value), "unit": m.unit}
                    for m in d.metrics
                ],
                "legs": [_leg_facts(leg) for leg in d.legs],
            }
            for d in record.decisions
        ],
        "cancellations": [
            {"status": c.status.value, "dispatch_status": _enum(c.dispatch_status)}
            for c in record.cancellations
        ],
        "cancellation_rationales": [c.rationale for c in record.cancellation_rationales],
        "unresolved_questions": [q.question for q in record.unresolved_questions],
        "unassociated_actions": [
            {
                "kind": u.kind.value,
                "occ_symbol": str(u.occ_symbol) if u.occ_symbol is not None else None,
                "status": _unassociated(u)[0],
                "broker_order_id": _unassociated(u)[1],
            }
            for u in record.unassociated_actions
        ],
        "gaps": [f"{g.field}: {g.detail}" for g in record.gaps],
        "metrics": [
            {"name": m.name, "value": _num(m.value.value), "unit": m.unit} for m in record.metrics
        ],
        "assembly_findings": [f"{f.code}: {f.detail}" for f in record.findings],
    }


def _candidate_facts(summary: RunSummaryInput) -> list[dict[str, JsonValue]]:
    decisions = summary.record.decisions if summary.record is not None else ()
    complete = (
        summary.record is not None and summary.record.decision_output_status.value == "parsed"
    )
    candidates: list[dict[str, JsonValue]] = []
    for candidate in summary.candidates:
        selected = [
            d
            for d in decisions
            if candidate.candidate_ref in (d.target_ref, d.replacement_ref)
            or any(str(leg.occ_symbol) == candidate.occ_symbol for leg in d.legs)
        ]
        candidates.append(
            {
                "candidate_ref": candidate.candidate_ref,
                "underlying": candidate.underlying,
                "occ_symbol": candidate.occ_symbol,
                "selection": "selected" if selected else "not selected" if complete else "unknown",
                "rationales": [d.rationale for d in selected],
                "gaps": list(candidate.gaps),
                "research_findings": [
                    finding.claim
                    for report in summary.research_reports
                    for finding in report.findings
                    if candidate.candidate_ref in finding.refs
                ],
            }
        )
    return candidates


def _unassociated(action: UnassociatedAction) -> tuple[str | None, str | None]:
    """(status, broker order ID) of an action no decision selected: the attempt's, else the
    cancellation's."""
    if action.attempt is not None:
        return action.attempt.status.value, action.attempt.broker_order_id
    if action.cancellation is not None:
        return action.cancellation.status.value, action.cancellation.broker_order_id
    return None, None


def _enum(value: object) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def summary_facts(summary: RunSummaryInput, redactor: Redactor) -> dict[str, JsonValue]:
    """The redacted JSON the prose writer summarizes. Numbers are strings, copied verbatim."""
    facts: dict[str, JsonValue] = {
        "run": {
            "run_id": summary.run_id,
            "environment": summary.environment.value,
            "slot": summary.slot.isoformat(),
            "status": summary.status.value,
            "reason": summary.reason,
            "requested_execution_mode": summary.requested_execution_mode.value,
            "effective_execution_mode": summary.effective_execution_mode.value,
            "order_venue": summary.order_venue.value,
            "orders_sent_to_broker": summary.order_venue is OrderVenue.BROKER,
        },
        "record": _record_facts(summary.record) if summary.record is not None else None,
        "candidates": [c for c in _candidate_facts(summary)],
        "research_reports": [r.model_dump(mode="json") for r in summary.research_reports],
        "research_unavailable": summary.research_unavailable,
        "diagnostic_details": list(summary.diagnostic_details),
        "audit": {
            "status": summary.audit_status,
            "violations": summary.audit_violations,
            "unverifiable": summary.audit_unverifiable,
            "details": list(summary.audit_details),
        },
        "alerts": list(summary.alerts),
        "next_run": {
            "not_before": _iso(summary.next_run_at),
            "source": summary.next_run_source,
            "agent_rationale": summary.next_run_rationale,
        },
    }
    redacted = redactor.redact(facts)
    if not isinstance(redacted, dict):  # redact() preserves mappings; narrowed for typing
        raise TypeError("redacted facts must stay a mapping")
    return redacted


def _leg_line(leg: LegRecord) -> str:
    contract = str(leg.occ_symbol) if leg.occ_symbol is not None else "contract unknown"
    parts = [leg.side.value, contract]
    if leg.strike is not None:
        parts.append(f"strike {leg.strike}")
    if leg.expiration is not None:
        parts.append(f"exp {leg.expiration.isoformat()}")
    if leg.target_quantity is not None:
        parts.append(f"qty {leg.target_quantity}")
    return " · ".join(parts)


def _attempt_line(attempt: Attempt) -> str:
    kind = "proposal (not sent)" if _is_proposal(attempt) else "order"
    parts = [f"{kind}: {attempt.status.value}"]
    if attempt.requested_quantity is not None:
        parts.append(f"qty {attempt.requested_quantity}")
    if attempt.limit_price is not None:
        parts.append(f"limit {attempt.limit_price}")
    if attempt.filled_quantity:
        parts.append(f"filled {attempt.filled_quantity}")
    if attempt.reason_codes:
        parts.append("codes " + ",".join(c.value for c in attempt.reason_codes))
    return " · ".join(parts)


def _candidate_lines(summary: RunSummaryInput) -> list[str]:
    """One row per contract/status; selected contracts already appear with their decisions."""
    groups: dict[tuple[str, str, str], tuple[list[str], list[str]]] = {}
    for model, candidate in zip(summary.candidates, _candidate_facts(summary), strict=True):
        selection = str(candidate["selection"])
        key = (model.underlying, model.occ_symbol, selection)
        refs, gaps = groups.setdefault(key, ([], []))
        if model.candidate_ref not in refs:
            refs.append(model.candidate_ref)
        gaps.extend(gap for gap in model.gaps if gap not in gaps)
    lines: list[str] = []
    statuses: set[str] = set()
    for (underlying, contract, selection), (refs, gaps) in groups.items():
        if selection == "selected" and not gaps:
            continue
        statuses.add(selection)
        lines.append(f"- {underlying} {contract} [{', '.join(refs)}]: {selection}")
        lines.extend(f"    Data gap: {gap}" for gap in gaps)
    if lines:
        lines.insert(0, "Other candidates / candidate gaps (selection is not execution):")
        if "not selected" in statuses:
            lines.append("Not selected does not establish a rejection reason; research follows.")
        if "unknown" in statuses:
            lines.append("Selection unknown because valid final decisions are unavailable.")
    elif not summary.candidates:
        lines.append("No candidate list recorded; this does not establish none were considered.")
    return lines


def _research_lines(summary: RunSummaryInput) -> list[str]:
    """Print each exact claim once, retaining all sources and task-scoped gaps/questions."""
    if not summary.research_reports:
        return []
    findings: dict[str, list[str]] = {}
    tasks: dict[str, None] = {}
    issues: dict[str, None] = {}
    for report in summary.research_reports:
        tasks[report.task] = None
        for finding in report.findings:
            # ADR-0056: a number resting only on fetched pages is labelled as such.
            claim = f"{finding.claim} (web-sourced)" if finding.web_sourced else finding.claim
            refs = findings.setdefault(claim, [])
            refs.extend(ref for ref in (*finding.refs, *finding.web_urls) if ref not in refs)
        for gap in report.gaps:
            issues[f"- Research gap ({report.task}): {gap}"] = None
        for question in report.follow_up_questions:
            issues[f"- Research question ({report.task}): {question}"] = None
    return [
        "Research (supporting claims, not final decisions): " + "; ".join(tasks),
        *(f"- {claim} [{', '.join(refs)}]" for claim, refs in findings.items()),
        *issues,
    ]


def render_facts_text(summary: RunSummaryInput, redactor: Redactor) -> str:
    """The deterministic facts block, from recorded values only."""
    lines = [
        f"Run {summary.run_id} ({summary.environment.value})",
        f"Slot: {summary.slot.isoformat()}",
        f"Status: {summary.status.value}" + (f" ({summary.reason})" if summary.reason else ""),
        f"Mode: {_mode_label(summary.effective_execution_mode, summary.order_venue)} "
        f"(requested {summary.requested_execution_mode.value})",
    ]
    record = summary.record
    if summary.status in (RunStatus.FAILED, RunStatus.TIMED_OUT, RunStatus.STOPPED):
        explanations = {
            "invalid_agent_output": "The agent's final decision output was missing or invalid.",
            "audit_failed": "The post-run audit could not complete.",
            "deadline": "The run exhausted its time budget and the session was stopped.",
            "sigterm": "The session was stopped by SIGTERM.",
            "sigint": "The session was stopped by SIGINT.",
            "infrastructure_failure": "An infrastructure failure stopped the session.",
        }
        lines.append(
            "Failure / stop reason: "
            + explanations.get(
                summary.reason or "", summary.reason or "No failure reason was recorded."
            )
        )
    lines.extend(f"Diagnostic: {detail}" for detail in dict.fromkeys(summary.diagnostic_details))
    if record is None:
        lines.append("Run record: not assembled (see the ledger for recorded events)")
    else:
        lines.append(f"Agent output: {record.decision_output_status.value}")
        lines.append(f"Summary: {record.summary}")
        if not record.decisions:
            lines.append("Decisions: none")
        for d in record.decisions:
            underlying = f" {d.underlying}" if d.underlying else ""
            lines.append(f"- {d.action.value}{underlying} [{d.decision_ref}]")
            lines.append(f"    Why: {d.rationale}")
            if d.thesis and d.thesis.strip() != d.rationale.strip():
                lines.append(f"    Thesis: {d.thesis}")
            lines.extend(
                f"    Reconsider if: {c}" for c in dict.fromkeys(d.invalidation_conditions)
            )
            lines.extend(f"    Data gap: {g.field}: {g.detail}" for g in d.gaps)
            lines.extend(
                f"    {m.name}: {_num(m.value.value) or 'unknown'} {m.unit}" for m in d.metrics
            )
            for leg in d.legs:
                lines.append(f"    {_leg_line(leg)}")
                lines.extend(f"      Data gap: {g.field}: {g.detail}" for g in leg.gaps)
                if leg.reason_codes:
                    lines.append("      Reasons: " + ", ".join(c.value for c in leg.reason_codes))
                for attempt in leg.attempts:
                    lines.append(f"      {_attempt_line(attempt)}")
        for c in record.cancellations:
            lines.append(f"- cancel: {c.status.value}")
        lines.extend(
            f"- cancellation rationale: {c.rationale}" for c in record.cancellation_rationales
        )
        for u in record.unassociated_actions:
            status, order_id = _unassociated(u)
            what = " · ".join(
                p for p in (str(u.occ_symbol) if u.occ_symbol else None, u.side_raw) if p
            )
            order = f" (order {order_id})" if order_id else ""
            lines.append(
                f"- unassociated {u.kind.value}{': ' + what if what else ''}{order}: "
                f"{status or 'unknown'}"
            )
        for q in record.unresolved_questions:
            lines.append(f"- open question: {q.question}")
        lines.extend(f"Data gap: {g.field}: {g.detail}" for g in record.gaps)
        lines.extend(
            f"{m.name}: {_num(m.value.value) or 'unknown'} {m.unit}" for m in record.metrics
        )
        lines.extend(f"Assembly finding: {f.code}: {f.detail}" for f in record.findings)
    lines.extend(_candidate_lines(summary))
    if summary.research_unavailable:
        lines.append(f"Research context unavailable: {summary.research_unavailable}")
    lines.extend(_research_lines(summary))
    audit = summary.audit_status or "not run"
    lines.append(
        f"Audit: {audit} · {summary.audit_violations} violation(s) · "
        f"{summary.audit_unverifiable} unverifiable check(s)"
    )
    lines.append("Alerts: " + (", ".join(summary.alerts) if summary.alerts else "none"))
    lines.extend(f"Audit detail: {detail}" for detail in dict.fromkeys(summary.audit_details))
    if summary.next_run_at is not None:
        source = f" ({summary.next_run_source})" if summary.next_run_source else ""
        lines.append(f"Next run not before: {summary.next_run_at.isoformat()}{source}")
        if summary.next_run_rationale:
            lines.append(f"Next run rationale: {summary.next_run_rationale}")
    return redactor.redact_text("\n".join(lines))


def render_bodies(prose: str | None, facts_text: str, redactor: Redactor) -> tuple[str, str]:
    """(text, html) bodies: the prose (or a fallback line), then the facts block."""
    lead = redactor.redact_text(prose.strip()) if prose and prose.strip() else PROSE_UNAVAILABLE
    text = f"{lead}\n\n{FACTS_HEADING}\n\n{facts_text}\n"
    paragraphs = "".join(
        f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in lead.split("\n\n") if p
    )
    body = (
        '<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;'
        'font-size:14px;line-height:1.5;color:#1a1a1a;max-width:680px">'
        f"{paragraphs}"
        f'<h3 style="margin:24px 0 8px;font-size:14px">{html.escape(FACTS_HEADING)}</h3>'
        '<pre style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;'
        'background:#f4f5f7;padding:12px;border-radius:6px;white-space:pre-wrap">'
        f"{html.escape(facts_text)}</pre></div>"
    )
    return text, body
