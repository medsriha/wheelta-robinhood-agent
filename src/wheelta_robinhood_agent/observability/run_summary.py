"""Run-summary email content (ADR-0029, ADR-0064). Pure: no HTTP, no clock, no LLM.

ADR-0057: one summary is sent per tick in which an agent session started. A tick runs the
Buy-to-Close agent, then the Sell Options agent, each its own run; the email
(`SlotSummaryInput`) has one section per agent run, in that order, a skipped agent included with
its reason, then the tick's next run. Each section comes from one `RunSummaryInput`, built from
what the run recorded (the assembled RunRecord, the audit status, alerts, the next run), never
from the agent's free text alone.

ADR-0064: the email says what each agent did and why. Each decision is written in plain words
with its order outcomes and the agent's rationale; anything that needs the owner follows under
"Needs attention" only when present. Identifiers, refs, reason codes, metrics, fact gaps and
research stay in the ledger (`scripts/trace_run.py`). `slot_summary_facts` is the redacted JSON
handed to the prose writer and `render_slot_facts_text` the deterministic block every email
carries; both come from the same view, so a figure in the prose can be checked against it.
Delivery is in ``integrations/notifications``.
"""

import html
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from wheelta_robinhood_agent.domain.enums import (
    AgentRole,
    AppEnv,
    AttemptStatus,
    DecisionAction,
    ExecutionMode,
    OrderSide,
    OrderVenue,
    RunStatus,
)
from wheelta_robinhood_agent.domain.gating import executes_orders, with_default_venue
from wheelta_robinhood_agent.domain.mignon_report import MignonReport
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import Attempt, ReasonCode
from wheelta_robinhood_agent.domain.run_record import (
    DecisionOutputStatus,
    DecisionRecord,
    LegRecord,
    RunRecord,
    UnassociatedAction,
)
from wheelta_robinhood_agent.observability.alerts import AlertKind
from wheelta_robinhood_agent.observability.redaction import Redactor

SUBJECT_PREFIX = "[Wheelta agent]"
FACTS_HEADING = "What the agents did"
PROSE_UNAVAILABLE = "The written overview is unavailable for this tick."


class ConsideredOption(BaseModel):
    """A candidate delivered during research. Meeting one is not proof it was evaluated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_ref: str
    underlying: str
    occ_symbol: str


class RunSummaryInput(BaseModel):
    """Everything one summary email may say, as recorded by the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    environment: AppEnv
    slot: AwareDatetime
    status: RunStatus
    reason: str | None
    # ADR-0057: which agent this run is, and whether its session started (a skipped or
    # unstarted agent is reported by status and reason only).
    agent: AgentRole = AgentRole.WHEEL
    session_started: bool = True
    requested_execution_mode: ExecutionMode
    effective_execution_mode: ExecutionMode
    # ADR-0038; defaults from the mode (live: broker, off: none) when not given.
    order_venue: OrderVenue
    # None when assembly failed; the email then says so instead of listing decisions.
    record: RunRecord | None
    # None when the audit did not run in this invocation.
    audit_status: str | None = None
    audit_violations: int = Field(default=0, ge=0)
    alerts: tuple[str, ...] = ()
    next_run_at: AwareDatetime | None = None
    next_run_source: str | None = None
    next_run_rationale: str | None = None
    diagnostic_details: tuple[str, ...] = ()
    # Audit violations and check errors only; passing and unverifiable checks are not news.
    audit_details: tuple[str, ...] = ()
    candidates: tuple[ConsideredOption, ...] = ()
    research_reports: tuple[MignonReport, ...] = ()
    # The loader's error type when research could not be read for the email.
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


# The plain-words view (ADR-0064) ----------------------------------------------------------

_ACTION_LABELS = {
    DecisionAction.OPEN_CSP: "Sell a cash-secured put",
    DecisionAction.OPEN_CC: "Sell a covered call",
    DecisionAction.CLOSE: "Close",
    DecisionAction.ROLL: "Roll",
    DecisionAction.HOLD: "Hold",
}
_SIDE_LABELS = {OrderSide.SELL_TO_OPEN: "Sell to open", OrderSide.BUY_TO_CLOSE: "Buy to close"}
_STOP_EXPLANATIONS = {
    "invalid_agent_output": "The agent's final decision output was missing or invalid.",
    "audit_failed": "The post-run audit could not complete.",
    "deadline": "The run exhausted its time budget and the session was stopped.",
    "sigterm": "The session was stopped by SIGTERM.",
    "sigint": "The session was stopped by SIGINT.",
    "infrastructure_failure": "An infrastructure failure stopped the session.",
}
_STOPPED = (RunStatus.FAILED, RunStatus.TIMED_OUT, RunStatus.STOPPED)
# Every order already appears with its decision; the order-activity alert repeats it.
_ROUTINE_ALERTS = frozenset({AlertKind.ORDER_ACTIVITY.value})


def _when(value: datetime) -> str:
    return f"{value.astimezone(UTC):%Y-%m-%d %H:%M} UTC"


def _contract_text(occ: OccSymbol, underlying: str | None) -> str:
    """`AAPL 2026-10-16 190 put`."""
    return f"{underlying or occ.root} {occ.expiration.isoformat()} {occ.strike} {occ.right.value}"


def _leg_text(leg: LegRecord, underlying: str | None) -> str:
    """`Sell to open 2 × AAPL 2026-10-16 190 put`, or the OCC symbol when a term is unknown."""
    if leg.strike is None or leg.expiration is None or leg.right is None:
        contract = str(leg.occ_symbol) if leg.occ_symbol is not None else "contract unknown"
    else:
        terms = (underlying, leg.expiration.isoformat(), str(leg.strike), leg.right.value)
        contract = " ".join(t for t in terms if t)
    quantity = f"{leg.target_quantity} × " if leg.target_quantity is not None else ""
    return f"{_SIDE_LABELS[leg.side]} {quantity}{contract}"


def _attempt_text(attempt: Attempt, venue: OrderVenue) -> str:
    """`Order filled: 2 at limit 1.30`; a proposal is never called an order."""
    if _is_proposal(attempt):
        head = "Proposed, not sent"
    else:
        kind = "Simulated order" if venue is OrderVenue.SIMULATED else "Order"
        head = f"{kind} {attempt.status.value.replace('_', ' ')}"
    details: list[str] = []
    if attempt.requested_quantity is not None:
        filled = attempt.filled_quantity
        partial = filled is not None and filled != attempt.requested_quantity
        details.append(
            f"{filled} of {attempt.requested_quantity}"
            if partial
            else str(attempt.requested_quantity)
        )
    if attempt.limit_price is not None:
        details.append(f"at limit {attempt.limit_price}")
    return f"{head}: {' '.join(details)}" if details else head


@dataclass(frozen=True, slots=True)
class _LegView:
    contract: str
    orders: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ActionView:
    action: str
    legs: tuple[_LegView, ...]
    why: str | None


def _decision_view(decision: DecisionRecord, venue: OrderVenue) -> _ActionView:
    target = decision.underlying
    if not decision.legs and decision.target_occ_symbol is not None:
        # A HOLD has no legs: name the held contract, not only the stock.
        target = _contract_text(decision.target_occ_symbol, decision.underlying)
    return _ActionView(
        action=_ACTION_LABELS[decision.action] + (f" {target}" if target else ""),
        legs=tuple(
            _LegView(
                contract=_leg_text(leg, decision.underlying),
                orders=tuple(_attempt_text(a, venue) for a in leg.attempts),
            )
            for leg in decision.legs
        ),
        why=decision.rationale,
    )


def _cancel_views(record: RunRecord) -> list[_ActionView]:
    """Each cancel the agent made, with its rationale when it cited the cancel call."""
    rationales = {
        r.cancel_tool_call_id: r.rationale
        for r in record.cancellation_rationales
        if r.cancel_tool_call_id is not None
    }
    views = [
        _ActionView(
            action=f"Cancel a working order ({c.status.value})",
            legs=(),
            why=rationales.get(c.cancel_tool_call_id),
        )
        for c in record.cancellations
    ]
    cited = {c.cancel_tool_call_id for c in record.cancellations}
    views.extend(
        _ActionView(action="Cancel a working order", legs=(), why=r.rationale)
        for r in record.cancellation_rationales
        if r.cancel_tool_call_id not in cited
    )
    return views


def _selected(candidate: ConsideredOption, decisions: tuple[DecisionRecord, ...]) -> bool:
    return any(
        candidate.candidate_ref in (d.target_ref, d.replacement_ref)
        or str(d.target_occ_symbol) == candidate.occ_symbol
        or any(str(leg.occ_symbol) == candidate.occ_symbol for leg in d.legs)
        for d in decisions
    )


def _passed_over(summary: RunSummaryInput) -> list[tuple[str, str]]:
    """(contract, research notes) for each candidate the agent did not select that research
    commented on. Only with parsed decisions: otherwise selection is unknown. The notes are
    the research's claims, not the agent's reason."""
    record = summary.record
    if record is None or record.decision_output_status is not DecisionOutputStatus.PARSED:
        return []
    notes: dict[str, list[str]] = {}
    for report in summary.research_reports:
        for finding in report.findings:
            # ADR-0056: a number resting only on fetched pages is labelled as such.
            claim = f"{finding.claim} (web-sourced)" if finding.web_sourced else finding.claim
            for ref in finding.refs:
                claims = notes.setdefault(ref, [])
                if claim not in claims:
                    claims.append(claim)
    rows: dict[str, list[str]] = {}
    for candidate in summary.candidates:
        if _selected(candidate, record.decisions) or candidate.candidate_ref not in notes:
            continue
        try:
            contract = _contract_text(OccSymbol.parse(candidate.occ_symbol), candidate.underlying)
        except ValueError:
            contract = f"{candidate.underlying} {candidate.occ_symbol}"
        claims = rows.setdefault(contract, [])
        claims.extend(c for c in notes[candidate.candidate_ref] if c not in claims)
    return [(contract, " ".join(claims)) for contract, claims in rows.items()]


def _unlinked_text(action: UnassociatedAction, venue: OrderVenue) -> str:
    if action.attempt is not None:
        what = " ".join(
            p for p in (action.side_raw, str(action.occ_symbol) if action.occ_symbol else None) if p
        )
        return (
            "An order no decision accounts for"
            + (f" ({what})" if what else "")
            + f": {_attempt_text(action.attempt, venue)}"
        )
    status = action.cancellation.status.value if action.cancellation is not None else "unknown"
    return f"A cancel no decision accounts for: {status}"


def _attention(summary: RunSummaryInput) -> list[str]:
    """What the owner should look at; empty for a clean run."""
    items: list[str] = []
    if summary.status in _STOPPED:
        reason = summary.reason or ""
        items.append(_STOP_EXPLANATIONS.get(reason, reason or "No failure reason was recorded."))
    items.extend(summary.diagnostic_details)
    record = summary.record
    if summary.session_started and record is None:
        items.append("The run record could not be assembled; the ledger has the recorded events.")
    if record is not None:
        items.extend(_unlinked_text(u, summary.order_venue) for u in record.unassociated_actions)
    if summary.audit_status == "failed" and summary.reason != "audit_failed":
        items.append("The post-run audit could not complete.")
    if summary.audit_violations:
        plural = "s" if summary.audit_violations != 1 else ""
        items.append(f"The post-run audit found {summary.audit_violations} violation{plural}.")
    items.extend(summary.audit_details)
    if summary.research_unavailable:
        items.append(f"Research notes could not be loaded ({summary.research_unavailable}).")
    alerts = [a.replace("_", " ") for a in summary.alerts if a not in _ROUTINE_ALERTS]
    if alerts:
        items.append("Alerts sent: " + ", ".join(dict.fromkeys(alerts)))
    return list(dict.fromkeys(items))


AGENT_TITLES = {
    AgentRole.CLOSE: "Buy-to-Close agent",
    AgentRole.SELL: "Sell Options agent",
    AgentRole.WHEEL: "Wheel agent",
}


@dataclass(frozen=True, slots=True)
class AgentView:
    """One agent's section: its outcome, its actions with their reasons, what needs attention.

    `decisions_known` is false when the agent returned no valid final output, so an empty
    action list must not be read as "did nothing"."""

    agent: str
    status: str
    reason: str | None
    session_started: bool
    mode: str
    orders_sent_to_broker: bool
    decisions_known: bool
    actions: tuple[_ActionView, ...]
    # (contract, research notes) for candidates the agent did not select.
    passed_over: tuple[tuple[str, str], ...]
    open_questions: tuple[str, ...]
    needs_attention: tuple[str, ...]
    # (when, why) of the agent's own next-run request; None when it made none.
    requested_next_run: tuple[str, str | None] | None

    def facts(self) -> dict[str, JsonValue]:
        """The JSON form the prose writer reads."""
        return {
            "agent": self.agent,
            "status": self.status,
            "reason": self.reason,
            "session_started": self.session_started,
            "mode": self.mode,
            "orders_sent_to_broker": self.orders_sent_to_broker,
            "decisions_known": self.decisions_known,
            "actions": [
                {
                    "action": a.action,
                    "legs": [
                        {"contract": leg.contract, "orders": list(leg.orders)} for leg in a.legs
                    ],
                    "why": a.why,
                }
                for a in self.actions
            ],
            "passed_over_research_notes": [
                {"contract": contract, "notes": notes} for contract, notes in self.passed_over
            ],
            "open_questions": list(self.open_questions),
            "needs_attention": list(self.needs_attention),
            "requested_next_run": (
                None
                if self.requested_next_run is None
                else {"at": self.requested_next_run[0], "why": self.requested_next_run[1]}
            ),
        }

    def lines(self) -> list[str]:
        """The plain-text form printed in every email."""
        if not self.session_started:
            return [f"{self.agent}: did not run ({self.reason or self.status})"]
        lines = [f"{self.agent}: {self.status}"]
        for action in self.actions:
            lines.append(f"- {action.action}")
            for leg in action.legs:
                lines.append(f"  {leg.contract}")
                lines.extend(f"  {order}" for order in leg.orders)
            if action.why:
                lines.append(f"  Why: {action.why}")
        if not self.actions:
            lines.append(
                "- No decisions."
                if self.decisions_known
                else "- Decisions unknown: no valid final output."
            )
        if self.passed_over:
            lines.append("Not selected (research notes, not the agent's stated reasons):")
            lines.extend(f"- {contract}: {notes}" for contract, notes in self.passed_over)
        if self.open_questions:
            lines.append("Open questions:")
            lines.extend(f"- {question}" for question in self.open_questions)
        if self.needs_attention:
            lines.append("Needs attention:")
            lines.extend(f"- {item}" for item in self.needs_attention)
        if self.requested_next_run is not None:
            at, why = self.requested_next_run
            lines.append(f"Asked to run next at {at}" + (f": {why}" if why else ""))
        return lines


def agent_view(summary: RunSummaryInput) -> AgentView:
    """The plain-words view of one agent run (ADR-0064)."""
    record = summary.record
    actions: list[_ActionView] = []
    if record is not None:
        actions.extend(_decision_view(d, summary.order_venue) for d in record.decisions)
        actions.extend(_cancel_views(record))
    requested = None
    if summary.next_run_source == "agent" and summary.next_run_at is not None:
        requested = (_when(summary.next_run_at), summary.next_run_rationale)
    return AgentView(
        agent=AGENT_TITLES[summary.agent],
        status=summary.status.value,
        reason=summary.reason,
        session_started=summary.session_started,
        mode=_mode_label(summary.effective_execution_mode, summary.order_venue),
        orders_sent_to_broker=summary.order_venue is OrderVenue.BROKER,
        decisions_known=record is not None
        and record.decision_output_status is DecisionOutputStatus.PARSED,
        actions=tuple(actions),
        passed_over=tuple(_passed_over(summary)),
        open_questions=tuple(
            dict.fromkeys(q.question for q in record.unresolved_questions) if record else ()
        ),
        needs_attention=tuple(_attention(summary)),
        requested_next_run=requested,
    )


def render_bodies(prose: str | None, facts_text: str, redactor: Redactor) -> tuple[str, str]:
    """(text, html) bodies: the prose (or a fallback line), then the actions block."""
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
        f'<div style="white-space:pre-wrap">{html.escape(facts_text)}</div></div>'
    )
    return text, body


# ADR-0057: one email per tick ------------------------------------------------------------


class SlotSummaryInput(BaseModel):
    """One tick's email: each agent run in order, then the tick's effective next run (the
    earliest agent request, else the fallback)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: AppEnv
    slot: AwareDatetime
    agents: tuple[RunSummaryInput, ...] = Field(min_length=1)
    next_run_at: AwareDatetime | None = None
    next_run_source: str | None = None

    @model_validator(mode="after")
    def _same_slot(self) -> Self:
        if any(a.slot != self.slot or a.environment != self.environment for a in self.agents):
            raise ValueError("every agent run must belong to the summary's slot")
        return self


def _agent_activity(summary: RunSummaryInput) -> str:
    if not summary.session_started:
        return summary.status.value
    activity = build_subject(summary).split(" · ")[2]
    return f"{summary.status.value}, {activity}"


def build_slot_subject(summary: SlotSummaryInput) -> str:
    """`[Wheelta agent] <mode> · close: <outcome> · sell: <outcome> · <slot UTC>`."""
    first = summary.agents[0]
    parts = [
        f"{SUBJECT_PREFIX} {_mode_label(first.effective_execution_mode, first.order_venue)}",
        *(f"{a.agent.value}: {_agent_activity(a)}" for a in summary.agents),
        f"{summary.slot:%Y-%m-%d %H:%M} UTC",
    ]
    return " · ".join(parts)


def _tick_next_run(summary: SlotSummaryInput) -> str | None:
    if summary.next_run_at is None:
        return None
    source = " (hourly fallback)" if summary.next_run_source == "fallback" else ""
    return f"Next run: {_when(summary.next_run_at)}{source}"


def slot_summary_facts(summary: SlotSummaryInput, redactor: Redactor) -> dict[str, JsonValue]:
    """The redacted JSON the prose writer summarizes: one entry per agent run, in order."""
    redacted = redactor.redact(
        {
            "agents": [agent_view(agent).facts() for agent in summary.agents],
            "next_run": _tick_next_run(summary),
        }
    )
    if not isinstance(redacted, dict):  # redact() preserves mappings; narrowed for typing
        raise TypeError("redacted facts must stay a mapping")
    return redacted


def render_slot_facts_text(summary: SlotSummaryInput, redactor: Redactor) -> str:
    """The deterministic block: one section per agent run, then the tick's next run."""
    sections = ["\n".join(agent_view(agent).lines()) for agent in summary.agents]
    next_run = _tick_next_run(summary)
    if next_run is not None:
        sections.append(next_run)
    return redactor.redact_text("\n\n".join(sections))
