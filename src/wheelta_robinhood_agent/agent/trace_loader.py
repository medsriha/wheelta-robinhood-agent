"""Load a run's decision-trace inputs from the ledger (read-only; observability/decision_trace.py).

Everything comes from recorded rows of the run: identity, run events (config snapshot, status,
tool-access and prompt metadata, schedule), tool-call projections, the effective RunRecord,
DecisionFacts, and audit findings (corrections applied). Nothing is written and nothing is
recomputed: the trace shows what the run recorded, including when assembly or the audit is
missing.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from pydantic import JsonValue

from wheelta_robinhood_agent.domain.enums import AppEnv, ExecutionMode, OrderVenue, RunStatus
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger.errors import UnknownEntity
from wheelta_robinhood_agent.ledger.runs import run_event_payloads, run_projection
from wheelta_robinhood_agent.ledger.tool_calls import tool_call_records
from wheelta_robinhood_agent.observability.decision_trace import (
    DecisionTrace,
    TraceInput,
    build_decision_trace,
)

Conn = psycopg.Connection[tuple[object, ...]]

__all__ = ["RunListing", "list_runs", "load_decision_trace", "load_trace_input", "run_id_at"]


def _last(payloads: tuple[Any, ...], key: str) -> Any:
    found = [p[key] for p in payloads if isinstance(p, dict) and key in p]
    return found[-1] if found else None


def _effective_findings(findings: tuple[AuditFinding, ...]) -> tuple[AuditFinding, ...]:
    superseded = {f.corrects_finding_id for f in findings if f.corrects_finding_id is not None}
    return tuple(f for f in findings if f.finding_id not in superseded)


def _json(payload: object) -> dict[str, JsonValue]:
    """A recorded JSON object payload (run events are stored as jsonb objects)."""
    return {str(k): v for k, v in payload.items()} if isinstance(payload, dict) else {}


def _enum(kind: Any, value: object) -> Any:
    try:
        return kind(value) if isinstance(value, str) else None
    except ValueError:
        return None


def load_trace_input(conn: Conn, run_id: uuid.UUID) -> TraceInput:
    """The recorded inputs for one run's trace. Raises UnknownEntity for an unknown run."""
    row = conn.execute("SELECT environment, slot FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    if row is None or not isinstance(row[1], datetime):
        raise UnknownEntity(f"runs has no row {run_id}")
    environment, slot = AppEnv(str(row[0])), row[1]
    projection = run_projection(conn, run_id)
    started = run_event_payloads(conn, run_id, RunEventType.STARTED)
    statuses = run_event_payloads(conn, run_id, RunEventType.STATUS)
    metadata = run_event_payloads(conn, run_id, RunEventType.METADATA)
    schedule = run_event_payloads(conn, run_id, RunEventType.SCHEDULE)
    snapshot = _last(started, "config_snapshot")
    mode = _enum(
        ExecutionMode,
        snapshot.get("effective_execution_mode") if isinstance(snapshot, dict) else None,
    )
    access = _last(metadata, "tool_access")
    venue = _enum(OrderVenue, access.get("order_venue") if isinstance(access, dict) else None)
    records = ledger_evidence.effective(ledger_evidence.run_records_for_run(conn, run_id))
    record = records[-1].record if records else None
    if venue is None and record is not None:
        venue = record.order_venue
    next_run: dict[str, JsonValue] | None = None
    if schedule:
        next_run = {"applied": True, **_json(schedule[-1])}
    elif (unapplied := _last(metadata, "next_run_not_applied")) is not None:
        next_run = {"applied": False, **_json(unapplied)}
    rules_version = _last(started, "rules_version")
    prompt_hash = _last(metadata, "rendered_prompt_hash")
    return TraceInput(
        run_id=run_id,
        environment=environment,
        slot=slot,
        status=projection.status,
        reason=_last(statuses, "reason"),
        effective_execution_mode=mode,
        order_venue=venue,
        prompt_execution_mode=_last(metadata, "prompt_execution_mode"),
        model_id=_last(metadata, "model_id"),
        rules_version=str(rules_version) if rules_version is not None else None,
        prompt_hash=prompt_hash if isinstance(prompt_hash, str) else None,
        record=record,
        tool_calls=tool_call_records(conn, run_id),
        facts=tuple(
            f.facts
            for f in ledger_evidence.effective(ledger_evidence.decision_facts_for_run(conn, run_id))
        ),
        findings=_effective_findings(ledger_evidence.audit_findings_for_run(conn, run_id)),
        next_run=next_run,
    )


def load_decision_trace(conn: Conn, run_id: uuid.UUID) -> DecisionTrace:
    return build_decision_trace(load_trace_input(conn, run_id))


def run_id_at(conn: Conn, environment: AppEnv, slot: datetime) -> uuid.UUID:
    """The run recorded for (environment, slot). Raises UnknownEntity if none."""
    row = conn.execute(
        "SELECT run_id FROM runs WHERE environment = %s AND slot = %s", (environment.value, slot)
    ).fetchone()
    if row is None or not isinstance(row[0], uuid.UUID):
        raise UnknownEntity(f"no run at {environment.value} {slot.isoformat()}")
    return row[0]


@dataclass(frozen=True)
class RunListing:
    """One line of `list_runs`: enough to pick dry and live runs to compare."""

    run_id: uuid.UUID
    slot: datetime
    status: RunStatus | None
    effective_execution_mode: ExecutionMode | None
    order_venue: OrderVenue | None
    decisions: int
    attempts: int
    placed: int
    violations: int
    unverifiable: int


def list_runs(conn: Conn, environment: AppEnv, limit: int = 20) -> tuple[RunListing, ...]:
    """The newest `limit` runs of the environment that started a session (have tool calls or
    a run record), newest first."""
    rows = conn.execute(
        "SELECT r.run_id FROM runs r WHERE r.environment = %s AND (EXISTS "
        "(SELECT 1 FROM tool_calls t WHERE t.run_id = r.run_id) OR EXISTS "
        "(SELECT 1 FROM assembled_run_records rr WHERE rr.run_id = r.run_id)) "
        "ORDER BY r.slot DESC LIMIT %s",
        (environment.value, limit),
    ).fetchall()
    out: list[RunListing] = []
    for (run_id,) in rows:
        if not isinstance(run_id, uuid.UUID):
            continue
        trace = load_decision_trace(conn, run_id)
        attempts = [a for d in trace.decisions for leg in d.legs for a in leg.attempts]
        out.append(
            RunListing(
                run_id=run_id,
                slot=trace.slot,
                status=trace.status,
                effective_execution_mode=trace.effective_execution_mode,
                order_venue=trace.order_venue,
                decisions=len(trace.decisions),
                attempts=len(attempts),
                placed=sum(1 for a in attempts if a.place is not None)
                + sum(1 for u in trace.unassociated if u.kind == "place"),
                violations=trace.audit_counts.get("violation", 0),
                unverifiable=trace.audit_counts.get("unverifiable", 0),
            )
        )
    return tuple(out)
