"""Append-only evidence and output repositories (INTERFACES.md "Ledger tables and recovery").

Covers `results`, `citations`, `account_snapshots`, `agent_outputs`, `agent_decisions`,
`decision_facts`, `assembled_run_records`, `audit_findings`, and `alerts_sent`. Every
function inserts a new row or reads rows for a run; nothing updates evidence. A correction is
a new row whose `corrects_*_id` references the original in the same run; readers return every
row (originals and corrections) in insertion order, and `effective()` drops superseded ones.

Raw/parsed agent choices (`agent_outputs`, `agent_decisions`) stay separate from code-issued
facts and assembled records, which carry their input hash and assembler/formula versions
(ADR-0011).
"""

import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Generic, Protocol, TypeVar

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    DecisionOutputParsed,
    DecisionOutputParseResult,
    ParseIssue,
)
from wheelta_robinhood_agent.domain.enums import AuditCheck, SourceTier
from wheelta_robinhood_agent.domain.evidence import Citation
from wheelta_robinhood_agent.domain.facts import DecisionFacts
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import RunRecord
from wheelta_robinhood_agent.ledger.errors import IdentityConflict, LedgerError, UnknownEntity
from wheelta_robinhood_agent.ledger.ids import new_id

Conn = psycopg.Connection[tuple[object, ...]]
Row = dict[str, Any]
T = TypeVar("T")


class SchemaGap(LedgerError):
    """The domain model carries data the ledger schema has no column for (would be lost)."""


class ResultKind(StrEnum):
    """results.kind (migrations/0001_initial.sql)."""

    VALIDATED = "validated"
    ERROR = "error"
    RAW_INVALID = "raw_invalid"
    DELIVERED = "delivered"


class ParseStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class DeliveryStatus(StrEnum):
    SENT = "sent"
    FAILED = "failed"


# Closed mapping of correctable evidence tables to (id column, corrects column).
_CORRECTABLE: Final[dict[str, tuple[str, str]]] = {
    "results": ("result_id", "corrects_result_id"),
    "citations": ("citation_id", "corrects_citation_id"),
    "account_snapshots": ("snapshot_id", "corrects_snapshot_id"),
    "agent_outputs": ("output_id", "corrects_output_id"),
    "agent_decisions": ("agent_decision_id", "corrects_decision_id"),
    "decision_facts": ("decision_facts_id", "corrects_facts_id"),
    "assembled_run_records": ("record_id", "corrects_record_id"),
    "audit_findings": ("finding_id", "corrects_finding_id"),
}


@dataclass(frozen=True)
class Recorded(Generic[T]):
    """A stored domain value with its row identity and correction reference."""

    record_id: uuid.UUID
    run_id: uuid.UUID
    value: T
    corrects_id: uuid.UUID | None
    recorded_at: datetime


class _Correctable(Protocol):
    @property
    def record_id(self) -> uuid.UUID: ...
    @property
    def corrects_id(self) -> uuid.UUID | None: ...


R = TypeVar("R", bound=_Correctable)


def effective(records: Iterable[R]) -> tuple[R, ...]:
    """Records not superseded by a later correction (corrections chain transitively)."""
    items = tuple(records)
    superseded = {r.corrects_id for r in items if r.corrects_id is not None}
    return tuple(r for r in items if r.record_id not in superseded)


def _rows(conn: Conn, query: str, params: Sequence[object]) -> list[Row]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(query.encode(), params).fetchall()


def _check_aware(name: str, value: datetime | None) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


def _check_correction(
    conn: Conn, table: str, corrects_id: uuid.UUID | None, run_id: uuid.UUID
) -> None:
    """A correction must reference an existing row of the same table and run."""
    if corrects_id is None:
        return
    id_col, _ = _CORRECTABLE[table]
    row = conn.execute(
        sql.SQL("SELECT run_id FROM {} WHERE {} = %s").format(
            sql.Identifier(table), sql.Identifier(id_col)
        ),
        (corrects_id,),
    ).fetchone()
    if row is None:
        raise UnknownEntity(f"{table} has no row {corrects_id} to correct")
    if row[0] != run_id:
        raise IdentityConflict(f"{table} correction must reference a row of the same run")


def _check_tool_call_run(conn: Conn, tool_call_id: uuid.UUID, run_id: uuid.UUID) -> None:
    row = conn.execute(
        "SELECT run_id FROM tool_calls WHERE tool_call_id = %s", (tool_call_id,)
    ).fetchone()
    if row is None:
        raise UnknownEntity(f"tool_calls has no row {tool_call_id}")
    if row[0] != run_id:
        raise IdentityConflict(f"tool call {tool_call_id} belongs to another run")


# ---------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StoredResult:
    record_id: uuid.UUID
    run_id: uuid.UUID
    tool_call_id: uuid.UUID | None
    kind: ResultKind
    payload: object
    corrects_id: uuid.UUID | None
    recorded_at: datetime


def insert_result(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    kind: ResultKind,
    payload: object,
    tool_call_id: uuid.UUID | None = None,
    corrects_result_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Store a validated/error/raw-invalid/delivered envelope. Payloads are already redacted."""
    result_id = new_id()
    with conn.transaction():
        if tool_call_id is not None:
            _check_tool_call_run(conn, tool_call_id, run_id)
        _check_correction(conn, "results", corrects_result_id, run_id)
        conn.execute(
            "INSERT INTO results (result_id, run_id, tool_call_id, kind, payload, "
            "corrects_result_id) VALUES (%s, %s, %s, %s, %s, %s)",
            (result_id, run_id, tool_call_id, kind.value, Jsonb(payload), corrects_result_id),
        )
    return result_id


def _result(row: Row) -> StoredResult:
    return StoredResult(
        record_id=row["result_id"],
        run_id=row["run_id"],
        tool_call_id=row["tool_call_id"],
        kind=ResultKind(row["kind"]),
        payload=row["payload"],
        corrects_id=row["corrects_result_id"],
        recorded_at=row["recorded_at"],
    )


def get_result(conn: Conn, result_id: uuid.UUID) -> StoredResult:
    rows = _rows(conn, "SELECT * FROM results WHERE result_id = %s", (result_id,))
    if not rows:
        raise UnknownEntity(f"results has no row {result_id}")
    return _result(rows[0])


def results_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[StoredResult, ...]:
    rows = _rows(
        conn, "SELECT * FROM results WHERE run_id = %s ORDER BY recorded_at, result_id", (run_id,)
    )
    return tuple(_result(r) for r in rows)


# ---------------------------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------------------------
def insert_citation(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    citation: Citation,
    corrects_citation_id: uuid.UUID | None = None,
) -> uuid.UUID:
    with conn.transaction():
        _check_tool_call_run(conn, citation.tool_call_id, run_id)
        _check_correction(conn, "citations", corrects_citation_id, run_id)
        conn.execute(
            "INSERT INTO citations (citation_id, run_id, tool_call_id, url, title, publisher, "
            "published_at, retrieved_at, tier, excerpt, corrects_citation_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                citation.citation_id,
                run_id,
                citation.tool_call_id,
                citation.url,
                citation.title,
                citation.publisher,
                citation.published_at,
                citation.retrieved_at,
                citation.tier.value,
                citation.excerpt,
                corrects_citation_id,
            ),
        )
    return citation.citation_id


def citations_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[Recorded[Citation], ...]:
    rows = _rows(
        conn,
        "SELECT * FROM citations WHERE run_id = %s ORDER BY recorded_at, citation_id",
        (run_id,),
    )
    return tuple(
        Recorded(
            record_id=r["citation_id"],
            run_id=r["run_id"],
            value=Citation(
                citation_id=r["citation_id"],
                url=r["url"],
                title=r["title"],
                publisher=r["publisher"],
                published_at=r["published_at"],
                retrieved_at=r["retrieved_at"],
                tier=SourceTier(r["tier"]),
                excerpt=r["excerpt"],
                tool_call_id=r["tool_call_id"],
            ),
            corrects_id=r["corrects_citation_id"],
            recorded_at=r["recorded_at"],
        )
        for r in rows
    )


# ---------------------------------------------------------------------------------------------
# account_snapshots
# ---------------------------------------------------------------------------------------------
def insert_account_snapshot(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    snapshot: AccountSnapshot,
    corrects_snapshot_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Store a snapshot, including the evidence behind `csp_cash_base_usd`."""
    dumped = snapshot.model_dump(mode="json")
    with conn.transaction():
        for tool_call_id in snapshot.tool_call_ids:
            _check_tool_call_run(conn, tool_call_id, run_id)
        _check_correction(conn, "account_snapshots", corrects_snapshot_id, run_id)
        conn.execute(
            "INSERT INTO account_snapshots (snapshot_id, run_id, as_of, retrieved_at, "
            "tool_call_ids, account_ref, agentic_verified, account_value_usd, "
            "available_settled_cash_usd, csp_reserved_cash_usd, csp_cash_base_usd, "
            "positions_ref, open_orders_ref, tax_lots_ref, reservation_evidence, quality, gaps, "
            "corrects_snapshot_id, csp_cash_base_evidence_ids) VALUES "
            "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                snapshot.snapshot_id,
                run_id,
                snapshot.as_of,
                snapshot.retrieved_at,
                list(snapshot.tool_call_ids),
                snapshot.account_ref,
                snapshot.agentic_verified,
                snapshot.account_value_usd,
                snapshot.available_settled_cash_usd,
                snapshot.csp_reserved_cash_usd,
                snapshot.csp_cash_base_usd,
                snapshot.positions_ref,
                snapshot.open_orders_ref,
                snapshot.tax_lots_ref,
                Jsonb(dumped["reservation_evidence"]),
                snapshot.quality.value,
                Jsonb(dumped["gaps"]),
                corrects_snapshot_id,
                list(snapshot.csp_cash_base_evidence_ids),
            ),
        )
    return snapshot.snapshot_id


def account_snapshots_for_run(
    conn: Conn, run_id: uuid.UUID
) -> tuple[Recorded[AccountSnapshot], ...]:
    rows = _rows(
        conn,
        "SELECT * FROM account_snapshots WHERE run_id = %s ORDER BY recorded_at, snapshot_id",
        (run_id,),
    )
    out: list[Recorded[AccountSnapshot]] = []
    for r in rows:
        snapshot = AccountSnapshot.model_validate(
            {
                "snapshot_id": r["snapshot_id"],
                "as_of": r["as_of"],
                "retrieved_at": r["retrieved_at"],
                "tool_call_ids": tuple(r["tool_call_ids"]),
                "account_ref": r["account_ref"],
                "agentic_verified": r["agentic_verified"],
                "account_value_usd": r["account_value_usd"],
                "available_settled_cash_usd": r["available_settled_cash_usd"],
                "csp_reserved_cash_usd": r["csp_reserved_cash_usd"],
                "csp_cash_base_usd": r["csp_cash_base_usd"],
                "csp_cash_base_evidence_ids": tuple(r["csp_cash_base_evidence_ids"]),
                "positions_ref": r["positions_ref"],
                "open_orders_ref": r["open_orders_ref"],
                "tax_lots_ref": r["tax_lots_ref"],
                "reservation_evidence": r["reservation_evidence"],
                "quality": r["quality"],
                "gaps": r["gaps"],
            }
        )
        out.append(
            Recorded(
                r["snapshot_id"], r["run_id"], snapshot, r["corrects_snapshot_id"], r["recorded_at"]
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# agent_outputs and agent_decisions
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StoredAgentOutput:
    record_id: uuid.UUID
    run_id: uuid.UUID
    raw_redacted: str | None
    observed_at: datetime
    corrects_id: uuid.UUID | None
    recorded_at: datetime


def insert_agent_output(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    raw_redacted: str | None,
    observed_at: datetime,
    corrects_output_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Store the raw, already-redacted final model response (None when there was none)."""
    _check_aware("observed_at", observed_at)
    output_id = new_id()
    with conn.transaction():
        _check_correction(conn, "agent_outputs", corrects_output_id, run_id)
        conn.execute(
            "INSERT INTO agent_outputs (output_id, run_id, raw_redacted, observed_at, "
            "corrects_output_id) VALUES (%s, %s, %s, %s, %s)",
            (output_id, run_id, raw_redacted, observed_at, corrects_output_id),
        )
    return output_id


def agent_outputs_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[StoredAgentOutput, ...]:
    rows = _rows(
        conn,
        "SELECT * FROM agent_outputs WHERE run_id = %s ORDER BY recorded_at, output_id",
        (run_id,),
    )
    return tuple(
        StoredAgentOutput(
            r["output_id"],
            r["run_id"],
            r["raw_redacted"],
            r["observed_at"],
            r["corrects_output_id"],
            r["recorded_at"],
        )
        for r in rows
    )


@dataclass(frozen=True)
class StoredAgentDecision:
    record_id: uuid.UUID
    run_id: uuid.UUID
    output_id: uuid.UUID
    schema_version: str
    parse_status: ParseStatus
    output: AgentDecisionOutput | None
    issues: tuple[ParseIssue, ...]
    corrects_id: uuid.UUID | None
    recorded_at: datetime


def insert_agent_decision(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    output_id: uuid.UUID,
    result: DecisionOutputParseResult,
    corrects_decision_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Store the parse result for an agent output. The raw text lives in agent_outputs."""
    decision_id = new_id()
    if isinstance(result, DecisionOutputParsed):
        status, parsed, errors = ParseStatus.VALID, result.output.model_dump(mode="json"), []
    else:
        status, parsed = ParseStatus.INVALID, None
        errors = [{"loc": i.loc, "message": i.message, "kind": i.kind} for i in result.issues]
    with conn.transaction():
        row = conn.execute(
            "SELECT run_id FROM agent_outputs WHERE output_id = %s", (output_id,)
        ).fetchone()
        if row is None:
            raise UnknownEntity(f"agent_outputs has no row {output_id}")
        if row[0] != run_id:
            raise IdentityConflict("agent decision must reference an output of the same run")
        _check_correction(conn, "agent_decisions", corrects_decision_id, run_id)
        conn.execute(
            "INSERT INTO agent_decisions (agent_decision_id, run_id, output_id, schema_version, "
            "parse_status, parsed, errors, corrects_decision_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                decision_id,
                run_id,
                output_id,
                str(result.schema_version),
                status.value,
                Jsonb(parsed) if parsed is not None else None,
                Jsonb(errors),
                corrects_decision_id,
            ),
        )
    return decision_id


def agent_decisions_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[StoredAgentDecision, ...]:
    rows = _rows(
        conn,
        "SELECT * FROM agent_decisions WHERE run_id = %s ORDER BY recorded_at, agent_decision_id",
        (run_id,),
    )
    return tuple(
        StoredAgentDecision(
            record_id=r["agent_decision_id"],
            run_id=r["run_id"],
            output_id=r["output_id"],
            schema_version=r["schema_version"],
            parse_status=ParseStatus(r["parse_status"]),
            output=(
                AgentDecisionOutput.model_validate(r["parsed"]) if r["parsed"] is not None else None
            ),
            issues=tuple(
                ParseIssue(loc=str(e["loc"]), message=str(e["message"]), kind=str(e["kind"]))
                for e in r["errors"]
            ),
            corrects_id=r["corrects_decision_id"],
            recorded_at=r["recorded_at"],
        )
        for r in rows
    )


# ---------------------------------------------------------------------------------------------
# decision_facts
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StoredDecisionFacts:
    record_id: uuid.UUID
    run_id: uuid.UUID
    facts: DecisionFacts
    input_hash: str
    corrects_id: uuid.UUID | None
    recorded_at: datetime


def insert_decision_facts(
    conn: Conn,
    *,
    facts: DecisionFacts,
    input_hash: str,
    corrects_facts_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Store code-issued facts. A correction keeps the facts_ref and gets a new facts_id."""
    if not input_hash:
        raise ValueError("input_hash must be non-empty")
    with conn.transaction():
        _check_correction(conn, "decision_facts", corrects_facts_id, facts.run_id)
        if corrects_facts_id is not None:
            ref = conn.execute(
                "SELECT facts_ref FROM decision_facts WHERE decision_facts_id = %s",
                (corrects_facts_id,),
            ).fetchone()
            if ref is None or ref[0] != facts.facts_ref:
                raise IdentityConflict("a facts correction must keep the same facts_ref")
        conn.execute(
            "INSERT INTO decision_facts (decision_facts_id, run_id, facts_ref, observed_at, "
            "formula_versions, input_hash, facts, corrects_facts_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                facts.facts_id,
                facts.run_id,
                facts.facts_ref,
                facts.observed_at,
                Jsonb({f.formula: f.version for f in facts.formula_versions}),
                input_hash,
                Jsonb(facts.model_dump(mode="json")),
                corrects_facts_id,
            ),
        )
    return facts.facts_id


def decision_facts_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[StoredDecisionFacts, ...]:
    rows = _rows(
        conn,
        "SELECT * FROM decision_facts WHERE run_id = %s ORDER BY recorded_at, decision_facts_id",
        (run_id,),
    )
    return tuple(
        StoredDecisionFacts(
            r["decision_facts_id"],
            r["run_id"],
            DecisionFacts.model_validate(r["facts"]),
            r["input_hash"],
            r["corrects_facts_id"],
            r["recorded_at"],
        )
        for r in rows
    )


# ---------------------------------------------------------------------------------------------
# assembled_run_records
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StoredRunRecord:
    record_id: uuid.UUID
    run_id: uuid.UUID
    record: RunRecord
    assembled_at: datetime
    corrects_id: uuid.UUID | None
    recorded_at: datetime


def insert_run_record(
    conn: Conn,
    *,
    record: RunRecord,
    assembled_at: datetime,
    corrects_record_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Store an assembled RunRecord with its schema/assembler versions and input hash."""
    _check_aware("assembled_at", assembled_at)
    record_id = new_id()
    with conn.transaction():
        _check_correction(conn, "assembled_run_records", corrects_record_id, record.run_id)
        conn.execute(
            "INSERT INTO assembled_run_records (record_id, run_id, schema_version, "
            "assembler_version, input_hash, record, assembled_at, corrects_record_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                record_id,
                record.run_id,
                str(record.schema_version),
                record.assembler_version,
                record.input_hash,
                Jsonb(record.model_dump(mode="json")),
                assembled_at,
                corrects_record_id,
            ),
        )
    return record_id


def run_records_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[StoredRunRecord, ...]:
    rows = _rows(
        conn,
        "SELECT * FROM assembled_run_records WHERE run_id = %s ORDER BY recorded_at, record_id",
        (run_id,),
    )
    return tuple(
        StoredRunRecord(
            r["record_id"],
            r["run_id"],
            RunRecord.model_validate(r["record"]),
            r["assembled_at"],
            r["corrects_record_id"],
            r["recorded_at"],
        )
        for r in rows
    )


# ---------------------------------------------------------------------------------------------
# audit_findings
# ---------------------------------------------------------------------------------------------
def _check_id_column(finding: AuditFinding) -> str:
    """`V<n>` or `V<n>.<sub_item>` (the column allows `[.:_-]` separators)."""
    if finding.sub_item is None:
        return finding.check_id.value
    return f"{finding.check_id.value}.{finding.sub_item}"


def _canonical(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def insert_audit_finding(conn: Conn, finding: AuditFinding) -> uuid.UUID:
    """Store a finding; a correction sets `finding.corrects_finding_id` (same run)."""
    with conn.transaction():
        for tool_call_id in finding.tool_call_ids:
            _check_tool_call_run(conn, tool_call_id, finding.run_id)
        _check_correction(conn, "audit_findings", finding.corrects_finding_id, finding.run_id)
        conn.execute(
            "INSERT INTO audit_findings (finding_id, run_id, check_id, outcome, "
            "effective_execution_mode, decision_ref, leg_ref, attempt_index, rule_key, "
            "rule_value, observed_value, tool_call_ids, detail, audit_version, context_hash, "
            "corrects_finding_id) VALUES "
            "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                finding.finding_id,
                finding.run_id,
                _check_id_column(finding),
                finding.outcome.value,
                finding.effective_execution_mode.value,
                finding.decision_ref,
                finding.leg_ref,
                finding.attempt_index,
                finding.rule_key,
                Jsonb(finding.rule_value) if finding.rule_value is not None else None,
                Jsonb(finding.observed_value) if finding.observed_value is not None else None,
                list(finding.tool_call_ids),
                finding.detail,
                finding.audit_version,
                finding.context_hash,
                finding.corrects_finding_id,
            ),
        )
    return finding.finding_id


def audit_findings_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[AuditFinding, ...]:
    rows = _rows(
        conn,
        "SELECT * FROM audit_findings WHERE run_id = %s ORDER BY recorded_at, finding_id",
        (run_id,),
    )
    out: list[AuditFinding] = []
    for r in rows:
        check_id: str = r["check_id"]
        out.append(
            AuditFinding(
                finding_id=r["finding_id"],
                run_id=r["run_id"],
                check_id=AuditCheck(check_id[:2]),
                sub_item=check_id[3:] if len(check_id) > 3 else None,
                outcome=r["outcome"],
                effective_execution_mode=r["effective_execution_mode"],
                decision_ref=r["decision_ref"],
                leg_ref=r["leg_ref"],
                attempt_index=r["attempt_index"],
                rule_key=r["rule_key"],
                rule_value=_canonical(r["rule_value"]),
                observed_value=_canonical(r["observed_value"]),
                tool_call_ids=tuple(r["tool_call_ids"]),
                detail=r["detail"],
                audit_version=r["audit_version"],
                context_hash=r["context_hash"],
                corrects_finding_id=r["corrects_finding_id"],
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# alerts_sent
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StoredAlert:
    alert_id: uuid.UUID
    run_id: uuid.UUID | None
    alert_kind: str
    dedup_key: str
    payload: Mapping[str, object]
    delivery_status: DeliveryStatus
    attempted_at: datetime
    recorded_at: datetime


def record_alert_sent(
    conn: Conn,
    *,
    run_id: uuid.UUID | None,
    alert_kind: str,
    dedup_key: str,
    payload: Mapping[str, object],
    delivery_status: DeliveryStatus,
    attempted_at: datetime,
) -> uuid.UUID:
    """Record one delivery attempt (each attempt is its own row; payload already redacted)."""
    _check_aware("attempted_at", attempted_at)
    if not alert_kind or not dedup_key:
        raise ValueError("alert_kind and dedup_key must be non-empty")
    alert_id = new_id()
    conn.execute(
        "INSERT INTO alerts_sent (alert_id, run_id, alert_kind, dedup_key, payload, "
        "delivery_status, attempted_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            alert_id,
            run_id,
            alert_kind,
            dedup_key,
            Jsonb(dict(payload)),
            delivery_status.value,
            attempted_at,
        ),
    )
    return alert_id


def alerts_for_run(conn: Conn, run_id: uuid.UUID) -> tuple[StoredAlert, ...]:
    rows = _rows(
        conn,
        "SELECT * FROM alerts_sent WHERE run_id = %s ORDER BY attempted_at, alert_id",
        (run_id,),
    )
    return tuple(
        StoredAlert(
            r["alert_id"],
            r["run_id"],
            r["alert_kind"],
            r["dedup_key"],
            r["payload"],
            DeliveryStatus(r["delivery_status"]),
            r["attempted_at"],
            r["recorded_at"],
        )
        for r in rows
    )
