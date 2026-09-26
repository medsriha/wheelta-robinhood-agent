"""Append-only evidence/output repositories: round trips, run scoping, and corrections."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import count

import psycopg
import pytest

from wheelta_robinhood_agent.domain.account import (
    AccountSnapshot,
    ReservationEvidence,
    ReservationKind,
)
from wheelta_robinhood_agent.domain.decision_output import parse_agent_decision_output
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AuditCheck,
    AuditOutcome,
    DataQuality,
    ExecutionMode,
    SourceTier,
    ToolTier,
)
from wheelta_robinhood_agent.domain.evidence import Citation, Gap
from wheelta_robinhood_agent.domain.facts import DecisionFacts, FactsPurpose
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import DecisionOutputStatus, RunRecord
from wheelta_robinhood_agent.ledger.errors import IdentityConflict, UnknownEntity
from wheelta_robinhood_agent.ledger.evidence import (
    DeliveryStatus,
    ParseStatus,
    ResultKind,
    account_snapshots_for_run,
    agent_decisions_for_run,
    agent_outputs_for_run,
    alerts_for_run,
    audit_findings_for_run,
    citations_for_run,
    decision_facts_for_run,
    effective,
    get_result,
    insert_account_snapshot,
    insert_agent_decision,
    insert_agent_output,
    insert_audit_finding,
    insert_citation,
    insert_decision_facts,
    insert_result,
    insert_run_record,
    record_alert_sent,
    results_for_run,
    run_records_for_run,
)
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import record_tool_call_requested

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 17, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=5)
_sdk = count(1)


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


def _call(conn: Conn, run_id: uuid.UUID) -> uuid.UUID:
    return record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id=f"toolu_{next(_sdk)}",
        stage="agent",
        server="robinhood",
        tool="get_account",
        tier=ToolTier.R,
        arguments_redacted={},
        requested_at=T0,
    ).tool_call_id


def test_results_round_trip_and_corrections_reference_originals(
    conn: Conn, run_id: uuid.UUID
) -> None:
    call = _call(conn, run_id)
    original = insert_result(
        conn, run_id=run_id, kind=ResultKind.VALIDATED, payload={"bid": "1.20"}, tool_call_id=call
    )
    delivered = insert_result(
        conn, run_id=run_id, kind=ResultKind.DELIVERED, payload=["envelope"], tool_call_id=call
    )
    fixed = insert_result(
        conn,
        run_id=run_id,
        kind=ResultKind.VALIDATED,
        payload={"bid": "1.21"},
        tool_call_id=call,
        corrects_result_id=original,
    )
    rows = results_for_run(conn, run_id)
    assert [r.record_id for r in rows] == [original, delivered, fixed]
    assert get_result(conn, fixed).corrects_id == original
    assert get_result(conn, delivered).payload == ["envelope"]
    assert [r.record_id for r in effective(rows)] == [delivered, fixed]
    # Nothing was updated: the original row is intact.
    assert get_result(conn, original).payload == {"bid": "1.20"}


def test_corrections_must_stay_in_the_same_run(conn: Conn, run_id: uuid.UUID) -> None:
    other = open_run_slot(conn, AppEnv.LOCAL, SLOT + timedelta(hours=1)).run_id
    original = insert_result(conn, run_id=run_id, kind=ResultKind.ERROR, payload={})
    with pytest.raises(IdentityConflict):
        insert_result(
            conn, run_id=other, kind=ResultKind.ERROR, payload={}, corrects_result_id=original
        )
    with pytest.raises(UnknownEntity):
        insert_result(
            conn, run_id=run_id, kind=ResultKind.ERROR, payload={}, corrects_result_id=new_id()
        )
    with pytest.raises(IdentityConflict):
        insert_result(
            conn,
            run_id=other,
            kind=ResultKind.RAW_INVALID,
            payload={},
            tool_call_id=_call(conn, run_id),
        )


def test_citations_round_trip(conn: Conn, run_id: uuid.UUID) -> None:
    citation = Citation(
        citation_id=new_id(),
        url="https://www.sec.gov/filing",
        title="10-Q",
        publisher="SEC",
        published_at=None,
        retrieved_at=T0,
        tier=SourceTier.TIER_1,
        excerpt="Revenue increased",
        tool_call_id=_call(conn, run_id),
    )
    insert_citation(conn, run_id=run_id, citation=citation)
    fixed = citation.model_copy(update={"citation_id": new_id(), "excerpt": "Revenue rose"})
    insert_citation(conn, run_id=run_id, citation=fixed, corrects_citation_id=citation.citation_id)
    stored = citations_for_run(conn, run_id)
    assert [r.value for r in stored] == [citation, fixed]
    assert stored[1].corrects_id == citation.citation_id
    assert [r.value for r in effective(stored)] == [fixed]


def _snapshot(call: uuid.UUID, *, with_base: bool = False) -> AccountSnapshot:
    cash, reserved = Decimal("10000.00"), Decimal("3600.00")
    return AccountSnapshot(
        snapshot_id=new_id(),
        as_of=T0,
        retrieved_at=T0,
        tool_call_ids=(call,),
        account_ref="****1234",
        agentic_verified=True,
        account_value_usd=Decimal("25000.00"),
        available_settled_cash_usd=cash,
        csp_reserved_cash_usd=reserved if with_base else None,
        csp_cash_base_usd=cash + reserved if with_base else None,
        csp_cash_base_evidence_ids=(call,) if with_base else (),
        positions_ref=None,
        open_orders_ref=None,
        tax_lots_ref=None,
        reservation_evidence=(
            ReservationEvidence(
                kind=ReservationKind.POSITION,
                broker_ref="pos-1",
                quantity=2,
                amount_usd=Decimal("3600.00"),
                source_tool_call_ids=(call,),
            ),
        ),
        quality=DataQuality.MISSING,
        gaps=()
        if with_base
        else (
            Gap(field="csp_reserved_cash_usd", kind=DataQuality.MISSING, detail="unverified"),
            Gap(field="csp_cash_base_usd", kind=DataQuality.MISSING, detail="not derived"),
        ),
    )


def test_account_snapshot_round_trip(conn: Conn, run_id: uuid.UUID) -> None:
    call = _call(conn, run_id)
    snapshot = _snapshot(call)
    insert_account_snapshot(conn, run_id=run_id, snapshot=snapshot)
    (stored,) = account_snapshots_for_run(conn, run_id)
    assert stored.value == snapshot


def test_account_snapshot_with_cash_base_evidence_round_trips(
    conn: Conn, run_id: uuid.UUID
) -> None:
    snapshot = _snapshot(_call(conn, run_id), with_base=True)
    insert_account_snapshot(conn, run_id=run_id, snapshot=snapshot)
    (stored,) = account_snapshots_for_run(conn, run_id)
    assert stored.value == snapshot
    assert stored.value.csp_cash_base_evidence_ids == snapshot.csp_cash_base_evidence_ids


def test_agent_output_and_decisions_are_kept_separately(conn: Conn, run_id: uuid.UUID) -> None:
    raw_bad = '{"decisions": 1}'
    bad_output = insert_agent_output(conn, run_id=run_id, raw_redacted=raw_bad, observed_at=T0)
    bad = parse_agent_decision_output(raw_bad)
    assert not bad.ok
    insert_agent_decision(conn, run_id=run_id, output_id=bad_output, result=bad)

    raw_ok = '{"decisions": [], "cancellation_rationales": [], "unresolved_questions": []}'
    ok_output = insert_agent_output(
        conn, run_id=run_id, raw_redacted=raw_ok, observed_at=T0, corrects_output_id=bad_output
    )
    good = parse_agent_decision_output(raw_ok)
    assert good.ok
    insert_agent_decision(conn, run_id=run_id, output_id=ok_output, result=good)

    outputs = agent_outputs_for_run(conn, run_id)
    assert [o.raw_redacted for o in outputs] == [raw_bad, raw_ok]
    assert outputs[1].corrects_id == bad_output
    decisions = agent_decisions_for_run(conn, run_id)
    assert [d.parse_status for d in decisions] == [ParseStatus.INVALID, ParseStatus.VALID]
    assert decisions[0].output is None and decisions[0].issues == bad.issues
    assert decisions[1].output == good.output
    assert decisions[1].schema_version == "5"


def test_decision_facts_round_trip_and_correction_keeps_ref(conn: Conn, run_id: uuid.UUID) -> None:
    facts = DecisionFacts(
        facts_id=new_id(),
        facts_ref="facts:1",
        run_id=run_id,
        subject_ref="candidate:AAPL",
        purpose=FactsPurpose.OPEN,
        observed_at=T0,
        rules_version="3",
        rules_hash="abc",
        input_evidence_ids=(new_id(),),
        snapshot_ref=None,
        initial_quantity=2,
        remaining_quantity=2,
        quality=DataQuality.OK,
    )
    insert_decision_facts(conn, facts=facts, input_hash="h1")
    corrected = facts.model_copy(update={"facts_id": new_id(), "remaining_quantity": 1})
    insert_decision_facts(conn, facts=corrected, input_hash="h2", corrects_facts_id=facts.facts_id)
    with pytest.raises(IdentityConflict):
        insert_decision_facts(
            conn,
            facts=facts.model_copy(update={"facts_id": new_id(), "facts_ref": "facts:2"}),
            input_hash="h3",
            corrects_facts_id=facts.facts_id,
        )
    stored = decision_facts_for_run(conn, run_id)
    assert [s.facts for s in stored] == [facts, corrected]
    assert [s.input_hash for s in stored] == ["h1", "h2"]
    assert [s.facts for s in effective(stored)] == [corrected]


def test_run_record_round_trip(conn: Conn, run_id: uuid.UUID) -> None:
    record = RunRecord(
        schema_version=4,
        assembler_version="assembler-1",
        input_hash="in-hash",
        run_id=run_id,
        environment=AppEnv.LOCAL,
        slot=SLOT,
        terminated_at=T0,
        requested_execution_mode=ExecutionMode.OFF,
        effective_execution_mode=ExecutionMode.OFF,
        rules_version="3",
        rules_hash="abc",
        prompt_id=None,
        prompt_hash=None,
        model_id=None,
        decision_output_status=DecisionOutputStatus.MISSING,
        summary="No output.",
    )
    first = insert_run_record(conn, record=record, assembled_at=T0)
    second = insert_run_record(
        conn,
        record=record.model_copy(update={"assembler_version": "assembler-2"}),
        assembled_at=T0 + timedelta(minutes=1),
        corrects_record_id=first,
    )
    stored = run_records_for_run(conn, run_id)
    assert [s.record_id for s in stored] == [first, second]
    assert stored[0].record == record
    assert stored[1].corrects_id == first
    versions = conn.execute(
        "SELECT schema_version, assembler_version, input_hash FROM assembled_run_records "
        "WHERE record_id = %s",
        (first,),
    ).fetchone()
    assert versions == ("4", "assembler-1", "in-hash")


def test_audit_findings_round_trip_with_correction(conn: Conn, run_id: uuid.UUID) -> None:
    finding = AuditFinding(
        finding_id=new_id(),
        run_id=run_id,
        check_id=AuditCheck.V7,
        sub_item="max_contracts",
        outcome=AuditOutcome.VIOLATION,
        effective_execution_mode=ExecutionMode.OFF,
        decision_ref="decision:1",
        leg_ref="leg:1",
        attempt_index=0,
        rule_key="limits.max_contracts",
        rule_value="5",
        observed_value="6",
        tool_call_ids=(_call(conn, run_id),),
        detail="six contracts proposed",
        audit_version="audit-1",
        context_hash="ctx",
    )
    insert_audit_finding(conn, finding)
    correction = finding.model_copy(
        update={
            "finding_id": new_id(),
            "outcome": AuditOutcome.UNVERIFIABLE,
            "corrects_finding_id": finding.finding_id,
            "sub_item": None,
        }
    )
    insert_audit_finding(conn, correction)
    assert audit_findings_for_run(conn, run_id) == (finding, correction)


def test_alerts_are_recorded_per_attempt(conn: Conn, run_id: uuid.UUID) -> None:
    for status in (DeliveryStatus.FAILED, DeliveryStatus.SENT):
        record_alert_sent(
            conn,
            run_id=run_id,
            alert_kind="order_placed",
            dedup_key="order:B-1",
            payload={"order": "B-1"},
            delivery_status=status,
            attempted_at=T0,
        )
    alerts = alerts_for_run(conn, run_id)
    assert [a.delivery_status for a in alerts] == [DeliveryStatus.FAILED, DeliveryStatus.SENT]
    assert alerts[0].payload == {"order": "B-1"}
