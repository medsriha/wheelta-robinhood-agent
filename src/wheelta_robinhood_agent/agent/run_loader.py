"""Load the immutable input bundles for assembly and audit from the ledger (OUTPUT_ASSEMBLY.md).

`assemble_run_record` and `run_audit` are pure; this module does the reading. Everything comes
from recorded rows of this run: tool-call projections, delivered result envelopes (which
define the code-issued references the model actually saw), persisted DecisionFacts, the
stored agent output/parse, and the PositionBook rendered into the prompt.

Loader decisions (documented, not guessed values):
- Only effective mode `off` is supported (phase 1, ADR-0013): no broker order is ever placed,
  so `orders` and `attempt_evidence` are empty. Live loading needs order projections for the
  run and is out of scope until phase 2.
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

from wheelta_robinhood_agent.agent.audit.context import AuditContext, InstrumentFact
from wheelta_robinhood_agent.agent.facts_tool import FACTS_TOOL_NAME
from wheelta_robinhood_agent.agent.result_boundary import (
    evidence_ref_for,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.config.rules import LoadedRules
from wheelta_robinhood_agent.domain.assembly import DecisionsInput
from wheelta_robinhood_agent.domain.assembly_context import (
    ASSEMBLER_VERSION,
    AssemblyContext,
    DeliveredRef,
    RefKind,
)
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    DecisionOutputParsed,
    DecisionOutputParseFailure,
)
from wheelta_robinhood_agent.domain.enums import AppEnv, ExecutionMode
from wheelta_robinhood_agent.domain.facts import DecisionFacts
from wheelta_robinhood_agent.domain.positions import PositionBook
from wheelta_robinhood_agent.domain.run_record import Quote, RunRecord
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
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


def _delivered_envelopes(conn: Conn, run_id: uuid.UUID) -> list[Mapping[str, Any]]:
    envelopes: list[Mapping[str, Any]] = []
    for stored in ledger_evidence.effective(ledger_evidence.results_for_run(conn, run_id)):
        if stored.kind is not ledger_evidence.ResultKind.DELIVERED:
            continue
        payload = stored.payload
        if not isinstance(payload, dict) or payload.get("replaced") is not True:
            continue
        envelope = payload.get("tool_output")
        if isinstance(envelope, dict):
            envelopes.append(envelope)
    return envelopes


def load_delivered(conn: Conn, meta: RunMeta, book: PositionBook | None) -> DeliveredEvidence:
    """Code-issued references delivered in this run (tool results and the rendered book)."""
    refs: dict[str, DeliveredRef] = {}
    quotes: dict[uuid.UUID, Quote] = {}
    instruments: dict[str, InstrumentFact] = {}
    scope = meta.account_scope_id
    for envelope in _delivered_envelopes(conn, meta.run_id):
        call_id = envelope.get("tool_call_id")
        data = envelope.get("data")
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
                tick_increment=None,
                source_tool_call_id=inst.source_tool_call_ids[0],
            )
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
    )


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
    if meta.effective_mode is not ExecutionMode.OFF:
        raise LoaderError("only effective mode off can be loaded in phase 1 (ADR-0013)")


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
    return AssemblyContext(
        run_id=meta.run_id,
        environment=meta.environment,
        slot=meta.slot,
        terminated_at=terminated_at,
        requested_execution_mode=meta.requested_mode,
        effective_execution_mode=meta.effective_mode,
        account_scope_id=meta.account_scope_id,
        rules_version=str(meta.rules.version),
        rules_hash=meta.rules.sha256,
        prompt_id=meta.prompt_id,
        prompt_hash=meta.prompt_hash,
        model_id=meta.model_id,
        output_record_id=output_id,
        time_in_force=meta.rules.rules.orders.time_in_force,
        tool_calls=tool_call_records(conn, meta.run_id),
        facts=facts,
        refs=delivered.refs,
        quotes=delivered.quotes,
    )


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
    return AuditContext(
        run_id=meta.run_id,
        effective_execution_mode=meta.effective_mode,
        rules=meta.rules.rules,
        rules_version=str(meta.rules.version),
        rules_hash=meta.rules.sha256,
        assembler_version=ASSEMBLER_VERSION if run_record is not None else None,
        tool_calls=tool_call_records(conn, meta.run_id),
        quotes=delivered.quotes,
        instruments=delivered.instruments,
        decision_facts=facts,
        decision_output=decision_output,
        run_record=run_record,
        position_book=book,
    )
