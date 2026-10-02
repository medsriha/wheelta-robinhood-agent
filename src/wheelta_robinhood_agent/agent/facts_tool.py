"""The local Tier R `get_decision_facts` tool (ADR-0011; OUTPUT_ASSEMBLY.md; CUSTOM_TOOLS.md).

`mcp__wra_local__get_decision_facts(subject_ref, purpose, limit_price?, close_quantity?)`:

1. Resolve the code-issued `subject_ref`: a `candidate:` ref issued at the result boundary
   (OPEN) or a durable `position:` ref from the PositionBook (CLOSE/MANAGE). A ref the code
   never issued is rejected; nothing is matched by ticker, price, or time.
2. Load this run's validated evidence from the ledger (`results` rows of kind `validated`
   whose envelope carries `MappedEvidence`), taking for each input the most recently recorded
   observation of the subject's instrument, its option quote, its underlying quote, and the
   account snapshot, positions, and open-orders reads. Freshness is judged by the pure
   function against `as_of`; an old observation is never refreshed by re-use.
3. Build `FactInputs`, call pure `compute_decision_facts` with `config.facts_rules`, persist
   the DecisionFacts via `ledger/evidence.py`, and return the code-issued `facts_ref`.

The model supplies only refs plus the discretionary `limit_price` (decimal string) and, for
CLOSE, `close_quantity` (ADR-0015). No balances, quotes, formulas, or rule values are ever
accepted. Missing evidence yields typed gaps (quality `missing`), never a guessed value: when
not even the instrument is known, no fact set can exist and an `unavailable` result with gaps
is returned instead. This tool performs no broker request or mutation and gates no order.

Inputs come only from `result_boundary.VERIFIED_MAPPERS`; a tool without a mapper yields no
evidence, and the dependent input is reported as a gap.
"""

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

import psycopg
from claude_agent_sdk import SdkMcpTool, ToolAnnotations, tool
from pydantic import JsonValue, ValidationError

from wheelta_robinhood_agent.agent.result_boundary import (
    CANDIDATE_REF_PREFIX,
    CandidateEvidence,
    MappedEvidence,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.enums import CandidateOrigin, DataQuality, PositionsCoverage
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.facts import DecisionFacts, FactsPurpose
from wheelta_robinhood_agent.domain.facts_compute import (
    BoardScreen,
    CandidateProvenance,
    CashBaseline,
    FactInputs,
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
    ShortOptionHolding,
    UnderlyingQuote,
    combine_positions,
    compute_decision_facts,
)
from wheelta_robinhood_agent.domain.facts_rules import FactsRules
from wheelta_robinhood_agent.domain.positions import PositionBookEntry
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger import positions as ledger_positions
from wheelta_robinhood_agent.ledger.ids import new_id

Conn = psycopg.Connection[tuple[object, ...]]

FACTS_TOOL_NAME: Final = "get_decision_facts"
FACTS_REF_PREFIX: Final = "facts:"
POSITION_REF_PREFIX: Final = "position:"
_DECIMAL_RE: Final = re.compile(r"^\d+(\.\d+)?$")
_ARGS: Final = frozenset({"subject_ref", "purpose", "limit_price", "close_quantity"})

INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "subject_ref": {"type": "string", "minLength": 1},
        "purpose": {"type": "string", "enum": [p.value for p in FactsPurpose]},
        "limit_price": {"type": "string", "pattern": _DECIMAL_RE.pattern},
        "close_quantity": {"type": "integer", "minimum": 1},
    },
    "required": ["subject_ref", "purpose"],
    "additionalProperties": False,
}


class FactsRequestError(ValueError):
    """The request itself is invalid (bad ref, purpose, or discretionary input)."""


@dataclass(frozen=True, slots=True)
class FactsRequest:
    subject_ref: str
    purpose: FactsPurpose
    limit_price: Decimal | None
    close_quantity: int | None


def parse_request(args: object) -> FactsRequest:
    """Validate the model's arguments. Numbers for prices are refused (never binary floats)."""
    if not isinstance(args, dict):
        raise FactsRequestError("arguments must be an object")
    extra = set(args) - _ARGS
    if extra:
        raise FactsRequestError(f"unexpected arguments: {sorted(extra)}")
    subject = args.get("subject_ref")
    if not isinstance(subject, str) or not subject:
        raise FactsRequestError("subject_ref must be a code-issued reference string")
    raw_purpose = args.get("purpose")
    try:
        purpose = FactsPurpose(raw_purpose if isinstance(raw_purpose, str) else "")
    except ValueError:
        raise FactsRequestError("purpose must be open, close, or manage") from None
    limit_price: Decimal | None = None
    raw_price = args.get("limit_price")
    if raw_price is not None:
        if not isinstance(raw_price, str) or not _DECIMAL_RE.fullmatch(raw_price):
            raise FactsRequestError("limit_price must be a decimal string such as '1.25'")
        try:
            limit_price = Decimal(raw_price)
        except InvalidOperation:
            raise FactsRequestError("limit_price is not a decimal") from None
        if limit_price <= 0:
            raise FactsRequestError("limit_price must be positive")
    close_quantity: int | None = None
    raw_qty = args.get("close_quantity")
    if raw_qty is not None:
        if isinstance(raw_qty, bool) or not isinstance(raw_qty, int) or raw_qty < 1:
            raise FactsRequestError("close_quantity must be a positive integer")
        if purpose is not FactsPurpose.CLOSE:
            raise FactsRequestError("close_quantity is a CLOSE-purpose input only")
        close_quantity = raw_qty
    return FactsRequest(subject, purpose, limit_price, close_quantity)


@dataclass(frozen=True)
class RunEvidence:
    """Validated, mapped evidence recorded in this run, oldest first."""

    items: tuple[MappedEvidence, ...]

    def _latest(self, values: list[Any]) -> Any:
        return values[-1] if values else None

    def candidate(self, ref: str) -> CandidateEvidence | None:
        found = [c for e in self.items for c in e.candidates if c.candidate_ref == ref]
        return found[0] if len(found) == 1 else None

    def instrument(self, broker_instrument_id: str) -> OptionInstrument | None:
        found = [
            i
            for e in self.items
            for i in e.instruments
            if i.broker_instrument_id == broker_instrument_id
        ]
        latest: OptionInstrument | None = self._latest(found)
        return latest

    def instrument_by_id(self, evidence_id: uuid.UUID) -> OptionInstrument | None:
        return next(
            (i for e in self.items for i in e.instruments if i.evidence_id == evidence_id), None
        )

    def option_quote(self, broker_instrument_id: str) -> Quote | None:
        found = [
            q
            for e in self.items
            for q in e.option_quotes
            if q.broker_instrument_id == broker_instrument_id
        ]
        latest: Quote | None = self._latest(found)
        return latest

    def underlying_quote(self, symbol: str) -> UnderlyingQuote | None:
        found = [u for e in self.items for u in e.underlying_quotes if u.symbol == symbol]
        latest: UnderlyingQuote | None = self._latest(found)
        return latest

    def account(self) -> AccountSnapshot | None:
        latest: AccountSnapshot | None = self._latest(
            [a for e in self.items for a in e.account_snapshots]
        )
        return latest

    def _resolved_option_reads(self) -> list[PositionsRead]:
        """Options-only reads built from pending short rows whose every contract this run's
        instrument evidence identifies (ADR-0034). A row whose instrument is unknown, or whose
        underlying or multiplier disagrees with it, leaves that read unresolved."""
        reads: list[PositionsRead] = []
        for e in self.items:
            for pending in e.pending_option_positions:
                holdings: list[ShortOptionHolding] = []
                for row in pending.rows:
                    inst = self.instrument(row.broker_instrument_id)
                    if (
                        inst is None
                        or inst.underlying != row.underlying
                        or inst.multiplier != row.multiplier
                    ):
                        break
                    holdings.append(
                        ShortOptionHolding(
                            underlying=row.underlying,
                            occ_symbol=inst.occ_symbol,
                            broker_instrument_id=row.broker_instrument_id,
                            short_quantity=row.short_quantity,
                            multiplier=row.multiplier,
                        )
                    )
                else:
                    reads.append(
                        PositionsRead(
                            evidence_id=pending.evidence_id,
                            as_of=pending.as_of,
                            source_tool_call_ids=pending.source_tool_call_ids,
                            covers=frozenset({PositionsCoverage.OPTIONS}),
                            short_options=tuple(holdings),
                        )
                    )
        return reads

    def positions(self, first: bool = False) -> PositionsRead | None:
        """The latest complete positions read: a read covering both kinds, or the latest
        shares-only and options-only reads of this run combined (ADR-0031), whichever is
        newer. A lone half is returned as is, so the facts record which half is missing.
        Options halves include pending short rows once resolved (ADR-0034). `first` picks
        the earliest complete read instead (the cash baseline, ADR-0072)."""
        reads = [p for e in self.items for p in e.positions] + self._resolved_option_reads()
        reads.sort(key=lambda p: p.as_of)
        complete = [p for p in reads if p.complete]
        shares = [p for p in reads if p.covers == {PositionsCoverage.SHARES}]
        options = [p for p in reads if p.covers == {PositionsCoverage.OPTIONS}]
        i = 0 if first else -1
        candidates: list[PositionsRead] = complete[i:][:1]
        if shares and options:
            candidates.append(combine_positions(shares[i], options[i]))
        if candidates:
            pick = min if first else max
            return pick(candidates, key=lambda p: p.as_of)
        latest: PositionsRead | None = None if first else self._latest(reads)
        return latest

    def cash_baseline(self, unsettled_usd: Decimal | None = None) -> CashBaseline | None:
        """The run's first account snapshot with its first complete positions and open-orders
        reads, and the session-start `unsettled_funds` (ADR-0072); None if a read is missing."""
        snapshots = [a for e in self.items for a in e.account_snapshots]
        pos = self.positions(first=True)
        orders = [o for e in self.items for o in e.open_orders]
        if not snapshots or pos is None or not pos.complete or not orders:
            return None
        return CashBaseline(
            account=snapshots[0],
            positions=pos,
            open_orders=orders[0],
            unsettled_usd=unsettled_usd,
        )

    def board_screens(self) -> dict[str, BoardScreen]:
        """Board rows of the run's current Wheelta build, by OCC symbol (ADR-0041).

        The current build is the build of the newest board row seen this run (by `as_of`); rows
        of any older build are never used (data_quality.freshness.wheelta_board). A contract
        listed more than once in the build keeps its latest row.
        """
        rows = sorted((b for e in self.items for b in e.board_screens), key=lambda b: b.as_of)
        if not rows:
            return {}
        build = rows[-1].build_id
        return {str(b.occ_symbol): b for b in rows if b.build_id == build}

    def open_orders(self) -> OpenOrdersRead | None:
        latest: OpenOrdersRead | None = self._latest([o for e in self.items for o in e.open_orders])
        return latest


def load_run_evidence(conn: Conn, run_id: uuid.UUID) -> RunEvidence:
    """Every validated envelope with mapped evidence in this run, in recorded order."""
    items: list[MappedEvidence] = []
    for stored in ledger_evidence.effective(ledger_evidence.results_for_run(conn, run_id)):
        if stored.kind is not ledger_evidence.ResultKind.VALIDATED:
            continue
        if not isinstance(stored.payload, dict):
            continue
        mapped = mapped_evidence_of(stored.payload)
        if mapped is not None:
            items.append(mapped)
    return RunEvidence(tuple(items))


def _gap(field: str, detail: str) -> dict[str, JsonValue]:
    return Gap(field=field, kind=DataQuality.MISSING, detail=detail).model_dump(mode="json")


def _unavailable(request: FactsRequest, gaps: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    return {
        "status": "unavailable",
        "subject_ref": request.subject_ref,
        "purpose": request.purpose.value,
        "quality": DataQuality.MISSING.value,
        "gaps": list(gaps),
        "note": "No fact set could be computed; collect the missing evidence with the read "
        "tools and request facts again. Do not supply substitute values.",
    }


@dataclass(frozen=True)
class DecisionFactsService:
    """Resolves refs to ledger evidence, computes, and persists DecisionFacts for one run."""

    conn: Conn
    run_id: uuid.UUID
    account_scope_id: str
    rules: FactsRules
    clock: Callable[[], datetime]
    id_factory: Callable[[], uuid.UUID] = new_id
    # ADR-0072: the trusted get_accounts check's unsettled_funds at session start.
    opening_unsettled_usd: Decimal | None = None

    def compute(self, request: FactsRequest) -> dict[str, JsonValue]:
        """The tool's result body. Raises FactsRequestError for an invalid subject/purpose;
        ledger errors propagate (the caller fails closed)."""
        as_of = self.clock()
        evidence = load_run_evidence(self.conn, self.run_id)
        candidate: CandidateProvenance | None = None
        position: PositionBookEntry | None = None
        gaps: list[dict[str, JsonValue]] = []
        instrument: OptionInstrument | None
        if request.subject_ref.startswith(CANDIDATE_REF_PREFIX):
            if request.purpose is not FactsPurpose.OPEN:
                raise FactsRequestError("a candidate reference takes purpose 'open'")
            found = evidence.candidate(request.subject_ref)
            if found is None:
                raise FactsRequestError("subject_ref is not a candidate issued in this run")
            candidate = CandidateProvenance(
                candidate_ref=found.candidate_ref,
                origin=found.origin,
                underlying=found.underlying,
            )
            instrument = evidence.instrument_by_id(found.instrument_evidence_id)
            latest = evidence.instrument(found.broker_instrument_id)
            if latest is not None:
                instrument = latest
        elif request.subject_ref.startswith(POSITION_REF_PREFIX):
            if request.purpose is FactsPurpose.OPEN:
                raise FactsRequestError("a position reference takes purpose 'close' or 'manage'")
            book = ledger_positions.position_book(self.conn, self.account_scope_id, as_of=as_of)
            position = next(
                (e for e in book.entries if e.position_ref == request.subject_ref), None
            )
            if position is None:
                raise FactsRequestError("subject_ref is not an active position of this account")
            if len(position.current_instruments) != 1:
                gaps.append(
                    _gap("instrument", "the position does not hold exactly one current contract")
                )
                return _unavailable(request, gaps)
            instrument = evidence.instrument(position.current_instruments[0].broker_instrument_id)
        else:
            raise FactsRequestError("subject_ref must be a code-issued candidate or position ref")
        if instrument is None:
            gaps.append(
                _gap("instrument", "no validated instrument evidence for the subject in this run")
            )
            return _unavailable(request, gaps)
        facts_id = self.id_factory()
        try:
            inputs = FactInputs(
                facts_id=facts_id,
                facts_ref=f"{FACTS_REF_PREFIX}{facts_id}",
                run_id=self.run_id,
                subject_ref=request.subject_ref,
                purpose=request.purpose,
                instrument=instrument,
                option_quote=evidence.option_quote(instrument.broker_instrument_id),
                underlying_quote=evidence.underlying_quote(instrument.underlying),
                account=evidence.account(),
                positions=evidence.positions(),
                open_orders=evidence.open_orders(),
                cash_baseline=evidence.cash_baseline(self.opening_unsettled_usd),
                candidate=candidate,
                board_screen=(
                    evidence.board_screens().get(str(instrument.occ_symbol))
                    if candidate is not None and candidate.origin is CandidateOrigin.BOARD
                    else None
                ),
                position=position,
                limit_price=request.limit_price,
                close_quantity=request.close_quantity,
            )
        except ValidationError as exc:
            gaps.append(_gap("fact_inputs", f"inputs are inconsistent ({exc.error_count()})"))
            return _unavailable(request, gaps)
        facts = compute_decision_facts(inputs, self.rules, as_of)
        ledger_evidence.insert_decision_facts(
            self.conn, facts=facts, input_hash=fact_inputs_hash(inputs, as_of)
        )
        return _available(facts)


def _available(facts: DecisionFacts) -> dict[str, JsonValue]:
    return {
        "status": "ok",
        "facts_ref": facts.facts_ref,
        "facts": facts.model_dump(mode="json"),
        "note": "Code-computed facts. Select facts_ref in proposed legs; do not restate or "
        "adjust these values.",
    }


def fact_inputs_hash(inputs: FactInputs, as_of: datetime) -> str:
    """SHA-256 of the canonical inputs plus observation time (recorded with the facts)."""
    payload = {"inputs": inputs.model_dump(mode="json"), "as_of": as_of.isoformat()}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_facts_tool(service: DecisionFactsService, run_control: RunControl) -> SdkMcpTool[Any]:
    """The in-process `get_decision_facts` tool for the `wra_local` server.

    A ledger failure sets the stop latch (the same persistence requirement as any other
    delivery, OUTPUT_ASSEMBLY.md) and returns an error; a bad request returns an error only.
    """

    @tool(
        FACTS_TOOL_NAME,
        "Compute code-owned decision facts (DTE, collateral, yields, spread, capacity, "
        "rule-derived contract quantity) for a code-issued candidate ref (purpose 'open') or "
        "position ref (purpose 'close' or 'manage') from evidence validated in this run. "
        "Optional: limit_price as a decimal string; close_quantity (CLOSE only). Returns a "
        "facts_ref, or typed gaps when evidence is missing. It never places or approves orders.",
        INPUT_SCHEMA,
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False),
    )
    async def get_decision_facts(args: dict[str, Any]) -> dict[str, Any]:
        try:
            request = parse_request(args)
            body = service.compute(request)
        except FactsRequestError as exc:
            return _error(str(exc))
        except Exception as exc:
            run_control.request_stop(StopReason.INFRASTRUCTURE_FAILURE, service.clock())
            return _error(f"decision facts could not be recorded ({type(exc).__name__})")
        return {"content": [{"type": "text", "text": json.dumps(body, sort_keys=True)}]}

    return get_decision_facts


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "is_error": True}
