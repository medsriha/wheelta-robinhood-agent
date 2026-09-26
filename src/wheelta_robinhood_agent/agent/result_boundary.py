"""The result boundary: what the model receives instead of a raw tool result.

docs/DATA_QUALITY.md "Delivery and failure contract"; CLAUDE.md §2.3-2.4, §8, §12.
`BoundaryValidator` is the `ResultValidator` injected into the PostToolUse hook:

- **Local tools** (`wra_local`): our own code produced the JSON text, so it is parsed and
  delivered as `validated` data. Unparseable output becomes an `error` envelope.
- **Built-ins** (WebSearch/WebFetch): the redacted payload is recorded as `validated` data.
  The hook records but cannot replace built-in output (hooks.py), so web content stays
  labeled untrusted by the prompt and source tiers (CLAUDE.md §11).
- **Remote MCP tools** (Robinhood, Wheelta): a result is `validated` only if a verified
  `EvidenceMapper` exists for that exact (server, tool). The mapper normalizes the payload
  into typed evidence (`MappedEvidence`) with code-issued evidence IDs and candidate refs.
  Without a mapper the envelope is `missing` with a typed gap and the redacted raw payload is
  stored as restricted `raw_invalid` evidence: **no mapping means no data, never a guess**.
  MCP `isError` results become `error` envelopes.

Every Robinhood/Wheelta result mapping is UNVERIFIED until our own `tools/list` and scrubbed
result fixtures are captured (CLAUDE.md §9), so `VERIFIED_MAPPERS` is empty. Tests inject
fixture mappers for fake servers only.

The shape of `tool_response` for MCP tools in the PostToolUse hook input is itself unverified
against the pinned CLI; `extract_mcp_payload` accepts the documented MCP `CallToolResult`
shapes and fails closed (error envelope) on anything else.
"""

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue, ValidationError

from wheelta_robinhood_agent.agent.hooks import (
    BUILTIN_SERVER,
    EnvelopeKind,
    ResultEnvelope,
    ValidationOutcome,
    ValidationRequest,
)
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.base import NonEmptyStr, Ref
from wheelta_robinhood_agent.domain.enums import CandidateOrigin
from wheelta_robinhood_agent.domain.facts_compute import (
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
    UnderlyingQuote,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.observability.redaction import Redactor

CANDIDATE_REF_PREFIX: Final = "candidate:"
EVIDENCE_REF_PREFIX: Final = "evidence:"
EVIDENCE_KEY: Final = "evidence"
EVIDENCE_REF_KEY: Final = "evidence_ref"


def evidence_ref_for(tool_call_id: uuid.UUID) -> str:
    """The code-issued evidence reference of one validated tool result."""
    return f"{EVIDENCE_REF_PREFIX}{tool_call_id}"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateEvidence(_Model):
    """A code-issued candidate reference and the instrument it names (OUTPUT_ASSEMBLY.md).

    The model may select `candidate_ref`; it can never mint one from a ticker string.
    """

    candidate_ref: Ref
    origin: CandidateOrigin
    underlying: NonEmptyStr
    instrument_evidence_id: uuid.UUID
    broker_instrument_id: NonEmptyStr
    occ_symbol: OccSymbol


class MappedEvidence(_Model):
    """Normalized, typed evidence produced from one validated tool result."""

    instruments: tuple[OptionInstrument, ...] = ()
    option_quotes: tuple[Quote, ...] = ()
    underlying_quotes: tuple[UnderlyingQuote, ...] = ()
    account_snapshots: tuple[AccountSnapshot, ...] = ()
    positions: tuple[PositionsRead, ...] = ()
    open_orders: tuple[OpenOrdersRead, ...] = ()
    candidates: tuple[CandidateEvidence, ...] = ()
    gaps: tuple[str, ...] = ()

    def evidence_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(i.evidence_id for i in self.instruments),
            *(q.quote_id for q in self.option_quotes),
            *(u.evidence_id for u in self.underlying_quotes),
            *(a.snapshot_id for a in self.account_snapshots),
            *(p.evidence_id for p in self.positions),
            *(o.evidence_id for o in self.open_orders),
        )

    def source_tool_call_ids(self) -> tuple[uuid.UUID, ...]:
        return (
            *(t for i in self.instruments for t in i.source_tool_call_ids),
            *(t for q in self.option_quotes for t in q.source_tool_call_ids),
            *(t for u in self.underlying_quotes for t in u.source_tool_call_ids),
            *(t for a in self.account_snapshots for t in a.tool_call_ids),
            *(t for p in self.positions for t in p.source_tool_call_ids),
            *(t for o in self.open_orders for t in o.source_tool_call_ids),
        )


class MappingRequest(_Model):
    """Input to an `EvidenceMapper`: the parsed (redacted) payload of one successful call."""

    tool_call_id: uuid.UUID
    server: str
    tool: str
    effective_input: dict[str, JsonValue]
    payload: JsonValue
    retrieved_at: AwareDatetime


class EvidenceMapper(Protocol):
    """Map one tool's verified result schema to typed evidence. Raise on any schema mismatch.

    `new_id` issues evidence IDs and candidate refs, so every identity is code-issued.
    """

    def __call__(
        self, request: MappingRequest, new_id: Callable[[], uuid.UUID]
    ) -> MappedEvidence: ...


# (server, tool) -> mapper. Empty until result schemas are captured and verified (CLAUDE.md §9).
VERIFIED_MAPPERS: Mapping[tuple[str, str], EvidenceMapper] = MappingProxyType({})


class PayloadError(ValueError):
    """The tool response does not have a recognized MCP result shape."""


class PayloadKind(StrEnum):
    OK = "ok"
    TOOL_ERROR = "tool_error"


def _texts(blocks: object) -> list[str]:
    if not isinstance(blocks, list):
        raise PayloadError("content is not a list")
    texts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            raise PayloadError("only text content blocks are supported")
        text = block.get("text")
        if not isinstance(text, str):
            raise PayloadError("text block without text")
        texts.append(text)
    return texts


def extract_mcp_payload(tool_response: object) -> tuple[PayloadKind, JsonValue]:
    """Parse an MCP tool response into (kind, JSON value).

    Accepted shapes: a `CallToolResult`-like dict (`content` blocks, optional `isError` /
    `is_error`, optional `structuredContent`), a bare list of text blocks, or a JSON string.
    A single text block must hold one JSON document. Anything else raises PayloadError.
    """
    is_error = False
    content: object = tool_response
    structured: object = None
    if isinstance(tool_response, dict):
        is_error = tool_response.get("isError") is True or tool_response.get("is_error") is True
        structured = tool_response.get("structuredContent")
        content = tool_response.get("content")
    if is_error:
        texts = _texts(content) if content is not None else []
        return PayloadKind.TOOL_ERROR, "\n".join(texts)
    if structured is not None:
        return PayloadKind.OK, json.loads(json.dumps(structured))
    if isinstance(content, str):
        texts = [content]
    else:
        texts = _texts(content)
    if len(texts) != 1:
        raise PayloadError("expected exactly one text block")
    try:
        value: JsonValue = json.loads(texts[0])
    except (ValueError, RecursionError) as exc:
        raise PayloadError(f"text is not JSON ({type(exc).__name__})") from None
    return PayloadKind.OK, value


@dataclass(frozen=True)
class BoundaryValidator:
    """`ResultValidator` for the PostToolUse hook (see module docstring)."""

    redactor: Redactor
    mappers: Mapping[tuple[str, str], EvidenceMapper] = field(
        default_factory=lambda: VERIFIED_MAPPERS
    )
    id_factory: Callable[[], uuid.UUID] = new_id

    def __call__(self, request: ValidationRequest) -> ValidationOutcome:
        if request.server == BUILTIN_SERVER:
            return self._envelope(
                request, EnvelopeKind.VALIDATED, data=self.redactor.redact(request.tool_response)
            )
        try:
            kind, payload = extract_mcp_payload(request.tool_response)
        except (PayloadError, TypeError, ValueError) as exc:
            return self._invalid(request, EnvelopeKind.ERROR, f"unrecognized result: {exc}")
        if kind is PayloadKind.TOOL_ERROR:
            return self._invalid(request, EnvelopeKind.ERROR, "the tool returned an error")
        redacted = self.redactor.redact(payload)
        if request.server == LOCAL_SERVER_NAME:
            return self._envelope(request, EnvelopeKind.VALIDATED, data=redacted)
        mapper = self.mappers.get((request.server, request.tool))
        if mapper is None:
            gap = f"no verified result mapping for {request.server}.{request.tool}"
            return self._invalid(request, EnvelopeKind.MISSING, gap)
        try:
            evidence = mapper(
                MappingRequest(
                    tool_call_id=request.tool_call_id,
                    server=request.server,
                    tool=request.tool,
                    effective_input=self.redactor.redact_mapping(request.effective_input),
                    payload=redacted,
                    retrieved_at=request.retrieved_at,
                ),
                self.id_factory,
            )
            _check_provenance(evidence, request.tool_call_id)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            gap = f"result failed schema validation ({type(exc).__name__})"
            return self._invalid(request, EnvelopeKind.MISSING, gap)
        data: dict[str, JsonValue] = {
            EVIDENCE_REF_KEY: evidence_ref_for(request.tool_call_id),
            EVIDENCE_KEY: evidence.model_dump(mode="json"),
        }
        return self._envelope(request, EnvelopeKind.VALIDATED, data=data, gaps=evidence.gaps)

    def _envelope(
        self,
        request: ValidationRequest,
        kind: EnvelopeKind,
        *,
        data: JsonValue = None,
        gaps: tuple[str, ...] = (),
    ) -> ValidationOutcome:
        return ValidationOutcome(
            envelope=ResultEnvelope(
                tool_call_id=request.tool_call_id,
                server=request.server,
                tool=request.tool,
                kind=kind,
                data=data,
                gaps=gaps,
                retrieved_at=request.retrieved_at,
            )
        )

    def _invalid(
        self, request: ValidationRequest, kind: EnvelopeKind, gap: str
    ) -> ValidationOutcome:
        """A missing/error envelope; the redacted raw payload is kept as restricted evidence."""
        raw = self.redactor.redact(request.tool_response)
        outcome = self._envelope(request, kind, gaps=(gap,))
        return outcome.model_copy(update={"raw_redacted": raw if raw is not None else ""})


def _check_provenance(evidence: MappedEvidence, tool_call_id: uuid.UUID) -> None:
    """Every observation must cite exactly this call; candidate refs are code-shaped."""
    if any(t != tool_call_id for t in evidence.source_tool_call_ids()):
        raise ValueError("mapped evidence cites another tool call")
    ids = evidence.evidence_ids()
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence ids")
    instruments = {i.evidence_id: i for i in evidence.instruments}
    for candidate in evidence.candidates:
        if not candidate.candidate_ref.startswith(CANDIDATE_REF_PREFIX):
            raise ValueError("candidate refs must be code-issued")
        instrument = instruments.get(candidate.instrument_evidence_id)
        if instrument is None or (
            instrument.broker_instrument_id != candidate.broker_instrument_id
            or instrument.occ_symbol != candidate.occ_symbol
            or instrument.underlying != candidate.underlying
        ):
            raise ValueError("a candidate must name an instrument from the same result")


def mapped_evidence_of(envelope: Mapping[str, Any]) -> MappedEvidence | None:
    """The typed evidence inside a stored validated envelope, or None if it carries none."""
    if envelope.get("kind") != EnvelopeKind.VALIDATED.value:
        return None
    data = envelope.get("data")
    if not isinstance(data, dict) or EVIDENCE_KEY not in data:
        return None
    return MappedEvidence.model_validate(data[EVIDENCE_KEY])
