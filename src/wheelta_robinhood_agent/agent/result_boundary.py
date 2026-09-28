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

`VERIFIED_MAPPERS` holds only the Robinhood tools whose results were captured as scrubbed
fixtures (ADR-0017; `robinhood_mappers.py`). Every other Robinhood/Wheelta result stays
`missing`. Tests inject fixture mappers for fake servers.

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
from typing import Any, Final, cast

from pydantic import JsonValue, ValidationError

from wheelta_robinhood_agent.agent.account_scope import LOGIN_SCOPED_TOOLS
from wheelta_robinhood_agent.agent.hooks import (
    BUILTIN_SERVER,
    EnvelopeKind,
    ResultEnvelope,
    ValidationOutcome,
    ValidationRequest,
)
from wheelta_robinhood_agent.agent.mapped_evidence import (  # re-exported
    CandidateEvidence as CandidateEvidence,
)
from wheelta_robinhood_agent.agent.mapped_evidence import (
    EvidenceMapper as EvidenceMapper,
)
from wheelta_robinhood_agent.agent.mapped_evidence import (
    MappedEvidence as MappedEvidence,
)
from wheelta_robinhood_agent.agent.mapped_evidence import (
    MappingRequest as MappingRequest,
)
from wheelta_robinhood_agent.agent.robinhood_mappers import ROBINHOOD_MAPPERS
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    SERVER_NAME as ROBINHOOD_SERVER,
)
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.observability.redaction import REDACTED, Redactor, is_account_key

CANDIDATE_REF_PREFIX: Final = "candidate:"
EVIDENCE_REF_PREFIX: Final = "evidence:"
EVIDENCE_KEY: Final = "evidence"
EVIDENCE_REF_KEY: Final = "evidence_ref"


def evidence_ref_for(tool_call_id: uuid.UUID) -> str:
    """The code-issued evidence reference of one validated tool result."""
    return f"{EVIDENCE_REF_PREFIX}{tool_call_id}"


# ADR-0026: login-scoped workspace reads without a verified mapper reach the model as redacted
# context, never as evidence (no evidence_ref, not citable, cannot support a number).
CONTEXT_ONLY_TOOLS: Final = frozenset((ROBINHOOD_SERVER, tool) for tool in LOGIN_SCOPED_TOOLS)
CONTEXT_ONLY_GAP: Final = "context only: no verified result mapping; not evidence"

# (server, tool) -> mapper. Only tools whose result shapes were captured and verified
# (ADR-0017 fixtures, tests/fixtures/robinhood/results/; robinhood_mappers.py).
VERIFIED_MAPPERS: Mapping[tuple[str, str], EvidenceMapper] = MappingProxyType(
    {(ROBINHOOD_SERVER, tool): mapper for tool, mapper in ROBINHOOD_MAPPERS.items()}
)


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
    # Set by the session from its trusted `get_accounts` check; never from tool output.
    account_eligible: bool = False

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
        if mapper is None and (request.server, request.tool) in CONTEXT_ONLY_TOOLS:
            # ADR-0026: delivered as redacted context with account values dropped. It carries
            # no evidence ref, so it can neither be cited nor back a number or a decision.
            context = _drop_account_values(redacted)
            gaps: tuple[str, ...] = (CONTEXT_ONLY_GAP,)
            if request.tool == RUN_SCAN_TOOL:
                context, scan_gaps = project_scan(context)
                gaps += scan_gaps
            return self._envelope(
                request,
                EnvelopeKind.VALIDATED,
                data={"context_only": True, "payload": context},
                gaps=gaps,
            )
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
                    account_eligible=self.account_eligible,
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
        raw = _drop_account_values(self.redactor.redact(_expand_text_json(request.tool_response)))
        outcome = self._envelope(request, kind, gaps=(gap,))
        return outcome.model_copy(update={"raw_redacted": raw if raw is not None else ""})


_MAX_EXPAND_DEPTH: Final = 32


def _expand_text_json(value: object, depth: int = 0) -> object:
    """Decode JSON carried inside MCP text blocks so redaction sees its keys.

    MCP results arrive as `[{"type": "text", "text": "<json>"}]`; key-based redaction can't see
    an account field inside that string (real-CLI acceptance test 9, DATA_QUALITY.md).
    """
    if depth > _MAX_EXPAND_DEPTH:
        return REDACTED
    if isinstance(value, Mapping):
        out = {str(k): _expand_text_json(v, depth + 1) for k, v in value.items()}
        text = out.get("text")
        if out.get("type") == "text" and isinstance(text, str):
            try:
                decoded = json.loads(text)
            except ValueError:
                return out
            if isinstance(decoded, dict | list):
                out["text"] = _expand_text_json(decoded, depth + 1)
        return out
    if isinstance(value, list | tuple):
        return [_expand_text_json(v, depth + 1) for v in value]
    return value


RUN_SCAN_TOOL: Final = "run_scan"
# ADR-0032: scan rows delivered per call. Well under the proxy's MAX_DELIVERED_CHARS, which
# also has to fit the envelope around the rows.
SCAN_CONTEXT_BUDGET_CHARS: Final = 20_000
# Row columns that repeat a row field (`Symbol` is the row's `ticker`).
_SCAN_DUPLICATE_COLUMNS: Final = frozenset({"Symbol"})


def project_scan(payload: JsonValue) -> tuple[JsonValue, tuple[str, ...]]:
    """`run_scan` context trimmed to fit delivery (ADR-0032). Pure and deterministic.

    Drops the tool's `guide` prose and each row's duplicate `Symbol` column. It keeps the
    scan's metadata and, in the scan's own order, as many rows as fit
    `SCAN_CONTEXT_BUDGET_CHARS`, with a gap naming how many rows were kept. A payload of any
    other shape is returned unchanged; the proxy's size cap still applies.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    result = data.get("result") if isinstance(data, dict) else None
    rows = result.get("results") if isinstance(result, dict) else None
    if not isinstance(result, dict) or not isinstance(rows, list):
        return payload, ()
    trimmed: list[JsonValue] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("columns"), dict):
            columns = cast(dict[str, JsonValue], row["columns"])
            row = {
                **row,
                "columns": {k: v for k, v in columns.items() if k not in _SCAN_DUPLICATE_COLUMNS},
            }
        trimmed.append(row)
    meta = {k: v for k, v in result.items() if k != "results"}
    kept: list[JsonValue] = []
    used = len(json.dumps(meta, sort_keys=True))
    for row in trimmed:
        size = len(json.dumps(row, sort_keys=True)) + 2
        if used + size > SCAN_CONTEXT_BUDGET_CHARS:
            break
        kept.append(row)
        used += size
    projected: JsonValue = {"data": {"result": {**meta, "results": kept}}}
    if len(kept) == len(rows):
        return projected, ()
    gap = (
        f"run_scan: showing the first {len(kept)} of {len(rows)} rows in the scan's own order "
        "(delivery size limit)"
    )
    return projected, (gap,)


def _drop_account_values(value: JsonValue) -> JsonValue:
    """Replace every account-keyed scalar outright (not last-4): restricted raw evidence must
    not keep another account's identifier (CLAUDE.md §24), and the configured account is
    already known to trusted code."""
    if isinstance(value, dict):
        return {
            k: REDACTED
            if is_account_key(k) and not isinstance(v, dict | list)
            else _drop_account_values(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_drop_account_values(v) for v in value]
    return value


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
