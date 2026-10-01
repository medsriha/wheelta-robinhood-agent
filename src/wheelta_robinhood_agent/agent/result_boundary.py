"""The result boundary: what the model receives instead of a raw tool result.

docs/DATA_QUALITY.md "Delivery and failure contract"; CLAUDE.md §2.3-2.4, §8, §12.
`BoundaryValidator` is the `ResultValidator` injected into the PostToolUse hook:

- **Local tools** (`wra_local`): our own code produced the JSON text, so it is parsed and
  delivered as `validated` data. Unparseable output becomes an `error` envelope.
- **Built-ins**: the redacted payload is recorded as `validated` data (no built-in web tool
  is enabled since ADR-0058; `Agent` hand-backs are validated in hooks.py).
- **Tavily** (ADR-0058, `integrations/websearch/results.py`): only a search/extract payload
  is delivered, normalized and labelled untrusted web content, with no evidence ref. Anything
  else (a cap notice, a changed shape) is `missing` with a fixed gap, and a tool error never
  carries Tavily's own text: web-facing text is untrusted (CLAUDE.md §11, §24).
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
    CANDIDATE_REF_PREFIX as CANDIDATE_REF_PREFIX,
)
from wheelta_robinhood_agent.agent.mapped_evidence import (
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
from wheelta_robinhood_agent.agent.model_view import (  # re-exported
    EVIDENCE_KEY as EVIDENCE_KEY,
)
from wheelta_robinhood_agent.agent.model_view import (
    EVIDENCE_REF_KEY as EVIDENCE_REF_KEY,
)
from wheelta_robinhood_agent.agent.model_view import (
    EVIDENCE_VIEW_KEY,
    flatten,
    tabulate,
)
from wheelta_robinhood_agent.agent.robinhood_mappers import ROBINHOOD_MAPPERS
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.agent.wheelta_mappers import WHEELTA_MAPPERS
from wheelta_robinhood_agent.domain.enums import CandidateOrigin
from wheelta_robinhood_agent.domain.facts_compute import BoardScreen
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    SERVER_NAME as ROBINHOOD_SERVER,
)
from wheelta_robinhood_agent.integrations.websearch.registry import (
    SERVER_NAME as TAVILY_SERVER,
)
from wheelta_robinhood_agent.integrations.websearch.results import (
    TavilyResultError,
    normalize_result,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import (
    SERVER_NAME as WHEELTA_SERVER,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.observability.redaction import REDACTED, Redactor, is_account_key

EVIDENCE_REF_PREFIX: Final = "evidence:"
# Characters of a tool's own error message kept in the error envelope's gap.
MAX_TOOL_ERROR_CHARS: Final = 500


def evidence_ref_for(tool_call_id: uuid.UUID) -> str:
    """The code-issued evidence reference of one validated tool result."""
    return f"{EVIDENCE_REF_PREFIX}{tool_call_id}"


PREVIEW_SCAN_TOOL: Final = "preview_scan"
# ADR-0026: login-scoped workspace reads without a verified mapper reach the model as redacted
# context, never as evidence (no evidence_ref, not citable, cannot support a number).
# ADR-0041: every Wheelta tool but the board query is delivered the same way (read-only
# research, macro, calendar, quotes, board detail and status).
CONTEXT_ONLY_TOOLS: Final = frozenset(
    {
        *((ROBINHOOD_SERVER, tool) for tool in LOGIN_SCOPED_TOOLS),
        # ADR-0041: the Robinhood scanner fallback; market data, projected like run_scan.
        (ROBINHOOD_SERVER, PREVIEW_SCAN_TOOL),
        *(
            (WHEELTA_SERVER, t.name)
            for t in WHEELTA_REGISTRY.tools
            if t.name not in WHEELTA_MAPPERS
        ),
    }
)
CONTEXT_ONLY_GAP: Final = "context only: no verified result mapping; not evidence"
TAVILY_ERROR_GAP: Final = (
    "the web tool returned an error (its text is not delivered); try other arguments or "
    "another source, and report what is missing as a gap"
)

# (server, tool) -> mapper. Only tools whose result shapes were captured and verified
# (ADR-0017 fixtures, tests/fixtures/robinhood/results/; robinhood_mappers.py).
VERIFIED_MAPPERS: Mapping[tuple[str, str], EvidenceMapper] = MappingProxyType(
    {
        **{(ROBINHOOD_SERVER, tool): mapper for tool, mapper in ROBINHOOD_MAPPERS.items()},
        **{(WHEELTA_SERVER, tool): mapper for tool, mapper in WHEELTA_MAPPERS.items()},
    }
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
    # ADR-0041: this run's current-build Wheelta board rows by OCC symbol (read from the
    # ledger by the session). A candidate whose contract is listed there has origin `board`.
    board_screens: Callable[[], Mapping[str, BoardScreen]] | None = None

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
            gap = (
                TAVILY_ERROR_GAP
                if request.server == TAVILY_SERVER
                else self._tool_error_gap(payload)
            )
            return self._invalid(request, EnvelopeKind.ERROR, gap)
        redacted = self.redactor.redact(payload)
        if request.server == LOCAL_SERVER_NAME:
            return self._envelope(request, EnvelopeKind.VALIDATED, data=redacted)
        if request.server == TAVILY_SERVER:
            try:
                web, web_gaps = normalize_result(request.tool, redacted)
            except TavilyResultError as exc:
                return self._invalid(request, EnvelopeKind.MISSING, str(exc))
            return self._envelope(request, EnvelopeKind.VALIDATED, data=web, gaps=web_gaps)
        mapper = self.mappers.get((request.server, request.tool))
        if mapper is None and (request.server, request.tool) in CONTEXT_ONLY_TOOLS:
            # ADR-0026: delivered as redacted context with account values dropped. It carries
            # no evidence ref, so it can neither be cited nor back a number or a decision.
            context = _drop_account_values(redacted)
            gaps: tuple[str, ...] = (CONTEXT_ONLY_GAP,)
            if request.tool in (RUN_SCAN_TOOL, PREVIEW_SCAN_TOOL):
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
            evidence = self._board_origin(evidence)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            gap = f"result failed schema validation ({type(exc).__name__})"
            return self._invalid(request, EnvelopeKind.MISSING, gap)
        data: dict[str, JsonValue] = {
            EVIDENCE_REF_KEY: evidence_ref_for(request.tool_call_id),
            EVIDENCE_KEY: evidence.model_dump(mode="json"),
        }
        return self._envelope(request, EnvelopeKind.VALIDATED, data=data, gaps=evidence.gaps)

    def _tool_error_gap(self, message: JsonValue) -> str:
        """The gap for a tool error, carrying the tool's own (redacted, truncated) message so
        the agent can correct its input (e.g. Wheelta's 45-day calendar window). The message
        is data, like any tool output; it supplies no fact."""
        text = " ".join(str(message).split()) if isinstance(message, str) else ""
        text = self.redactor.redact_text(text)
        if not text:
            return "the tool returned an error"
        if len(text) > MAX_TOOL_ERROR_CHARS:
            text = text[:MAX_TOOL_ERROR_CHARS] + "…"
        return f"the tool returned an error: {text}"

    def _board_origin(self, evidence: MappedEvidence) -> MappedEvidence:
        """Relabel candidates the run's current board lists (selection.board_comparison: a
        board-derived candidate must be compared, never treated as independently found)."""
        if not evidence.candidates or self.board_screens is None:
            return evidence
        listed = self.board_screens()
        if not listed:
            return evidence
        relabeled = tuple(
            c.model_copy(update={"origin": CandidateOrigin.BOARD})
            if str(c.occ_symbol) in listed
            else c
            for c in evidence.candidates
        )
        return evidence.model_copy(update={"candidates": relabeled})

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
    """`run_scan` context trimmed to fit delivery (ADR-0032, ADR-0037). Pure, deterministic.

    Drops the tool's `guide` prose and each row's duplicate `Symbol` column. It keeps the
    scan's metadata and, in the scan's own order, as many rows as fit
    `SCAN_CONTEXT_BUDGET_CHARS`, with a gap naming how many rows were kept. Rows of one shape
    are delivered as a table (`model_view.tabulate`: `common`, `columns`, `rows`; a row's
    scan columns become `columns.<name>`); rows of mixed shapes stay objects. A payload of
    any other shape is returned unchanged; the proxy's size cap still applies.
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
    table = tabulate(trimmed) if len(trimmed) >= 2 else None
    used = len(json.dumps(meta, sort_keys=True))
    if isinstance(table, dict):
        # A table row is at most its flattened values: `common` only removes values.
        used += len(json.dumps(table["columns"])) + len('"common":{},"columns":,"rows":[]')
        sizes = [
            len(json.dumps(list(flatten(cast(dict[str, JsonValue], r)).values()))) for r in trimmed
        ]
    else:
        sizes = [len(json.dumps(r, sort_keys=True)) for r in trimmed]
    kept: list[JsonValue] = []
    for row, size in zip(trimmed, sizes, strict=True):
        if used + size + 2 > SCAN_CONTEXT_BUDGET_CHARS:
            break
        kept.append(row)
        used += size + 2
    results: JsonValue = tabulate(kept) if isinstance(table, dict) and len(kept) >= 2 else kept
    projected: JsonValue = {"data": {"result": {**meta, "results": results}}}
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
    """The typed evidence inside a stored validated envelope, or None if it carries none.

    A model view (`model_view`) is not typed evidence and yields None; resolve it through
    the validated envelope it was built from.
    """
    if envelope.get("kind") != EnvelopeKind.VALIDATED.value:
        return None
    data = envelope.get("data")
    if not isinstance(data, dict) or EVIDENCE_KEY not in data or EVIDENCE_VIEW_KEY in data:
        return None
    return MappedEvidence.model_validate(data[EVIDENCE_KEY])
