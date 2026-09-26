"""AgentDecisionOutput v5 models and a strict parser (OUTPUT_ASSEMBLY.md; ADR-0011).

Field-for-field mirror of `prompts/agent_decision_output.v5.schema.json` (parity is tested).
The model emits choices, rationale, and existing reference selections only. Extra fields are
rejected on every object (so `schema_version`, quantities, statuses, broker IDs, or
`rejected_candidates` fail parsing). The version comes from trusted prompt configuration,
never from the model. `limit_price` must be a JSON decimal string and is parsed to Decimal;
JSON numbers anywhere are rejected.

Context-dependent semantic validation (reference resolution, action/leg compatibility) is
a separate, later step and is not performed here.
"""

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Final, Literal, Self

from pydantic import (
    PlainSerializer,
    PlainValidator,
    StrictStr,
    StringConstraints,
    ValidationError,
    model_validator,
)

from wheelta_robinhood_agent.domain.base import DomainModel, require_unique
from wheelta_robinhood_agent.domain.enums import DecisionAction

SCHEMA_VERSION: Final = 5
LIMIT_PRICE_PATTERN: Final = r"^(0|[1-9][0-9]*)(\.[0-9]+)?$"
_LIMIT_PRICE_RE = re.compile(LIMIT_PRICE_PATTERN)

Reference = Annotated[StrictStr, StringConstraints(min_length=1)]
"""`$defs/reference`: a non-empty string naming a code-issued reference."""

Text = Annotated[StrictStr, StringConstraints(min_length=1)]


def _parse_limit_price(value: object) -> Decimal:
    """A decimal string matching the schema pattern; floats/ints/Decimals are rejected."""
    if not isinstance(value, str):
        raise ValueError("limit_price must be a decimal string")
    if not _LIMIT_PRICE_RE.fullmatch(value):
        raise ValueError("limit_price must match ^(0|[1-9][0-9]*)(\\.[0-9]+)?$")
    return Decimal(value)


LimitPrice = Annotated[
    Decimal,
    PlainValidator(_parse_limit_price),
    PlainSerializer(lambda d: str(d), return_type=str),
]


class ProposedLeg(DomainModel):
    """`$defs/proposed_leg`: a fact set plus a discretionary intended limit price."""

    facts_ref: Reference
    limit_price: LimitPrice


class Decision(DomainModel):
    """`$defs/decision`. All fields required; nullable ones must be present as null."""

    action: DecisionAction
    target_ref: Reference
    replacement_ref: Reference | None
    funding_close_refs: tuple[Reference, ...]
    proposed_legs: tuple[ProposedLeg, ...]
    execution_refs: tuple[Reference, ...]
    rationale: Text
    thesis: Text | None
    invalidation_conditions: tuple[Text, ...]
    evidence_refs: tuple[Reference, ...]

    @model_validator(mode="after")
    def _check_decision_unique(self) -> Self:
        require_unique(self.funding_close_refs, "funding_close_ref")
        require_unique(self.execution_refs, "execution_ref")
        require_unique(self.evidence_refs, "evidence_ref")
        return self


class CancellationRationale(DomainModel):
    """`$defs/cancellation_rationale`."""

    cancel_call_ref: Reference
    rationale: Text
    evidence_refs: tuple[Reference, ...]

    @model_validator(mode="after")
    def _check_cancellation_unique(self) -> Self:
        require_unique(self.evidence_refs, "evidence_ref")
        return self


class ResearchQuestion(DomainModel):
    """`$defs/research_question`."""

    target_ref: Reference | None
    question: Text
    evidence_refs: tuple[Reference, ...]

    @model_validator(mode="after")
    def _check_question_unique(self) -> Self:
        require_unique(self.evidence_refs, "evidence_ref")
        return self


class AgentDecisionOutput(DomainModel):
    """Top-level AgentDecisionOutput v5."""

    decisions: tuple[Decision, ...]
    cancellation_rationales: tuple[CancellationRationale, ...]
    unresolved_questions: tuple[ResearchQuestion, ...]


@dataclass(frozen=True, slots=True)
class ParseIssue:
    """One parse problem: JSON path (`loc`), a message, and a machine-readable kind."""

    loc: str
    message: str
    kind: str


@dataclass(frozen=True, slots=True)
class DecisionOutputParsed:
    ok: Literal[True]
    output: AgentDecisionOutput
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class DecisionOutputParseFailure:
    """Invalid model output. `raw_text` preserves the response (the caller redacts it)."""

    ok: Literal[False]
    raw_text: str
    issues: tuple[ParseIssue, ...]
    schema_version: int = SCHEMA_VERSION


DecisionOutputParseResult = DecisionOutputParsed | DecisionOutputParseFailure


class _RejectedJson(ValueError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _reject_number(text: str) -> object:
    raise _RejectedJson("json_number", f"JSON numbers are not allowed in this schema: {text}")


def _reject_constant(text: str) -> object:
    raise _RejectedJson("json_constant", f"non-standard JSON constant: {text}")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _RejectedJson("duplicate_key", f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _failure(raw_text: str, *issues: ParseIssue) -> DecisionOutputParseFailure:
    return DecisionOutputParseFailure(ok=False, raw_text=raw_text, issues=tuple(issues))


def parse_agent_decision_output(raw: str | bytes) -> DecisionOutputParseResult:
    """Strictly parse the model's final response into AgentDecisionOutput v5.

    Never raises on bad model output: returns `DecisionOutputParseFailure` with the raw text
    and typed issues. The response must be exactly one JSON object (surrounding whitespace
    allowed; no code fences or prose). Duplicate keys, JSON numbers, NaN/Infinity, invalid
    UTF-8, unknown fields, and type mismatches are all failures (INTERFACES.md: forbid extra
    fields recursively; preserve the raw response on failure).
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return _failure(
                raw.decode("utf-8", errors="replace"),
                ParseIssue(loc="", message=str(exc), kind="invalid_utf8"),
            )
    else:
        text = raw
    try:
        data = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_float=_reject_number,
            parse_int=_reject_number,
            parse_constant=_reject_constant,
        )
    except _RejectedJson as exc:
        return _failure(text, ParseIssue(loc="", message=str(exc), kind=exc.kind))
    except (ValueError, RecursionError) as exc:
        return _failure(text, ParseIssue(loc="", message=str(exc), kind="invalid_json"))
    if not isinstance(data, dict):
        return _failure(
            text, ParseIssue(loc="", message="top level must be a JSON object", kind="not_object")
        )
    try:
        output = AgentDecisionOutput.model_validate(data)
    except ValidationError as exc:
        issues = tuple(
            ParseIssue(
                loc=".".join(str(part) for part in err["loc"]),
                message=err["msg"],
                kind=err["type"],
            )
            for err in exc.errors()
        )
        return _failure(text, *issues)
    except RecursionError as exc:
        return _failure(text, ParseIssue(loc="", message=str(exc), kind="too_deep"))
    return DecisionOutputParsed(ok=True, output=output)
