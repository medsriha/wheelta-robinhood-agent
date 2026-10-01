"""MignonBrief v1: the typed task the orchestrator gives a research Mignon (ADR-0061).

The orchestrator states *what* it needs; the Mignon decides *how* (its prompt carries its
tools' know-how). The brief is the `prompt` input of the `Agent` call: one strict JSON object,
checked by the PreToolUse hook before the spawn (`parse_mignon_brief`). A brief that fails is
denied with its issues as feedback, so the orchestrator corrects and spawns again.

Why typed (ADR-0061): free-text tasks restated trading-rule values, and got them wrong (a
2026-10-01 dry run asked for |delta| 0.17-0.28 against the rules' 0.15-0.30, so a Mignon
rejected a passing contract). A brief names rules by key (`criteria`), never by value; the
Mignon reads the values from the rules rendered into its own prompt. Code checks every key.

Fields:

- `objective`: the question, in a sentence or two.
- `subjects`: what to work on: symbols, OCC symbols, or code-issued refs. Empty for a
  discovery task.
- `criteria`: rule keys (`filters`, `filters.min_abs_delta`, `events.earnings_exclusion`)
  the Mignon applies; each must exist in the rules.
- `exclude`: symbols or OCC symbols to leave out (earlier rejections).
- `source`: `board`, `scanner`, or `any` (`selection.sources`, `selection.discovery`).
- `want`: the values to return per subject (snake_case names); code checks the report
  covers them (`coverage_gaps`).
- `max_results`: a digit string; JSON numbers are rejected everywhere in model output.
- `notes`: approach details with no rule value (sort order, sectors, expirations).

Pure: no I/O, no clock.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import Field, StrictStr, StringConstraints, ValidationError, model_validator

from wheelta_robinhood_agent.domain.base import DomainModel, require_unique
from wheelta_robinhood_agent.domain.decision_output import (
    ParseIssue,
    load_strict_json,
    validation_issues,
)

SCHEMA_VERSION: Final = 1
MAX_SUBJECTS: Final = 30
MAX_WANT: Final = 20

Objective = Annotated[StrictStr, StringConstraints(min_length=1, max_length=1000)]
Notes = Annotated[StrictStr, StringConstraints(max_length=1500)]
# A symbol, an OCC symbol (root padded to 6, then yymmdd, C/P, 8-digit strike), or a
# code-issued reference.
Subject = Annotated[
    StrictStr,
    StringConstraints(
        pattern=r"^([A-Z]{1,5}|[A-Z ]{6}\d{6}[CP]\d{8}|(evidence|candidate):\S+)$",
        max_length=200,
    ),
]
RuleKey = Annotated[StrictStr, StringConstraints(pattern=r"^[a-z_]+(\.[a-z_]+)*$", max_length=100)]
FieldName = Annotated[StrictStr, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
MaxResults = Annotated[StrictStr, StringConstraints(pattern=r"^([1-9]|[1-4][0-9]|50)$")]


class BriefSource(StrEnum):
    BOARD = "board"
    SCANNER = "scanner"
    ANY = "any"


class MignonBrief(DomainModel):
    """Top-level MignonBrief v1."""

    objective: Objective
    subjects: Annotated[tuple[Subject, ...], Field(max_length=MAX_SUBJECTS)] = ()
    criteria: tuple[RuleKey, ...] = ()
    exclude: tuple[Subject, ...] = ()
    source: BriefSource = BriefSource.ANY
    want: Annotated[tuple[FieldName, ...], Field(max_length=MAX_WANT)] = ()
    max_results: MaxResults | None = None
    notes: Notes | None = None

    @model_validator(mode="after")
    def _check(self) -> "MignonBrief":
        require_unique(self.subjects, "subject")
        require_unique(self.criteria, "criterion")
        require_unique(self.want, "want field")
        if set(self.subjects) & set(self.exclude):
            raise ValueError("a subject cannot also be excluded")
        return self


@dataclass(frozen=True, slots=True)
class MignonBriefParsed:
    ok: Literal[True]
    brief: MignonBrief


@dataclass(frozen=True, slots=True)
class MignonBriefParseFailure:
    ok: Literal[False]
    issues: tuple[ParseIssue, ...]


MignonBriefParseResult = MignonBriefParsed | MignonBriefParseFailure


def parse_mignon_brief(raw: str, rule_keys: Iterable[str]) -> MignonBriefParseResult:
    """Parse an `Agent` prompt as a MignonBrief. Never raises.

    The whole prompt must be one strict JSON object (no prose: the brief is the task). Every
    `criteria` key must be in `rule_keys` (dotted paths of the loaded rules)."""
    loaded = load_strict_json(raw.strip())
    if isinstance(loaded, ParseIssue):
        issue = ParseIssue(
            loc=loaded.loc,
            message=f"not one strict JSON object, with no other text ({loaded.message})",
            kind=loaded.kind,
        )
        return MignonBriefParseFailure(ok=False, issues=(issue,))
    _, data = loaded
    if not isinstance(data, dict):
        issue = ParseIssue(loc="", message="the brief must be one JSON object", kind="not_object")
        return MignonBriefParseFailure(ok=False, issues=(issue,))
    try:
        brief = MignonBrief.model_validate(data)
    except ValidationError as exc:
        return MignonBriefParseFailure(ok=False, issues=validation_issues(exc))
    except RecursionError as exc:
        issue = ParseIssue(loc="", message=str(exc), kind="too_deep")
        return MignonBriefParseFailure(ok=False, issues=(issue,))
    known = frozenset(rule_keys)
    unknown = tuple(
        ParseIssue(
            loc=f"criteria.{i}",
            message=f"no rule {key!r}: name a rules section or key (e.g. filters.min_abs_delta)",
            kind="unknown_rule",
        )
        for i, key in enumerate(brief.criteria)
        if key not in known
    )
    if unknown:
        return MignonBriefParseFailure(ok=False, issues=unknown)
    return MignonBriefParsed(ok=True, brief=brief)


def coverage_gaps(
    brief: MignonBrief, reported: Iterable[tuple[str | None, Iterable[str]]]
) -> tuple[str, ...]:
    """What the report left out of the brief (ADR-0061), as code-written gaps.

    `reported` is (subject, value names) per finding. Every brief subject needs a finding
    with that subject; every finding with a subject needs a value for each `want` field. A
    gap does not reject the report: it tells the orchestrator what is still unknown.
    Deterministic order: brief subjects first, then findings in report order."""
    rows = [(s, frozenset(v)) for s, v in reported]
    covered = {s for s, _ in rows if s is not None}
    gaps = [f"subject {s}: not reported" for s in brief.subjects if s not in covered]
    seen: set[tuple[str, str]] = set()
    for subject, values in rows:
        if subject is None:
            continue
        for want in brief.want:
            if want not in values and (subject, want) not in seen:
                seen.add((subject, want))
                gaps.append(f"subject {subject}: no value for {want!r}")
    return tuple(gaps)
