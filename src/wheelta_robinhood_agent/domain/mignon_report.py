"""MignonReport v1: the typed hand-back of one research Mignon (ADR-0025, INTERFACES.md).

A Mignon's final response must be exactly one JSON object of this shape. It carries claims,
the code-issued references that support them, fetched web pages, gaps, and questions; never
a number of its own that a reference does not back. The PostToolUse hook parses it strictly
(`parse_mignon_report`) and then resolves its references against what that Mignon was
actually delivered this run (`check_report_sources`). Only a report that passes both reaches
the orchestrator; anything else is replaced by an invalid-report envelope.

Rules (each checked here, in pure code):

- extra fields are rejected on every object; JSON numbers are rejected anywhere;
- every finding cites at least one source: a reference or a fetched URL;
- a claim containing a digit needs at least one reference (web pages never supply prices,
  strikes, premiums, Greeks, positions, or dates of record: CLAUDE.md §11);
- references are `evidence:` or `candidate:` strings issued by code; a cited reference or
  URL that was not delivered to this Mignon is an issue.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Self

from pydantic import StrictStr, StringConstraints, ValidationError, model_validator

from wheelta_robinhood_agent.domain.base import DomainModel, require_unique
from wheelta_robinhood_agent.domain.decision_output import (
    ParseIssue,
    load_strict_json,
    validation_issues,
)

SCHEMA_VERSION: Final = 1
REF_PREFIXES: Final = ("evidence:", "candidate:")

Text = Annotated[StrictStr, StringConstraints(min_length=1, max_length=2000)]
SourceRef = Annotated[StrictStr, StringConstraints(pattern=r"^(evidence|candidate):\S+$")]
WebUrl = Annotated[StrictStr, StringConstraints(pattern=r"^https://\S+$", max_length=2000)]


class Finding(DomainModel):
    """One claim and the sources that support it."""

    claim: Text
    refs: tuple[SourceRef, ...]
    web_urls: tuple[WebUrl, ...]

    @model_validator(mode="after")
    def _check_sources(self) -> Self:
        require_unique(self.refs, "ref")
        require_unique(self.web_urls, "web_url")
        if not self.refs and not self.web_urls:
            raise ValueError("a finding must cite at least one ref or web_url")
        if not self.refs and any(ch.isdigit() for ch in self.claim):
            raise ValueError("a claim containing a number must cite a code-issued ref")
        return self


class MignonReport(DomainModel):
    """Top-level MignonReport v1."""

    task: Text
    findings: tuple[Finding, ...]
    gaps: tuple[Text, ...]
    follow_up_questions: tuple[Text, ...]

    def cited_refs(self) -> frozenset[str]:
        return frozenset(r for f in self.findings for r in f.refs)

    def cited_urls(self) -> frozenset[str]:
        return frozenset(u for f in self.findings for u in f.web_urls)


@dataclass(frozen=True, slots=True)
class MignonReportParsed:
    ok: Literal[True]
    report: MignonReport
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MignonReportParseFailure:
    ok: Literal[False]
    issues: tuple[ParseIssue, ...]
    schema_version: int = SCHEMA_VERSION


MignonReportParseResult = MignonReportParsed | MignonReportParseFailure


def parse_mignon_report(raw: str | bytes) -> MignonReportParseResult:
    """Strictly parse a Mignon's final text into MignonReport v1. Never raises.

    The text must be exactly one JSON object (surrounding whitespace allowed; no code fences
    or prose), under the same strict JSON rules as AgentDecisionOutput.
    """
    loaded = load_strict_json(raw)
    if isinstance(loaded, ParseIssue):
        return MignonReportParseFailure(ok=False, issues=(loaded,))
    _, data = loaded
    if not isinstance(data, dict):
        issue = ParseIssue(loc="", message="top level must be a JSON object", kind="not_object")
        return MignonReportParseFailure(ok=False, issues=(issue,))
    try:
        report = MignonReport.model_validate(data)
    except ValidationError as exc:
        return MignonReportParseFailure(ok=False, issues=validation_issues(exc))
    except RecursionError as exc:
        issue = ParseIssue(loc="", message=str(exc), kind="too_deep")
        return MignonReportParseFailure(ok=False, issues=(issue,))
    return MignonReportParsed(ok=True, report=report)


def check_report_sources(
    report: MignonReport, delivered_refs: Iterable[str], fetched_urls: Iterable[str]
) -> tuple[ParseIssue, ...]:
    """Issues for every cited ref or URL that was not delivered to the reporting Mignon.

    `delivered_refs` are the code-issued references in results delivered to that Mignon this
    run; `fetched_urls` the URLs of its successful WebFetch calls. Sorted, so deterministic.
    """
    refs, urls = frozenset(delivered_refs), frozenset(fetched_urls)
    issues = [
        ParseIssue(
            loc="findings.refs",
            message=f"ref not delivered to this Mignon: {r}",
            kind="unknown_ref",
        )
        for r in sorted(report.cited_refs() - refs)
    ]
    issues.extend(
        ParseIssue(
            loc="findings.web_urls",
            message=f"URL not fetched by this Mignon: {u}",
            kind="unknown_url",
        )
        for u in sorted(report.cited_urls() - urls)
    )
    return tuple(issues)
