"""MignonReport v1: the typed hand-back of one research Mignon (ADR-0025, INTERFACES.md).

A Mignon's final response must contain one JSON object of this shape; prose or a code fence
around it is dropped (ADR-0043). It carries claims, the code-issued references that support
them, fetched web pages, gaps, and questions; never a number of its own that a reference does
not back. The PostToolUse hook parses it strictly
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

from pydantic import Field, StrictStr, StringConstraints, ValidationError, model_validator

from wheelta_robinhood_agent.domain.base import DomainModel, require_unique
from wheelta_robinhood_agent.domain.decision_output import (
    ParseIssue,
    extract_json_object,
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


def extract_report_object(raw: str | bytes) -> str | bytes:
    """The report's JSON object text from a Mignon's final response (ADR-0043): the last
    top-level JSON object with a `findings` key, else the last one (`extract_json_object`)."""
    return extract_json_object(raw, "findings")


def parse_mignon_report(raw: str | bytes) -> MignonReportParseResult:
    """Parse a Mignon's final text into MignonReport v1. Never raises.

    ADR-0043: the report object is located with `extract_report_object`, so prose or a code
    fence around it is allowed. The object itself is parsed under the same strict JSON rules as
    AgentDecisionOutput, and every schema and citation rule still applies.
    """
    loaded = load_strict_json(extract_report_object(raw))
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
    issues: list[ParseIssue] = []
    for index, finding in enumerate(report.findings):
        issues.extend(
            ParseIssue(
                loc=f"findings.{index}.refs",
                message=f"ref not delivered to this Mignon: {r}",
                kind="unknown_ref",
            )
            for r in sorted(set(finding.refs) - refs)
        )
    for index, finding in enumerate(report.findings):
        issues.extend(
            ParseIssue(
                loc=f"findings.{index}.web_urls",
                message=f"URL not fetched by this Mignon: {u}",
                kind="unknown_url",
            )
            for u in sorted(set(finding.web_urls) - urls)
        )
    return tuple(issues)


# -- ADR-0047: repairing a report by patch ---------------------------------------------------

MAX_PATCHES: Final = 50


class FindingPatch(DomainModel):
    """A correction to one finding of the original report, by its index there.

    Given fields replace the finding's; omitted fields are kept. `drop` removes the finding.
    """

    finding: Annotated[int, Field(strict=False, ge=0)]
    claim: Text | None = None
    refs: tuple[SourceRef, ...] | None = None
    web_urls: tuple[WebUrl, ...] | None = None
    drop: bool = False

    @model_validator(mode="after")
    def _check_change(self) -> Self:
        edits = (self.claim, self.refs, self.web_urls)
        if self.drop and any(e is not None for e in edits):
            raise ValueError("a dropped finding takes no other field")
        if not self.drop and all(e is None for e in edits):
            raise ValueError("a patch must change claim, refs, or web_urls, or drop")
        return self


class ReportPatch(DomainModel):
    """A Mignon's reply to repair feedback: only the findings it corrects (ADR-0047)."""

    patches: Annotated[tuple[FindingPatch, ...], Field(min_length=1, max_length=MAX_PATCHES)]

    @model_validator(mode="after")
    def _one_patch_per_finding(self) -> Self:
        require_unique(tuple(p.finding for p in self.patches), "finding")
        return self


def _finding_index(value: object) -> int:
    """JSON numbers are rejected by the strict loader, so indexes arrive as digit strings."""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError('finding must be a digit string such as "1"')


def parse_report_patch(raw: str | bytes) -> ReportPatch | tuple[ParseIssue, ...]:
    """Parse a Mignon's patch reply (prose around the object is dropped). Never raises."""
    loaded = load_strict_json(extract_json_object(raw, "patches"))
    if isinstance(loaded, ParseIssue):
        return (loaded,)
    _, data = loaded
    if not isinstance(data, dict) or not isinstance(data.get("patches"), list):
        return (ParseIssue(loc="", message='expected {"patches": [...]}', kind="not_patch"),)
    try:
        items = [
            {**p, "finding": _finding_index(p.get("finding"))} if isinstance(p, dict) else p
            for p in data["patches"]
        ]
        return ReportPatch.model_validate({**data, "patches": items})
    except ValidationError as exc:
        return validation_issues(exc)
    except ValueError as exc:
        return (ParseIssue(loc="patches.finding", message=str(exc), kind="bad_index"),)


@dataclass(frozen=True, slots=True)
class PatchedReport:
    """The original report with patches applied, and where each finding came from.

    `origins[i]` is the original index of merged finding `i`; `patched` and `dropped` name
    original indexes, so every merged finding maps back to the original text.
    """

    data: dict[str, object]
    origins: tuple[int, ...]
    patched: tuple[int, ...]
    dropped: tuple[int, ...]


def apply_report_patch(
    base: PatchedReport, patch: ReportPatch
) -> PatchedReport | tuple[ParseIssue, ...]:
    """Apply `patch` to `base`. Patch indexes name findings of the ORIGINAL report, so repeated
    repairs keep one numbering. A patch for a dropped or unknown finding is an issue."""
    position = {origin: i for i, origin in enumerate(base.origins)}
    findings = base.data.get("findings")
    if not isinstance(findings, list):
        return (ParseIssue(loc="findings", message="the report has no findings list", kind="bad"),)
    issues = [
        ParseIssue(
            loc=f"patches.{n}.finding",
            message=f"no finding {p.finding} in the original report",
            kind="bad_index",
        )
        for n, p in enumerate(patch.patches)
        if p.finding not in position
    ]
    if issues:
        return tuple(issues)
    merged = list(findings)
    dropped = set(base.dropped)
    patched = set(base.patched)
    for p in patch.patches:
        i = position[p.finding]
        if p.drop:
            dropped.add(p.finding)
            continue
        current = merged[i] if isinstance(merged[i], dict) else {}
        updated = dict(current)
        if p.claim is not None:
            updated["claim"] = p.claim
        if p.refs is not None:
            updated["refs"] = list(p.refs)
        if p.web_urls is not None:
            updated["web_urls"] = list(p.web_urls)
        merged[i] = updated
        patched.add(p.finding)
    keep = [i for i, origin in enumerate(base.origins) if origin not in dropped]
    return PatchedReport(
        data={**base.data, "findings": [merged[i] for i in keep]},
        origins=tuple(base.origins[i] for i in keep),
        patched=tuple(sorted(patched - dropped)),
        dropped=tuple(sorted(dropped)),
    )


def original_report(data: dict[str, object]) -> PatchedReport:
    """An unpatched report: every finding maps to itself."""
    findings = data.get("findings")
    count = len(findings) if isinstance(findings, list) else 0
    return PatchedReport(data=data, origins=tuple(range(count)), patched=(), dropped=())


def report_issues(
    data: dict[str, object], delivered_refs: Iterable[str], fetched_urls: Iterable[str]
) -> tuple[ParseIssue, ...]:
    """Schema, citation, and source issues of a report object, with `findings.<i>` locs."""
    try:
        report = MignonReport.model_validate(data)
    except ValidationError as exc:
        return validation_issues(exc)
    except RecursionError as exc:
        return (ParseIssue(loc="", message=str(exc), kind="too_deep"),)
    return check_report_sources(report, delivered_refs, fetched_urls)
