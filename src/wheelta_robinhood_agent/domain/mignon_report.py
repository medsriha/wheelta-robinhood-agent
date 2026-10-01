"""MignonReport v2: the typed hand-back of one research Mignon (ADR-0025, INTERFACES.md).

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
- ADR-0056: a claim containing a digit needs a reference or a fetched URL. A finding whose
  only sources are web pages is `web_sourced` (`web_sourced_indices`); the rules still forbid
  a web page to supply prices, strikes, premiums, Greeks, positions, or buying power
  (CLAUDE.md §11), and orders use only the orchestrator's own quotes and decision facts;
- references are `evidence:` or `candidate:` strings issued by code; a cited reference or
  URL that was not delivered to this Mignon is an issue;
- ADR-0061 (v2): a finding may name the brief `subject` it answers and carry `values`, the
  brief's `want` fields as text (a value with a digit makes a web-only finding web-sourced,
  like its claim). `coverage_gaps` (domain/mignon_brief.py) lists what the report left out.

ADR-0056: `salvage_report` drops only the findings that break these rules (each recorded with
its reasons, never its claim text) and keeps the rest; a problem outside the findings still
rejects the whole report.
"""

import re
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
from wheelta_robinhood_agent.domain.mignon_brief import FieldName, Subject

SCHEMA_VERSION: Final = 2
MAX_VALUES: Final = 20
REF_PREFIXES: Final = ("evidence:", "candidate:")

Text = Annotated[StrictStr, StringConstraints(min_length=1, max_length=2000)]
# Code-issued refs are a prefix plus a UUID-like id, far below this bound (ADR-0056).
SourceRef = Annotated[
    StrictStr, StringConstraints(pattern=r"^(evidence|candidate):\S+$", max_length=200)
]
WebUrl = Annotated[StrictStr, StringConstraints(pattern=r"^https://\S+$", max_length=2000)]
ValueText = Annotated[StrictStr, StringConstraints(min_length=1, max_length=300)]


class Finding(DomainModel):
    """One claim and the sources that support it."""

    claim: Text
    refs: tuple[SourceRef, ...]
    web_urls: tuple[WebUrl, ...]
    # ADR-0061: the brief subject this finding answers, and its requested values.
    subject: Subject | None = None
    values: Annotated[dict[FieldName, ValueText], Field(max_length=MAX_VALUES)] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _check_sources(self) -> Self:
        require_unique(self.refs, "ref")
        require_unique(self.web_urls, "web_url")
        if not self.refs and not self.web_urls:
            raise ValueError("a finding must cite at least one ref or web_url")
        return self

    @property
    def web_sourced(self) -> bool:
        """A claim or value with a number whose only sources are fetched web pages
        (ADR-0056, ADR-0061)."""
        texts = (self.claim, *self.values.values())
        return not self.refs and any(ch.isdigit() for t in texts for ch in t)


class MignonReport(DomainModel):
    """Top-level MignonReport v2."""

    task: Text
    findings: tuple[Finding, ...]
    gaps: tuple[Text, ...]
    follow_up_questions: tuple[Text, ...]

    def cited_refs(self) -> frozenset[str]:
        return frozenset(r for f in self.findings for r in f.refs)

    def cited_urls(self) -> frozenset[str]:
        return frozenset(u for f in self.findings for u in f.web_urls)

    def reported_values(self) -> tuple[tuple[str | None, tuple[str, ...]], ...]:
        """(subject, value names) per finding, for `mignon_brief.coverage_gaps`."""
        return tuple((f.subject, tuple(f.values)) for f in self.findings)


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
    """Parse a Mignon's final text into MignonReport v2. Never raises.

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
    return tuple(
        issue
        for index, finding in enumerate(report.findings)
        for issue in _finding_source_issues(index, finding, refs, urls)
    )


def _finding_source_issues(
    index: int, finding: Finding, refs: frozenset[str], urls: frozenset[str]
) -> list[ParseIssue]:
    """Cited refs not delivered to the Mignon and URLs it did not fetch, for one finding."""
    issues = [
        ParseIssue(
            loc=f"findings.{index}.refs",
            message=f"ref not delivered to this Mignon: {r}",
            kind="unknown_ref",
        )
        for r in sorted(set(finding.refs) - refs)
    ]
    issues.extend(
        ParseIssue(
            loc=f"findings.{index}.web_urls",
            message=f"URL not fetched by this Mignon: {u}",
            kind="unknown_url",
        )
        for u in sorted(set(finding.web_urls) - urls)
    )
    return issues


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
    subject: Subject | None = None
    values: Annotated[dict[FieldName, ValueText], Field(max_length=MAX_VALUES)] | None = None
    drop: bool = False

    @model_validator(mode="after")
    def _check_change(self) -> Self:
        edits = (self.claim, self.refs, self.web_urls, self.subject, self.values)
        if self.drop and any(e is not None for e in edits):
            raise ValueError("a dropped finding takes no other field")
        if not self.drop and all(e is None for e in edits):
            raise ValueError("a patch must change a finding field or drop the finding")
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
        # ADR-0056: only the finding fields survive a patch, so a stray key the Mignon added
        # cannot keep the finding invalid through every round.
        updated = {k: current[k] for k in _FINDING_FIELDS if k in current}
        if p.claim is not None:
            updated["claim"] = p.claim
        if p.refs is not None:
            updated["refs"] = list(p.refs)
        if p.web_urls is not None:
            updated["web_urls"] = list(p.web_urls)
        if p.subject is not None:
            updated["subject"] = p.subject
        if p.values is not None:
            updated["values"] = dict(p.values)
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
    """Schema, citation, and source issues of a report object, with `findings.<i>` locs.

    ADR-0056: each finding is checked on its own (schema and sources together), so every
    finding's problems are found in one pass rather than hidden behind another finding's
    schema error. Issues outside the findings keep their own locs.
    """
    refs, urls = frozenset(delivered_refs), frozenset(fetched_urls)
    findings = data.get("findings")
    issues: list[ParseIssue] = []
    top = data
    if isinstance(findings, list):
        top = {**data, "findings": []}
        for index, item in enumerate(findings):
            try:
                finding = Finding.model_validate(item)
            except ValidationError as exc:
                issues.extend(
                    ParseIssue(
                        loc=f"findings.{index}" + (f".{i.loc}" if i.loc else ""),
                        message=i.message,
                        kind=i.kind,
                    )
                    for i in validation_issues(exc)
                )
            except RecursionError as exc:
                issues.append(
                    ParseIssue(loc=f"findings.{index}", message=str(exc), kind="too_deep")
                )
            else:
                issues.extend(_finding_source_issues(index, finding, refs, urls))
    try:
        MignonReport.model_validate(top)
    except ValidationError as exc:
        issues.extend(validation_issues(exc))
    except RecursionError as exc:
        issues.append(ParseIssue(loc="", message=str(exc), kind="too_deep"))
    return tuple(issues)


# -- ADR-0056: keep the valid part of a report -------------------------------------------------

_FINDING_AT: Final = re.compile(r"^findings\.(\d+)(?:\.|$)")


_FINDING_FIELDS: Final = ("claim", "refs", "web_urls", "subject", "values")
_SAFE_REASONS: Final = {
    "unknown_ref": "a cited ref was not delivered to this Mignon",
    "unknown_url": "a cited URL was not fetched by this Mignon (or returned an HTTP error)",
    "extra_forbidden": (
        "the finding has a field other than claim, refs, web_urls, subject, and values"
    ),
}
_ECHO: Final = re.compile(r"^(.*?\bduplicate [a-z_ ]+?): .*$", re.DOTALL)
MAX_REASON_CHARS: Final = 200


def _safe_reason(issue: ParseIssue) -> str:
    """A dropped finding's reason without any text the Mignon wrote (refs, URLs, values): the
    reason travels to the orchestrator, the claim and its citations do not (ADR-0056)."""
    if issue.kind in _SAFE_REASONS:
        return _SAFE_REASONS[issue.kind]
    reason = _ECHO.sub(r"\1", issue.message)
    return reason[:MAX_REASON_CHARS]


@dataclass(frozen=True, slots=True)
class DroppedFinding:
    """A finding code removed from a report: its index in the report the Mignon sent (the
    original, before any patch) and why. The claim text is not kept: it was not supported."""

    index: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SalvagedReport:
    """The report with only valid findings, or None with the problems outside any finding."""

    report: MignonReport | None
    fatal: tuple[ParseIssue, ...]
    dropped: tuple[DroppedFinding, ...]
    origins: tuple[int, ...]  # original index of each kept finding


def salvage_report(
    data: dict[str, object],
    delivered_refs: Iterable[str],
    fetched_urls: Iterable[str],
    origins: tuple[int, ...] | None = None,
) -> SalvagedReport:
    """Drop every finding that fails the schema, citation, or source rules; keep the rest.

    `origins` maps each finding of `data` to its index in the original report (a patched
    report's `PatchedReport.origins`); identity by default. Any issue not located at one finding
    (a missing field, a bad top level) is fatal: no report. Pure and deterministic.
    """
    refs, urls = frozenset(delivered_refs), frozenset(fetched_urls)
    findings = data.get("findings")
    if not isinstance(findings, list):
        return SalvagedReport(None, report_issues(data, refs, urls), (), ())
    current = list(findings)
    kept = list(origins) if origins is not None else list(range(len(current)))
    if len(kept) != len(current):
        raise ValueError("origins must name every finding")
    dropped: list[DroppedFinding] = []
    # Each pass removes at least one finding, so this ends after len(findings) + 1 passes.
    for _ in range(len(current) + 1):
        issues = report_issues({**data, "findings": current}, refs, urls)
        if not issues:
            report = MignonReport.model_validate({**data, "findings": current})
            return SalvagedReport(report, (), tuple(dropped), tuple(kept))
        located: dict[int, list[ParseIssue]] = {}
        fatal: list[ParseIssue] = []
        for issue in issues:
            match = _FINDING_AT.match(issue.loc)
            if match is None or int(match.group(1)) >= len(current):
                fatal.append(issue)
            else:
                located.setdefault(int(match.group(1)), []).append(issue)
        if fatal:
            return SalvagedReport(None, tuple(fatal), tuple(dropped), ())
        dropped.extend(
            DroppedFinding(
                index=kept[i],
                reasons=tuple(dict.fromkeys(_safe_reason(issue) for issue in located[i])),
            )
            for i in sorted(located)
        )
        current = [f for i, f in enumerate(current) if i not in located]
        kept = [o for i, o in enumerate(kept) if i not in located]
    raise AssertionError("unreachable: every pass drops a finding")  # pragma: no cover


def web_sourced_indices(report: MignonReport) -> tuple[int, ...]:
    """Indexes of findings whose number rests only on fetched web pages (ADR-0056)."""
    return tuple(i for i, f in enumerate(report.findings) if f.web_sourced)
