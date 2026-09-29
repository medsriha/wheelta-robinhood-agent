"""MignonReport v1 strict parser and source check (ADR-0025, domain/mignon_report.py)."""

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wheelta_robinhood_agent.domain.mignon_report import (
    MignonReportParsed,
    PatchedReport,
    ReportPatch,
    apply_report_patch,
    check_report_sources,
    extract_report_object,
    original_report,
    parse_mignon_report,
    parse_report_patch,
    report_issues,
)


def doc(*findings: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {"task": "t", "findings": list(findings), "gaps": [], "follow_up_questions": [], **top}


def f(claim: str, refs: tuple[str, ...] = (), urls: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"claim": claim, "refs": list(refs), "web_urls": list(urls)}


def parsed(value: Any) -> MignonReportParsed:
    result = parse_mignon_report(json.dumps(value))
    assert isinstance(result, MignonReportParsed), result
    return result


def failure_kinds(raw: str) -> list[str]:
    result = parse_mignon_report(raw)
    assert not result.ok
    return [i.kind for i in result.issues]


def test_valid_report_parses() -> None:
    r = parsed(
        doc(
            f("Bid is 1.20.", ("evidence:e-1", "candidate:c-1")),
            f("Guidance held.", urls=("https://ir.example.com/q3",)),
        )
    )
    assert r.report.cited_refs() == {"evidence:e-1", "candidate:c-1"}
    assert r.report.cited_urls() == {"https://ir.example.com/q3"}


def test_empty_findings_with_gaps_is_valid() -> None:
    assert parsed(doc(gaps=["No chain returned."])).report.findings == ()


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("No report here.", "invalid_json"),
        ("Prose {not json} only.", "invalid_json"),
        ("```json\n{}\n```", "missing"),  # the fence is removed; the schema still applies
        ("[]", "not_object"),
        ('{"task": "t", "task": "u"}', "duplicate_key"),
        (json.dumps(doc(n=1)), "json_number"),
        (json.dumps(doc(extra="x")), "extra_forbidden"),
        (json.dumps(doc(f("x", ("facts:f-1",)))), "string_pattern_mismatch"),
        (json.dumps(doc(f("x", urls=("http://insecure.example.com",)))), "string_pattern_mismatch"),
    ],
)
def test_malformed_reports_fail(raw: str, kind: str) -> None:
    assert kind in failure_kinds(raw)


@pytest.mark.parametrize(
    "finding",
    [
        f("No source at all."),
        f("The bid is 1.20.", urls=("https://news.example.com/a",)),
        f("Dup.", ("evidence:e-1", "evidence:e-1")),
    ],
)
def test_finding_source_rules(finding: dict[str, Any]) -> None:
    assert "value_error" in failure_kinds(json.dumps(doc(finding)))


def test_check_sources_names_every_undelivered_citation() -> None:
    r = parsed(
        doc(f("Bid 1.2.", ("evidence:a", "evidence:b")), f("Page.", urls=("https://x.example/p",)))
    )
    issues = check_report_sources(r.report, {"evidence:a"}, set())
    assert [i.kind for i in issues] == ["unknown_ref", "unknown_url"]
    assert (
        check_report_sources(r.report, {"evidence:a", "evidence:b"}, {"https://x.example/p"}) == ()
    )


@given(st.text())
def test_parser_never_raises(raw: str) -> None:
    parse_mignon_report(raw)


@given(
    st.recursive(
        st.none() | st.booleans() | st.text(), lambda c: st.lists(c) | st.dictionaries(st.text(), c)
    )
)
def test_parser_never_raises_on_json(value: Any) -> None:
    parse_mignon_report(json.dumps(value))


# -- ADR-0032: one enclosing code fence is removed before the strict parse ----------------------


@pytest.mark.parametrize("fence", ["```json", "```", "```json  "])
def test_one_enclosing_fence_is_accepted(fence: str) -> None:
    raw = f"  {fence}\n{json.dumps(doc(f('Bid is 1.20.', ('evidence:e-1',))), indent=2)}\n```\n"
    result = parse_mignon_report(raw)
    assert isinstance(result, MignonReportParsed), result
    assert result.report.cited_refs() == {"evidence:e-1"}


def test_a_fence_inside_a_claim_is_left_alone() -> None:
    report = doc(f("Scan note: ```x```", ("evidence:e-1",)))
    assert parsed(report).report.findings[0].claim == "Scan note: ```x```"


# -- ADR-0043: prose or a code fence around the report is dropped --------------------------------


@pytest.mark.parametrize(
    "wrap",
    [
        lambda body: "Here is my report.\n" + body,
        lambda body: body + "\nDone.",
        lambda body: "I have everything needed. Compiling the final report.\n\n" + body,
        lambda body: "```json\n" + body + "\n```\n\nSources:\n- [BLS](https://www.bls.gov/)",
        lambda body: "```json\n```json\n" + body + "\n```\n```",
    ],
)
def test_prose_or_fence_around_the_report_is_accepted(wrap: Any) -> None:
    body = json.dumps(doc(f("Bid is 1.20.", ("evidence:e-1",))), indent=2)
    result = parse_mignon_report(wrap(body))
    assert isinstance(result, MignonReportParsed), result
    assert result.report.cited_refs() == {"evidence:e-1"}


def test_the_last_object_with_findings_is_the_report() -> None:
    example = json.dumps({"example": "shape"})
    first = json.dumps(doc(task="draft"))
    final = json.dumps(doc(task="final"))
    raw = f'Shape: {example}. Draft: {first}\nFinal: {final}\nNote {{braces}} and {{"a": 1}}.'
    assert parsed_raw(raw).report.task == "final"


def test_without_findings_the_last_object_is_parsed() -> None:
    assert failure_kinds('Notes. {"task": "t"}') == ["missing", "missing", "missing"]


def test_strict_rules_still_apply_to_the_extracted_report() -> None:
    assert failure_kinds("Report:\n" + json.dumps(doc(n=1))) == ["json_number"]
    assert failure_kinds("Report:\n" + json.dumps(doc(extra="x"))) == ["extra_forbidden"]
    uncited = json.dumps(doc(f("Last trade 22.23.", urls=("https://example.com/q",))))
    assert failure_kinds("Report:\n" + uncited) == ["value_error"]


def test_extract_leaves_text_without_an_object_and_bytes_unchanged() -> None:
    assert extract_report_object("no object [1, 2]") == "no object [1, 2]"
    assert extract_report_object(b"{}") == b"{}"


def parsed_raw(raw: str) -> MignonReportParsed:
    result = parse_mignon_report(raw)
    assert isinstance(result, MignonReportParsed), result
    return result


# -- ADR-0047: patches ---------------------------------------------------------------------------


def _base(*findings: dict[str, Any]) -> PatchedReport:
    return original_report(doc(*findings))


def _patch(*patches: dict[str, Any]) -> ReportPatch:
    result = parse_report_patch(json.dumps({"patches": list(patches)}))
    assert isinstance(result, ReportPatch), result
    return result


def test_patch_replaces_only_given_fields_and_drops_by_original_index() -> None:
    base = _base(f("A.", ("evidence:a",)), f("B 1.2."), f("C.", urls=("https://x.example/c",)))
    merged = apply_report_patch(
        base, _patch({"finding": "1", "refs": ["evidence:b"]}, {"finding": "2", "drop": True})
    )
    assert isinstance(merged, PatchedReport)
    assert merged.data["findings"] == [
        f("A.", ("evidence:a",)),
        {"claim": "B 1.2.", "refs": ["evidence:b"], "web_urls": []},
    ]
    assert (merged.origins, merged.patched, merged.dropped) == ((0, 1), (1,), (2,))
    assert report_issues(merged.data, {"evidence:a", "evidence:b"}, set()) == ()
    again = apply_report_patch(merged, _patch({"finding": "2", "claim": "C again."}))
    assert isinstance(again, tuple) and again[0].kind == "bad_index"  # finding 2 was dropped


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ('{"patches": [{"finding": 1, "drop": true}]}', "json_number"),
        ('{"patches": [{"finding": "x", "drop": true}]}', "bad_index"),
        ('{"patches": [{"finding": "1"}]}', "value_error"),  # changes nothing
        ('{"patches": [{"finding": "1", "drop": true, "claim": "c"}]}', "value_error"),
        (
            '{"patches": [{"finding": "1", "drop": true}, {"finding": "1", "claim": "c"}]}',
            "value_error",
        ),
        ('{"patches": []}', "too_short"),
        ('{"other": []}', "not_patch"),
        ('{"patches": [{"finding": "1", "refs": ["facts:x"]}]}', "string_pattern_mismatch"),
    ],
)
def test_malformed_patches_are_issues(raw: str, kind: str) -> None:
    result = parse_report_patch(raw)
    assert isinstance(result, tuple) and kind in [i.kind for i in result], result


def test_prose_around_a_patch_is_dropped() -> None:
    result = parse_report_patch('Fixed:\n{"patches": [{"finding": "0", "drop": true}]}\nDone.')
    assert isinstance(result, ReportPatch) and result.patches[0].finding == 0


def test_report_issues_are_located_by_finding() -> None:
    data = doc(f("Fine.", ("evidence:a",)), f("Cites nothing."), f("Old.", ("evidence:z",)))
    locs = [i.loc for i in report_issues(data, {"evidence:a"}, set())]
    assert locs == ["findings.1"]  # schema issues come first; source checks need a valid report
    fixed = doc(f("Fine.", ("evidence:a",)), f("Old.", ("evidence:z",)))
    assert [i.loc for i in report_issues(fixed, {"evidence:a"}, set())] == ["findings.1.refs"]
