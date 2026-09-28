"""MignonReport v1 strict parser and source check (ADR-0025, domain/mignon_report.py)."""

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wheelta_robinhood_agent.domain.mignon_report import (
    MignonReportParsed,
    check_report_sources,
    parse_mignon_report,
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
        ("Here is my report.\n" + json.dumps(doc()), "invalid_json"),  # prose before
        (json.dumps(doc()) + "\nDone.", "invalid_json"),  # prose after
        ("```json\n" + json.dumps(doc()) + "\n```\nDone.", "invalid_json"),  # text after fence
        ("```json\n```json\n" + json.dumps(doc()) + "\n```\n```", "invalid_json"),  # two fences
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
