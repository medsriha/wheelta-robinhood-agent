"""MignonBrief v1 and report coverage (ADR-0061)."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wheelta_robinhood_agent.config.rules import load_rules, rule_keys
from wheelta_robinhood_agent.domain.mignon_brief import (
    BriefSource,
    MignonBrief,
    MignonBriefParsed,
    MignonBriefParseFailure,
    coverage_gaps,
    parse_mignon_brief,
)
from wheelta_robinhood_agent.domain.mignon_report import (
    MignonReportParsed,
    apply_report_patch,
    original_report,
    parse_mignon_report,
    parse_report_patch,
    web_sourced_indices,
)

KEYS = rule_keys(load_rules().rules)


def _parse(brief: object) -> MignonBriefParsed | MignonBriefParseFailure:
    return parse_mignon_brief(json.dumps(brief), KEYS)


def test_rule_keys_name_sections_and_rules_but_not_meta() -> None:
    assert {"filters", "filters.min_abs_delta", "events.earnings_exclusion"} <= KEYS
    assert "data_quality.freshness.option_quote_max_age_seconds" in KEYS
    assert not any(k == "meta" or k.startswith("meta.") for k in KEYS)


def test_a_full_brief_parses() -> None:
    parsed = _parse(
        {
            "objective": "Quote these puts.",
            "subjects": ["AAPL", "NU    261009P00012000", "candidate:01a0-x"],
            "criteria": ["filters", "events.earnings_exclusion"],
            "exclude": ["EWZ"],
            "source": "scanner",
            "want": ["bid", "delta"],
            "max_results": "5",
            "notes": "Single stocks only.",
        }
    )
    assert isinstance(parsed, MignonBriefParsed)
    assert parsed.brief.source is BriefSource.SCANNER
    assert parsed.brief.subjects[1] == "NU    261009P00012000"


def test_only_an_objective_is_required() -> None:
    parsed = _parse({"objective": "Macro regime."})
    assert isinstance(parsed, MignonBriefParsed)
    assert parsed.brief == MignonBrief(objective="Macro regime.")


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("Screen AAPL puts.", "JSON"),
        ('Here is the brief: {"objective": "x"}', "JSON"),
        ("[]", "one JSON object"),
        (json.dumps({"objective": "x", "max_results": 5}), "number"),
        (json.dumps({"objective": "x", "criteria": ["filters.min_delta"]}), "no rule"),
        (json.dumps({"objective": "x", "subjects": ["aapl"]}), "pattern"),
        (json.dumps({"objective": "x", "subjects": ["AAPL"], "exclude": ["AAPL"]}), "excluded"),
        (json.dumps({"objective": "x", "want": ["Bid"]}), "pattern"),
        (json.dumps({"objective": "x", "max_results": "51"}), "pattern"),
        (json.dumps({"objective": "x", "delta": "0.2"}), "Extra inputs"),
        (json.dumps({"objective": "x", "want": ["bid", "bid"]}), "duplicate"),
    ],
)
def test_invalid_briefs_name_the_problem(raw: str, fragment: str) -> None:
    parsed = parse_mignon_brief(raw, KEYS)
    assert isinstance(parsed, MignonBriefParseFailure)
    assert any(fragment in i.message for i in parsed.issues), parsed.issues


@given(st.text(max_size=200))
def test_parse_never_raises(raw: str) -> None:
    parse_mignon_brief(raw, KEYS)


def test_coverage_names_missing_subjects_and_values_in_order() -> None:
    brief = MignonBrief(objective="q", subjects=("AAPL", "MSFT"), want=("bid", "delta"))
    gaps = coverage_gaps(brief, [("AAPL", ["bid"]), (None, []), ("NU", ["bid", "delta"])])
    assert gaps == ("subject MSFT: not reported", "subject AAPL: no value for 'delta'")
    assert coverage_gaps(MignonBrief(objective="q"), [(None, [])]) == ()


def _report(finding: dict[str, object]) -> str:
    return json.dumps({"task": "t", "findings": [finding], "gaps": [], "follow_up_questions": []})


def test_a_v2_finding_carries_subject_and_values() -> None:
    finding = {
        "claim": "Quoted.",
        "refs": ["evidence:e-1"],
        "web_urls": [],
        "subject": "AAPL",
        "values": {"bid": "1.20"},
    }
    parsed = parse_mignon_report(_report(finding))
    assert isinstance(parsed, MignonReportParsed)
    assert parsed.report.reported_values() == (("AAPL", ("bid",)),)
    # A v1 finding (no subject, no values) still parses.
    plain = parse_mignon_report(_report({"claim": "c", "refs": ["evidence:e"], "web_urls": []}))
    assert isinstance(plain, MignonReportParsed)


def test_a_number_in_a_web_only_value_is_web_sourced() -> None:
    finding = {
        "claim": "Election day per the court.",
        "refs": [],
        "web_urls": ["https://www.tse.jus.br/x"],
        "values": {"election_date": "2026-10-04"},
    }
    parsed = parse_mignon_report(_report(finding))
    assert isinstance(parsed, MignonReportParsed)
    assert web_sourced_indices(parsed.report) == (0,)


def test_values_reject_json_numbers_and_bad_names() -> None:
    for values in ('{"bid": 1.2}', '{"Bid": "1.2"}'):
        raw = (
            '{"task": "t", "findings": [{"claim": "c", "refs": ["evidence:e"], "web_urls": [], '
            f'"values": {values}}}], "gaps": [], "follow_up_questions": []}}'
        )
        assert not parse_mignon_report(raw).ok


def test_a_patch_may_set_subject_and_values() -> None:
    base = original_report(
        {
            "task": "t",
            "findings": [{"claim": "c", "refs": ["evidence:e"], "web_urls": []}],
            "gaps": [],
            "follow_up_questions": [],
        }
    )
    patch = parse_report_patch(
        '{"patches": [{"finding": "0", "subject": "AAPL", "values": {"bid": "1.20"}}]}'
    )
    assert not isinstance(patch, tuple)
    merged = apply_report_patch(base, patch)
    assert not isinstance(merged, tuple)
    findings = merged.data["findings"]
    assert isinstance(findings, list)
    assert findings[0] == {
        "claim": "c",
        "refs": ["evidence:e"],
        "web_urls": [],
        "subject": "AAPL",
        "values": {"bid": "1.20"},
    }
