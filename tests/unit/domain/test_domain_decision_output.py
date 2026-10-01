"""AgentDecisionOutput v6 parser: strictness, failure shape, and parity with the JSON schema."""

import copy
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from wheelta_robinhood_agent.config.prompts import load_agent_prompts
from wheelta_robinhood_agent.domain.decision_output import (
    LIMIT_PRICE_PATTERN,
    NEXT_RUN_AT_PATTERN,
    ROLE_ACTIONS,
    SCHEMA_VERSION,
    AgentDecisionOutput,
    CancellationRationale,
    Decision,
    DecisionOutputParsed,
    DecisionOutputParseFailure,
    NextRun,
    ProposedLeg,
    ResearchQuestion,
    parse_agent_decision_output,
)
from wheelta_robinhood_agent.domain.enums import AgentRole, DecisionAction

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/wheelta_robinhood_agent/prompts/agent_decision_output.v6.schema.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text())
JsonObj = dict[str, object]


def _valid() -> JsonObj:
    return {
        "decisions": [
            {
                "action": "OPEN_CSP",
                "target_ref": "candidate:1",
                "replacement_ref": None,
                "funding_close_refs": [],
                "proposed_legs": [{"facts_ref": "facts:1", "limit_price": "1.25"}],
                "execution_refs": [],
                "rationale": "Supported by the cited filing.",
                "thesis": "Durable demand.",
                "invalidation_conditions": ["Guidance withdrawn."],
                "evidence_refs": ["evidence:1"],
            }
        ],
        "cancellation_rationales": [
            {"cancel_call_ref": "call:1", "rationale": "Stale order.", "evidence_refs": []}
        ],
        "unresolved_questions": [
            {"target_ref": None, "question": "Guidance date?", "evidence_refs": []}
        ],
        "next_run": {"at": "2026-09-28T15:30:00Z", "rationale": "Re-check after the open."},
    }


# Where one instance of each schema object lives inside _valid().
LOCATIONS: dict[str, tuple[str | int, ...]] = {
    "#": (),
    "decision": ("decisions", 0),
    "proposed_leg": ("decisions", 0, "proposed_legs", 0),
    "cancellation_rationale": ("cancellation_rationales", 0),
    "research_question": ("unresolved_questions", 0),
    "next_run": ("next_run",),
}
MODELS: dict[str, type[BaseModel]] = {
    "#": AgentDecisionOutput,
    "decision": Decision,
    "proposed_leg": ProposedLeg,
    "cancellation_rationale": CancellationRationale,
    "research_question": ResearchQuestion,
    "next_run": NextRun,
}


def _obj(doc: JsonObj, path: tuple[str | int, ...]) -> JsonObj:
    node: object = doc
    for part in path:
        node = node[part]  # type: ignore[index]
    return cast(JsonObj, node)


def _schema_def(name: str) -> JsonObj:
    return cast(JsonObj, SCHEMA if name == "#" else SCHEMA["$defs"][name])


def _parse(doc: object) -> DecisionOutputParsed | DecisionOutputParseFailure:
    return parse_agent_decision_output(json.dumps(doc))


def _ok(doc: object) -> bool:
    return _parse(doc).ok


def _resolve(prop: JsonObj) -> JsonObj:
    ref = prop.get("$ref")
    if isinstance(ref, str):
        return cast(JsonObj, SCHEMA["$defs"][ref.rsplit("/", 1)[1]])
    return prop


# ---------------------------------------------------------------- basic behavior


def test_valid_document_parses() -> None:
    result = _parse(_valid())
    assert isinstance(result, DecisionOutputParsed)
    assert result.schema_version == SCHEMA_VERSION == 6
    decision = result.output.decisions[0]
    assert decision.action is DecisionAction.OPEN_CSP
    assert decision.proposed_legs[0].limit_price == Decimal("1.25")
    assert isinstance(decision.proposed_legs[0].limit_price, Decimal)


def test_prompt_examples_parse_for_their_role() -> None:
    for role, template in load_agent_prompts().items():
        example = template.body.split("```json", 1)[1].split("```", 1)[0]
        assert parse_agent_decision_output(example, ROLE_ACTIONS[role]).ok


def test_role_actions_split_management_from_opening() -> None:
    """ADR-0057: the close agent manages shorts, the sell agent opens; WHEEL is unrestricted."""
    assert ROLE_ACTIONS[AgentRole.CLOSE] == {
        DecisionAction.CLOSE,
        DecisionAction.ROLL,
        DecisionAction.HOLD,
    }
    assert ROLE_ACTIONS[AgentRole.SELL] == {DecisionAction.OPEN_CSP, DecisionAction.OPEN_CC}
    assert ROLE_ACTIONS[AgentRole.WHEEL] == frozenset(DecisionAction)


def test_disallowed_action_is_a_parse_issue() -> None:
    raw = json.dumps(_valid())  # an OPEN_CSP decision
    assert parse_agent_decision_output(raw, ROLE_ACTIONS[AgentRole.SELL]).ok
    result = parse_agent_decision_output(raw, ROLE_ACTIONS[AgentRole.CLOSE])
    assert isinstance(result, DecisionOutputParseFailure)
    assert [(i.loc, i.kind) for i in result.issues] == [
        ("decisions.0.action", "action_not_allowed")
    ]
    assert "OPEN_CSP is not an action of this agent" in result.issues[0].message
    assert result.raw_text == raw


def test_empty_output_and_bytes() -> None:
    doc = {
        "decisions": [],
        "cancellation_rationales": [],
        "unresolved_questions": [],
        "next_run": None,
    }
    assert parse_agent_decision_output(json.dumps(doc).encode()).ok


def test_serializes_limit_price_as_string() -> None:
    result = _parse(_valid())
    assert isinstance(result, DecisionOutputParsed)
    dumped = result.output.model_dump(mode="json")
    assert dumped["decisions"][0]["proposed_legs"][0]["limit_price"] == "1.25"


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("not json", "invalid_json"),
        ("```json\n{}\n```", "missing"),  # ADR-0035: the fence is removed; the schema applies
        # ADR-0044: prose and fences around the object are dropped; the schema still applies.
        ("Here it is:\n```json\n{}\n```", "missing"),
        ("```json\n{}\n```\nDone.", "missing"),
        ("```json\n```json\n{}\n```\n```", "missing"),
        ("Prose {not json} only.", "invalid_json"),
        ("[]", "not_object"),
        ('"x"', "not_object"),
        ('{"decisions": [], "decisions": []}', "duplicate_key"),
        ('{"decisions": 1}', "json_number"),
        ('{"decisions": 1.5}', "json_number"),
        ('{"decisions": NaN}', "json_constant"),
        ("[" * 100000 + "]" * 100000, "invalid_json"),
    ],
)
def test_malformed_json_fails_without_raising(raw: str, kind: str) -> None:
    result = parse_agent_decision_output(raw)
    assert isinstance(result, DecisionOutputParseFailure)
    assert result.raw_text == raw
    assert result.issues[0].kind == kind


@pytest.mark.parametrize("fence", ["```json", "```", "```json  "])
def test_one_enclosing_fence_is_accepted(fence: str) -> None:
    # ADR-0035: the 2026-09-28 19:45 production output was valid JSON inside a ```json fence.
    raw = f"{fence}\n{json.dumps(_valid(), indent=2)}\n```\n"
    result = parse_agent_decision_output(raw)
    assert isinstance(result, DecisionOutputParsed), result
    assert result.output.decisions[0].proposed_legs[0].limit_price == Decimal("1.25")


@pytest.mark.parametrize(
    "wrap",
    [
        # 2026-09-29 dry run 006a73a1: one line of prose before the JSON failed the run.
        lambda body: "NCLH is resting at the mid as a day order. I'll end the run here.\n\n" + body,
        lambda body: body + "\nDone.",
        lambda body: 'Example: {"decisions": "x"}\nFinal:\n```json\n' + body + "\n```",
    ],
)
def test_prose_around_the_output_is_accepted(wrap: Any) -> None:
    result = parse_agent_decision_output(wrap(json.dumps(_valid(), indent=2)))
    assert isinstance(result, DecisionOutputParsed), result
    assert result.output.decisions[0].proposed_legs[0].limit_price == Decimal("1.25")


def test_prose_wrapped_failure_preserves_the_raw_text() -> None:
    raw = 'Here:\n{"decisions": 1}'
    result = parse_agent_decision_output(raw)
    assert isinstance(result, DecisionOutputParseFailure)
    assert result.raw_text == raw
    assert result.issues[0].kind == "json_number"


def test_fenced_failure_preserves_the_raw_text() -> None:
    raw = '```json\n{"decisions": 1}\n```'
    result = parse_agent_decision_output(raw)
    assert isinstance(result, DecisionOutputParseFailure)
    assert result.raw_text == raw
    assert result.issues[0].kind == "json_number"


def test_invalid_utf8() -> None:
    result = parse_agent_decision_output(b'{"decisions": "\xff"}')
    assert isinstance(result, DecisionOutputParseFailure)
    assert result.issues[0].kind == "invalid_utf8"
    assert "�" in result.raw_text


def test_failure_preserves_raw_and_locates_issue() -> None:
    doc = _valid()
    _obj(doc, LOCATIONS["decision"])["action"] = "BUY_STOCK"
    raw = json.dumps(doc)
    result = parse_agent_decision_output(raw)
    assert isinstance(result, DecisionOutputParseFailure)
    assert result.raw_text == raw
    assert result.schema_version == 6
    assert any(i.loc == "decisions.0.action" for i in result.issues)


@pytest.mark.parametrize(
    ("path", "key", "value"),
    [
        ((), "schema_version", "6"),
        (("next_run",), "not_before", "2026-09-28T15:30:00Z"),
        ((), "rejected_candidates", []),
        (("decisions", 0), "quantity", "1"),
        (("decisions", 0), "status", "filled"),
        (("decisions", 0), "priority", "1"),
        (("decisions", 0, "proposed_legs", 0), "quantity", "1"),
        (("decisions", 0, "proposed_legs", 0), "broker_order_id", "x"),
        (("cancellation_rationales", 0), "status", "confirmed"),
        (("unresolved_questions", 0), "gap", "x"),
    ],
)
def test_forbidden_fields_rejected(path: tuple[str | int, ...], key: str, value: object) -> None:
    doc = _valid()
    _obj(doc, path)[key] = value
    assert not _ok(doc)


@pytest.mark.parametrize("price", ["1.25", "0", "0.05", "10", "123.4500"])
def test_limit_price_valid(price: str) -> None:
    doc = _valid()
    _obj(doc, LOCATIONS["proposed_leg"])["limit_price"] = price
    result = _parse(doc)
    assert isinstance(result, DecisionOutputParsed)
    assert result.output.decisions[0].proposed_legs[0].limit_price == Decimal(price)


@pytest.mark.parametrize(
    "price", ["01.25", "1.", ".5", "-1.25", "1e3", " 1.25", "1,25", "NaN", "Infinity", ""]
)
def test_limit_price_invalid_strings(price: str) -> None:
    doc = _valid()
    _obj(doc, LOCATIONS["proposed_leg"])["limit_price"] = price
    assert not _ok(doc)


def test_limit_price_non_string_rejected_by_model() -> None:
    leg = {"facts_ref": "facts:1", "limit_price": 1.25}
    with pytest.raises(ValueError):
        ProposedLeg.model_validate(leg)
    with pytest.raises(ValueError):
        ProposedLeg.model_validate({**leg, "limit_price": Decimal("1.25")})


# ---------------------------------------------------------------- parity with the schema


def test_every_schema_object_has_a_model() -> None:
    object_defs = {n for n, d in SCHEMA["$defs"].items() if d.get("type") == "object"}
    assert object_defs | {"#"} == set(MODELS)
    assert set(LOCATIONS) == set(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fields_and_required_match(name: str) -> None:
    schema = _schema_def(name)
    model = MODELS[name]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert model.model_config.get("extra") == "forbid"
    assert model.model_config.get("frozen") is True
    props = cast(JsonObj, schema["properties"])
    assert set(props) == set(model.model_fields)
    assert set(cast(list[str], schema["required"])) == set(model.model_fields)
    assert all(f.is_required() for f in model.model_fields.values())


# (matching, non-matching) value per schema string pattern.
PATTERN_SAMPLES: dict[str, tuple[str, str]] = {
    LIMIT_PRICE_PATTERN: ("2.50", "2.5.0"),
    NEXT_RUN_AT_PATTERN: ("2026-09-28T19:45:00-04:00", "2026-09-28 19:45:00"),
}


def _probes(name: str) -> list[tuple[str, str, object, bool]]:
    """(description, field, replacement value, expected ok) derived from the schema only."""
    schema = _schema_def(name)
    probes: list[tuple[str, str, object, bool]] = []
    for field, raw_prop in cast(dict[str, JsonObj], schema["properties"]).items():
        options = cast(list[JsonObj], raw_prop.get("anyOf", [raw_prop]))
        nullable = any(o.get("type") == "null" for o in options)
        prop = _resolve(next(o for o in options if o.get("type") != "null"))
        probes.append(("null", field, None, nullable))
        probes.append(("bool", field, True, False))
        probes.append(("object", field, {}, False))
        kind = prop["type"]
        if kind == "string":
            probes.append(("array-for-string", field, [], False))
            if prop.get("minLength") == 1:
                probes.append(("empty string", field, "", False))
            if "enum" in prop:
                for value in cast(list[str], prop["enum"]):
                    probes.append((f"enum {value}", field, value, True))
                probes.append(("unknown enum", field, "open_csp", False))
            elif "pattern" in prop:
                good, bad = PATTERN_SAMPLES[cast(str, prop["pattern"])]
                probes.append(("pattern ok", field, good, True))
                probes.append(("pattern bad", field, bad, False))
            else:
                probes.append(("string", field, "some-text", True))
        elif kind == "array":
            probes.append(("empty array", field, [], True))
            probes.append(("string-for-array", field, "x", False))
            item = _resolve(cast(JsonObj, prop["items"]))
            if item["type"] == "string":
                probes.append(("item", field, ["a"], True))
                probes.append(("empty item", field, [""], item.get("minLength") != 1))
                probes.append(("non-string item", field, [True], False))
                probes.append(("duplicates", field, ["a", "a"], not prop.get("uniqueItems")))
            else:
                probes.append(("non-object item", field, ["a"], False))
        elif kind == "object":
            probes.append(("string-for-object", field, "x", False))
            probes.append(("array-for-object", field, [], False))
        else:  # pragma: no cover - the v6 schema has no other property kinds
            raise AssertionError(f"unhandled schema kind {kind} for {name}.{field}")
    return probes


@pytest.mark.parametrize("name", sorted(MODELS))
def test_schema_constraints_enforced(name: str) -> None:
    for description, field, value, expected in _probes(name):
        doc = _valid()
        _obj(doc, LOCATIONS[name])[field] = value
        assert _ok(doc) is expected, f"{name}.{field}: {description}"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_required_field_missing_fails(name: str) -> None:
    for field in cast(list[str], _schema_def(name)["required"]):
        doc = _valid()
        del _obj(doc, LOCATIONS[name])[field]
        assert not _ok(doc), f"{name}.{field} missing should fail"


def test_enum_and_pattern_match_code() -> None:
    action = SCHEMA["$defs"]["decision"]["properties"]["action"]
    assert action["enum"] == [a.value for a in DecisionAction]
    leg = SCHEMA["$defs"]["proposed_leg"]["properties"]["limit_price"]
    assert leg["pattern"] == LIMIT_PRICE_PATTERN
    at = SCHEMA["$defs"]["next_run"]["properties"]["at"]
    assert at["pattern"] == NEXT_RUN_AT_PATTERN


# ---------------------------------------------------------------- next_run (ADR-0028)


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        ("2026-09-28T15:30:00Z", datetime(2026, 9, 28, 15, 30, tzinfo=UTC)),
        ("2026-09-28T11:30:00-04:00", datetime(2026, 9, 28, 15, 30, tzinfo=UTC)),
        ("2026-09-28T15:30:00.250+00:00", datetime(2026, 9, 28, 15, 30, 0, 250000, tzinfo=UTC)),
    ],
)
def test_next_run_at_parses_to_utc(at: str, expected: datetime) -> None:
    doc = _valid()
    _obj(doc, LOCATIONS["next_run"])["at"] = at
    result = _parse(doc)
    assert isinstance(result, DecisionOutputParsed)
    assert result.output.next_run is not None
    assert result.output.next_run.at == expected
    assert result.output.next_run.at.tzinfo is UTC


@pytest.mark.parametrize(
    "at",
    [
        "2026-09-28T15:30:00",  # no offset
        "2026-09-28T15:30Z",  # no seconds
        "2026-09-28",
        "2026-13-01T15:30:00Z",  # matches the pattern, not a date
        "2026-09-28T25:00:00Z",
        "in 45 minutes",
        "",
        "2026-09-28T15:30:00+99:00",  # offset out of range
        # In range locally, out of range in UTC: must fail parsing, never raise.
        "0001-01-01T00:00:00+14:00",
        "9999-12-31T23:59:59-14:00",
    ],
)
def test_next_run_at_invalid(at: str) -> None:
    doc = _valid()
    _obj(doc, LOCATIONS["next_run"])["at"] = at
    assert not _ok(doc)


def test_next_run_null_and_round_trip() -> None:
    doc = _valid()
    doc["next_run"] = None
    result = _parse(doc)
    assert isinstance(result, DecisionOutputParsed)
    assert result.output.next_run is None
    stored = _parse(_valid())
    assert isinstance(stored, DecisionOutputParsed)
    dumped = stored.output.model_dump(mode="json")
    assert dumped["next_run"]["at"] == "2026-09-28T15:30:00+00:00"
    assert AgentDecisionOutput.model_validate(dumped) == stored.output


# ---------------------------------------------------------------- property-based

_KEYS = st.text(min_size=1, max_size=12)
_PATHS = st.sampled_from(sorted(LOCATIONS.values()))


@settings(max_examples=150)
@given(path=_PATHS, key=_KEYS, value=st.one_of(st.text(), st.none(), st.booleans()))
def test_any_extra_key_rejected(path: tuple[str | int, ...], key: str, value: object) -> None:
    doc = _valid()
    target = _obj(doc, path)
    if key in target:
        return
    target[key] = value
    result = _parse(doc)
    assert isinstance(result, DecisionOutputParseFailure)
    assert result.raw_text == json.dumps(doc)


_REF = st.text(min_size=1, max_size=20)
_TEXT = st.text(min_size=1, max_size=40)
_PRICE = st.from_regex(r"\A(0|[1-9][0-9]{0,4})(\.[0-9]{1,4})?\Z")


def _unique_list(elements: st.SearchStrategy[str]) -> st.SearchStrategy[list[str]]:
    return st.lists(elements, max_size=3, unique=True)


_LEG = st.fixed_dictionaries({"facts_ref": _REF, "limit_price": _PRICE})
_DECISION = st.fixed_dictionaries(
    {
        "action": st.sampled_from([a.value for a in DecisionAction]),
        "target_ref": _REF,
        "replacement_ref": st.none() | _REF,
        "funding_close_refs": _unique_list(_REF),
        "proposed_legs": st.lists(_LEG, max_size=2),
        "execution_refs": _unique_list(_REF),
        "rationale": _TEXT,
        "thesis": st.none() | _TEXT,
        "invalidation_conditions": st.lists(_TEXT, max_size=2),
        "evidence_refs": _unique_list(_REF),
    }
)
_CANCEL = st.fixed_dictionaries(
    {"cancel_call_ref": _REF, "rationale": _TEXT, "evidence_refs": _unique_list(_REF)}
)
_QUESTION = st.fixed_dictionaries(
    {"target_ref": st.none() | _REF, "question": _TEXT, "evidence_refs": _unique_list(_REF)}
)
_NEXT_RUN = st.fixed_dictionaries(
    {
        "at": st.datetimes(
            min_value=datetime(2026, 1, 1),  # noqa: DTZ001 - hypothesis bounds are naive
            max_value=datetime(2030, 12, 31),  # noqa: DTZ001
            timezones=st.just(UTC),
        ).map(lambda d: d.replace(microsecond=0).isoformat()),
        "rationale": _TEXT,
    }
)
_OUTPUT = st.fixed_dictionaries(
    {
        "decisions": st.lists(_DECISION, max_size=3),
        "cancellation_rationales": st.lists(_CANCEL, max_size=2),
        "unresolved_questions": st.lists(_QUESTION, max_size=2),
        "next_run": st.none() | _NEXT_RUN,
    }
)


@settings(max_examples=150)
@given(doc=_OUTPUT)
def test_valid_structures_accepted_and_round_trip(doc: JsonObj) -> None:
    result = _parse(doc)
    assert isinstance(result, DecisionOutputParsed), result
    dumped = result.output.model_dump(mode="json")
    expected = copy.deepcopy(doc)
    for decision in cast(list[JsonObj], expected["decisions"]):
        for leg in cast(list[JsonObj], decision["proposed_legs"]):
            leg["limit_price"] = str(Decimal(cast(str, leg["limit_price"])))
    assert dumped == expected


@settings(max_examples=100)
@given(raw=st.text(max_size=200) | st.binary(max_size=200))
def test_arbitrary_input_never_raises(raw: str | bytes) -> None:
    result = parse_agent_decision_output(raw)
    if isinstance(result, DecisionOutputParseFailure):
        assert result.issues
