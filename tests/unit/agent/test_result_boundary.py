"""BoundaryValidator and the facts-tool request parser (DATA_QUALITY.md; ADR-0011)."""

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from wheelta_robinhood_agent.agent.facts_tool import FactsRequestError, parse_request
from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, ValidationRequest
from wheelta_robinhood_agent.agent.result_boundary import (
    CONTEXT_ONLY_TOOLS,
    VERIFIED_MAPPERS,
    BoundaryValidator,
    CandidateEvidence,
    MappedEvidence,
    MappingRequest,
    PayloadError,
    PayloadKind,
    extract_mcp_payload,
    mapped_evidence_of,
)
from wheelta_robinhood_agent.domain.enums import CandidateOrigin, ToolTier
from wheelta_robinhood_agent.domain.facts import FactsPurpose
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.observability.redaction import Redactor

NOW = datetime(2026, 9, 23, 15, 30, tzinfo=UTC)
CALL = uuid.UUID("0190a0a0-0000-7000-8000-000000000001")
OCC = OccSymbol.parse("AAPL  261016P00150000")


def _req(server: str, tool: str, response: object) -> ValidationRequest:
    return ValidationRequest(
        tool_call_id=CALL,
        server=server,
        tool=tool,
        tier=ToolTier.R,
        effective_input={},
        tool_response=response,
        retrieved_at=NOW,
    )


def _text(value: object) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value)}]}


def test_only_captured_tools_have_verified_mappers() -> None:
    # The Robinhood set is pinned in test_robinhood_mappers.py; Wheelta maps only the board
    # query (ADR-0041); every other Wheelta tool is context only.
    assert {t for s, t in VERIFIED_MAPPERS if s == "wheelta"} == {"wheelta_board_query"}
    assert {s for s, _ in VERIFIED_MAPPERS} == {"robinhood", "wheelta"}
    wheelta_context = {t for s, t in CONTEXT_ONLY_TOOLS if s == "wheelta"}
    assert len(wheelta_context) == 11 and "wheelta_board_query" not in wheelta_context


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (_text({"a": 1}), PayloadKind.OK),
        ([{"type": "text", "text": "[1]"}], PayloadKind.OK),
        ('{"a": 1}', PayloadKind.OK),
        ({"structuredContent": {"a": 1}, "content": []}, PayloadKind.OK),
        ({"isError": True, "content": [{"type": "text", "text": "boom"}]}, PayloadKind.TOOL_ERROR),
    ],
)
def test_extract_accepts_mcp_shapes(response: object, kind: PayloadKind) -> None:
    assert extract_mcp_payload(response)[0] is kind


@pytest.mark.parametrize(
    "response",
    [
        42,
        {"content": [{"type": "image"}]},
        {"content": []},
        _text("x") | {"content": [{"type": "text", "text": "no json"}]},
    ],
)
def test_extract_rejects_unknown_shapes(response: object) -> None:
    with pytest.raises(PayloadError):
        extract_mcp_payload(response)


def test_unmapped_remote_result_is_missing_and_raw_is_restricted() -> None:
    outcome = BoundaryValidator(Redactor())(
        _req("robinhood", "get_equity_historicals", _text({"p": "1"}))
    )
    assert outcome.envelope.kind is EnvelopeKind.MISSING
    assert outcome.envelope.data is None
    assert "no verified result mapping" in outcome.envelope.gaps[0]
    assert outcome.raw_redacted is not None


def test_tool_error_becomes_an_error_envelope() -> None:
    response = {"isError": True, "content": [{"type": "text", "text": "denied"}]}
    outcome = BoundaryValidator(Redactor())(_req("robinhood", "x", response))
    assert outcome.envelope.kind is EnvelopeKind.ERROR


def test_local_and_builtin_results_are_validated() -> None:
    local = BoundaryValidator(Redactor())(_req("wra_local", "get_decision_facts", _text({"s": 1})))
    assert local.envelope.kind is EnvelopeKind.VALIDATED and local.envelope.data == {"s": 1}
    web = BoundaryValidator(Redactor())(_req("builtin", "WebSearch", {"results": []}))
    assert web.envelope.kind is EnvelopeKind.VALIDATED


def _instrument(call: uuid.UUID, evidence: uuid.UUID) -> OptionInstrument:
    return OptionInstrument(
        evidence_id=evidence,
        as_of=NOW,
        source_tool_call_ids=(call,),
        occ_symbol=OCC,
        broker_instrument_id="inst-1",
        underlying="AAPL",
        multiplier=100,
    )


def _mapper(call: uuid.UUID, ref: str = "candidate:1") -> Any:
    def mapper(req: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
        inst = _instrument(call, new_id())
        return MappedEvidence(
            instruments=(inst,),
            candidates=(
                CandidateEvidence(
                    candidate_ref=ref,
                    origin=CandidateOrigin.ROBINHOOD,
                    underlying="AAPL",
                    instrument_evidence_id=inst.evidence_id,
                    broker_instrument_id="inst-1",
                    occ_symbol=OCC,
                ),
            ),
        )

    return mapper


def test_mapped_result_carries_typed_evidence_and_an_evidence_ref() -> None:
    validator = BoundaryValidator(Redactor(), mappers={("robinhood", "chains"): _mapper(CALL)})
    outcome = validator(_req("robinhood", "chains", _text({"any": "shape"})))
    envelope = outcome.envelope.model_dump(mode="json")
    assert envelope["kind"] == "validated"
    assert envelope["data"]["evidence_ref"] == f"evidence:{CALL}"
    mapped = mapped_evidence_of(envelope)
    assert mapped is not None and mapped.candidates[0].candidate_ref == "candidate:1"


@pytest.mark.parametrize(
    "mapper",
    [_mapper(uuid.UUID(int=7)), _mapper(CALL, ref="AAPL")],
    ids=["cites-another-call", "model-shaped-ref"],
)
def test_mapper_provenance_violations_are_missing(mapper: Any) -> None:
    validator = BoundaryValidator(Redactor(), mappers={("robinhood", "chains"): mapper})
    outcome = validator(_req("robinhood", "chains", _text({})))
    assert outcome.envelope.kind is EnvelopeKind.MISSING


# -- facts tool arguments ------------------------------------------------------------------


def test_parse_request_accepts_refs_and_discretionary_inputs() -> None:
    req = parse_request(
        {
            "subject_ref": "position:x",
            "purpose": "close",
            "limit_price": "0.35",
            "close_quantity": 1,
        }
    )
    assert req.purpose is FactsPurpose.CLOSE and str(req.limit_price) == "0.35"
    assert req.close_quantity == 1


@pytest.mark.parametrize(
    "args",
    [
        {"subject_ref": "candidate:x", "purpose": "open", "limit_price": 1.25},
        {"subject_ref": "candidate:x", "purpose": "open", "limit_price": "0"},
        {"subject_ref": "candidate:x", "purpose": "open", "close_quantity": 1},
        {"subject_ref": "candidate:x", "purpose": "buy"},
        {"subject_ref": "", "purpose": "open"},
        {"subject_ref": "candidate:x", "purpose": "open", "cash": "100000"},
        {"subject_ref": "position:x", "purpose": "close", "close_quantity": True},
        "not an object",
    ],
)
def test_parse_request_rejects_model_supplied_facts_and_bad_inputs(args: Any) -> None:
    with pytest.raises(FactsRequestError):
        parse_request(args)


def test_raw_invalid_evidence_drops_account_numbers_inside_text_blocks() -> None:
    import json as _json

    from wheelta_robinhood_agent.agent.result_boundary import (
        _drop_account_values,
        _expand_text_json,
    )

    other = "987654321"
    raw = [{"type": "text", "text": _json.dumps({"accounts": [{"account_number": other}]})}]
    expanded = _expand_text_json(raw)
    assert _json.dumps(expanded).count(other) == 1  # visible to redaction now
    cleaned = _drop_account_values(expanded)  # type: ignore[arg-type]
    assert other not in _json.dumps(cleaned)
    assert _expand_text_json([{"type": "text", "text": "not json"}]) == [
        {"type": "text", "text": "not json"}
    ]


def test_login_scoped_reads_are_delivered_as_context_never_as_evidence() -> None:
    """ADR-0026: no mapper, but a redacted context-only envelope with account values dropped."""
    from wheelta_robinhood_agent.agent.result_boundary import CONTEXT_ONLY_GAP

    payload = {"watchlists": [{"name": "Tech", "account_number": "9ZZ11223344"}]}
    outcome = BoundaryValidator(redactor=Redactor())(
        _req("robinhood", "get_watchlists", _text(payload))
    )
    envelope = outcome.envelope
    assert envelope.kind is EnvelopeKind.VALIDATED and envelope.gaps == (CONTEXT_ONLY_GAP,)
    assert isinstance(envelope.data, dict) and envelope.data["context_only"] is True
    assert "evidence_ref" not in envelope.data and "evidence" not in envelope.data
    assert "9ZZ11223344" not in json.dumps(envelope.data)
    assert envelope.data["payload"]["watchlists"][0]["name"] == "Tech"  # type: ignore[index]
