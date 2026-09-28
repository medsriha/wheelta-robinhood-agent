"""Fake Robinhood/Wheelta servers, fixture result mappers, and scripted agents for e2e runs.

The fake broker's payload shapes are INVENTED for these tests. The mappers below are fixture
mappers for these fake shapes only; production uses `VERIFIED_MAPPERS` (agent/robinhood_mappers.py,
built from real captured responses). Prices are decimal strings, never floats.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from e2e_fake_cli import FakeModel, FakeWorld

from wheelta_robinhood_agent.agent.account_scope import (
    ROBINHOOD_ACCOUNT_SCOPE,
    AccountScopeSpec,
)
from wheelta_robinhood_agent.agent.result_boundary import (
    CANDIDATE_REF_PREFIX,
    CandidateEvidence,
    MappedEvidence,
    MappingRequest,
)
from wheelta_robinhood_agent.domain.account import AccountSnapshot
from wheelta_robinhood_agent.domain.enums import CandidateOrigin, DataQuality
from wheelta_robinhood_agent.domain.facts_compute import (
    OpenOrdersRead,
    OptionInstrument,
    PositionsRead,
)
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY

ACCOUNT_NUMBER = "5550001234"
OTHER_ACCOUNT_NUMBER = "5550009876"
OTHER_ACCOUNT_NICKNAME = "OTHER-ACCOUNT-MUST-NOT-PERSIST"
INSTRUMENT_ID = "inst-aapl-150p"
OCC = "AAPL  261016P00150000"
RAW_MARKER = "RAW-UNMAPPED-PAYLOAD-MARKER"


def _text(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def build_world(as_of: datetime) -> FakeWorld:
    stamp = as_of.isoformat()

    def chains(args: dict[str, Any]) -> dict[str, Any]:
        return _text(
            {
                "as_of": stamp,
                "contracts": [
                    {
                        "instrument_id": INSTRUMENT_ID,
                        "occ": OCC,
                        "underlying": "AAPL",
                        "multiplier": 100,
                    }
                ],
            }
        )

    def quotes(args: dict[str, Any]) -> dict[str, Any]:
        return _text(
            {
                "quotes": [
                    {"instrument_id": INSTRUMENT_ID, "bid": "1.20", "ask": "1.30", "as_of": stamp}
                ]
            }
        )

    def portfolio(args: dict[str, Any]) -> dict[str, Any]:
        return _text(
            {
                "as_of": stamp,
                "agentic": True,
                "account_value": "150000.00",
                "settled_cash": "30000.00",
                "csp_reserved": "0.00",
            }
        )

    def empty(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return lambda args: _text({"as_of": stamp, key: []})

    def accounts(args: dict[str, Any]) -> dict[str, Any]:
        # The captured get_accounts shape (tests/fixtures/robinhood/results/), with a second,
        # non-Agentic account that trusted code must drop.
        base = {
            "type": "cash",
            "state": "active",
            "deactivated": False,
            "permanently_deactivated": False,
            "option_level": "option_level_2",
        }
        return _text(
            {
                "data": {
                    "accounts": [
                        {**base, "account_number": ACCOUNT_NUMBER, "agentic_allowed": True},
                        {
                            **base,
                            "account_number": OTHER_ACCOUNT_NUMBER,
                            "agentic_allowed": False,
                            "nickname": OTHER_ACCOUNT_NICKNAME,
                        },
                    ]
                },
                "guide": "prose",
            }
        )

    def unmapped(args: dict[str, Any]) -> dict[str, Any]:
        return _text({"symbol": "AAPL", "price": "180.00", "note": RAW_MARKER})

    def place(args: dict[str, Any]) -> dict[str, Any]:  # must never be reached
        raise AssertionError("an order tool reached the fake broker")

    return FakeWorld(
        handlers={
            "robinhood": {
                "get_accounts": accounts,
                "get_option_chains": chains,
                "get_option_quotes": quotes,
                "get_portfolio": portfolio,
                "get_option_positions": empty("positions"),
                "get_option_orders": empty("orders"),
                "get_equity_quotes": unmapped,
                "place_option_order": place,
                "review_option_order": place,
                "cancel_option_order": place,
                # Every other registered tool exists on the fake so discovery is complete.
                **{
                    t.name: unmapped
                    for t in ROBINHOOD_REGISTRY.tools
                    if t.name
                    not in {
                        "get_accounts",
                        "get_option_chains",
                        "get_option_quotes",
                        "get_portfolio",
                        "get_option_positions",
                        "get_option_orders",
                        "get_equity_quotes",
                        "place_option_order",
                        "review_option_order",
                        "cancel_option_order",
                    }
                },
            }
        },
        web_results={"AAPL earnings date": [{"title": "AAPL earnings", "url": "https://x.test"}]},
    )


# -- fixture mappers (fake schemas only) -------------------------------------------------------


def _as_of(req: MappingRequest, payload: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(payload["as_of"]))


def map_chains(req: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    payload = req.payload
    assert isinstance(payload, dict)
    instruments, candidates = [], []
    for row in payload["contracts"]:
        occ = OccSymbol.parse(row["occ"])
        inst = OptionInstrument(
            evidence_id=new_id(),
            as_of=_as_of(req, payload),
            source_tool_call_ids=(req.tool_call_id,),
            occ_symbol=occ,
            broker_instrument_id=row["instrument_id"],
            underlying=row["underlying"],
            multiplier=row["multiplier"],
        )
        instruments.append(inst)
        candidates.append(
            CandidateEvidence(
                candidate_ref=f"{CANDIDATE_REF_PREFIX}{new_id()}",
                origin=CandidateOrigin.ROBINHOOD,
                underlying=inst.underlying,
                instrument_evidence_id=inst.evidence_id,
                broker_instrument_id=inst.broker_instrument_id,
                occ_symbol=occ,
            )
        )
    return MappedEvidence(instruments=tuple(instruments), candidates=tuple(candidates))


def map_quotes(req: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    payload = req.payload
    assert isinstance(payload, dict)
    return MappedEvidence(
        option_quotes=tuple(
            Quote(
                quote_id=new_id(),
                broker_instrument_id=q["instrument_id"],
                bid=Decimal(q["bid"]),
                ask=Decimal(q["ask"]),
                as_of=datetime.fromisoformat(q["as_of"]),
                source_tool_call_ids=(req.tool_call_id,),
            )
            for q in payload["quotes"]
        )
    )


def map_portfolio(req: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    payload = req.payload
    assert isinstance(payload, dict)
    snapshot_id = new_id()
    cash, reserved = Decimal(payload["settled_cash"]), Decimal(payload["csp_reserved"])
    return MappedEvidence(
        account_snapshots=(
            AccountSnapshot(
                snapshot_id=snapshot_id,
                as_of=_as_of(req, payload),
                retrieved_at=req.retrieved_at,
                tool_call_ids=(req.tool_call_id,),
                account_ref="x1234",
                agentic_verified=payload["agentic"] is True,
                account_value_usd=Decimal(payload["account_value"]),
                available_settled_cash_usd=cash,
                csp_reserved_cash_usd=reserved,
                csp_cash_base_usd=cash + reserved,
                csp_cash_base_evidence_ids=(snapshot_id,),
                positions_ref=None,
                open_orders_ref=None,
                tax_lots_ref=None,
                quality=DataQuality.OK,
            ),
        )
    )


def map_positions(req: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    payload = req.payload
    assert isinstance(payload, dict) and payload["positions"] == []
    return MappedEvidence(
        positions=(
            PositionsRead(
                evidence_id=new_id(),
                as_of=_as_of(req, payload),
                source_tool_call_ids=(req.tool_call_id,),
            ),
        )
    )


def map_orders(req: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    payload = req.payload
    assert isinstance(payload, dict) and payload["orders"] == []
    return MappedEvidence(
        open_orders=(
            OpenOrdersRead(
                evidence_id=new_id(),
                as_of=_as_of(req, payload),
                source_tool_call_ids=(req.tool_call_id,),
            ),
        )
    )


FIXTURE_MAPPERS = {
    ("robinhood", "get_option_chains"): map_chains,
    ("robinhood", "get_option_quotes"): map_quotes,
    ("robinhood", "get_portfolio"): map_portfolio,
    ("robinhood", "get_option_positions"): map_positions,
    ("robinhood", "get_option_orders"): map_orders,
}

# The fake broker's account-scoped reads take `account_number` (fixture schema, not verified).
FIXTURE_SCOPE_TABLE: dict[str, AccountScopeSpec] = {
    **ROBINHOOD_ACCOUNT_SCOPE,
    "get_portfolio": AccountScopeSpec.verified("account_number"),
    "get_option_positions": AccountScopeSpec.verified("account_number"),
    "get_option_orders": AccountScopeSpec.verified("account_number"),
}


# -- scripted agents ---------------------------------------------------------------------------


def decision_json(
    candidate_ref: str,
    facts_ref: str,
    limit_price: str = "1.25",
    next_run: dict[str, str] | None = None,
) -> str:
    return json.dumps(
        {
            "decisions": [
                {
                    "action": "OPEN_CSP",
                    "target_ref": candidate_ref,
                    "replacement_ref": None,
                    "funding_close_refs": [],
                    "proposed_legs": [{"facts_ref": facts_ref, "limit_price": limit_price}],
                    "execution_refs": [],
                    "rationale": "Scripted e2e choice.",
                    "thesis": "Scripted thesis.",
                    "invalidation_conditions": ["Scripted invalidation."],
                    "evidence_refs": [facts_ref],
                }
            ],
            "cancellation_rationales": [],
            "unresolved_questions": [],
            "next_run": next_run,
        }
    )


# Mignon agent names on the e2e session model (tests/e2e/conftest.py AGENT_MODEL), the
# default allowlist when MIGNON_AGENT_MODELS is unset (ADR-0025).
E2E_MODEL = "claude-test-model"
MARKET, COMPANY, MACRO = (f"mignon-{t}--{E2E_MODEL}" for t in ("market", "company", "macro"))


def mignon_report(task: str, *findings: tuple[str, list[str]]) -> str:
    """A MignonReport v1 JSON text (domain/mignon_report.py)."""
    return json.dumps(
        {
            "task": task,
            "findings": [{"claim": c, "refs": refs, "web_urls": []} for c, refs in findings],
            "gaps": [],
            "follow_up_questions": [],
        }
    )


async def market_mignon(model: FakeModel) -> str:
    """A market Mignon: screen AAPL puts from the chain and a live quote (ADR-0025)."""
    chains = await model.call("mcp__robinhood__get_option_chains", {"symbol": "AAPL"})
    candidate_ref = chains.data["evidence"]["candidates"][0]["candidate_ref"]
    quotes = await model.call(
        "mcp__robinhood__get_option_quotes", {"instrument_ids": [INSTRUMENT_ID]}
    )
    return mignon_report(
        "Screen AAPL cash-secured puts.",
        ("The AAPL put candidate from the chain.", [chains.data["evidence_ref"], candidate_ref]),
        ("Its live quote was returned.", [quotes.data["evidence_ref"]]),
    )


async def research(model: FakeModel) -> tuple[str, str]:
    """Delegate screening to a market Mignon, then establish state and facts as the
    orchestrator; return (candidate_ref, facts_ref)."""
    turn = await model.spawn(MARKET, "Screen AAPL cash-secured puts.", market_mignon)
    assert turn.output["kind"] == "validated", turn.output
    refs = [r for f in turn.data["report"]["findings"] for r in f["refs"]]
    candidate_ref = next(r for r in refs if r.startswith("candidate:"))
    await model.call("mcp__robinhood__get_option_quotes", {"instrument_ids": [INSTRUMENT_ID]})
    account = {"account_number": ACCOUNT_NUMBER}
    await model.call("mcp__robinhood__get_portfolio", account)
    await model.call("mcp__robinhood__get_option_positions", account)
    await model.call("mcp__robinhood__get_option_orders", account)
    facts = await model.call(
        "mcp__wra_local__get_decision_facts",
        {"subject_ref": candidate_ref, "purpose": "open", "limit_price": "1.25"},
    )
    return candidate_ref, facts.data["facts_ref"]


async def dry_run_script(model: FakeModel) -> str | None:
    candidate_ref, facts_ref = await research(model)
    return decision_json(candidate_ref, facts_ref)
