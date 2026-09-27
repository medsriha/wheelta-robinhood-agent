"""Agentic-account eligibility from `get_accounts` (CLAUDE.md §9, §24; ADR-0023).

Built from the scrubbed capture `tests/fixtures/robinhood/results/get_accounts.agentic_only.json`.
"""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from wheelta_robinhood_agent.domain.account import AgenticEligibility
from wheelta_robinhood_agent.integrations.robinhood.accounts import check_eligibility

NOW = datetime(2026, 9, 28, 13, 35, tzinfo=UTC)
FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "robinhood"
    / "results"
    / "get_accounts.agentic_only.json"
)
NUMBER = "5QR12345678"


def listing(**overrides: Any) -> dict[str, Any]:
    """The captured payload (provenance dropped) with the scrubbed number replaced."""
    captured = json.loads(FIXTURE.read_text())
    payload = {"data": copy.deepcopy(captured["data"]), "guide": "prose"}
    (account,) = payload["data"]["accounts"]
    account.update(account_number=NUMBER, **overrides)
    return payload


def check(payload: Any, number: str = NUMBER) -> AgenticEligibility:
    return check_eligibility(payload, SecretStr(number), NOW)


def test_captured_agentic_account_is_eligible() -> None:
    result = check(listing())
    assert result.eligible and result.reasons == ()
    assert result.account_ref == "****5678"
    assert result.account_type == "cash" and result.option_level == "option_level_2"
    assert result.retrieved_at == NOW


def test_other_accounts_are_dropped_and_never_returned() -> None:
    payload = listing()
    payload["data"]["accounts"].append(
        {"account_number": "9ZZ00000001", "nickname": "Retirement", "agentic_allowed": False}
    )
    result = check(payload)
    assert result.eligible
    dumped = result.model_dump_json()
    assert "9ZZ00000001" not in dumped and "Retirement" not in dumped


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"agentic_allowed": False}, "agentic_allowed is false"),
        ({"state": "restricted"}, "account state is not active"),
        ({"deactivated": True}, "account is deactivated"),
        ({"permanently_deactivated": True}, "account is deactivated"),
    ],
)
def test_each_failed_condition_is_named(overrides: dict[str, Any], reason: str) -> None:
    result = check(listing(**overrides))
    assert not result.eligible and reason in result.reasons


def test_last_four_digits_are_not_identity() -> None:
    other = "9ZZ99995678"  # same last four, different account
    result = check(listing(), number=other)
    assert not result.eligible
    assert result.reasons == ("the configured account is not in the account listing",)


def test_duplicate_listing_is_ineligible() -> None:
    payload = listing()
    payload["data"]["accounts"].append(dict(payload["data"]["accounts"][0]))
    assert "more than once" in check(payload).reasons[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {}},
        {"data": {"accounts": "nope"}},
        listing(agentic_allowed="true"),  # strings are not booleans
        "not an object",
    ],
)
def test_malformed_listing_raises(payload: Any) -> None:
    with pytest.raises((ValidationError, ValueError)):
        check(payload)


def test_malformed_error_never_echoes_the_account_number() -> None:
    with pytest.raises(ValidationError) as info:
        check(listing(agentic_allowed="yes"))
    assert NUMBER not in str(info.value)


def test_eligibility_model_invariants() -> None:
    with pytest.raises(ValidationError, match="redacted"):
        AgenticEligibility(account_ref="12345678", eligible=True, retrieved_at=NOW)
    with pytest.raises(ValidationError, match="exactly when"):
        AgenticEligibility(account_ref="****5678", eligible=True, reasons=("x",), retrieved_at=NOW)
    with pytest.raises(ValidationError, match="exactly when"):
        AgenticEligibility(account_ref="****5678", eligible=False, retrieved_at=NOW)
