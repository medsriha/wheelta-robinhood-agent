"""Agentic-account eligibility from `get_accounts` (CLAUDE.md §9 "Agentic workspace", §24).

Trusted code only: the session calls `get_accounts` through the validating proxy's upstream
before the model starts (agent/session.py), never through the model. `check_eligibility`
reduces the listing to the configured account's redacted eligibility; every other account is
dropped here and never returned, logged, or persisted.

Shape from the scrubbed capture `tests/fixtures/robinhood/results/get_accounts.agentic_only.json`
(robinhood-trading 1.6.0, 2026-09-27): `{"data": {"accounts": [...]}, "guide": ...}`. Only the
fields relied on are validated; unknown fields are ignored (external payload, CLAUDE.md §4).
Matching is on the full `account_number`; last-four digits are never identity.

The configured account is eligible only if the listing contains it exactly once with
`agentic_allowed` true, `state` "active", and both `deactivated` flags false. Anything else
is ineligible with a named reason. A malformed listing raises `ValueError` (the caller treats
the check as failed, never as eligible).
"""

import hmac
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, SecretStr, StrictBool, StrictStr

from wheelta_robinhood_agent.domain.account import AgenticEligibility
from wheelta_robinhood_agent.observability.redaction import mask_account

__all__ = ["check_eligibility"]


class _Account(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    account_number: StrictStr
    agentic_allowed: StrictBool
    state: StrictStr
    deactivated: StrictBool
    permanently_deactivated: StrictBool
    type: StrictStr | None = None
    option_level: StrictStr | None = None


class _Listing(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    accounts: tuple[dict[str, Any], ...]


def _unwrap(payload: JsonValue) -> JsonValue:
    if isinstance(payload, Mapping) and "data" in payload:
        return payload["data"]
    return payload


def check_eligibility(
    payload: JsonValue, account_number: SecretStr, retrieved_at: datetime
) -> AgenticEligibility:
    """Reduce a `get_accounts` payload to the configured account's eligibility."""
    listing = _Listing.model_validate(_unwrap(payload))
    wanted = account_number.get_secret_value()
    matches = []
    for raw in listing.accounts:
        number = raw.get("account_number")
        if isinstance(number, str) and hmac.compare_digest(number, wanted):
            # Only the configured account is parsed; others are never inspected further.
            matches.append(_Account.model_validate(raw))
    ref = mask_account(wanted)
    if len(matches) != 1:
        reason = (
            "the configured account is not in the account listing"
            if not matches
            else "the configured account appears more than once in the listing"
        )
        return AgenticEligibility(
            account_ref=ref, eligible=False, reasons=(reason,), retrieved_at=retrieved_at
        )
    (account,) = matches
    reasons: list[str] = []
    if not account.agentic_allowed:
        reasons.append("agentic_allowed is false")
    if account.state != "active":
        reasons.append("account state is not active")
    if account.deactivated or account.permanently_deactivated:
        reasons.append("account is deactivated")
    return AgenticEligibility(
        account_ref=ref,
        eligible=not reasons,
        reasons=tuple(reasons),
        account_type=account.type,
        option_level=account.option_level,
        retrieved_at=retrieved_at,
    )
