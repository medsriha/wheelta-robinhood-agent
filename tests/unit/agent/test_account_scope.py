import pytest
from pydantic import SecretStr

from wheelta_robinhood_agent.agent.account_scope import (
    NOT_SCOPED,
    ROBINHOOD_ACCOUNT_SCOPE,
    UNVERIFIED,
    AccountScope,
    AccountScopeSpec,
    account_scope_for,
    check_account_scope,
)
from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

ACCOUNT = SecretStr("5QR12345678")


def test_every_registered_robinhood_tool_has_a_spec_and_account_tools_are_unverified() -> None:
    for tool in ROBINHOOD_REGISTRY.tools:
        if tool.tier is ToolTier.EXCLUDED or (tool.tier is ToolTier.X and not tool.live_order_tool):
            continue  # never callable; absent from the table means unverified anyway
        assert tool.name in ROBINHOOD_ACCOUNT_SCOPE, tool.name
    for name in ("get_accounts", "get_portfolio", "get_option_positions", "get_option_orders"):
        assert ROBINHOOD_ACCOUNT_SCOPE[name] is UNVERIFIED
    for tool in ROBINHOOD_REGISTRY.by_tier(ToolTier.S) + ROBINHOOD_REGISTRY.by_tier(ToolTier.X):
        assert account_scope_for("robinhood", tool.name).scope is AccountScope.UNVERIFIED


def test_no_spec_is_verified_while_schemas_are_uncaptured() -> None:
    assert all(s.scope is not AccountScope.VERIFIED for s in ROBINHOOD_ACCOUNT_SCOPE.values())


def test_market_data_tools_are_not_scoped() -> None:
    for name in ("get_option_chains", "get_option_quotes", "get_earnings_calendar"):
        assert account_scope_for("robinhood", name) is NOT_SCOPED


def test_unknown_robinhood_tool_and_unknown_server_are_unverified() -> None:
    assert account_scope_for("robinhood", "get_new_thing") is UNVERIFIED
    assert account_scope_for("other", "anything") is UNVERIFIED


def test_wheelta_tools_are_not_scoped() -> None:
    for tool in WHEELTA_REGISTRY.tools:
        assert account_scope_for("wheelta", tool.name) is NOT_SCOPED


def test_custom_table_is_used() -> None:
    spec = AccountScopeSpec.verified("account_number")
    assert account_scope_for("robinhood", "x", {"x": spec}) is spec


def test_spec_invariant() -> None:
    with pytest.raises(ValueError):
        AccountScopeSpec(AccountScope.VERIFIED)
    with pytest.raises(ValueError):
        AccountScopeSpec(AccountScope.UNVERIFIED, "account_number")


def test_unverified_always_denied() -> None:
    reason = check_account_scope(UNVERIFIED, {"account_number": "5QR12345678"}, ACCOUNT)
    assert reason is not None and "unverified" in reason


def test_not_scoped_passes_without_account_argument() -> None:
    assert check_account_scope(NOT_SCOPED, {"symbol": "AAPL"}, ACCOUNT) is None


def test_not_scoped_with_account_argument_is_denied() -> None:
    assert check_account_scope(NOT_SCOPED, {"account_number": "x"}, ACCOUNT) is not None


VERIFIED = AccountScopeSpec.verified("account_number")


def test_verified_full_match_passes() -> None:
    assert check_account_scope(VERIFIED, {"account_number": "5QR12345678"}, ACCOUNT) is None


@pytest.mark.parametrize(
    "tool_input",
    [
        {},  # the broker's default account is never assumed
        {"account_number": ""},
        {"account_number": 5678},
        {"account_number": "5678"},  # last four is not identity
        {"account_number": "OTHER0005678"},  # non-Agentic account
        {"account_number": "5QR12345678", "account_id": "OTHER"},
    ],
)
def test_verified_mismatch_is_denied(tool_input: dict[str, object]) -> None:
    assert check_account_scope(VERIFIED, tool_input, ACCOUNT) is not None
