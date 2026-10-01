import pytest
from pydantic import SecretStr

from wheelta_robinhood_agent.agent.account_scope import (
    AGENTIC_ACCOUNT_PLACEHOLDER,
    NOT_SCOPED,
    ROBINHOOD_ACCOUNT_SCOPE,
    UNVERIFIED,
    AccountScope,
    AccountScopeSpec,
    account_scope_for,
    check_account_scope,
    resolve_account_argument,
)
from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

ACCOUNT = SecretStr("5QR12345678")


def test_every_registered_robinhood_tool_has_a_spec() -> None:
    for tool in ROBINHOOD_REGISTRY.tools:
        if tool.tier is ToolTier.EXCLUDED or (tool.tier is ToolTier.X and not tool.live_order_tool):
            continue  # never callable; absent from the table means unverified anyway
        assert tool.name in ROBINHOOD_ACCOUNT_SCOPE, tool.name


def test_account_reads_verified_on_account_number_from_capture() -> None:
    # ADR-0017: every account-specific read in robinhood-trading 1.6.0 requires account_number.
    for name in (
        "get_portfolio", "get_realized_pnl", "get_pnl_trade_history",
        "get_limited_margin_upgrade_info", "get_option_level_upgrade_info",
        "get_equity_tradability", "get_equity_tax_lots", "get_equity_positions",
        "get_equity_orders", "get_option_positions", "get_option_orders",
    ):  # fmt: skip
        assert ROBINHOOD_ACCOUNT_SCOPE[name] == AccountScopeSpec.verified("account_number"), name


def test_workspace_reads_are_login_scoped_discovery_unverified_orders_account_scoped() -> None:
    """ADR-0026: the owner accepted login-scoped workspace reads."""
    for name in (
        "get_scans",
        "run_scan",
        "get_watchlists",
        "get_watchlist_items",
        "get_option_watchlist",
        "get_alerts",
        "get_alert_log",
    ):
        assert ROBINHOOD_ACCOUNT_SCOPE[name].scope is AccountScope.LOGIN_SCOPED, name
        assert check_account_scope(ROBINHOOD_ACCOUNT_SCOPE[name], {}, SecretStr("5QR1")) is None
        assert check_account_scope(
            ROBINHOOD_ACCOUNT_SCOPE[name], {"account_number": "5QR1"}, SecretStr("5QR1")
        )  # an account argument on a login-scoped tool is still denied
    assert ROBINHOOD_ACCOUNT_SCOPE["get_accounts"] is UNVERIFIED
    for tool in ROBINHOOD_REGISTRY.by_tier(ToolTier.S):  # no account argument (ADR-0027)
        assert account_scope_for("robinhood", tool.name).scope is AccountScope.LOGIN_SCOPED
    for tool in ROBINHOOD_REGISTRY.by_tier(ToolTier.X):
        scope = account_scope_for("robinhood", tool.name).scope
        if tool.live_order_tool:  # ADR-0034: required account_number, verified
            assert scope is AccountScope.VERIFIED, tool.name
            spec = account_scope_for("robinhood", tool.name)
            assert check_account_scope(spec, {}, SecretStr("5QR1"))  # missing: denied
            assert check_account_scope(spec, {"account_number": "5QR1"}, SecretStr("5QR1")) is None
        else:
            assert scope is AccountScope.UNVERIFIED, tool.name


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


# -- ADR-0030: the agent passes a placeholder; code substitutes the configured number ------------


def test_placeholder_passes_and_resolves_to_the_configured_number() -> None:
    tool_input = {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER, "limit": 5}
    assert check_account_scope(VERIFIED, tool_input, ACCOUNT) is None
    resolved = resolve_account_argument(VERIFIED, tool_input, ACCOUNT)
    assert resolved == {"account_number": "5QR12345678", "limit": 5}
    assert tool_input["account_number"] == AGENTIC_ACCOUNT_PLACEHOLDER  # input left unchanged


def test_nothing_to_resolve_without_the_placeholder() -> None:
    assert resolve_account_argument(VERIFIED, {"account_number": "5QR12345678"}, ACCOUNT) is None
    assert resolve_account_argument(NOT_SCOPED, {"symbol": "AAPL"}, ACCOUNT) is None
    # A placeholder under a non-account key, or on an unscoped tool, is never substituted.
    assert (
        resolve_account_argument(NOT_SCOPED, {"note": AGENTIC_ACCOUNT_PLACEHOLDER}, ACCOUNT) is None
    )


@pytest.mark.parametrize("value", ["agentic_account", "AGENTIC_ACCOUNT ", "****5678", "AGENTIC"])
def test_near_placeholders_are_denied(value: str) -> None:
    assert check_account_scope(VERIFIED, {"account_number": value}, ACCOUNT) is not None


def test_placeholder_is_not_account_shaped_so_it_stays_readable_in_the_ledger() -> None:
    from wheelta_robinhood_agent.observability.redaction import Redactor

    redacted = Redactor(account_number=ACCOUNT).redact_mapping(
        {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER}
    )
    assert redacted == {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER}


def test_the_order_work_server_name_is_pinned_and_unscoped() -> None:
    """ADR-0066: spelled out in account_scope.py (order_walk imports it); never scoped."""
    from wheelta_robinhood_agent.agent.account_scope import (
        ORDER_WORK_SERVER_NAME,
        AccountScope,
        account_scope_for,
    )
    from wheelta_robinhood_agent.agent.order_walk import AWAIT_TOOL, ORDER_WORK_SERVER, WORK_TOOL

    assert ORDER_WORK_SERVER_NAME == ORDER_WORK_SERVER
    for tool in (WORK_TOOL, AWAIT_TOOL):
        assert account_scope_for(ORDER_WORK_SERVER, tool).scope is AccountScope.NOT_ACCOUNT_SCOPED
