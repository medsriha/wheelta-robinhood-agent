"""Per-tool account scope for the PreToolUse hook (CLAUDE.md §9, §18, §24).

Every account-specific Robinhood call (reads included) must be confined to the full configured
`ROBINHOOD_AGENTIC_ACCOUNT_NUMBER`. Read-tool input schemas were captured from
`robinhood-trading` 1.6.0 on 2026-09-26 (ADR-0017): every account-specific read takes a required
`account_number`, so those tools are VERIFIED on that argument.

`LOGIN_SCOPED` (allowed by owner decision, ADR-0026): workspace reads (scans, watchlists,
alerts), whose schemas carry no account argument because they belong to the Robinhood login
rather than an account. Their results may include the login's non-Agentic objects.

Still `UNVERIFIED` (denied, CLAUDE.md §18): every Tier S/X tool, whose schemas weren't
captured. A Robinhood tool missing from the table is `UNVERIFIED` (fail closed).

Account discovery (`get_accounts`) is trusted-code-only (CLAUDE.md §9): the model never
calls it, so it stays `UNVERIFIED` here. Matching is on the full account number only;
last-four display digits are never identity.

ADR-0030: the model never sees the full number. It passes `AGENTIC_ACCOUNT_PLACEHOLDER` as
the account argument, and trusted code substitutes the configured number only on the call
sent upstream (`resolve_account_argument`: the validating proxy's upstream call, or the
hook's `updatedInput` for a direct server). The full number itself is still accepted.
"""

import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from pydantic import SecretStr

from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME as ROBINHOOD
from wheelta_robinhood_agent.integrations.wheelta.registry import SERVER_NAME as WHEELTA
from wheelta_robinhood_agent.observability.redaction import is_account_key

# ADR-0030: the account argument the agent passes; not account-shaped, so it stays readable
# in redacted ledger rows. Code replaces it with the configured number before the call leaves.
AGENTIC_ACCOUNT_PLACEHOLDER = "AGENTIC_ACCOUNT"


class AccountScope(StrEnum):
    """How a tool relates to the Agentic account."""

    NOT_ACCOUNT_SCOPED = "not_account_scoped"  # market/reference data; no account argument
    VERIFIED = "verified"  # the account argument's name is captured from tools/list
    UNVERIFIED = "unverified"  # account-specific, or might be; schema not captured: withheld
    # Belongs to the Robinhood login, not an account (no account argument). Allowed by owner
    # decision (ADR-0026): results may include the login's non-Agentic objects.
    LOGIN_SCOPED = "login_scoped"


@dataclass(frozen=True, slots=True)
class AccountScopeSpec:
    scope: AccountScope
    account_arg: str | None = None

    def __post_init__(self) -> None:
        if (self.scope is AccountScope.VERIFIED) != (self.account_arg is not None):
            raise ValueError("account_arg is required exactly when the scope is verified")

    @classmethod
    def verified(cls, account_arg: str) -> "AccountScopeSpec":
        return cls(AccountScope.VERIFIED, account_arg)


NOT_SCOPED = AccountScopeSpec(AccountScope.NOT_ACCOUNT_SCOPED)
LOGIN_SCOPED = AccountScopeSpec(AccountScope.LOGIN_SCOPED)
UNVERIFIED = AccountScopeSpec(AccountScope.UNVERIFIED)

# Market and reference data: quotes, chains, instruments, fundamentals, earnings, filings,
# public lists. None of these read account state.
_NOT_SCOPED_TOOLS = (
    "search",
    "get_equity_quotes", "get_equity_historicals", "get_equity_fundamentals",
    "get_equity_price_book", "get_equity_technical_indicators", "get_equity_analyst_ratings",
    "get_option_chains", "get_option_instruments", "get_option_quotes", "get_option_historicals",
    "get_indexes", "get_index_quotes", "get_index_historicals",
    "get_earnings_calendar", "get_earnings_results", "get_financials",
    "get_sec_filing_index", "get_sec_filing", "get_sec_filing_facts",
    "get_sec_filing_facts_catalog", "get_politician_trades", "get_scanner_filter_specs",
    "get_scanner_datapoints", "preview_scan", "get_popular_watchlists",
)  # fmt: skip

# Account-specific reads whose captured schema requires `account_number` (ADR-0017).
_ACCOUNT_NUMBER_TOOLS = (
    "get_portfolio", "get_realized_pnl", "get_pnl_trade_history",
    "get_limited_margin_upgrade_info", "get_option_level_upgrade_info",
    "get_equity_tradability", "get_equity_tax_lots", "get_equity_positions", "get_equity_orders",
    "get_option_positions", "get_option_orders",
)  # fmt: skip

# Workspace objects belong to the Robinhood login, not an account: no account argument in the
# captured schemas (reads: ADR-0017, allowed by ADR-0026; Tier S writes: tools/list of
# 2026-09-27, ADR-0027, still subject to the hook's ownership checks).
LOGIN_SCOPED_TOOLS = (
    "get_option_watchlist", "get_scans", "run_scan", "get_watchlists", "get_watchlist_items",
    "get_alerts", "get_alert_log",
    "create_scan", "update_scan_filters", "update_scan_config",
    "create_watchlist", "update_watchlist", "add_to_watchlist", "remove_from_watchlist",
    "follow_watchlist", "unfollow_watchlist",
    "add_option_to_watchlist", "remove_option_from_watchlist",
    "create_alert", "update_alert", "delete_alert", "mark_alerts_read",
)  # fmt: skip

# Account discovery (trusted code only) and the Tier X order tools (their account argument
# is not wired yet: phase 2). Listed explicitly so the table documents them.
_UNVERIFIED_TOOLS = (
    "get_accounts",
    "review_option_order", "place_option_order", "cancel_option_order",
)  # fmt: skip

ROBINHOOD_ACCOUNT_SCOPE: Mapping[str, AccountScopeSpec] = MappingProxyType(
    {
        **{name: NOT_SCOPED for name in _NOT_SCOPED_TOOLS},
        **{name: AccountScopeSpec.verified("account_number") for name in _ACCOUNT_NUMBER_TOOLS},
        **{name: LOGIN_SCOPED for name in LOGIN_SCOPED_TOOLS},
        **{name: UNVERIFIED for name in _UNVERIFIED_TOOLS},
    }
)

# Servers whose tools are never account-scoped: Wheelta is read-only market data (§10); the
# local server's tools read only this run's already account-scoped ledger evidence and take
# no account argument (web_cache_lookup, get_decision_facts).
_UNSCOPED_SERVERS = frozenset({WHEELTA, LOCAL_SERVER_NAME})


def account_scope_for(
    server: str,
    tool: str,
    table: Mapping[str, AccountScopeSpec] = ROBINHOOD_ACCOUNT_SCOPE,
) -> AccountScopeSpec:
    """The spec for one registered tool. Robinhood tools missing from `table` are unverified;
    tools of any server that is neither Robinhood nor known-unscoped are unverified too."""
    if server == ROBINHOOD:
        return table.get(tool, UNVERIFIED)
    if server in _UNSCOPED_SERVERS:
        return NOT_SCOPED
    return UNVERIFIED


def check_account_scope(
    spec: AccountScopeSpec,
    tool_input: Mapping[str, object],
    account_number: SecretStr,
) -> str | None:
    """Return None if the call is confined to the Agentic account, else the deny reason.

    - NOT_ACCOUNT_SCOPED: allowed, unless a top-level argument looks like an account
      identifier (the schema would then be wrong about the tool; fail closed).
    - UNVERIFIED: always denied (CLAUDE.md §18).
    - LOGIN_SCOPED: allowed (ADR-0026), unless an argument looks like an account identifier.
    - VERIFIED: the account argument must be present, a string, and either the placeholder
      (ADR-0030) or equal to the full configured account number (constant-time compare). A
      missing argument is denied: the broker's default account is never assumed.
    """
    if spec.scope is AccountScope.UNVERIFIED:
        return "account scope unverified for this tool; withheld until tools/list is captured"
    if spec.scope in (AccountScope.NOT_ACCOUNT_SCOPED, AccountScope.LOGIN_SCOPED):
        if any(is_account_key(key) for key in tool_input):
            return "unexpected account argument on a tool without account scope"
        return None
    arg = str(spec.account_arg)  # set whenever VERIFIED (AccountScopeSpec.__post_init__)
    if any(is_account_key(key) and key != arg for key in tool_input):
        return "unexpected second account argument"
    value = tool_input.get(arg)
    if not isinstance(value, str) or not value:
        return f"account argument {arg!r} missing; the default account is never assumed"
    if value == AGENTIC_ACCOUNT_PLACEHOLDER:
        return None
    expected = account_number.get_secret_value()
    if not hmac.compare_digest(value.encode(), expected.encode()):
        return "account argument does not match the configured Agentic account"
    return None


def resolve_account_argument(
    spec: AccountScopeSpec,
    tool_input: Mapping[str, object],
    account_number: SecretStr,
) -> dict[str, object] | None:
    """The upstream arguments with the placeholder replaced by the configured account number,
    or None when nothing is replaced (ADR-0030). Call only after `check_account_scope` passed.
    """
    if spec.scope is not AccountScope.VERIFIED:
        return None
    arg = str(spec.account_arg)
    if tool_input.get(arg) != AGENTIC_ACCOUNT_PLACEHOLDER:
        return None
    return {**tool_input, arg: account_number.get_secret_value()}


# scrypt parameters for `account_scope_id`. Never change them: every ledger row keyed by the
# account scope would stop matching. The salt is a fixed domain separator, not a secret.
_SCOPE_SALT = b"wheelta-robinhood-agent/account-scope/v1"
_SCOPE_N, _SCOPE_R, _SCOPE_P, _SCOPE_LEN = 2**14, 8, 1, 16


def account_scope_id(account_number: SecretStr) -> str:
    """The ledger's `account_scope_id` for the configured Agentic account.

    Deterministic and one-way, so ledger rows are keyed to the full account number without
    storing it (CLAUDE.md §7). A plain hash of a short digit string is cheap to brute force;
    scrypt makes that costly. Last-four digits are never identity (CLAUDE.md §18).
    """
    raw = account_number.get_secret_value()
    if not raw:
        raise ValueError("account number must be non-empty")
    digest = hashlib.scrypt(
        raw.encode("utf-8"),
        salt=_SCOPE_SALT,
        n=_SCOPE_N,
        r=_SCOPE_R,
        p=_SCOPE_P,
        dklen=_SCOPE_LEN,
    )
    return f"agentic:{digest.hex()}"
