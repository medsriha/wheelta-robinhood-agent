"""Robinhood Trading MCP tool-tier registry (ADR-0005, amended by ADR-0006 and ADR-0017).

Names are verified against our own capture of `robinhood-trading` 1.6.0 on 2026-09-26
(tests/fixtures/robinhood/tools_2026-09-26.json, ADR-0017). Runtime discovery still diffs
against this registry: unknown tools are denied, expected tools that are missing fail the stage.
Tier S and Tier X input schemas were not captured (Claude Code deny rules hide them).
"""

from wheelta_robinhood_agent.domain.enums import ToolTier
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, make_registry

SERVER_NAME = "robinhood"

# ADR-0006: the only Tier X tools the agent may ever call, and only in armed live mode.
LIVE_ORDER_TOOLS = ("review_option_order", "place_option_order", "cancel_option_order")

_TIER_R = (
    # account and portfolio
    "get_accounts", "get_portfolio", "search", "get_realized_pnl", "get_pnl_trade_history",
    "get_limited_margin_upgrade_info",
    # equities
    "get_equity_quotes", "get_equity_historicals", "get_equity_fundamentals",
    "get_equity_price_book", "get_equity_technical_indicators", "get_equity_tradability",
    "get_equity_tax_lots", "get_equity_positions", "get_equity_orders",
    "get_equity_analyst_ratings",
    # options
    "get_option_chains", "get_option_instruments", "get_option_quotes",
    "get_option_historicals", "get_option_positions", "get_option_orders",
    "get_option_watchlist", "get_option_level_upgrade_info",
    # indexes, earnings, financials, filings
    "get_indexes", "get_index_quotes", "get_index_historicals", "get_earnings_calendar",
    "get_earnings_results", "get_financials", "get_sec_filing_index", "get_sec_filing",
    "get_sec_filing_facts", "get_sec_filing_facts_catalog", "get_politician_trades",
    # scanner, watchlists, alerts
    "get_scans", "run_scan", "get_scanner_filter_specs", "get_scanner_datapoints",
    # a read-only, non-persisting scan evaluation; not an order preview (ADR-0017)
    "preview_scan",
    "get_watchlists", "get_watchlist_items", "get_popular_watchlists", "get_alerts",
    "get_alert_log",
)  # fmt: skip

_TIER_S = (
    "create_scan", "update_scan_filters", "update_scan_config",
    "create_watchlist", "update_watchlist", "add_to_watchlist", "remove_from_watchlist",
    "follow_watchlist", "unfollow_watchlist",
    "add_option_to_watchlist", "remove_option_from_watchlist",
    "create_alert", "update_alert", "delete_alert", "mark_alerts_read",
)  # fmt: skip

_TIER_X = (
    *LIVE_ORDER_TOOLS,
    # denied in every mode; the advanced-order tools are absent from 1.6.0 and stay registered
    # so they remain denied if they return
    "review_equity_order", "place_equity_order", "cancel_equity_order",
    "review_advanced_order", "place_advanced_order", "cancel_advanced_order",
    "exercise_option", "cancel_option_exercise",
)  # fmt: skip

_EXCLUDED = (
    "get_crypto_quotes", "get_currency_pairs", "get_crypto_positions", "get_crypto_orders",
    "get_crypto_account_onboarding_info", "preview_crypto_order", "place_crypto_order",
    "cancel_crypto_order",
)  # fmt: skip

ROBINHOOD_REGISTRY: ToolRegistry = make_registry(
    SERVER_NAME,
    {ToolTier.R: _TIER_R, ToolTier.S: _TIER_S, ToolTier.X: _TIER_X, ToolTier.EXCLUDED: _EXCLUDED},
    verified=True,
    live_order_tools=LIVE_ORDER_TOOLS,
)
