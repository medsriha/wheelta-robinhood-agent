"""Map loaded trading rules to the domain `FactsRules` and `PretradeRules` (config may import
domain; §3).

The mapping is 1:1 by key. Rule markers map to `FactsRuleMarker` by meaning; no value is
defaulted, derived, or reinterpreted here.
"""

from decimal import Decimal

from wheelta_robinhood_agent.config.rules import LoadedRules, RuleMarker
from wheelta_robinhood_agent.domain.facts_rules import FactsRuleMarker, FactsRules
from wheelta_robinhood_agent.domain.pretrade import PretradeRules

_MARKERS: dict[RuleMarker, FactsRuleMarker] = {
    RuleMarker.TBD: FactsRuleMarker.UNSET,
    RuleMarker.NONE: FactsRuleMarker.NONE,
    RuleMarker.AGENT_DISCRETION: FactsRuleMarker.AGENT_DISCRETION,
}


def _decimal(value: Decimal | RuleMarker) -> Decimal | FactsRuleMarker:
    return _MARKERS[value] if isinstance(value, RuleMarker) else value


def _count(value: int | RuleMarker) -> int | FactsRuleMarker:
    return _MARKERS[value] if isinstance(value, RuleMarker) else value


def facts_rules_from(loaded: LoadedRules) -> FactsRules:
    """Build the `FactsRules` a facts computation needs from validated, hashed rules."""
    rules = loaded.rules
    limits = rules.limits
    freshness = rules.data_quality.freshness
    return FactsRules(
        rules_version=str(loaded.version),
        rules_hash=loaded.sha256,
        max_contracts_per_order=_count(limits.max_contracts_per_order),
        max_collateral_per_underlying_usd=_decimal(limits.max_collateral_per_underlying_usd),
        max_collateral_per_underlying_ratio=_decimal(limits.max_collateral_per_underlying_ratio),
        max_total_csp_collateral_ratio=_decimal(limits.max_total_csp_collateral_ratio),
        min_cash_reserve_usd=_decimal(limits.min_cash_reserve_usd),
        cc_max_lot_coverage_ratio=_decimal(limits.cc_max_lot_coverage_ratio),
        roll_dte_window=_count(rules.management.roll_dte_window),
        option_quote_max_age_seconds=_count(freshness.option_quote_max_age_seconds),
        equity_quote_max_age_seconds=_count(freshness.equity_quote_max_age_seconds),
        account_state_max_age_seconds=_count(freshness.account_state_max_age_seconds),
    )


def pretrade_rules_from(loaded: LoadedRules) -> PretradeRules:
    """Build the `PretradeRules` pre-trade validation needs (ADR-0048)."""
    rules = loaded.rules
    filters = rules.filters
    freshness = rules.data_quality.freshness
    return PretradeRules(
        min_dte=_count(filters.min_dte),
        max_dte=_count(filters.max_dte),
        min_abs_delta=_decimal(filters.min_abs_delta),
        max_abs_delta=_decimal(filters.max_abs_delta),
        min_cushion_ratio=_decimal(filters.min_cushion_ratio),
        min_annualized_yield_ratio=_decimal(filters.min_annualized_yield_ratio),
        option_quote_max_age_seconds=_count(freshness.option_quote_max_age_seconds),
        equity_quote_max_age_seconds=_count(freshness.equity_quote_max_age_seconds),
    )
