"""FactsRules: the rule values `compute_decision_facts` needs (OUTPUT_ASSEMBLY.md; ADR-0007).

Domain never imports `config`; `config.facts_rules.facts_rules_from` maps the loaded
`rules/trading_rules.toml` into this model. A rule set to `"TBD"`, `"none"`, or
`"agent_discretion"` arrives as a `FactsRuleMarker`, never as a number:

- `UNSET` ("TBD"): any fact depending on it is unavailable, with a gap naming the rule.
- `NONE` ("none"): deliberately no limit; the capacity it would impose does not constrain.
- `AGENT_DISCRETION`: no fixed value; code cannot compute with it, so a numeric fact that
  would need it is unavailable, with a gap (code never picks a number).
"""

from enum import StrEnum
from typing import Self

from pydantic import model_validator

from wheelta_robinhood_agent.domain.base import Count, DomainModel, NonEmptyStr, NonNegDec


class FactsRuleMarker(StrEnum):
    """Domain mirror of the rules file's value conventions (`conventions.value_conventions`)."""

    UNSET = "TBD"
    NONE = "none"
    AGENT_DISCRETION = "agent_discretion"


DecimalSetting = FactsRuleMarker | NonNegDec
CountSetting = FactsRuleMarker | Count


class FactsRules(DomainModel):
    """Exactly the rule values the facts computation reads, with the rules version and hash.

    Keys mirror `rules/trading_rules.toml` (section.key in each docstring line):
    - limits.max_contracts_per_order, limits.max_collateral_per_underlying_usd,
      limits.max_collateral_per_underlying_ratio, limits.max_total_csp_collateral_ratio,
      limits.min_cash_reserve_usd, limits.cc_max_lot_coverage_ratio (at most 1.00)
    - management.roll_dte_window
    - data_quality.freshness.{option_quote,equity_quote,account_state}_max_age_seconds
    """

    rules_version: NonEmptyStr
    rules_hash: NonEmptyStr
    max_contracts_per_order: CountSetting
    max_collateral_per_underlying_usd: DecimalSetting
    max_collateral_per_underlying_ratio: DecimalSetting
    max_total_csp_collateral_ratio: DecimalSetting
    min_cash_reserve_usd: DecimalSetting
    cc_max_lot_coverage_ratio: DecimalSetting
    roll_dte_window: CountSetting
    option_quote_max_age_seconds: CountSetting
    equity_quote_max_age_seconds: CountSetting
    account_state_max_age_seconds: CountSetting

    @model_validator(mode="after")
    def _check_rules(self) -> Self:
        ratio = self.cc_max_lot_coverage_ratio
        if not isinstance(ratio, FactsRuleMarker) and ratio > 1:
            raise ValueError("cc_max_lot_coverage_ratio must be at most 1.00")
        return self
