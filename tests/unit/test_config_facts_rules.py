"""config.facts_rules: loaded trading rules -> domain FactsRules (1:1, markers by meaning)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.config.facts_rules import facts_rules_from
from wheelta_robinhood_agent.config.rules import DEFAULT_RULES_PATH, load_rules, parse_rules
from wheelta_robinhood_agent.domain.facts_rules import FactsRuleMarker, FactsRules


def test_maps_the_committed_rules_file() -> None:
    loaded = load_rules()
    rules = facts_rules_from(loaded)
    assert rules.rules_version == str(loaded.version) == "5"
    assert rules.rules_hash == loaded.sha256
    assert rules.max_contracts_per_order == 10
    assert rules.max_collateral_per_underlying_usd is FactsRuleMarker.NONE
    assert rules.max_collateral_per_underlying_ratio == Decimal("1.00")
    assert rules.max_total_csp_collateral_ratio == Decimal("1.00")
    assert rules.min_cash_reserve_usd == Decimal("0.00")
    assert rules.cc_max_lot_coverage_ratio == Decimal("1.00")
    assert rules.roll_dte_window == 14
    assert rules.option_quote_max_age_seconds == 60
    assert rules.equity_quote_max_age_seconds == 60
    assert rules.account_state_max_age_seconds == 120


def _modified(*replacements: tuple[str, str]) -> FactsRules:
    text = DEFAULT_RULES_PATH.read_text()
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new, 1)
    return facts_rules_from(parse_rules(text.encode()))


def test_markers_map_by_meaning() -> None:
    rules = _modified(
        ("max_contracts_per_order = 10", 'max_contracts_per_order = "TBD"'),
        ('min_cash_reserve_usd = "0.00"', 'min_cash_reserve_usd = "agent_discretion"'),
        ("roll_dte_window = 14", 'roll_dte_window = "none"'),
    )
    assert rules.max_contracts_per_order is FactsRuleMarker.UNSET
    assert rules.min_cash_reserve_usd is FactsRuleMarker.AGENT_DISCRETION
    assert rules.roll_dte_window is FactsRuleMarker.NONE


def test_coverage_ratio_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cc_max_lot_coverage_ratio"):
        _modified(('cc_max_lot_coverage_ratio = "1.00"', 'cc_max_lot_coverage_ratio = "1.01"'))


def test_facts_rules_is_frozen_and_strict() -> None:
    rules = facts_rules_from(load_rules())
    with pytest.raises(ValidationError):
        rules.max_contracts_per_order = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        FactsRules.model_validate({**rules.model_dump(), "unexpected": 1})
    with pytest.raises(ValidationError):
        FactsRules.model_validate({**rules.model_dump(), "max_contracts_per_order": 1.5})
