"""ADR-0009 item 4 mapping, ADR-0010 null semantics, ADR-0014 premium basis."""

import json
from decimal import Decimal
from typing import Any

import pytest

from wheelta_robinhood_agent.config.rules import RuleMarker, TradingRules, load_rules
from wheelta_robinhood_agent.integrations.wheelta.board_filters import (
    MAX_BOARD_FILTERS,
    BoardFilterError,
    append_rules_filters,
    rules_board_filters,
)

RULES = load_rules().rules

EXPECTED_CURRENT = [
    {"field": "contract.dte", "op": "gte", "value": 7},
    {"field": "contract.dte", "op": "lte", "value": 45},
    {"field": "contract.greeks.delta", "op": "lte", "value": -0.15},
    {"field": "contract.greeks.delta", "op": "gte", "value": -0.3},
    {"field": "contract.openInterest", "op": "gte", "value": 100},
    {"field": "contract.spreadPct", "op": "lte", "value": 0.3},
    {"field": "contract.bid", "op": "gte", "value": 0.1},
]


def _with(section: str, **values: Any) -> TradingRules:
    part = getattr(RULES, section).model_copy(update=values)
    return RULES.model_copy(update={section: part})


def test_current_rules_produce_exact_filters() -> None:
    assert append_rules_filters({}, RULES)["filters"] == EXPECTED_CURRENT


def test_wire_values_are_json_numbers_of_the_right_type() -> None:
    wire = append_rules_filters({}, RULES)["filters"]
    assert isinstance(wire, list)
    assert [type(f["value"]) for f in wire] == [int, int, float, float, int, float, float]
    # Serializes to the plain decimal text, with no binary rounding artefacts.
    assert '"value": -0.15' in json.dumps(wire)


def test_agent_filters_kept_first_and_unchanged() -> None:
    mine = [{"field": "sector", "op": "eq", "value": "Technology"}]
    out = append_rules_filters({"filters": mine}, RULES)
    assert out["filters"] == [*mine, *EXPECTED_CURRENT]
    assert mine == [{"field": "sector", "op": "eq", "value": "Technology"}]  # not mutated


def test_non_filter_arguments_untouched() -> None:
    args: dict[str, object] = {
        "select": ["symbol", "contract.bid"],
        "sort_by": "risk.cushionCoverage",
        "order": "asc",
        "limit": 7,
        "offset": 14,
        "group_by": None,
        "metrics": ["wheelIq.score"],
    }
    out = append_rules_filters(args, RULES)
    assert {k: v for k, v in out.items() if k != "filters"} == args
    assert "filters" not in args


def test_null_filters_treated_as_empty() -> None:
    assert append_rules_filters({"filters": None}, RULES)["filters"] == EXPECTED_CURRENT


def test_non_list_filters_raise() -> None:
    with pytest.raises(BoardFilterError):
        append_rules_filters({"filters": {"field": "x"}}, RULES)


@pytest.mark.parametrize("marker", list(RuleMarker))
def test_markers_inject_nothing(marker: RuleMarker) -> None:
    rules = _with(
        "filters",
        min_dte=marker,
        max_dte=marker,
        min_abs_delta=marker,
        max_abs_delta=marker,
        min_open_interest=marker,
        max_spread_ratio_of_mid=marker,
        min_premium_usd=marker,
    )
    rules = rules.model_copy(
        update={"scope": rules.scope.model_copy(update={"underlying_denylist": marker})}
    )
    assert rules_board_filters(rules) == ()


def test_unmapped_rules_inject_nothing() -> None:
    fields = {f.field.value for f in rules_board_filters(RULES)}
    assert not any(f.startswith("fund.") or f.startswith("risk.") for f in fields)
    assert "isEtf" not in fields


def test_no_fund_predicate_even_when_leveraged_disallowed() -> None:
    """ADR-0010 regression: stocks with fund=null must not be dropped by an injected predicate."""
    assert RULES.scope.leveraged_inverse_etfs_allowed is False
    for flt in rules_board_filters(RULES):
        assert not flt.field.value.startswith("fund")
    stock_row = {
        "symbol": "MSFT",
        "isEtf": False,
        "fund": None,
        "contract": {
            "dte": 20,
            "bid": 0.43,
            "openInterest": 2298,
            "spreadPct": 0.0455,
            "greeks": {"delta": -0.29},
        },
    }
    assert all(_matches(stock_row, f) for f in append_rules_filters({}, RULES)["filters"])


def _matches(row: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Wheelta query.py semantics for the ops we inject: null never satisfies a comparison."""
    actual: Any = row
    for part in flt["field"].split("."):
        actual = actual.get(part) if isinstance(actual, dict) else None
    if actual is None:
        return False
    op, value = flt["op"], flt["value"]
    if op == "gte":
        return bool(actual >= value)
    if op == "lte":
        return bool(actual <= value)
    if op == "notIn":
        return actual not in value
    raise AssertionError(op)


def test_denylist_maps_to_symbol_not_in() -> None:
    rules = _with("scope", underlying_denylist=("TSLA", "GME"))
    wire = append_rules_filters({}, rules)["filters"]
    assert {"field": "symbol", "op": "notIn", "value": ["TSLA", "GME"]} in wire


def test_empty_denylist_injects_nothing() -> None:
    rules = _with("scope", underlying_denylist=())
    assert all(f.field.value != "symbol" for f in rules_board_filters(rules))


def test_delta_mapping_is_negative_for_puts() -> None:
    rules = _with("filters", min_abs_delta=Decimal("0.20"), max_abs_delta=Decimal("0.25"))
    deltas = [f for f in rules_board_filters(rules) if f.field.value == "contract.greeks.delta"]
    assert [(f.op.value, f.value) for f in deltas] == [
        ("lte", Decimal("-0.20")),
        ("gte", Decimal("-0.25")),
    ]


def test_over_limit_raises() -> None:
    mine = [{"field": "sector", "op": "eq", "value": "X"}] * (
        MAX_BOARD_FILTERS - len(EXPECTED_CURRENT) + 1
    )
    with pytest.raises(BoardFilterError, match="limit of 20"):
        append_rules_filters({"filters": mine}, RULES)


def test_exactly_at_limit_is_allowed() -> None:
    mine = [{"field": "sector", "op": "eq", "value": "X"}] * (
        MAX_BOARD_FILTERS - len(EXPECTED_CURRENT)
    )
    assert len(append_rules_filters({"filters": mine}, RULES)["filters"]) == MAX_BOARD_FILTERS  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("section", "values"),
    [
        ("filters", {"min_abs_delta": Decimal("-0.15")}),
        ("filters", {"max_abs_delta": Decimal("1.5")}),
        ("filters", {"max_spread_ratio_of_mid": Decimal("-0.1")}),
        ("filters", {"min_premium_usd": Decimal("-1")}),
        ("filters", {"min_premium_usd": Decimal("0.1000000000000000000001")}),
    ],
)
def test_unmappable_values_raise(section: str, values: dict[str, Any]) -> None:
    with pytest.raises(BoardFilterError):
        append_rules_filters({}, _with(section, **values))
