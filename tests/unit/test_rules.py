import tomllib
from decimal import Decimal
from pathlib import Path

import pytest

from wheelta_robinhood_agent.config.rules import (
    DEFAULT_RULES_PATH,
    RuleMarker,
    RulesError,
    load_rules,
    parse_rules,
    render_policy,
)

REAL = DEFAULT_RULES_PATH.read_bytes()


def _patch(old: str, new: str) -> bytes:
    text = REAL.decode()
    assert old in text, old
    return text.replace(old, new, 1).encode()


def test_real_file_loads() -> None:
    loaded = load_rules()
    assert loaded.version == loaded.rules.meta.version == 5
    assert len(loaded.sha256) == 64
    r = loaded.rules
    assert r.limits.max_contracts_per_order == 10
    assert r.limits.max_collateral_per_underlying_usd is RuleMarker.NONE
    assert r.limits.cc_min_strike_vs_cost_basis is RuleMarker.AGENT_DISCRETION
    assert r.filters.min_abs_delta == Decimal("0.15")
    assert isinstance(r.filters.min_abs_delta, Decimal)
    assert r.scope.leveraged_inverse_etfs_allowed is False
    assert [rule.id for rule in r.management.rules] == [1, 2, 3, 4, 5, 6, 7]
    assert r.meta.signed_off_by == "TBD"
    assert r.management.close_quantity is RuleMarker.AGENT_DISCRETION
    assert "365" in r.definitions.annualization


def test_hash_covers_exact_bytes() -> None:
    a = parse_rules(REAL)
    b = parse_rules(REAL + b"\n# trailing comment\n")
    assert a.rendered == b.rendered
    assert a.sha256 != b.sha256


def test_tbd_loads_as_unset() -> None:
    loaded = parse_rules(_patch("max_contracts_per_order = 10", 'max_contracts_per_order = "TBD"'))
    assert loaded.rules.limits.max_contracts_per_order is RuleMarker.TBD
    assert "- max_contracts_per_order: TBD" in loaded.rendered


@pytest.mark.parametrize(
    ("old", "new", "where"),
    [
        (
            "max_contracts_per_order = 10",
            "max_contracts_per_order = 10\nsurprise = 1",
            "limits.surprise",
        ),
        ('min_abs_delta = "0.15"', "min_abs_delta = 0.15", "filters.min_abs_delta"),
        ('min_abs_delta = "0.15"', 'min_abs_delta = "0.1.5"', "filters.min_abs_delta"),
        ('min_abs_delta = "0.15"', 'min_abs_delta = "15%"', "filters.min_abs_delta"),
        ('min_abs_delta = "0.15"', "min_abs_delta = 1", "filters.min_abs_delta"),
        (
            "max_contracts_per_order = 10",
            'max_contracts_per_order = "10"',
            "limits.max_contracts_per_order",
        ),
        (
            "max_contracts_per_order = 10",
            "max_contracts_per_order = true",
            "limits.max_contracts_per_order",
        ),
        (
            "max_contracts_per_order = 10",
            "max_contracts_per_order = -1",
            "limits.max_contracts_per_order",
        ),
        ("etfs_allowed = true", 'etfs_allowed = "yes"', "scope.etfs_allowed"),
        (
            'underlying_denylist = "none"',
            'underlying_denylist = ["brk.b"]',
            "scope.underlying_denylist",
        ),
        ("version = 5", 'version = "5"', "meta.version"),
        ('time_in_force = "day"', "time_in_force = 1", "orders.time_in_force"),
    ],
)
def test_invalid_values_abort(old: str, new: str, where: str) -> None:
    with pytest.raises(RulesError, match=where.replace(".", r"\.")):
        parse_rules(_patch(old, new))


def test_missing_section_aborts() -> None:
    text = REAL.decode()
    start = text.index("[workspace]")
    end = text.index("# ----", start)
    with pytest.raises(RulesError, match="workspace"):
        parse_rules((text[:start] + text[end:]).encode())


def test_bool_rule_accepts_tbd() -> None:
    loaded = parse_rules(_patch("etfs_allowed = true", 'etfs_allowed = "TBD"'))
    assert loaded.rules.scope.etfs_allowed is RuleMarker.TBD


def test_denylist_accepts_tickers() -> None:
    loaded = parse_rules(_patch('underlying_denylist = "none"', 'underlying_denylist = ["TQQQ"]'))
    assert loaded.rules.scope.underlying_denylist == ("TQQQ",)


def test_invalid_toml_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="not valid TOML"):
        parse_rules(b"[meta\n")
    with pytest.raises(RulesError, match="not valid TOML"):
        parse_rules(b"\xff\xfe")
    with pytest.raises(RulesError, match="cannot read"):
        load_rules(tmp_path / "absent.toml")


def test_rendering_is_deterministic_and_covers_every_section_but_meta() -> None:
    raw = tomllib.loads(REAL.decode())
    first = render_policy(raw)
    assert first == render_policy(tomllib.loads(REAL.decode()))
    assert "### meta" not in first
    assert "rules_id" not in first
    for section in raw:
        if section != "meta":
            assert f"### {section}\n" in first
    for heading in (
        "### management.rules",
        "### data_quality.freshness",
        "### data_quality.precedence",
        "### data_quality.source_tiers",
    ):
        assert heading in first
    # file order is preserved
    assert first.index("### conventions") < first.index("### scope") < first.index("### limits")


def test_render_formats() -> None:
    raw = {
        "meta": {"version": 1},
        "sec": {
            "flag": False,
            "text": "a",
            "items": ["x", "y"],
            "sub": {"n": 1},
            "rows": [{"a": "1|2", "b": ["p", "q"]}, {"a": "", "c": 3}],
        },
    }
    assert render_policy(raw) == (
        "### sec\n\n- flag: false\n- text: a\n- items:\n  1. x\n  2. y\n\n"
        "### sec.sub\n\n- n: 1\n\n"
        "### sec.rows\n\n| a | b | c |\n|---|---|---|\n| 1\\|2 | p → q |  |\n|  |  | 3 |\n"
    )
