"""Load, validate, hash, and render `rules/trading_rules.toml` (ADR-0007, rules/README.md).

The file is read once at startup with the stdlib `tomllib`. Validation is strict: an unknown
key, a wrong type, a TOML float, or a malformed decimal aborts the run. `"TBD"`, `"none"`,
and `"agent_discretion"` load as `RuleMarker` values, never as numbers.
"""

import hashlib
import re
import tomllib
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "trading_rules.toml"

_DECIMAL_RE = re.compile(r"^-?\d+(\.\d+)?$")


class RulesError(Exception):
    """The trading-rules file is missing or invalid. The run must abort."""


class RuleMarker(StrEnum):
    """Value conventions from `conventions.value_conventions`."""

    TBD = "TBD"  # unset: any check depending on it fails
    NONE = "none"  # deliberately no limit: the check always passes
    AGENT_DISCRETION = "agent_discretion"  # the agent decides from the named guidance


_MARKERS = {marker.value: marker for marker in RuleMarker}


def _parse_decimal_rule(value: object) -> object:
    """Decimals are strings ("0.50"); floats and bare ints are rejected (rules/README.md)."""
    if isinstance(value, str):
        if value in _MARKERS:
            return _MARKERS[value]
        if _DECIMAL_RE.fullmatch(value):
            return Decimal(value)
        raise ValueError(f"malformed decimal string: {value!r}")
    raise ValueError(f"decimals must be strings, got {type(value).__name__}")


def _parse_int_rule(value: object) -> object:
    """Counts are TOML integers; bools and floats are rejected."""
    if isinstance(value, str) and value in _MARKERS:
        return _MARKERS[value]
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"counts must be non-negative, got {value}")
        return value
    raise ValueError(f"expected an integer or a rule marker, got {value!r}")


def _parse_bool_rule(value: object) -> object:
    if isinstance(value, str) and value in _MARKERS:
        return _MARKERS[value]
    if isinstance(value, bool):
        return value
    raise ValueError(f"expected a boolean or a rule marker, got {value!r}")


def _parse_symbols_rule(value: object) -> object:
    if isinstance(value, str) and value in _MARKERS:
        return _MARKERS[value]
    if isinstance(value, list) and all(
        isinstance(s, str) and re.fullmatch(r"[A-Z]{1,5}", s) for s in value
    ):
        return tuple(value)
    raise ValueError(f"expected a list of uppercase tickers or a rule marker, got {value!r}")


DecimalRule = Annotated[Decimal | RuleMarker, BeforeValidator(_parse_decimal_rule)]
IntRule = Annotated[int | RuleMarker, BeforeValidator(_parse_int_rule)]
BoolRule = Annotated[bool | RuleMarker, BeforeValidator(_parse_bool_rule)]
SymbolsRule = Annotated[tuple[str, ...] | RuleMarker, BeforeValidator(_parse_symbols_rule)]
Notes = tuple[StrictStr, ...]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Meta(_Section):
    rules_id: StrictStr
    version: StrictInt = Field(ge=1)
    status: StrictStr
    adr: StrictStr
    signed_off_by: StrictStr
    signed_off_on: StrictStr


class Conventions(_Section):
    value_conventions: Notes


class Scope(_Section):
    notes: Notes = ()
    underlying_allowlist: SymbolsRule
    underlying_denylist: SymbolsRule
    etfs_allowed: BoolRule
    leveraged_inverse_etfs_allowed: BoolRule


class Limits(_Section):
    notes: Notes = ()
    max_contracts_per_order: IntRule
    max_new_positions_per_run: IntRule
    max_new_positions_per_day: IntRule
    max_collateral_per_underlying_usd: DecimalRule
    max_collateral_per_underlying_ratio: DecimalRule
    max_total_csp_collateral_ratio: DecimalRule
    min_cash_reserve_usd: DecimalRule
    max_sector_concentration_ratio: DecimalRule
    max_correlated_exposure: DecimalRule
    cc_max_lot_coverage_ratio: DecimalRule
    cc_min_strike_vs_cost_basis: DecimalRule


class Filters(_Section):
    notes: Notes = ()
    min_dte: IntRule
    max_dte: IntRule
    min_abs_delta: DecimalRule
    max_abs_delta: DecimalRule
    min_open_interest: IntRule
    min_volume: IntRule
    max_spread_usd: DecimalRule
    max_spread_ratio_of_mid: DecimalRule
    min_premium_usd: DecimalRule
    min_annualized_yield_ratio: DecimalRule


class Orders(_Section):
    execution_order: StrictStr
    open_checks: StrictStr
    close_checks: StrictStr
    limit_price_rule: StrictStr
    limit_price_bounds: StrictStr
    time_in_force: StrictStr
    working: Notes


class Events(_Section):
    earnings_exclusion: StrictStr
    ex_dividend_exclusion: StrictStr
    exclude_pending_ma_halt_delisting: BoolRule
    exclude_unverified_material_catalyst: BoolRule


class Definitions(_Section):
    tested: StrictStr
    broken_thesis: StrictStr
    remaining_yield: StrictStr
    annualization: StrictStr
    captured_ratio: StrictStr
    cash_accounting: StrictStr


class ManagementRule(_Section):
    id: StrictInt = Field(ge=1)
    condition: StrictStr
    action: StrictStr
    evaluated: StrictStr


class Management(_Section):
    rule_order: StrictStr
    take_profit_capture_ratio: DecimalRule
    take_profit_guidance: StrictStr
    roll_dte_window: IntRule
    roll_requires_net_credit: BoolRule
    max_roll_dte: IntRule
    max_rolls_per_position: IntRule
    close_quantity: DecimalRule
    close_quantity_guidance: StrictStr
    reallocation_min_yield_advantage_ratio: DecimalRule
    reallocation_guidance: StrictStr
    rules: tuple[ManagementRule, ...]


class Roll(_Section):
    definition: StrictStr
    requirements: Notes
    sequencing: StrictStr


class Assignment(_Section):
    assigned_shares: StrictStr


class Selection(_Section):
    sources: Notes
    underlying_filters: Notes
    contract_filters: Notes
    board_comparison: StrictStr
    sizing: Notes
    ranking: Notes
    portfolio_caps: Notes


class CircuitBreakers(_Section):
    max_daily_drawdown_usd: DecimalRule
    on_max_daily_drawdown: StrictStr
    max_consecutive_run_failures: IntRule
    on_max_consecutive_run_failures: StrictStr
    max_consecutive_unverifiable_runs: IntRule
    on_max_consecutive_unverifiable_runs: StrictStr


class Workspace(_Section):
    max_owned_watchlists: IntRule
    max_items_per_owned_watchlist: IntRule
    max_owned_scans: IntRule
    max_owned_alerts: IntRule
    max_mutations_per_run: IntRule


class Freshness(_Section):
    notes: Notes = ()
    option_quote_max_age_seconds: IntRule
    equity_quote_max_age_seconds: IntRule
    account_state_max_age_seconds: IntRule
    wheelta_board: StrictStr
    wheelta_macro_snapshot_max_age_seconds: IntRule
    earnings_calendar_max_age_seconds: IntRule
    news_max_age_seconds: IntRule


class Sanity(_Section):
    notes: Notes = ()
    min_iv_ratio: DecimalRule
    max_iv_ratio: DecimalRule


class Tolerances(_Section):
    board_vs_live_premium_divergence_ratio: DecimalRule
    fundamentals_divergence_ratio: DecimalRule


class Precedence(_Section):
    fact: StrictStr
    sources: Notes
    on_divergence: StrictStr


class SourceTier(_Section):
    tier: StrictInt = Field(ge=1)
    sources: StrictStr
    use: StrictStr


class DataQuality(_Section):
    precedence_rule: StrictStr
    freshness: Freshness
    sanity: Sanity
    tolerances: Tolerances
    precedence: tuple[Precedence, ...]
    source_tiers: tuple[SourceTier, ...]


class TradingRules(_Section):
    meta: Meta
    conventions: Conventions
    scope: Scope
    limits: Limits
    filters: Filters
    orders: Orders
    events: Events
    definitions: Definitions
    management: Management
    roll: Roll
    assignment: Assignment
    selection: Selection
    circuit_breakers: CircuitBreakers
    workspace: Workspace
    data_quality: DataQuality


class LoadedRules(BaseModel):
    """Validated rules plus what every run records: version, file hash, rendered text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rules: TradingRules
    version: int
    sha256: StrictStr
    rendered: StrictStr


def load_rules(path: Path = DEFAULT_RULES_PATH) -> LoadedRules:
    """Read, validate, hash, and render the rules file. Raises RulesError on any problem."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RulesError(f"cannot read trading rules at {path}: {exc.strerror}") from None
    return parse_rules(data)


def parse_rules(data: bytes) -> LoadedRules:
    """Validate and render rules from the file's bytes. The hash covers the exact bytes."""
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RulesError(f"trading rules are not valid TOML: {exc}") from None
    try:
        rules = TradingRules.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise RulesError(f"invalid trading rules: {problems}") from None
    return LoadedRules(
        rules=rules,
        version=rules.meta.version,
        sha256=hashlib.sha256(data).hexdigest(),
        rendered=render_policy(raw),
    )


def render_policy(raw: dict[str, Any]) -> str:
    """Render every section except `meta` into the prompt's `{{policy}}` block.

    Deterministic, in file order (rules/README.md "Loading and rendering"): each table and
    each array of tables gets a `###` heading named by its path; scalars become `- key: value`
    lines; string arrays become a `key:` line and a numbered list; arrays of tables become a
    Markdown table with the keys as columns, string arrays in cells joined with ` → `.
    """
    lines: list[str] = []
    for key, value in raw.items():
        if key != "meta":
            _render_table(key, value, lines)
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_table(path: str, table: dict[str, Any], lines: list[str]) -> None:
    lines.append(f"### {path}")
    lines.append("")
    nested: list[tuple[str, Any]] = []
    for key, value in table.items():
        if isinstance(value, dict) or _is_table_array(value):
            nested.append((key, value))
        elif isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  {i}. {_scalar(item)}" for i, item in enumerate(value, start=1))
        else:
            lines.append(f"- {key}: {_scalar(value)}")
    lines.append("")
    for key, value in nested:
        child = f"{path}.{key}"
        if isinstance(value, dict):
            _render_table(child, value, lines)
        else:
            _render_table_array(child, value, lines)


def _render_table_array(path: str, rows: list[dict[str, Any]], lines: list[str]) -> None:
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    lines.append(f"### {path}")
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in rows:
        cells = [_cell(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")


def _is_table_array(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _cell(value: object) -> str:
    text = " → ".join(_scalar(v) for v in value) if isinstance(value, list) else _scalar(value)
    return text.replace("|", "\\|")
