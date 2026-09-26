"""Rules-derived filters for `wheelta_board_query` (ADR-0009 item 4, ADR-0010, ADR-0014).

Pure: no I/O, no clock. The `agent/` `PreToolUse` hook calls `append_rules_filters` and
returns the result as `updatedInput`; on `BoardFilterError` the hook denies the call
(ADR-0009 item 6, fail closed). It never sends a board query without these filters.

Wheelta contract (read-only source, `/Users/deepset/wheelta-mcp`):

- `src/wheelta_mcp/tools/screener.py`: `filters: list[{field, op, value}]`, at most 20
  (`Field(max_length=20)`), ANDed. Ops used here: `gte`, `lte`, `notIn`.
- `src/wheelta_mcp/query.py`: a null never satisfies a positive comparison, and `notIn`
  takes a list. Numeric strings are coerced to float, but we send JSON numbers.
- `src/wheelta_mcp/fields.py` units: `contract.dte` days (integer), `contract.greeks.delta`
  negative for these put rows, `contract.openInterest` contracts (integer),
  `contract.spreadPct` a decimal fraction (full width / midpoint, 0.30 = 30%),
  `contract.bid` USD/share.

Board values are build-time screens (CLAUDE.md §10). These filters narrow what the board
returns; they are not order enforcement (ADR-0009 "Consequences").

Wire values: rules hold `Decimal`. The board compares against float columns, so a decimal
is sent as a JSON number (Python `float`) only at the wire boundary, in `BoardFilter.to_wire`.
The conversion is rejected unless the float's shortest repr round-trips to the same decimal,
so no rule value is silently rounded.
"""

from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from wheelta_robinhood_agent.config.rules import RuleMarker, TradingRules

# screener.py: `filters: Annotated[list[FilterInput] | None, Field(max_length=20)]`.
MAX_BOARD_FILTERS = 20

FILTERS_ARG = "filters"


class BoardFilterError(Exception):
    """The rules-derived filters can't be appended. The caller denies the call (ADR-0009 §6)."""


class BoardFilterOp(StrEnum):
    """The subset of Wheelta filter operators the mapping uses (screener.py `FilterInput.op`)."""

    GTE = "gte"
    LTE = "lte"
    NOT_IN = "notIn"


class BoardField(StrEnum):
    """Board columns the mapping targets (fields.py catalog paths)."""

    DTE = "contract.dte"
    DELTA = "contract.greeks.delta"
    OPEN_INTEREST = "contract.openInterest"
    SPREAD_PCT = "contract.spreadPct"
    BID = "contract.bid"
    SYMBOL = "symbol"


WireValue = int | float | list[str]


class BoardFilter(BaseModel):
    """One rules-derived board filter. `value` keeps rules' exact type (int or Decimal)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    field: BoardField
    op: BoardFilterOp
    value: int | Decimal | tuple[str, ...]
    rule: str  # the rules key it came from, for the ledger and PostToolUse context

    def to_wire(self) -> dict[str, str | WireValue]:
        """The `{field, op, value}` dict the Wheelta server accepts."""
        return {"field": self.field.value, "op": self.op.value, "value": _wire_value(self.value)}


def _wire_value(value: int | Decimal | tuple[str, ...]) -> WireValue:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Decimal):
        as_float = float(value)
        if Decimal(repr(as_float)) != value:
            raise BoardFilterError(f"rule value {value} is not exactly representable on the wire")
        return as_float
    return value


def _is_set(value: object) -> bool:
    """`none`, `agent_discretion`, and `TBD` inject nothing (ADR-0009 item 4)."""
    return not isinstance(value, RuleMarker)


def _require_int(rule: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BoardFilterError(f"{rule} must be a non-negative integer to map to a board filter")
    return value


def _require_ratio(rule: str, value: object, *, upper: Decimal | None = None) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise BoardFilterError(f"{rule} must be a non-negative decimal to map to a board filter")
    if upper is not None and value > upper:
        raise BoardFilterError(f"{rule} must be at most {upper} to map to a board filter")
    return value


def rules_board_filters(rules: TradingRules) -> tuple[BoardFilter, ...]:
    """Map the rules to board filters, in ADR-0009 item 4 table order.

    - `filters.min_dte` → `contract.dte gte v`; `filters.max_dte` → `contract.dte lte v`
    - `filters.min_abs_delta` → `contract.greeks.delta lte -v` (board rows are puts)
    - `filters.max_abs_delta` → `contract.greeks.delta gte -v`
    - `filters.min_open_interest` → `contract.openInterest gte v`
    - `filters.max_spread_ratio_of_mid` → `contract.spreadPct lte v` (same definition)
    - `scope.underlying_denylist` (non-empty list) → `symbol notIn [...]`
    - `filters.min_premium_usd` → `contract.bid gte v` (ADR-0014 item 1)

    Deliberately not mapped: `scope.leveraged_inverse_etfs_allowed` (ADR-0010: stocks have
    `fund=null` and an AND-only predicate would drop them), `filters.min_annualized_yield_ratio`
    (ADR-0014 item 2: `risk.annualizedYield` is mid-based), and every rule absent from the
    ADR table. Raises `BoardFilterError` if a set value can't produce a filter.
    """
    f = rules.filters
    out: list[BoardFilter] = []

    def add(
        field: BoardField, op: BoardFilterOp, value: int | Decimal | tuple[str, ...], rule: str
    ) -> None:
        out.append(BoardFilter(field=field, op=op, value=value, rule=rule))

    if _is_set(f.min_dte):
        add(
            BoardField.DTE,
            BoardFilterOp.GTE,
            _require_int("filters.min_dte", f.min_dte),
            "filters.min_dte",
        )
    if _is_set(f.max_dte):
        add(
            BoardField.DTE,
            BoardFilterOp.LTE,
            _require_int("filters.max_dte", f.max_dte),
            "filters.max_dte",
        )
    one = Decimal("1")
    if _is_set(f.min_abs_delta):
        v = _require_ratio("filters.min_abs_delta", f.min_abs_delta, upper=one)
        add(BoardField.DELTA, BoardFilterOp.LTE, -v, "filters.min_abs_delta")
    if _is_set(f.max_abs_delta):
        v = _require_ratio("filters.max_abs_delta", f.max_abs_delta, upper=one)
        add(BoardField.DELTA, BoardFilterOp.GTE, -v, "filters.max_abs_delta")
    if _is_set(f.min_open_interest):
        v_oi = _require_int("filters.min_open_interest", f.min_open_interest)
        add(BoardField.OPEN_INTEREST, BoardFilterOp.GTE, v_oi, "filters.min_open_interest")
    if _is_set(f.max_spread_ratio_of_mid):
        v = _require_ratio("filters.max_spread_ratio_of_mid", f.max_spread_ratio_of_mid)
        add(BoardField.SPREAD_PCT, BoardFilterOp.LTE, v, "filters.max_spread_ratio_of_mid")
    denylist = rules.scope.underlying_denylist
    if _is_set(denylist):
        if not isinstance(denylist, tuple):
            raise BoardFilterError("scope.underlying_denylist must be a list of tickers")
        if denylist:
            add(BoardField.SYMBOL, BoardFilterOp.NOT_IN, denylist, "scope.underlying_denylist")
    if _is_set(f.min_premium_usd):
        v = _require_ratio("filters.min_premium_usd", f.min_premium_usd)
        add(BoardField.BID, BoardFilterOp.GTE, v, "filters.min_premium_usd")
    return tuple(out)


def append_rules_filters(
    agent_input: Mapping[str, object], rules: TradingRules
) -> dict[str, object]:
    """Return `agent_input` with the rules-derived filters appended (ADR-0009 items 1, 3, 6).

    The agent's own filters are kept first and unchanged (ANDed, so they can only narrow).
    Every other argument (`select`, `sort_by`, `order`, `limit`, `offset`, `group_by`,
    `metrics`, ...) is copied untouched. The input is not mutated. Raises `BoardFilterError`
    if `filters` is present but not a list, if the total exceeds `MAX_BOARD_FILTERS`, or if a
    rule value can't produce a filter.
    """
    requested = agent_input.get(FILTERS_ARG)
    if requested is None:
        agent_filters: list[object] = []
    elif isinstance(requested, list):
        agent_filters = list(requested)
    else:
        raise BoardFilterError("the agent's `filters` argument must be a list")
    injected = [flt.to_wire() for flt in rules_board_filters(rules)]
    total = len(agent_filters) + len(injected)
    if total > MAX_BOARD_FILTERS:
        raise BoardFilterError(
            f"{len(agent_filters)} agent filters + {len(injected)} rules filters = {total}, "
            f"over the board's limit of {MAX_BOARD_FILTERS}"
        )
    updated = dict(agent_input)
    updated[FILTERS_ARG] = [*agent_filters, *injected]
    return updated
