"""V5 No order on an unset rule (docs/VALIDATION.md "V5"; ADR-0014 item 4).

A placed/intended sell-to-open violates V5 if a rule it depends on is `TBD`: the limits and
circuit-breaker keys used by V1/V7, `orders.limit_price_rule`, the option-quote freshness
threshold, or any `filters` value. A buy-to-close violates V5 if a management value its
decision necessarily depends on is `TBD` (`close_quantity` for CLOSE, the roll keys for
ROLL). Management values that only some CLOSE decisions depend on (take-profit,
reallocation) cannot be attributed from recorded data: a `TBD` there is unverifiable.
`none` and `agent_discretion` are set markers. Entry filters are never applied to a BTC.
"""

from wheelta_robinhood_agent.agent.audit._common import (
    AuditAttempt,
    Findings,
    Reason,
    Unknown,
    audited_attempts,
    instrument,
    is_tbd,
)
from wheelta_robinhood_agent.agent.audit.context import AuditContext
from wheelta_robinhood_agent.config.rules import TradingRules
from wheelta_robinhood_agent.domain.enums import (
    AuditCheck,
    DecisionAction,
    OptionRight,
    OrderSide,
)
from wheelta_robinhood_agent.domain.run import AuditFinding

_FILTER_KEYS = (
    "min_dte",
    "max_dte",
    "min_abs_delta",
    "max_abs_delta",
    "min_open_interest",
    "min_volume",
    "max_spread_usd",
    "max_spread_ratio_of_mid",
    "min_premium_usd",
    "min_annualized_yield_ratio",
)
_PUT_LIMIT_KEYS = (
    "min_cash_reserve_usd",
    "max_collateral_per_underlying_usd",
    "max_collateral_per_underlying_ratio",
    "max_total_csp_collateral_ratio",
)
_NEW_POSITION_KEYS = ("max_new_positions_per_run", "max_new_positions_per_day")
_ROLL_KEYS = (
    "roll_dte_window",
    "roll_requires_net_credit",
    "max_roll_dte",
    "max_rolls_per_position",
)
_CLOSE_DISCRETIONARY_KEYS = ("take_profit_capture_ratio", "reallocation_min_yield_advantage_ratio")


def _value(rules: TradingRules, dotted: str) -> object:
    section, key = dotted.split(".", 1)
    obj: object = rules
    for part in (section, *key.split(".")):
        obj = getattr(obj, part)
    return obj


def check_v5(ctx: AuditContext) -> tuple[AuditFinding, ...]:
    """Run V5 over every audited attempt."""
    out = Findings(ctx, AuditCheck.V5)
    attempts = audited_attempts(ctx)
    if isinstance(attempts, Unknown):
        out.unknown("all", attempts.reason, attempts.detail)
        return out.result()
    if not attempts:
        out.ok("all", "no placed or intended attempts")
        return out.result()
    for attempt in attempts:
        if attempt.side is OrderSide.SELL_TO_OPEN:
            _open(ctx, out, attempt)
        elif attempt.side is OrderSide.BUY_TO_CLOSE:
            _close(ctx, out, attempt)
        else:
            out.unknown(
                "1", Reason.MISSING_EVIDENCE, "side unknown or not STO/BTC", attempt=attempt
            )
    return out.result()


def _tbd(ctx: AuditContext, keys: list[str]) -> list[str]:
    return [k for k in keys if is_tbd(_value(ctx.rules, k))]


def _open(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    required = [
        "limits.max_contracts_per_order",
        "circuit_breakers.max_daily_drawdown_usd",
        "orders.limit_price_rule",
        "data_quality.freshness.option_quote_max_age_seconds",
        *(f"filters.{k}" for k in _FILTER_KEYS),
    ]
    fact = instrument(ctx, attempt.instrument_id)
    right = fact.occ_symbol.right if fact is not None else None
    uncertain: list[str] = []
    if right is OptionRight.PUT:
        required += [f"limits.{k}" for k in _PUT_LIMIT_KEYS]
    elif right is None:
        uncertain += [f"limits.{k}" for k in _PUT_LIMIT_KEYS]
    new_keys = [f"limits.{k}" for k in _NEW_POSITION_KEYS]
    if attempt.is_new_open:
        required += new_keys
    elif not attempt.is_roll_replacement:
        uncertain += new_keys  # unassociated: new lineage or roll replacement unknown
    _report(ctx, out, attempt, required, uncertain, Reason.LINEAGE_UNKNOWN)


def _close(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    action = attempt.action
    required: list[str] = []
    uncertain: list[str] = []
    if action is DecisionAction.CLOSE:
        required.append("management.close_quantity")
        uncertain += [f"management.{k}" for k in _CLOSE_DISCRETIONARY_KEYS]
    elif action is DecisionAction.ROLL:
        required += [f"management.{k}" for k in _ROLL_KEYS]
    else:
        uncertain += [
            "management.close_quantity",
            *(f"management.{k}" for k in (*_ROLL_KEYS, *_CLOSE_DISCRETIONARY_KEYS)),
        ]
    _report(ctx, out, attempt, required, uncertain, Reason.RULE_ATTRIBUTION_UNKNOWN)


def _report(
    ctx: AuditContext,
    out: Findings,
    attempt: AuditAttempt,
    required: list[str],
    uncertain: list[str],
    reason: Reason,
) -> None:
    hard = _tbd(ctx, required)
    soft = _tbd(ctx, uncertain)
    if hard:
        out.bad(
            "1",
            "order depends on a TBD rule",
            attempt=attempt,
            rule_key=",".join(hard),
            rule_value="TBD",
        )
    elif soft:
        out.unknown(
            "1",
            reason,
            "a possibly applicable rule is TBD",
            attempt=attempt,
            rule_key=",".join(soft),
            rule_value="TBD",
        )
    else:
        out.ok("1", "every rule the order depends on is set", attempt=attempt)
