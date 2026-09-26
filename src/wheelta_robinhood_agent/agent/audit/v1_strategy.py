"""V1 Strategy identity (docs/VALIDATION.md "V1 Strategy identity").

For every placed (live) or intended (off) attempt:
1. limit order, side sell_to_open or buy_to_close;
2. BTC quantity <= short quantity available to close net of other working BTC orders;
3. STO put: N <= C - limits.min_cash_reserve_usd;
4. STO call: multiplier x quantity <= owned shares not otherwise reserved.
Live uses the pre-order broker state as reported; off uses the dry-run reservation projection.
A missing broker field is unverifiable, never a zero reservation.
"""

from decimal import Decimal

from wheelta_robinhood_agent.agent.audit._common import (
    AuditAttempt,
    Findings,
    PreState,
    Reason,
    Unknown,
    audited_attempts,
    available_shares,
    available_to_close,
    collateral,
    instrument,
    is_tbd,
    pre_states,
)
from wheelta_robinhood_agent.agent.audit.context import AuditContext
from wheelta_robinhood_agent.config.rules import RuleMarker
from wheelta_robinhood_agent.domain.enums import AuditCheck, OptionRight, OrderSide
from wheelta_robinhood_agent.domain.run import AuditFinding

LIMIT = "limit"


def check_v1(ctx: AuditContext) -> tuple[AuditFinding, ...]:
    """Run V1 over every audited attempt."""
    out = Findings(ctx, AuditCheck.V1)
    attempts = audited_attempts(ctx)
    if isinstance(attempts, Unknown):
        out.unknown("all", attempts.reason, attempts.detail)
        return out.result()
    if not attempts:
        out.ok("all", "no placed or intended attempts")
        return out.result()
    states = pre_states(ctx, attempts)
    for attempt in attempts:
        _identity(out, attempt)
        side = attempt.side
        if side is None:
            continue
        pre = states[attempt.key]
        if side is OrderSide.BUY_TO_CLOSE:
            _close_quantity(ctx, out, attempt, pre)
            continue
        fact = instrument(ctx, attempt.instrument_id)
        if fact is None:
            out.unknown(
                "3-4", Reason.MISSING_EVIDENCE, "no validated instrument result", attempt=attempt
            )
        elif fact.occ_symbol.right is OptionRight.PUT:
            _put_cash(ctx, out, attempt, pre)
        else:
            _call_shares(ctx, out, attempt, pre, fact.occ_symbol.root)
    return out.result()


def _identity(out: Findings, attempt: AuditAttempt) -> None:
    """V1.1: order type is limit and side is STO/BTC."""
    if attempt.side_raw is None or attempt.order_type_raw is None:
        out.unknown(
            "1", Reason.MISSING_EVIDENCE, "side or order type not recorded", attempt=attempt
        )
        return
    observed = f"{attempt.side_raw}/{attempt.order_type_raw}"
    if attempt.side is None or attempt.order_type_raw != LIMIT:
        out.bad(
            "1", "not a limit sell_to_open/buy_to_close order", attempt=attempt, observed=observed
        )
    else:
        out.ok("1", "limit order with a permitted side", attempt=attempt, observed=observed)


def _close_quantity(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt, pre: PreState | Unknown
) -> None:
    """V1.2: BTC quantity <= short quantity available to close (other working BTC netted)."""
    if isinstance(pre, Unknown):
        out.unknown("2", pre.reason, pre.detail, attempt=attempt)
        return
    if attempt.quantity is None or attempt.instrument_id is None:
        out.unknown("2", Reason.MISSING_EVIDENCE, "quantity or instrument unknown", attempt=attempt)
        return
    available = available_to_close(ctx, pre, attempt.instrument_id)
    ids = pre.tool_call_ids
    if available is None:
        out.unknown(
            "2",
            Reason.UNMAPPED_BROKER_FIELD,
            "short position or working BTC quantity unknown",
            attempt=attempt,
            tool_call_ids=ids,
        )
    elif attempt.quantity > available:
        out.bad(
            "2",
            "buy-to-close quantity exceeds the short quantity available to close",
            attempt=attempt,
            observed=f"{attempt.quantity}>{available}",
            tool_call_ids=ids,
        )
    else:
        out.ok(
            "2",
            "close quantity within available short quantity",
            attempt=attempt,
            observed=f"{attempt.quantity}<={available}",
            tool_call_ids=ids,
        )


def _put_cash(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt, pre: PreState | Unknown
) -> None:
    """V1.3: N <= C - limits.min_cash_reserve_usd (`none` means no reserve requirement)."""
    key = "limits.min_cash_reserve_usd"
    reserve_rule = ctx.rules.limits.min_cash_reserve_usd
    if isinstance(pre, Unknown):
        out.unknown("3", pre.reason, pre.detail, attempt=attempt, rule_key=key)
        return
    ids = pre.tool_call_ids
    if is_tbd(reserve_rule):
        out.unknown("3", Reason.RULE_TBD, f"{key} is TBD", attempt=attempt, rule_key=key)
        return
    if reserve_rule is RuleMarker.AGENT_DISCRETION:
        out.unknown(
            "3", Reason.OPEN_DEFINITION, f"{key} is agent_discretion", attempt=attempt, rule_key=key
        )
        return
    reserve = reserve_rule if isinstance(reserve_rule, Decimal) else Decimal(0)
    n = collateral(ctx, attempt)
    if n is None or pre.cash is None:
        out.unknown(
            "3",
            Reason.UNMAPPED_BROKER_FIELD,
            "collateral or available settled cash unknown",
            attempt=attempt,
            rule_key=key,
            rule_value=reserve_rule,
            tool_call_ids=ids,
        )
        return
    observed = f"N={n} C={pre.cash}"
    if n > pre.cash - reserve:
        out.bad(
            "3",
            "put collateral exceeds available settled cash after the reserve",
            attempt=attempt,
            rule_key=key,
            rule_value=reserve_rule,
            observed=observed,
            tool_call_ids=ids,
        )
    else:
        out.ok(
            "3",
            "put collateral fits available settled cash after the reserve",
            attempt=attempt,
            rule_key=key,
            rule_value=reserve_rule,
            observed=observed,
            tool_call_ids=ids,
        )


def _call_shares(
    ctx: AuditContext,
    out: Findings,
    attempt: AuditAttempt,
    pre: PreState | Unknown,
    underlying: str,
) -> None:
    """V1.4: multiplier x quantity <= owned shares not reserved elsewhere."""
    if isinstance(pre, Unknown):
        out.unknown("4", pre.reason, pre.detail, attempt=attempt)
        return
    fact = instrument(ctx, attempt.instrument_id)
    ids = pre.tool_call_ids
    free = available_shares(ctx, pre, underlying)
    if fact is None or fact.multiplier is None or attempt.quantity is None or free is None:
        out.unknown(
            "4",
            Reason.UNMAPPED_BROKER_FIELD,
            "multiplier, quantity, or unreserved shares unknown",
            attempt=attempt,
            tool_call_ids=ids,
        )
        return
    needed = fact.multiplier * attempt.quantity
    observed = f"needed={needed} free={free}"
    if needed > free:
        out.bad(
            "4",
            "covered call exceeds unreserved owned shares",
            attempt=attempt,
            observed=observed,
            tool_call_ids=ids,
        )
    else:
        out.ok(
            "4",
            "covered call fits unreserved owned shares",
            attempt=attempt,
            observed=observed,
            tool_call_ids=ids,
        )
