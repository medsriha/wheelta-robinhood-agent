"""V7 Hard caps (docs/VALIDATION.md "V7"; ADR-0014 item 3).

1. Each STO attempt quantity <= limits.max_contracts_per_order.
2. Distinct new position lineages first filled this run (live) / unconditional opening intents
   (off) within max_new_positions_per_run, and with earlier same-day lineages within
   max_new_positions_per_day. Roll replacements continue their lineage and do not count.
3. Per-underlying CSP exposure (held shorts + unfilled working STO puts + N) within the USD
   cap and the account-value ratio.
4. R + N <= limits.max_total_csp_collateral_ratio x B.
5. No STO after the daily loss breaker: `none` passes; an enabled breaker is unverifiable
   until its data source is verified (open definition, VALIDATION.md "Decisions still open").
A cap set to `none` passes without data; `TBD` is unverifiable here (V5 reports it).
"""

from decimal import Decimal

from wheelta_robinhood_agent.agent.audit._common import (
    AuditAttempt,
    Findings,
    PreState,
    Reason,
    Unknown,
    audited_attempts,
    collateral,
    csp_exposure,
    instrument,
    is_tbd,
    pre_states,
)
from wheelta_robinhood_agent.agent.audit.context import AuditContext
from wheelta_robinhood_agent.config.rules import RuleMarker
from wheelta_robinhood_agent.domain.enums import (
    AuditCheck,
    AuditOutcome,
    ExecutionMode,
    OptionRight,
    OrderSide,
)
from wheelta_robinhood_agent.domain.run import AuditFinding


def check_v7(ctx: AuditContext) -> tuple[AuditFinding, ...]:
    """Run V7 over every audited sell-to-open attempt and the run's new lineages."""
    out = Findings(ctx, AuditCheck.V7)
    attempts = audited_attempts(ctx)
    if isinstance(attempts, Unknown):
        out.unknown("all", attempts.reason, attempts.detail)
        return out.result()
    opens = [a for a in attempts if a.side is OrderSide.SELL_TO_OPEN]
    states = pre_states(ctx, attempts)
    for attempt in opens:
        _per_order(ctx, out, attempt)
        fact = instrument(ctx, attempt.instrument_id)
        if fact is None:
            out.unknown(
                "3-4", Reason.MISSING_EVIDENCE, "no validated instrument result", attempt=attempt
            )
        elif fact.occ_symbol.right is OptionRight.PUT:
            _per_underlying(ctx, out, attempt, states[attempt.key], fact.occ_symbol.root)
            _total_ratio(ctx, out, attempt, states[attempt.key])
        _drawdown(ctx, out, attempt)
    _new_positions(ctx, out, attempts)
    return out.result()


def _per_order(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    key = "limits.max_contracts_per_order"
    cap = ctx.rules.limits.max_contracts_per_order
    if cap is RuleMarker.NONE:
        out.ok("1", f"{key} is none", attempt=attempt, rule_key=key, rule_value=cap)
    elif not isinstance(cap, int):
        out.unknown("1", Reason.RULE_TBD, f"{key} is {cap}", attempt=attempt, rule_key=key)
    elif attempt.quantity is None:
        out.unknown("1", Reason.MISSING_EVIDENCE, "quantity unknown", attempt=attempt, rule_key=key)
    elif attempt.quantity > cap:
        out.bad(
            "1",
            "order quantity exceeds the per-order cap",
            attempt=attempt,
            rule_key=key,
            rule_value=cap,
            observed=attempt.quantity,
        )
    else:
        out.ok(
            "1",
            "order quantity within the per-order cap",
            attempt=attempt,
            rule_key=key,
            rule_value=cap,
            observed=attempt.quantity,
        )


def _combine(parts: list[tuple[AuditOutcome, str]]) -> tuple[AuditOutcome, str]:
    for wanted in (AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE):
        hits = [d for o, d in parts if o is wanted]
        if hits:
            return wanted, "; ".join(hits)
    return AuditOutcome.PASS, "; ".join(d for _, d in parts)


def _per_underlying(
    ctx: AuditContext,
    out: Findings,
    attempt: AuditAttempt,
    pre: PreState | Unknown,
    underlying: str,
) -> None:
    usd_cap = ctx.rules.limits.max_collateral_per_underlying_usd
    ratio = ctx.rules.limits.max_collateral_per_underlying_ratio
    key = "limits.max_collateral_per_underlying_usd,limits.max_collateral_per_underlying_ratio"
    caps = (usd_cap, ratio)
    if all(c is RuleMarker.NONE for c in caps):
        out.ok("3", "per-underlying caps are none", attempt=attempt, rule_key=key)
        return
    if any(not isinstance(c, Decimal) and c is not RuleMarker.NONE for c in caps):
        out.unknown(
            "3", Reason.RULE_TBD, "a per-underlying cap is not set", attempt=attempt, rule_key=key
        )
        return
    if isinstance(pre, Unknown):
        out.unknown("3", pre.reason, pre.detail, attempt=attempt, rule_key=key)
        return
    n = collateral(ctx, attempt)
    held = csp_exposure(ctx, pre, underlying)
    ids = pre.tool_call_ids
    if n is None or held is None:
        out.unknown(
            "3",
            Reason.UNMAPPED_BROKER_FIELD,
            "collateral or existing exposure unknown",
            attempt=attempt,
            rule_key=key,
            tool_call_ids=ids,
        )
        return
    exposure = held + n
    parts: list[tuple[AuditOutcome, str]] = []
    if isinstance(usd_cap, Decimal):
        ok = exposure <= usd_cap
        parts.append(
            (
                AuditOutcome.PASS if ok else AuditOutcome.VIOLATION,
                f"exposure {exposure} vs {usd_cap}",
            )
        )
    if isinstance(ratio, Decimal):
        value = pre.snapshot.account_value_usd
        if value is None:
            parts.append((AuditOutcome.UNVERIFIABLE, "account value unknown"))
        else:
            ok = exposure <= ratio * value
            parts.append(
                (
                    AuditOutcome.PASS if ok else AuditOutcome.VIOLATION,
                    f"exposure {exposure} vs {ratio} x {value}",
                )
            )
    outcome, detail = _combine(parts)
    if outcome is AuditOutcome.UNVERIFIABLE:
        detail = f"{Reason.UNMAPPED_BROKER_FIELD.value}: {detail}"
    out.add("3", outcome, detail, attempt=attempt, rule_key=key, tool_call_ids=ids)


def _total_ratio(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt, pre: PreState | Unknown
) -> None:
    key = "limits.max_total_csp_collateral_ratio"
    ratio = ctx.rules.limits.max_total_csp_collateral_ratio
    if ratio is RuleMarker.NONE:
        out.ok("4", f"{key} is none", attempt=attempt, rule_key=key, rule_value=ratio)
        return
    if not isinstance(ratio, Decimal):
        out.unknown("4", Reason.RULE_TBD, f"{key} is {ratio}", attempt=attempt, rule_key=key)
        return
    if isinstance(pre, Unknown):
        out.unknown("4", pre.reason, pre.detail, attempt=attempt, rule_key=key)
        return
    n = collateral(ctx, attempt)
    ids = pre.tool_call_ids
    if n is None or pre.reserved is None or pre.base is None:
        out.unknown(
            "4",
            Reason.UNMAPPED_BROKER_FIELD,
            "collateral, reserved cash, or cash base unknown",
            attempt=attempt,
            rule_key=key,
            rule_value=ratio,
            tool_call_ids=ids,
        )
        return
    observed = f"R={pre.reserved} N={n} B={pre.base}"
    if pre.reserved + n > ratio * pre.base:
        out.bad(
            "4",
            "total CSP collateral exceeds the ratio of the pre-order cash base",
            attempt=attempt,
            rule_key=key,
            rule_value=ratio,
            observed=observed,
            tool_call_ids=ids,
        )
    else:
        out.ok(
            "4",
            "total CSP collateral within the ratio of the cash base",
            attempt=attempt,
            rule_key=key,
            rule_value=ratio,
            observed=observed,
            tool_call_ids=ids,
        )


def _drawdown(ctx: AuditContext, out: Findings, attempt: AuditAttempt) -> None:
    key = "circuit_breakers.max_daily_drawdown_usd"
    value = ctx.rules.circuit_breakers.max_daily_drawdown_usd
    if value is RuleMarker.NONE:
        out.ok("5", "daily loss breaker disabled (none)", attempt=attempt, rule_key=key)
    elif is_tbd(value):
        out.unknown("5", Reason.RULE_TBD, f"{key} is TBD", attempt=attempt, rule_key=key)
    else:
        out.unknown(
            "5",
            Reason.OPEN_DEFINITION,
            "V7.5 daily realized+unrealized loss source is not verified",
            attempt=attempt,
            rule_key=key,
            rule_value=value,
        )


def _new_positions(ctx: AuditContext, out: Findings, attempts: tuple[AuditAttempt, ...]) -> None:
    """V7.2: count distinct new lineages (known, unknown) and compare with run/day caps."""
    known, unknown = _count_new(ctx, attempts)
    run_cap = ctx.rules.limits.max_new_positions_per_run
    day_cap = ctx.rules.limits.max_new_positions_per_day
    _compare(out, "limits.max_new_positions_per_run", run_cap, known, unknown)
    if day_cap is RuleMarker.NONE or not isinstance(day_cap, int):
        _compare(out, "limits.max_new_positions_per_day", day_cap, known, unknown)
    elif ctx.day_history is None:
        out.unknown(
            "2",
            Reason.MISSING_EVIDENCE,
            "earlier same-day lineages unknown",
            rule_key="limits.max_new_positions_per_day",
            rule_value=day_cap,
        )
    else:
        prior = len(ctx.day_history.prior_new_lineage_ids)
        _compare(out, "limits.max_new_positions_per_day", day_cap, known + prior, unknown)


def _compare(out: Findings, key: str, cap: int | RuleMarker, known: int, unknown: int) -> None:
    observed = f"known={known} unknown={unknown}"
    if cap is RuleMarker.NONE:
        out.ok("2", f"{key} is none", rule_key=key, rule_value=cap)
    elif not isinstance(cap, int):
        out.unknown("2", Reason.RULE_TBD, f"{key} is {cap}", rule_key=key)
    elif known > cap:
        out.bad(
            "2",
            "new position lineages exceed the cap",
            rule_key=key,
            rule_value=cap,
            observed=observed,
        )
    elif known + unknown > cap:
        out.unknown(
            "2",
            Reason.LINEAGE_UNKNOWN,
            "unattributed openings could exceed the cap",
            rule_key=key,
            rule_value=cap,
            observed=observed,
        )
    else:
        out.ok(
            "2",
            "new position lineages within the cap",
            rule_key=key,
            rule_value=cap,
            observed=observed,
        )


def _count_new(ctx: AuditContext, attempts: tuple[AuditAttempt, ...]) -> tuple[int, int]:
    """(known new lineages, possibly-new unattributed ones). Roll replacements never count."""
    opens = [a for a in attempts if a.side is OrderSide.SELL_TO_OPEN and not a.is_roll_replacement]
    if ctx.effective_execution_mode is not ExecutionMode.LIVE:
        legs = {a.leg_ref for a in opens if a.is_new_open and not a.conditional}
        return len(legs), 0
    known: set[str] = set()
    unknown = 0
    for attempt in opens:
        filled = _filled(attempt)
        if filled == 0:
            continue
        if attempt.is_new_open and attempt.leg_ref is not None and filled is not None:
            known.add(attempt.leg_ref)
        elif not (attempt.is_new_open and attempt.leg_ref in known):
            unknown += 1
    return len(known), unknown


def _filled(attempt: AuditAttempt) -> int | None:
    """Deduplicated broker fills of a live step; a denied step filled nothing."""
    if attempt.order is None:
        return None if attempt.dispatched else 0
    return attempt.order.filled_quantity
