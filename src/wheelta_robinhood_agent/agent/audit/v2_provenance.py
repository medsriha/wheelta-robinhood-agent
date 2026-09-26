"""V2 Number provenance (docs/VALIDATION.md "V2 Number provenance").

1. Each attempt's quote is an original, validated Robinhood quote/chain result from this run,
   joined to a validated instrument result; assembled quote copies equal the original.
2. Instrument identity agrees across leg, quote, instrument result, and broker arguments.
3. The quote is fresh at placement dispatch (live) or at the proposal's fact snapshot (off).
"""

from wheelta_robinhood_agent.agent.audit._common import (
    QUOTE_TOOLS,
    AuditAttempt,
    Findings,
    Reason,
    Unknown,
    audited_attempts,
    call_by_id,
    instrument,
    is_tbd,
    linked_quotes,
    quote_before,
    quote_source_completed,
    tool_name,
)
from wheelta_robinhood_agent.agent.audit.context import AuditContext
from wheelta_robinhood_agent.domain.enums import AuditCheck, ToolCallStatus
from wheelta_robinhood_agent.domain.run import AuditFinding
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.domain.sanity import is_fresh


def check_v2(ctx: AuditContext) -> tuple[AuditFinding, ...]:
    """Run V2 over every audited attempt."""
    out = Findings(ctx, AuditCheck.V2)
    attempts = audited_attempts(ctx)
    if isinstance(attempts, Unknown):
        out.unknown("all", attempts.reason, attempts.detail)
        return out.result()
    if not attempts:
        out.ok("all", "no placed or intended attempts")
        return out.result()
    for attempt in attempts:
        quotes = _provenance(ctx, out, attempt)
        _identity(ctx, out, attempt, quotes)
        _freshness(ctx, out, attempt, quotes)
    return out.result()


def _source_ok(ctx: AuditContext, quote: Quote) -> bool:
    for call_id in quote.source_tool_call_ids:
        call = call_by_id(ctx, call_id)
        if (
            call is None
            or tool_name(call) not in QUOTE_TOOLS
            or call.status is not ToolCallStatus.SUCCEEDED
        ):
            return False
    return True


def _provenance(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt
) -> tuple[Quote, ...] | None:
    """V2.1. Returns the linked original quotes for later sub-items (None if unknown)."""
    if attempt.record_attempt is None:
        reason = Reason.MISSING_RUN_RECORD if ctx.run_record is None else Reason.MISSING_EVIDENCE
        out.unknown("1", reason, "attempt not in the assembled record", attempt=attempt)
        return None
    quotes = linked_quotes(ctx, attempt)
    if quotes is None:
        out.bad("1", "attempt references a quote that is not a recorded result", attempt=attempt)
        return None
    if not quotes:
        any_quote = any(
            q.broker_instrument_id == attempt.instrument_id
            and (done := quote_source_completed(ctx, q)) is not None
            and attempt.at is not None
            and done <= attempt.at
            for q in ctx.quotes
        )
        if any_quote or attempt.instrument_id is None:
            out.unknown(
                "1", Reason.MISSING_EVIDENCE, "quote not linked to the attempt", attempt=attempt
            )
        else:
            out.bad("1", "no Robinhood quote for the contract before the attempt", attempt=attempt)
        return None
    ids = tuple(i for q in quotes for i in q.source_tool_call_ids)
    if not all(_source_ok(ctx, q) for q in quotes):
        out.bad(
            "1",
            "quote source is not a successful Robinhood quote/chain call in this run",
            attempt=attempt,
            tool_call_ids=ids,
        )
        return quotes
    copies = {q.quote_id: q for q in attempt.leg.quotes} if attempt.leg is not None else {}
    if any(q.quote_id in copies and copies[q.quote_id] != q for q in quotes):
        out.bad(
            "1",
            "assembled quote fields differ from the original quote result",
            attempt=attempt,
            tool_call_ids=ids,
        )
        return quotes
    fact = instrument(ctx, attempt.instrument_id)
    if fact is None:
        out.unknown(
            "1",
            Reason.MISSING_EVIDENCE,
            "no validated instrument result for the contract",
            attempt=attempt,
            tool_call_ids=ids,
        )
        return quotes
    out.ok(
        "1",
        "quote traces to a Robinhood quote/chain result and a validated instrument",
        attempt=attempt,
        tool_call_ids=(*ids, fact.source_tool_call_id),
    )
    return quotes


def _identity(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt, quotes: tuple[Quote, ...] | None
) -> None:
    """V2.2: instrument identity consistent across leg, quote, instrument, broker args."""
    fact = instrument(ctx, attempt.instrument_id)
    ids: list[str | None] = [attempt.instrument_id]
    occs = [attempt.occ]
    if attempt.leg is not None:
        ids.append(attempt.leg.broker_instrument_id)
        occs.append(attempt.leg.occ_symbol)
    if quotes:
        ids.extend(q.broker_instrument_id for q in quotes)
    known_ids = {i for i in ids if i is not None}
    known_occs = {str(o) for o in occs if o is not None}
    if fact is not None:
        known_occs.add(str(fact.occ_symbol))
    if len(known_ids) > 1 or len(known_occs) > 1:
        out.bad(
            "2",
            "instrument identity differs across leg, quote, instrument, or broker arguments",
            attempt=attempt,
            observed=",".join(sorted(known_ids | known_occs)),
        )
    elif attempt.instrument_id is None or fact is None or not known_occs:
        out.unknown("2", Reason.MISSING_EVIDENCE, "instrument identity incomplete", attempt=attempt)
    else:
        out.ok("2", "instrument identity consistent", attempt=attempt)


def _freshness(
    ctx: AuditContext, out: Findings, attempt: AuditAttempt, quotes: tuple[Quote, ...] | None
) -> None:
    """V2.3: the quote relied on is fresh at dispatch (live) / proposal snapshot time (off)."""
    key = "data_quality.freshness.option_quote_max_age_seconds"
    max_age = ctx.rules.data_quality.freshness.option_quote_max_age_seconds
    if is_tbd(max_age):
        out.unknown("3", Reason.RULE_TBD, f"{key} is TBD", attempt=attempt, rule_key=key)
        return
    if not quotes or attempt.at is None:
        out.unknown(
            "3",
            Reason.MISSING_EVIDENCE,
            "linked quote or attempt time unknown",
            attempt=attempt,
            rule_key=key,
        )
        return
    quote = quote_before(ctx, quotes, attempt.at)
    if quote is None:
        out.bad(
            "3",
            "no linked quote was retrieved before the attempt",
            attempt=attempt,
            rule_key=key,
        )
        return
    if not isinstance(max_age, int):  # `none`: deliberately no freshness limit
        out.ok("3", f"{key} is {max_age}", attempt=attempt, rule_key=key, rule_value=max_age)
        return
    age = (attempt.at - quote.as_of).total_seconds()
    ids = quote.source_tool_call_ids
    if is_fresh(quote.as_of, attempt.at, max_age):
        out.ok(
            "3",
            "quote fresh at the attempt time",
            attempt=attempt,
            rule_key=key,
            rule_value=max_age,
            observed=f"{age}s",
            tool_call_ids=ids,
        )
    else:
        out.bad(
            "3",
            "quote stale at the attempt time",
            attempt=attempt,
            rule_key=key,
            rule_value=max_age,
            observed=f"{age}s",
            tool_call_ids=ids,
        )
