"""V2 Number provenance: pass / violation / unverifiable, live and dry run."""

from test_audit_builders import (
    PUT,
    STO,
    LegSpec,
    Scenario,
    live_csp,
    off_csp,
    outcomes,
    reasons,
    rules_with,
)

from wheelta_robinhood_agent.agent.audit import check_v2
from wheelta_robinhood_agent.config.rules import RuleMarker
from wheelta_robinhood_agent.domain.enums import (
    AuditOutcome,
    DecisionAction,
    ExecutionMode,
    OrderSide,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.run_record import DecisionOutputStatus

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE


def _live(s: Scenario, **quote_kw: object) -> None:
    s.inst("put-1", PUT)
    state = s.state(90)
    q = s.quote("put-1", 95, **quote_kw)  # type: ignore[arg-type]
    place = s.place(100, "put-1", PUT, STO, 1, "1.10", quotes=(q,), snapshot=state)
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [place]))


def test_live_pass() -> None:
    s = Scenario()
    live_csp(s)
    f = check_v2(s.ctx())
    assert outcomes(f) == [P, P, P]


def test_no_attempts() -> None:
    assert outcomes(check_v2(Scenario().ctx())) == [P]


def test_unrecorded_quote_reference_is_violation() -> None:
    s = Scenario()
    live_csp(s)
    s.quotes.clear()
    f = check_v2(s.ctx())
    assert outcomes(f, "1") == [V]
    assert reasons(f, "3") == ["missing_evidence"]


def test_no_quote_at_all_is_violation_unlinked_is_unverifiable() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    place = s.place(100, "put-1", PUT, STO, 1, "1.10")
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [place]))
    assert outcomes(check_v2(s.ctx()), "1") == [V]
    s.quote("put-1", 95)
    assert reasons(check_v2(s.ctx()), "1") == ["missing_evidence"]


def test_quote_source_must_be_a_quote_call() -> None:
    s = Scenario()
    _live(s, tool="get_stock_quotes")
    assert outcomes(check_v2(s.ctx()), "1") == [V]
    s2 = Scenario()
    _live(s2, status=ToolCallStatus.FAILED)
    assert outcomes(check_v2(s2.ctx()), "1") == [V]


def test_assembled_quote_copy_must_equal_original() -> None:
    s = Scenario()
    live_csp(s)
    ctx = s.ctx()
    assert ctx.run_record is not None
    d = ctx.run_record.decisions[0]
    leg = d.legs[0]
    changed = leg.quotes[0].model_copy(update={"bid": leg.quotes[0].bid - 1 + 1 + 0})
    changed = changed.model_copy(update={"ask": leg.quotes[0].ask + 1})
    record = ctx.run_record.model_copy(
        update={
            "decisions": (
                d.model_copy(update={"legs": (leg.model_copy(update={"quotes": (changed,)}),)}),
            )
        }
    )
    f = check_v2(ctx.model_copy(update={"run_record": record}))
    assert outcomes(f, "1") == [V]


def test_missing_instrument_is_unverifiable() -> None:
    s = Scenario()
    live_csp(s)
    s.instruments.clear()
    f = check_v2(s.ctx())
    assert reasons(f, "1") == ["missing_evidence"]
    assert reasons(f, "2") == ["missing_evidence"]


def test_identity_mismatch_is_violation() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.inst("put-2", PUT)
    q = s.quote("put-2", 95)
    place = s.place(100, "put-1", PUT, STO, 1, "1.10", quotes=(q,))
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [place]))
    assert outcomes(check_v2(s.ctx()), "2") == [V]


def test_freshness() -> None:
    s = Scenario()
    _live(s)
    assert outcomes(check_v2(s.ctx()), "3") == [P]
    stale = Scenario()
    stale.inst("put-1", PUT)
    q = stale.quote("put-1", 10)
    place = stale.place(100, "put-1", PUT, STO, 1, "1.10", quotes=(q,))
    stale.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [place]))
    assert outcomes(check_v2(stale.ctx()), "3") == [V]
    stale.rules = rules_with(freshness={"option_quote_max_age_seconds": RuleMarker.NONE})
    assert outcomes(check_v2(stale.ctx()), "3") == [P]
    stale.rules = rules_with(freshness={"option_quote_max_age_seconds": RuleMarker.TBD})
    assert reasons(check_v2(stale.ctx()), "3") == ["rule_tbd"]


def test_quote_taken_after_placement_is_violation() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    q = s.quote("put-1", 110)
    place = s.place(100, "put-1", PUT, STO, 1, "1.10", quotes=(q,))
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [place]))
    assert outcomes(check_v2(s.ctx()), "3") == [V]


def test_live_without_record_uses_tool_events() -> None:
    s = Scenario()
    live_csp(s)
    s.record = False
    f = check_v2(s.ctx())
    assert reasons(f, "1") == ["missing_run_record"]
    assert outcomes(f, "2") == [P]


def test_unassociated_attempt_not_in_record() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.place(100, "put-1", PUT, STO, 1, "1.10")
    s.place_unrecorded = set(s.orders)
    assert reasons(check_v2(s.ctx()), "1") == ["missing_evidence"]


def test_dry_run_pass_and_unknown_time() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    assert outcomes(check_v2(s.ctx())) == [P, P, P]
    s.fact_sets.clear()
    s.auto_output = False
    assert reasons(check_v2(s.ctx()), "3") == ["missing_evidence"]
    s.output_status = DecisionOutputStatus.MISSING
    assert reasons(check_v2(s.ctx())) == ["missing_final_output"]
