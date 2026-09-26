"""V3 Reviewed = placed = assembled: pass / violation / unverifiable, live and dry run."""

from decimal import Decimal

from test_audit_builders import (
    PUT,
    STO,
    LegSpec,
    Scenario,
    live_csp,
    off_csp,
    only,
    outcomes,
    reasons,
    rules_with,
)

from wheelta_robinhood_agent.agent.audit import check_v3
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditOutcome,
    DecisionAction,
    ExecutionMode,
    OrderSide,
)

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE


def _one(s: Scenario, price: str = "1.10", **place_kw: object) -> LegSpec:
    s.inst("put-1", PUT)
    q = s.quote("put-1", 95)
    place = s.place(100, "put-1", PUT, STO, 1, price, quotes=(q,), **place_kw)  # type: ignore[arg-type]
    leg = LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [place])
    s.decide(DecisionAction.OPEN_CSP, leg)
    return leg


def test_live_pass() -> None:
    s = Scenario()
    live_csp(s)
    assert outcomes(check_v3(s.ctx())) == [P, P, P, P]


def test_no_attempts() -> None:
    assert outcomes(check_v3(Scenario().ctx())) == [P]


def test_review_missing_warning_or_mismatch_is_violation() -> None:
    s = Scenario()
    _one(s, review=False)
    assert outcomes(check_v3(s.ctx()), "1") == [V]
    s.review(99, "put-1", STO, 1, "1.10", warnings=("wide spread",))
    assert outcomes(check_v3(s.ctx()), "1") == [V]
    s.review(99.2, "put-1", STO, 2, "1.10")
    assert outcomes(check_v3(s.ctx()), "1") == [V]
    s.review(99.3, "put-1", STO, 1, "1.10", succeeded=False)
    assert outcomes(check_v3(s.ctx()), "1") == [V]


def test_review_outcome_unknown_is_unverifiable() -> None:
    s = Scenario()
    _one(s, review=False)
    s.review(99, "put-1", STO, 1, "1.10", record=False)
    assert reasons(check_v3(s.ctx()), "1") == ["missing_evidence"]
    s2 = Scenario()
    _one(s2, review=False)
    s2.review(99, "put-1", STO, 1, "1.10", tif=None)
    assert reasons(check_v3(s2.ctx()), "1") == ["missing_evidence"]


def test_each_price_step_needs_its_own_review() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    first = s.place(100, "put-1", PUT, STO, 1, "1.15")
    s.status(first, 110, AttemptStatus.CANCELLED, filled=0)
    second = s.place(200, "put-1", PUT, STO, 1, "1.15", review=False)
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [first, second])
    )
    assert outcomes(check_v3(s.ctx()), "1") == [P, V]


def test_denied_place_and_missing_arguments() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    s.place(100, "put-1", PUT, STO, 1, "1.10", denied=True)
    assert outcomes(check_v3(s.ctx()), "1") == [P]
    s2 = Scenario()
    s2.inst("put-1", PUT)
    call = s2.place(100, "put-1", PUT, STO, 1, "1.10")
    s2.orders.pop(call.identity.tool_call_id)
    f = check_v3(s2.ctx())
    assert reasons(f, "1") == ["missing_evidence"]
    assert reasons(f, "2") == ["missing_evidence"]
    assert reasons(f, "4") == ["missing_evidence"]


def test_assembled_parameters() -> None:
    s = Scenario()
    leg = _one(s)
    leg.record_overrides[0] = {"limit_price": Decimal("1.05")}
    assert outcomes(check_v3(s.ctx()), "2") == [V]
    s2 = Scenario()
    leg2 = _one(s2)
    leg2.facts_ref = "facts:x"
    s2.facts("facts:x", 90, qty=2)
    f = only(check_v3(s2.ctx()), "2")
    assert f.outcome is P and "differs from decision-facts" in f.detail
    s2.record = False
    assert reasons(check_v3(s2.ctx()), "2") == ["missing_run_record"]


def test_limit_bounds_and_tick() -> None:
    s = Scenario()
    _one(s, price="1.25")
    assert outcomes(check_v3(s.ctx()), "3") == [V]
    s2 = Scenario()
    _one(s2, price="1.105")
    assert outcomes(check_v3(s2.ctx()), "3") == [V]
    s3 = Scenario()
    s3.inst("put-1", PUT, tick=None)
    q = s3.quote("put-1", 95)
    p = s3.place(100, "put-1", PUT, STO, 1, "1.10", quotes=(q,))
    s3.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [p]))
    assert reasons(check_v3(s3.ctx()), "3") == ["missing_evidence"]


def test_time_in_force() -> None:
    s = Scenario()
    _one(s, tif="gtc")
    assert outcomes(check_v3(s.ctx()), "4") == [V]
    s.rules = rules_with(orders={"time_in_force": "TBD"})
    assert reasons(check_v3(s.ctx()), "4") == ["rule_tbd"]


def test_dry_run() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    f = check_v3(s.ctx())
    assert reasons(f, "1") == ["dry_run_no_execution"]
    assert reasons(f, "2") == ["dry_run_no_execution"]
    assert outcomes(f, "3") == [P]
    assert outcomes(f, "4") == [P]
    s.record = False
    assert reasons(check_v3(s.ctx())) == ["missing_run_record"]
