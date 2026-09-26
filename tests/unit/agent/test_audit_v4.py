"""V4 Decision and assembled-record invariants: pass / violation / unverifiable."""

from decimal import Decimal

from test_audit_builders import (
    BTC,
    CALL,
    NEW_PUT,
    PUT,
    PUT2,
    STO,
    T0,
    LegSpec,
    Scenario,
    live_csp,
    off_csp,
    outcomes,
    reasons,
)

from wheelta_robinhood_agent.agent.audit import check_v4
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    CancellationRationale,
    Decision,
    ProposedLeg,
)
from wheelta_robinhood_agent.domain.enums import (
    AttemptStatus,
    AuditOutcome,
    DataQuality,
    DecisionAction,
    ExecutionMode,
    OrderSide,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.orders import FillObservationKind
from wheelta_robinhood_agent.domain.positions import PositionBook
from wheelta_robinhood_agent.domain.run_record import AssemblyFinding, DecisionOutputStatus

P, V, U = AuditOutcome.PASS, AuditOutcome.VIOLATION, AuditOutcome.UNVERIFIABLE


def test_missing_record_is_unverifiable() -> None:
    s = Scenario()
    s.record = False
    assert reasons(check_v4(s.ctx())) == ["missing_run_record"]


# V4.1 -----------------------------------------------------------------------------------------


def test_one_management_decision_per_lineage() -> None:
    s = Scenario()
    s.position("position:1", "put-1", PUT)
    s.decide(DecisionAction.HOLD, target_ref="position:1")
    assert outcomes(check_v4(s.ctx()), "1") == [P]
    s.decide(DecisionAction.HOLD, target_ref="position:1")
    assert outcomes(check_v4(s.ctx()), "1") == [V]
    s.decisions.clear()
    assert outcomes(check_v4(s.ctx()), "1") == [V]
    s.decide(DecisionAction.HOLD, target_ref="position:unknown")
    assert outcomes(check_v4(s.ctx()), "1") == [V, V]


def test_coverage_unverifiable_without_book_or_output() -> None:
    s = Scenario()
    s.book = None
    assert reasons(check_v4(s.ctx()), "1") == ["missing_evidence"]
    s.book = PositionBook(
        as_of=T0,
        entries=(),
        gaps=(Gap(field="lineage", kind=DataQuality.CONTRADICTORY, detail="ambiguous"),),
    )
    assert reasons(check_v4(s.ctx()), "1") == ["missing_evidence"]
    s.output_status = DecisionOutputStatus.MISSING
    assert reasons(check_v4(s.ctx()), "1") == ["missing_final_output"]


# V4.2 -----------------------------------------------------------------------------------------


def test_leg_shapes_live() -> None:
    s = Scenario()
    live_csp(s)
    s.decide(DecisionAction.HOLD, target_ref="position:none")
    s.book = None
    assert outcomes(check_v4(s.ctx()), "2") == [P, P]
    s2 = Scenario()
    s2.inst("call-1", CALL)
    p = s2.place(100, "call-1", CALL, STO, 1, "1.10")
    s2.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, CALL, "call-1", 1, [p]))
    assert outcomes(check_v4(s2.ctx()), "2") == [V]


def test_no_decisions_and_missing_output() -> None:
    s = Scenario()
    assert outcomes(check_v4(s.ctx()), "2") == [P]
    s.output_status = DecisionOutputStatus.INVALID
    assert reasons(check_v4(s.ctx()), "2") == ["missing_final_output"]


def _roll(
    s: Scenario, *, with_open: bool, open_first: bool = False, occ2: object = NEW_PUT
) -> None:
    entry = s.position("position:1", "put-1", PUT)
    s.inst("put-1", PUT)
    s.inst("put-new", occ2)  # type: ignore[arg-type]
    close_place = s.place(100, "put-1", PUT, BTC, 1, "0.50")
    s.status(close_place, 110, AttemptStatus.PARTIALLY_FILLED, filled=0)
    close = LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1, [close_place], leg_ref="leg:c")
    legs = [close]
    if with_open:
        open_place = s.place(200, "put-new", occ2, STO, 1, "1.10")  # type: ignore[arg-type]
        legs.append(
            LegSpec(
                OrderSide.SELL_TO_OPEN,
                occ2,
                "put-new",
                1,
                [open_place],
                depends_on="leg:c",  # type: ignore[arg-type]
            )
        )
    if open_first:
        legs.reverse()
        legs[1].depends_on = None
    s.decide(DecisionAction.ROLL, *legs, target_ref="position:1", position_id=entry.position_id)


def test_partial_roll_close_preserved_live() -> None:
    s = Scenario()
    _roll(s, with_open=False)
    f = check_v4(s.ctx())
    assert outcomes(f, "1") == [P]
    assert outcomes(f, "2") == [P]


def test_roll_shape_violations() -> None:
    s = Scenario()
    _roll(s, with_open=True, open_first=True)
    assert outcomes(check_v4(s.ctx()), "2") == [V]
    s2 = Scenario()
    _roll(s2, with_open=True, occ2=CALL)
    assert outcomes(check_v4(s2.ctx()), "2") == [V]
    s3 = Scenario()
    _roll(s3, with_open=True, occ2=PUT2)
    assert outcomes(check_v4(s3.ctx()), "2") == [V]
    s4 = Scenario()
    _roll(s4, with_open=True)
    assert outcomes(check_v4(s4.ctx()), "2") == [P]


def test_close_leg_must_be_held_in_lineage() -> None:
    s = Scenario()
    entry = s.position("position:1", "put-1", PUT)
    s.inst("put-2", PUT2)
    p = s.place(100, "put-2", PUT2, BTC, 1, "0.50")
    s.decide(
        DecisionAction.CLOSE,
        LegSpec(OrderSide.BUY_TO_CLOSE, PUT2, "put-2", 1, [p]),
        target_ref="position:1",
        position_id=entry.position_id,
    )
    assert outcomes(check_v4(s.ctx()), "2") == [V]
    s.book = None
    assert reasons(check_v4(s.ctx()), "2") == ["missing_evidence"]


def test_leg_right_unknown() -> None:
    s = Scenario(ExecutionMode.OFF)
    leg = off_csp(s)
    leg.occ = None
    s.decide(DecisionAction.OPEN_CSP, leg)
    assert reasons(check_v4(s.ctx()), "2") == ["missing_evidence"]


def test_dry_run_exact_shape() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.CLOSE, target_ref="position:1")
    assert outcomes(check_v4(s.ctx()), "2") == [V]


# V4.3 -----------------------------------------------------------------------------------------


def test_every_place_and_cancel_recorded() -> None:
    s = Scenario()
    place, _ = live_csp(s)
    s.cancel(place, 150)
    f = check_v4(s.ctx())
    assert outcomes(f, "3") == [P, P]
    s.call("cancel_option_order", 160)
    s.place_unrecorded = set()
    s.inst("put-9", PUT)
    extra = s.place(170, "put-9", PUT, STO, 1, "1.10")
    s.place_unrecorded = {extra.identity.tool_call_id}
    f = check_v4(s.ctx())
    assert outcomes(f, "3").count(V) == 2


def test_known_status_needs_matching_broker_id() -> None:
    s = Scenario()
    _, leg = live_csp(s)
    leg.record_overrides[0] = {"broker_order_id": "someone-else"}
    assert outcomes(check_v4(s.ctx()), "3") == [V]


def test_unassociated_action_makes_attribution_unverifiable() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    place = s.place(100, "put-1", PUT, STO, 1, "1.10")
    s.cancel(place, 120)
    s.orders[place.identity.tool_call_id].cancellations.clear()
    ctx = s.ctx()
    assert ctx.run_record is not None
    from wheelta_robinhood_agent.domain.orders import Cancellation
    from wheelta_robinhood_agent.domain.run_record import UnassociatedAction, UnassociatedActionKind

    cancel_call = [c for c in s.calls if c.identity.tool.endswith("cancel_option_order")][0]
    s.unassociated.append(
        UnassociatedAction(
            kind=UnassociatedActionKind.CANCEL,
            cancellation=Cancellation(
                cancel_tool_call_id=cancel_call.identity.tool_call_id,
                broker_order_id=None,
                status="unknown",
            ),
        )
    )
    f = check_v4(s.ctx())
    assert reasons(f, "3") == ["unassociated_action", "unassociated_action"]


# V4.4 -----------------------------------------------------------------------------------------


def test_price_walk_with_fill_during_cancel() -> None:
    """Step 1 (5) is cancelled after 2 fill during cancellation; step 2 works the other 3."""
    s = Scenario()
    s.inst("put-1", PUT)
    q = s.quote("put-1", 95)
    first = s.place(100, "put-1", PUT, STO, 5, "1.15", quotes=(q,))
    s.status(first, 110, AttemptStatus.PLACED)
    confirm_at = 125
    s.cancel(first, 120)
    s.status(first, confirm_at, AttemptStatus.CANCELLED, filled=2)
    q2 = s.quote("put-1", 195)
    second = s.place(200, "put-1", PUT, STO, 3, "1.10", quotes=(q2,))
    s.status(second, 210, AttemptStatus.FILLED, filled=3)
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 5, [first, second])
    )
    f = check_v4(s.ctx())
    assert outcomes(f, "4") == [P]
    assert [x.observed_value for x in f if x.sub_item == "4"] == ["5<=5"]
    assert outcomes(f, "3") == [P, P, P]


def test_later_step_exceeding_remainder_and_fills_over_target() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    first = s.place(100, "put-1", PUT, STO, 2, "1.15")
    s.status(first, 110, AttemptStatus.CANCELLED, filled=1)
    second = s.place(200, "put-1", PUT, STO, 2, "1.10")
    s.status(second, 210, AttemptStatus.FILLED, filled=2)
    s.decide(
        DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 2, [first, second])
    )
    assert outcomes(check_v4(s.ctx()), "4") == [V, V]


def test_duplicated_fill_observations_are_deduplicated() -> None:
    s = Scenario()
    place, leg = live_csp(s, qty=2)
    s.orders[place.identity.tool_call_id].fills.clear()
    s.fill(place, 105, 1, execution_id="x-1")
    s.fill(place, 106, 1, execution_id="x-1")  # the same execution observed twice
    s.fill(place, 107, 1, execution_id="x-2")
    f = check_v4(s.ctx())
    assert outcomes(f, "4") == [P]


def test_assembled_fill_mismatch_and_unknowns() -> None:
    s = Scenario()
    place, leg = live_csp(s)
    leg.record_filled[0] = 0
    assert outcomes(check_v4(s.ctx()), "4") == [V, P]
    s2 = Scenario()
    place2, leg2 = live_csp(s2, qty=2)
    s2.fill(place2, 120, 1, kind=FillObservationKind.CUMULATIVE)  # total decreases: inconsistent
    leg2.record_filled[0] = 1
    assert reasons(check_v4(s2.ctx()), "4") == ["missing_evidence"]
    s3 = Scenario()
    _, leg3 = live_csp(s3)
    leg3.target = None
    assert reasons(check_v4(s3.ctx()), "4") == ["target_unknown"]


def test_denied_step_counts_no_fill_and_unknown_step_is_unverifiable() -> None:
    s = Scenario()
    s.inst("put-1", PUT)
    denied = s.place(100, "put-1", PUT, STO, 1, "1.10", denied=True)
    s.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [denied]))
    ctx = s.ctx()
    no_order = ctx.model_copy(update={"order_records": ()})
    assert outcomes(check_v4(no_order), "4") == [P]
    s2 = Scenario()
    s2.inst("put-1", PUT)
    unknown = s2.place(100, "put-1", PUT, STO, 1, "1.10")
    s2.decide(DecisionAction.OPEN_CSP, LegSpec(OrderSide.SELL_TO_OPEN, PUT, "put-1", 1, [unknown]))
    ctx2 = s2.ctx()
    assert reasons(check_v4(ctx2.model_copy(update={"order_records": ()})), "4") == [
        "missing_evidence"
    ]


# V4.5 -----------------------------------------------------------------------------------------


def test_dry_run_invariants_pass() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    assert outcomes(check_v4(s.ctx()), "5") == [P]


def test_dry_run_actual_action_is_violation() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    s.call("place_option_order", 100)
    assert outcomes(check_v4(s.ctx()), "5") == [V]
    s2 = Scenario(ExecutionMode.OFF)
    s2.inst("put-1", PUT)
    s2.place(100, "put-1", PUT, STO, 1, "1.10", denied=True)
    assert outcomes(check_v4(s2.ctx()), "5") == [V]


def test_conditional_opening_after_hypothetical_close() -> None:
    s = Scenario(ExecutionMode.OFF)
    entry = s.position("position:1", "put-1", PUT)
    close = off_csp(s)
    close.side = OrderSide.BUY_TO_CLOSE
    close.leg_ref = "leg:close"
    replacement = off_csp(s, iid="put-new", occ=NEW_PUT, conditional=True, depends_on="leg:close")
    s.decide(
        DecisionAction.ROLL,
        close,
        replacement,
        target_ref="position:1",
        position_id=entry.position_id,
    )
    f = check_v4(s.ctx())
    assert outcomes(f, "5") == [P, P]
    assert outcomes(f, "2") == [P]
    replacement.conditional = False
    assert outcomes(check_v4(s.ctx()), "5") == [V, P]


def test_conditional_without_dependency_and_funding_dependency() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, conditional=True))
    assert outcomes(check_v4(s.ctx()), "5") == [V, P]
    s2 = Scenario(ExecutionMode.OFF)
    s2.decide(DecisionAction.CLOSE, LegSpec(OrderSide.BUY_TO_CLOSE, PUT, "put-1", 1))
    s2.decide(DecisionAction.OPEN_CSP, off_csp(s2), depends_on=("decision:0",))
    assert outcomes(check_v4(s2.ctx()), "5") == [V, P]


# V4.6 -----------------------------------------------------------------------------------------


def test_model_boundary() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s))
    assert outcomes(check_v4(s.ctx()), "6") == [P]
    s.findings.append(AssemblyFinding(code="unresolved_ref", detail="evidence:9 not delivered"))
    assert outcomes(check_v4(s.ctx()), "6") == [P, V]
    s.output_status = DecisionOutputStatus.INVALID
    s.findings.clear()
    assert outcomes(check_v4(s.ctx()), "6") == [V]
    s.output_status = DecisionOutputStatus.MISSING
    assert reasons(check_v4(s.ctx()), "6") == ["missing_final_output"]


def _output(
    facts_ref: str, price: str, *, execution: tuple[str, ...] = (), cancel: bool = False
) -> AgentDecisionOutput:
    return AgentDecisionOutput(
        decisions=(
            Decision(
                action=DecisionAction.OPEN_CSP,
                target_ref="candidate:0",
                replacement_ref=None,
                funding_close_refs=(),
                proposed_legs=(ProposedLeg(facts_ref=facts_ref, limit_price=price),),
                execution_refs=execution,
                rationale="r",
                thesis=None,
                invalidation_conditions=(),
                evidence_refs=(),
            ),
        ),
        cancellation_rationales=(
            CancellationRationale(cancel_call_ref="call:1", rationale="r", evidence_refs=()),
        )
        if cancel
        else (),
        unresolved_questions=(),
    )


def test_fabricated_refs_and_dry_run_execution_refs() -> None:
    s = Scenario(ExecutionMode.OFF)
    s.decide(DecisionAction.OPEN_CSP, off_csp(s, ref="facts:a"))
    s.fact_sets.append(
        s.fact_sets[0].model_copy(
            update={
                "facts_ref": "facts:priced",
                "limit_price": Decimal("1.30"),
                "facts_id": s.fact_sets[0].facts_id,
            }
        )
    )
    s.auto_output = False
    s.output = _output("facts:missing", "1.10")
    assert outcomes(check_v4(s.ctx()), "6") == [V]
    s.output = _output("facts:priced", "1.10")
    assert outcomes(check_v4(s.ctx()), "6") == [V]
    s.output = _output("facts:a", "1.10", execution=("call:9",), cancel=True)
    f = [x for x in check_v4(s.ctx()) if x.sub_item == "6"]
    assert f[0].outcome is V and "execution_refs" in f[0].detail and "cancellation" in f[0].detail
