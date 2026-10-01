"""Reference feedback built from an assembled record (ADR-0052, domain/reference_check.py)."""

from test_domain_assembly import (
    ExecutionMode,
    _live_bundle,
    assemble_run_record,
    call,
    cand,
    cref,
    ctx,
    decision,
    intent,
    order,
    parsed,
    uid,
)

from wheelta_robinhood_agent.domain.enums import DecisionAction, ToolCallStatus
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.reference_check import reference_issues
from wheelta_robinhood_agent.domain.run_record import AssemblyFinding

PUT_A = OccSymbol.parse("AAPL  261016P00019000")
PLACE_REF = f"order_call:{uid(11)}"
CITABLE = frozenset({PLACE_REF})


def test_a_linked_order_leaves_no_issue() -> None:
    record = assemble_run_record(
        _live_bundle(), parsed(decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=("call:11",)))
    )
    assert reference_issues(record, CITABLE) == ()


def test_a_bare_call_id_is_returned_with_the_order_it_should_cite() -> None:
    """The dry run of 2026-09-30: the agent cited bare tool call IDs, so the place stayed
    unlinked. Both the unresolved ref and the unclaimed place come back, naming the ref."""
    out = parsed(decision(DecisionAction.OPEN_CSP, "cand:a", exec_refs=(str(uid(11)),)))
    issues = reference_issues(assemble_run_record(_live_bundle(), out), CITABLE)
    unresolved, unclaimed = issues
    assert unresolved.startswith("decisions[0]: execution_ref was never issued by code")
    assert unresolved.endswith("[unknown_reference]")
    assert unclaimed.startswith(f"place call {PLACE_REF} (sell_to_open AAPL  261016P00019000)")
    assert unclaimed.endswith("it matches decisions[0]")


def test_an_unlinked_place_without_a_fit_asks_for_its_decision() -> None:
    out = parsed(decision(DecisionAction.OPEN_CSP, "cand:b"))
    context = _live_bundle().model_copy(
        update={"refs": (*_live_bundle().refs, cand("cand:b", PUT_A, "inst-b"))}
    )
    (issue,) = reference_issues(assemble_run_record(context, out), CITABLE)
    assert issue.endswith("leave it: code records it as unassociated")


def test_orders_without_a_delivered_ref_are_not_listed() -> None:
    """A place the model never got a ref for (not citable), and one denied before dispatch,
    cannot be cited, so neither is returned."""
    out = parsed(decision(DecisionAction.OPEN_CSP, "cand:a"))
    record = assemble_run_record(_live_bundle(), out)
    assert reference_issues(record, frozenset()) == ()
    denied = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(11, "place_option_order", ToolCallStatus.DENIED, t=1),),
        orders=(order(11, intent(11, PUT_A, 1), status=None, broker=False),),
        refs=(cand("cand:a", PUT_A),),
    )
    assert reference_issues(assemble_run_record(denied, out), CITABLE) == ()


def test_an_unclaimed_cancel_is_returned() -> None:
    context = ctx(
        ExecutionMode.LIVE,
        tool_calls=(call(17, "cancel_option_order", t=1),),
        refs=(cref("call:17", 17),),
    )
    (issue,) = reference_issues(
        assemble_run_record(context, parsed()), frozenset({f"order_call:{uid(17)}"})
    )
    assert issue.startswith(f"cancel call order_call:{uid(17)} is associated with no decision")


def test_findings_a_reference_cannot_fix_are_not_returned() -> None:
    prior = AssemblyFinding(code="execution_ref_unverifiable", detail="identity unknown")
    record = assemble_run_record(ctx(prior_findings=(prior,)), parsed())
    assert reference_issues(record, CITABLE) == ()
