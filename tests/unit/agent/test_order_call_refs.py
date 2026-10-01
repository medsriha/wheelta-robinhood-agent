"""Order-call refs on the loading and session side (ADR-0052)."""

import uuid
from types import SimpleNamespace
from typing import Any

from wheelta_robinhood_agent.agent.run_loader import _order_call_ref
from wheelta_robinhood_agent.agent.session import (
    MAX_REFERENCE_REPAIRS,
    SessionResult,
    SessionStatus,
    _reference_issues,
    reference_message,
)
from wheelta_robinhood_agent.domain.assembly_context import RefKind
from wheelta_robinhood_agent.domain.decision_output import (
    AgentDecisionOutput,
    DecisionOutputParsed,
)

RUN = uuid.UUID(int=1)
CALL = uuid.UUID(int=11)
SCOPE = "acct-scope"


def _envelope(tool: str = "place_option_order", **kw: Any) -> dict[str, Any]:
    return {
        "tool_call_id": str(CALL),
        "server": "robinhood",
        "tool": tool,
        "kind": "error",
        "order_call_ref": f"order_call:{CALL}",
        **kw,
    }


def test_a_delivered_order_call_ref_is_registered_for_its_call() -> None:
    ref = _order_call_ref(_envelope(), RUN, SCOPE)
    assert ref is not None
    assert (ref.ref, ref.kind, ref.tool_call_id) == (f"order_call:{CALL}", RefKind.TOOL_CALL, CALL)
    assert (ref.run_id, ref.account_scope_id, ref.delivered) == (RUN, SCOPE, True)


def test_only_an_order_tool_envelope_naming_its_own_call_registers_a_ref() -> None:
    other = f"order_call:{uuid.UUID(int=12)}"
    assert _order_call_ref(_envelope(order_call_ref=other), RUN, SCOPE) is None
    assert _order_call_ref(_envelope(tool="get_option_quotes"), RUN, SCOPE) is None
    assert _order_call_ref(_envelope(tool_call_id="not-a-uuid"), RUN, SCOPE) is None
    no_ref = {k: v for k, v in _envelope().items() if k != "order_call_ref"}
    assert _order_call_ref(no_ref, RUN, SCOPE) is None


def _parsed() -> DecisionOutputParsed:
    output = AgentDecisionOutput(
        decisions=(), cancellation_rationales=(), unresolved_questions=(), next_run=None
    )
    return DecisionOutputParsed(ok=True, output=output)


def test_a_failing_reference_check_is_recorded_and_accepts_the_output() -> None:
    def fail(_: DecisionOutputParsed) -> tuple[str, ...]:
        raise RuntimeError("ledger unavailable")

    result = SessionResult(status=SessionStatus.COMPLETED)
    deps: Any = SimpleNamespace(reference_check=fail)
    assert _reference_issues(deps, _parsed(), result) == ()
    assert result.reference_check_error == "RuntimeError"
    # A failed check is not retried within the session.
    deps.reference_check = lambda _: ("x",)
    assert _reference_issues(deps, _parsed(), result) == ()


def test_reference_issues_are_recorded_and_the_message_lists_them() -> None:
    result = SessionResult(status=SessionStatus.COMPLETED)
    deps: Any = SimpleNamespace(reference_check=lambda _: ("decisions[0]: bad ref",))
    assert _reference_issues(deps, _parsed(), result) == ("decisions[0]: bad ref",)
    assert result.reference_issues == ("decisions[0]: bad ref",)
    message = reference_message(["decisions[0]: bad ref"], 1)
    assert f"reference check 1 of {MAX_REFERENCE_REPAIRS}" in message
    assert "- decisions[0]: bad ref" in message and "never its bare tool_call_id" in message
