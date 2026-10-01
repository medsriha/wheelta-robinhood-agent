"""Reference feedback for the agent before its output is accepted (ADR-0052).

`reference_issues(record, citable)` is pure. The session assembles the agent's parsed output
with the real assembler (`assemble_run_record`) while the session is still open and turns
the result into issues the agent can fix by changing only its references:

- each assembly finding whose code is in `REFERENCE_FINDING_CODES` (a selected ref that does
  not resolve, is of the wrong kind, or does not fit its decision);
- each dispatched place and each cancel that no decision or cancellation rationale claims,
  when its `order_call:` ref was delivered to the model (`citable`). A place denied before
  dispatch reached no venue and has no ref, so it is not listed. ADR-0066: a place or cancel
  an order-work job made (`work_of`: call ID -> its job's `order_work:` ref, when delivered)
  asks for that job's ref instead, since its own call was never delivered.

Nothing here decides a trade, repairs a choice, or attaches an action: the agent restates
its output, and the post-run assembly applies the same rules to whatever it returns.
"""

from collections.abc import Mapping
from typing import Final
from uuid import UUID

from wheelta_robinhood_agent.domain.assembly_context import order_call_ref_for
from wheelta_robinhood_agent.domain.orders import ReasonCode
from wheelta_robinhood_agent.domain.run_record import (
    AssemblyFinding,
    RunRecord,
    UnassociatedActionKind,
)

# Assembly findings the agent can resolve by selecting other references, with tools disabled.
# Findings about data, quantities, or ranking are not listed, nor findings that need a new
# decision-facts call (facts_not_recorded, facts_price_mismatch, proposal_shape_incomplete).
REFERENCE_FINDING_CODES: Final = frozenset(
    {
        "unknown_reference",
        "wrong_reference_kind",
        "wrong_run_reference",
        "wrong_account_reference",
        "undelivered_reference",
        "incompatible_reference",
        "missing_replacement",
        "incompatible_proposal",
        "duplicate_proposal",
        "proposal_order",
        "mismatched_contract",
        "duplicate_execution_association",
        "incompatible_execution_ref",
        "contradictory_cancel_association",
        "invalid_funding_dependency",
        "execution_refs_in_off_mode",
        "cancellation_rationale_in_off_mode",
    }
)
LIKELY_OWNER_CODE: Final = "unassociated_place_matches_decision"


def _where(finding: AssemblyFinding, positions: dict[str, int]) -> str:
    if finding.decision_ref is not None and finding.decision_ref in positions:
        return f"decisions[{positions[finding.decision_ref]}]"
    return "output"


def reference_issues(
    record: RunRecord,
    citable: frozenset[str],
    work_of: Mapping[UUID, str] | None = None,
) -> tuple[str, ...]:
    """Issues to return to the agent, in a stable order (module docstring). Empty when every
    selected reference resolves and every order action it can cite is claimed."""
    work_of = work_of or {}
    jobs: list[str] = []
    positions = {d.decision_ref: i for i, d in enumerate(record.decisions)}
    issues = [
        f"{_where(f, positions)}: {f.detail} [{f.code}]"
        for f in record.findings
        if f.code in REFERENCE_FINDING_CODES
    ]
    owner: dict[UUID, str] = {
        call_id: _where(f, positions)
        for f in record.findings
        if f.code == LIKELY_OWNER_CODE
        for call_id in f.tool_call_ids
    }
    for action in record.unassociated_actions:
        if action.kind is UnassociatedActionKind.PLACE and action.attempt is not None:
            call_id = action.attempt.place_tool_call_id
            if call_id is None or ReasonCode.NOT_DISPATCHED in action.attempt.reason_codes:
                continue
            if call_id in work_of:
                jobs.append(work_of[call_id])
                continue
            ref = order_call_ref_for(call_id)
            if ref not in citable:
                continue
            what = " ".join(str(v) for v in (action.side_raw, action.occ_symbol) if v is not None)
            hint = (
                f"it matches {owner[call_id]}"
                if call_id in owner
                else "if none of your decisions carried it out, leave it: code records it as "
                "unassociated"
            )
            issues.append(
                f"place call {ref} ({what or 'contract unknown'}) is associated with no "
                f"decision: add it, with its review's order_call ref, to the execution_refs "
                f"of the decision it executed; {hint}"
            )
        elif action.kind is UnassociatedActionKind.CANCEL and action.cancellation is not None:
            if action.cancellation.cancel_tool_call_id in work_of:
                jobs.append(work_of[action.cancellation.cancel_tool_call_id])
                continue
            ref = order_call_ref_for(action.cancellation.cancel_tool_call_id)
            if ref not in citable:
                continue
            issues.append(
                f"cancel call {ref} is associated with no decision: add it to the "
                "execution_refs of its decision, or explain it in cancellation_rationales"
            )
    issues.extend(
        f"order work {ref} is associated with no decision: add {ref} to the execution_refs of "
        "the decision it executed"
        for ref in dict.fromkeys(jobs)
    )
    return tuple(issues)


__all__ = ["LIKELY_OWNER_CODE", "REFERENCE_FINDING_CODES", "reference_issues"]
