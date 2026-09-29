"""Read the research already delivered during a run for its informational email."""

from uuid import UUID

from wheelta_robinhood_agent.agent.result_boundary import mapped_evidence_of
from wheelta_robinhood_agent.agent.run_loader import Conn, _delivered_envelopes
from wheelta_robinhood_agent.domain.mignon_report import MignonReport
from wheelta_robinhood_agent.ledger import evidence
from wheelta_robinhood_agent.observability.run_summary import ConsideredOption


def load_summary_research(
    conn: Conn, run_id: UUID
) -> tuple[tuple[ConsideredOption, ...], tuple[MignonReport, ...]]:
    """Include delivered candidates and accepted reports, never raw or rejected model text.

    Encountering a candidate is not proof that the agent evaluated or rejected it. Keep its
    recorded fact gaps separate from the agent's rationale; the renderer makes that clear.
    """
    candidates: dict[str, ConsideredOption] = {}
    reports: list[MignonReport] = []
    gaps: dict[str, list[str]] = {}
    for stored in evidence.effective(evidence.decision_facts_for_run(conn, run_id)):
        facts = stored.facts
        gaps.setdefault(facts.subject_ref, []).extend(f"{g.field}: {g.detail}" for g in facts.gaps)
    for envelope in _delivered_envelopes(conn, run_id):
        mapped = mapped_evidence_of(envelope)
        if mapped is not None:
            for candidate in mapped.candidates:
                candidates[candidate.candidate_ref] = ConsideredOption(
                    candidate_ref=candidate.candidate_ref,
                    underlying=candidate.underlying,
                    occ_symbol=str(candidate.occ_symbol),
                    gaps=tuple(dict.fromkeys(gaps.get(candidate.candidate_ref, []))),
                )
        data = envelope.get("data")
        if (
            envelope.get("kind") == "validated"
            and envelope.get("tool") == "Agent"
            and isinstance(data, dict)
            and isinstance(data.get("report"), dict)
        ):
            reports.append(MignonReport.model_validate(data["report"]))
    return tuple(candidates.values()), tuple(reports)
