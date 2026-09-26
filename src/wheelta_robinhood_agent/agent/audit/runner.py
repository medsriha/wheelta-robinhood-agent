"""Run all V1–V7 checks and derive the repeat-unverifiable alert (VALIDATION.md, ADR-0014).

`run_audit` isolates an exception in one check: the other checks still run, the failed
check is listed, and the result's status is `failed` (the caller alerts and exits non-zero).
The audit is advisory: nothing here blocks, cancels, retries, or changes an order.
"""

from collections.abc import Callable, Sequence
from uuid import UUID

from wheelta_robinhood_agent.agent.audit.context import AuditContext
from wheelta_robinhood_agent.agent.audit.v1_strategy import check_v1
from wheelta_robinhood_agent.agent.audit.v2_provenance import check_v2
from wheelta_robinhood_agent.agent.audit.v3_execution import check_v3
from wheelta_robinhood_agent.agent.audit.v4_invariants import check_v4
from wheelta_robinhood_agent.agent.audit.v5_rules_set import check_v5
from wheelta_robinhood_agent.agent.audit.v6_sequence import check_v6
from wheelta_robinhood_agent.agent.audit.v7_caps import check_v7
from wheelta_robinhood_agent.domain.base import DomainModel, NonEmptyStr
from wheelta_robinhood_agent.domain.enums import AuditCheck, AuditOutcome
from wheelta_robinhood_agent.domain.run import AuditFinding, AuditStatus

Check = Callable[[AuditContext], tuple[AuditFinding, ...]]

CHECKS: tuple[tuple[AuditCheck, Check], ...] = (
    (AuditCheck.V1, check_v1),
    (AuditCheck.V2, check_v2),
    (AuditCheck.V3, check_v3),
    (AuditCheck.V4, check_v4),
    (AuditCheck.V5, check_v5),
    (AuditCheck.V6, check_v6),
    (AuditCheck.V7, check_v7),
)


class CheckError(DomainModel):
    """An exception raised inside one check; recorded, never suppressing other checks."""

    check_id: AuditCheck
    error_type: NonEmptyStr
    message: str


class AuditResult(DomainModel):
    """All findings of one audit run plus its status, version, and context hash."""

    run_id: UUID
    audit_version: NonEmptyStr
    context_hash: NonEmptyStr
    status: AuditStatus
    findings: tuple[AuditFinding, ...]
    errors: tuple[CheckError, ...] = ()

    @property
    def violations(self) -> tuple[AuditFinding, ...]:
        return tuple(f for f in self.findings if f.outcome is AuditOutcome.VIOLATION)

    @property
    def unverifiable_checks(self) -> tuple[AuditCheck, ...]:
        return unverifiable_checks(self.findings)


def run_audit(
    ctx: AuditContext, checks: Sequence[tuple[AuditCheck, Check]] = CHECKS
) -> AuditResult:
    """Run every check; an exception in one check marks the audit failed, not passed."""
    findings: list[AuditFinding] = []
    errors: list[CheckError] = []
    for check_id, check in checks:
        try:
            findings.extend(check(ctx))
        except Exception as exc:  # noqa: BLE001 - isolate one check's failure (VALIDATION.md)
            errors.append(
                CheckError(check_id=check_id, error_type=type(exc).__name__, message=str(exc))
            )
    return AuditResult(
        run_id=ctx.run_id,
        audit_version=ctx.audit_version,
        context_hash=ctx.context_hash,
        status=AuditStatus.FAILED if errors else AuditStatus.COMPLETED,
        findings=tuple(findings),
        errors=tuple(errors),
    )


def unverifiable_checks(findings: Sequence[AuditFinding]) -> tuple[AuditCheck, ...]:
    """Checks with at least one unverifiable finding, in V1..V7 order."""
    hit = {f.check_id for f in findings if f.outcome is AuditOutcome.UNVERIFIABLE}
    return tuple(c for c in AuditCheck if c in hit)


def repeat_unverifiable_alerts(
    history: Sequence[Sequence[AuditFinding]],
    current: Sequence[AuditFinding],
    threshold: int,
) -> tuple[AuditCheck, ...]:
    """Checks unverifiable in `threshold` consecutive completed runs ending with `current`.

    `history` holds earlier completed runs' findings, oldest first; the caller passes only
    completed runs (ADR-0014 item 5, `circuit_breakers.max_consecutive_unverifiable_runs`).
    """
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise ValueError("threshold must be a positive integer")
    runs = [*history, current]
    if len(runs) < threshold:
        return ()
    window = [set(unverifiable_checks(run)) for run in runs[-threshold:]]
    return tuple(c for c in AuditCheck if all(c in run for run in window))
