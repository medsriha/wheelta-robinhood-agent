"""DecisionFacts: code-computed facts delivered before a decision (OUTPUT_ASSEMBLY.md).

Model only. `compute_decision_facts` is a separate, later task. A fact set is immutable and
does not become fresh when referenced again; unknown mappings/definitions produce None with
a typed gap, and zero capacity is a real zero. Quantities are never silently rounded to one.
"""

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import model_validator

from wheelta_robinhood_agent.domain.base import (
    Count,
    Dec,
    DomainModel,
    NonEmptyStr,
    PosCount,
    PosDec,
    Ref,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import DataQuality
from wheelta_robinhood_agent.domain.evidence import Gap, SourcedValue


class FactsPurpose(StrEnum):
    """`get_decision_facts` purpose (OUTPUT_ASSEMBLY.md "Facts available before the decision")."""

    OPEN = "open"
    CLOSE = "close"
    MANAGE = "manage"


class FormulaVersion(DomainModel):
    formula: NonEmptyStr
    version: NonEmptyStr


class DerivedMetric(DomainModel):
    """A named derived metric with explicit unit; its value carries the derivation."""

    name: NonEmptyStr
    unit: NonEmptyStr
    value: SourcedValue[Dec]

    @model_validator(mode="after")
    def _check_metric(self) -> Self:
        if self.value.derivation is None:
            raise ValueError("a derived metric must record its derivation")
        return self


_NULLABLE_QUANTITIES = ("initial_quantity", "remaining_quantity")


class DecisionFacts(DomainModel):
    """Facts for one subject and purpose (INTERFACES.md DecisionFacts; OUTPUT_ASSEMBLY.md).

    Rules:
    - Discretionary inputs: `limit_price` (any purpose) and `close_quantity` (CLOSE only,
      ADR-0015; at least one). No model-supplied balances or formulas.
    - `initial_quantity` is the rule-derived contract quantity; `remaining_quantity` accounts
      for confirmed fills and never exceeds it. None requires a gap naming the field.
    - Metric names and formula names are unique; every metric formula is listed in
      `formula_versions`. `quality=ok` requires no gaps.
    """

    facts_id: UUID
    facts_ref: Ref
    run_id: UUID
    subject_ref: Ref
    purpose: FactsPurpose
    observed_at: UtcDatetime
    rules_version: NonEmptyStr
    rules_hash: NonEmptyStr
    input_evidence_ids: tuple[UUID, ...]
    snapshot_ref: UUID | None
    quote_refs: tuple[UUID, ...] = ()
    candidate_ref: Ref | None = None
    position_id: UUID | None = None
    limit_price: PosDec | None = None
    close_quantity: PosCount | None = None
    formula_versions: tuple[FormulaVersion, ...] = ()
    metrics: tuple[DerivedMetric, ...] = ()
    initial_quantity: Count | None
    remaining_quantity: Count | None
    quality: DataQuality
    gaps: tuple[Gap, ...] = ()

    @model_validator(mode="after")
    def _check_facts(self) -> Self:
        if not self.input_evidence_ids:
            raise ValueError("decision facts need input evidence ids")
        require_unique(self.input_evidence_ids, "input evidence id")
        require_unique(tuple(m.name for m in self.metrics), "metric name")
        require_unique(tuple(f.formula for f in self.formula_versions), "formula")
        if self.close_quantity is not None and self.purpose is not FactsPurpose.CLOSE:
            raise ValueError("close_quantity is a CLOSE-purpose input only")
        versions = {(f.formula, f.version) for f in self.formula_versions}
        for metric in self.metrics:
            derivation = metric.value.derivation
            if derivation is not None and (
                (derivation.formula, derivation.formula_version) not in versions
            ):
                raise ValueError(f"metric {metric.name} formula is not in formula_versions")
        if (
            self.initial_quantity is not None
            and self.remaining_quantity is not None
            and self.remaining_quantity > self.initial_quantity
        ):
            raise ValueError("remaining_quantity cannot exceed initial_quantity")
        gap_fields = {g.field for g in self.gaps}
        for name in _NULLABLE_QUANTITIES:
            if getattr(self, name) is None and name not in gap_fields:
                raise ValueError(f"{name} is None but no gap names it")
        if self.quality is DataQuality.OK and self.gaps:
            raise ValueError("facts with gaps cannot have quality 'ok'")
        return self

    def metric(self, name: str) -> DerivedMetric | None:
        """The metric with `name`, or None."""
        return next((m for m in self.metrics if m.name == name), None)
