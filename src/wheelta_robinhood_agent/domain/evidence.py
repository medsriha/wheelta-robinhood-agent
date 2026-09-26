"""Sourced values, derivations, gaps, and citations (INTERFACES.md "Evidence and citations").

A financial value exists only if a named source returned it, or code derived it with an
approved formula from recorded inputs (DATA_QUALITY.md "Rule zero"). Everything else is
`None` with a recorded gap.
"""

import re
from typing import Generic, Self, TypeVar
from uuid import UUID

from pydantic import field_validator, model_validator

from wheelta_robinhood_agent.domain.base import (
    DomainModel,
    NonEmptyStr,
    UtcDatetime,
    require_unique,
)
from wheelta_robinhood_agent.domain.enums import DataQuality, SourceTier

T = TypeVar("T")

_HTTP_URL = re.compile(r"^https?://[^\s/]+\S*$")


class Gap(DomainModel):
    """A named missing/unusable fact (DATA_QUALITY.md "Gaps are first-class").

    `field` names the missing value; `kind` is its quality label and is never `ok`.
    """

    field: NonEmptyStr
    kind: DataQuality
    detail: NonEmptyStr
    evidence_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def _check_gap(self) -> Self:
        if self.kind is DataQuality.OK:
            raise ValueError("a gap cannot have quality 'ok'")
        return self


class Derivation(DomainModel):
    """How code derived a value: formula name/version and input evidence IDs.

    A derived value is not an observed broker fact (INTERFACES.md "Evidence and citations").
    """

    formula: NonEmptyStr
    formula_version: NonEmptyStr
    input_evidence_ids: tuple[UUID, ...]

    @model_validator(mode="after")
    def _check_derivation(self) -> Self:
        if not self.input_evidence_ids:
            raise ValueError("a derivation needs at least one input evidence id")
        require_unique(self.input_evidence_ids, "input evidence id")
        return self


class SourcedValue(DomainModel, Generic[T]):
    """A typed value (or None) with its provenance and quality label.

    Rules (INTERFACES.md "Evidence and citations"; CLAUDE.md §13):
    - Provenance: an observation carries the `tool_call_id` that returned it; a derived value
      carries a `derivation`. At least one must be present.
    - `quality=ok` requires a value and a known `as_of`; a datum whose freshness cannot be
      evaluated is not ok (DATA_QUALITY.md status note).
    - A `None` value is never `ok`, and any non-ok value names its reason.
    - `source_tier` is the web/source tier where one applies (`data_quality.source_tiers`);
      it is `None` for MCP tool results, which are not tiered.
    """

    evidence_id: UUID
    value: T | None
    source: NonEmptyStr
    source_tier: SourceTier | None = None
    as_of: UtcDatetime | None
    retrieved_at: UtcDatetime
    tool_call_id: UUID | None
    quality: DataQuality
    reason: NonEmptyStr | None = None
    derivation: Derivation | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _no_float(cls, value: object) -> object:
        """Financial values are never binary floats, whatever `T` is (CLAUDE.md §4)."""
        if isinstance(value, float):
            raise ValueError("sourced values must not be float; use Decimal")
        return value

    @model_validator(mode="after")
    def _check_sourced(self) -> Self:
        if self.tool_call_id is None and self.derivation is None:
            raise ValueError("a sourced value needs a tool_call_id or a derivation")
        if self.quality is DataQuality.OK:
            if self.value is None:
                raise ValueError("a missing value cannot have quality 'ok'")
            if self.as_of is None:
                raise ValueError("a value without as_of cannot have quality 'ok'")
            if self.source_tier is SourceTier.TIER_3:
                raise ValueError("a tier-3-only value cannot have quality 'ok'")
        elif self.reason is None:
            raise ValueError("a non-ok value must state its reason")
        if self.derivation is not None and self.evidence_id in self.derivation.input_evidence_ids:
            raise ValueError("a derived value cannot be its own input")
        return self

    @property
    def usable(self) -> bool:
        """Only `ok` values can support a dependent decision (DATA_QUALITY.md pipeline)."""
        return self.quality is DataQuality.OK


class Citation(DomainModel):
    """A web or document citation (CLAUDE.md §11; INTERFACES.md "Evidence and citations").

    Only tier 1–2 citations can support a decision; tier-3-only claims are low confidence.
    """

    citation_id: UUID
    url: NonEmptyStr
    title: NonEmptyStr
    publisher: NonEmptyStr
    published_at: UtcDatetime | None
    retrieved_at: UtcDatetime
    tier: SourceTier
    excerpt: NonEmptyStr
    tool_call_id: UUID

    @model_validator(mode="after")
    def _check_citation(self) -> Self:
        if not _HTTP_URL.fullmatch(self.url):
            raise ValueError("citation url must be an absolute http(s) URL")
        return self

    @property
    def supports_decision(self) -> bool:
        """Tier 1–2 only (CLAUDE.md §11)."""
        return self.tier is not SourceTier.TIER_3
