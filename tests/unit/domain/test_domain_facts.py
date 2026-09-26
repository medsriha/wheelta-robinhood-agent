from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.base import Dec
from wheelta_robinhood_agent.domain.enums import DataQuality
from wheelta_robinhood_agent.domain.evidence import Derivation, Gap, SourcedValue
from wheelta_robinhood_agent.domain.facts import (
    DecisionFacts,
    DerivedMetric,
    FactsPurpose,
    FormulaVersion,
)

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)


def _metric(name: str = "collateral_usd", formula: str = "csp_collateral") -> DerivedMetric:
    return DerivedMetric(
        name=name,
        unit="USD",
        value=SourcedValue[Dec](
            evidence_id=uuid4(),
            value=Decimal("19000.00"),
            source="derived",
            as_of=T0,
            retrieved_at=T0,
            tool_call_id=None,
            quality=DataQuality.OK,
            derivation=Derivation(
                formula=formula, formula_version="1", input_evidence_ids=(uuid4(),)
            ),
        ),
    )


def _facts(**kw: object) -> DecisionFacts:
    base: dict[str, object] = {
        "facts_id": uuid4(),
        "facts_ref": "facts:1",
        "run_id": uuid4(),
        "subject_ref": "candidate:1",
        "purpose": FactsPurpose.OPEN,
        "observed_at": T0,
        "rules_version": "5",
        "rules_hash": "sha256:abc",
        "input_evidence_ids": (uuid4(),),
        "snapshot_ref": uuid4(),
        "formula_versions": (FormulaVersion(formula="csp_collateral", version="1"),),
        "metrics": (_metric(),),
        "initial_quantity": 2,
        "remaining_quantity": 2,
        "quality": DataQuality.OK,
    }
    base.update(kw)
    return DecisionFacts.model_validate(base)


def test_valid_and_lookup() -> None:
    f = _facts()
    assert f.metric("collateral_usd") is not None
    assert f.metric("nope") is None


def test_zero_capacity_is_real_zero() -> None:
    assert _facts(initial_quantity=0, remaining_quantity=0).initial_quantity == 0


def test_unavailable_quantity_needs_gap() -> None:
    gaps = (
        Gap(field="initial_quantity", kind=DataQuality.MISSING, detail="multiplier unverified"),
        Gap(field="remaining_quantity", kind=DataQuality.MISSING, detail="depends on initial"),
    )
    f = _facts(
        initial_quantity=None, remaining_quantity=None, gaps=gaps, quality=DataQuality.MISSING
    )
    assert f.initial_quantity is None
    with pytest.raises(ValidationError, match="no gap names it"):
        _facts(initial_quantity=None, quality=DataQuality.MISSING)
    with pytest.raises(ValidationError, match="quality 'ok'"):
        _facts(initial_quantity=None, remaining_quantity=None, gaps=gaps)


def test_close_quantity_only_for_close() -> None:
    _facts(purpose=FactsPurpose.CLOSE, subject_ref="position:1", close_quantity=1)
    with pytest.raises(ValidationError, match="CLOSE"):
        _facts(close_quantity=1)
    with pytest.raises(ValidationError):
        _facts(purpose=FactsPurpose.CLOSE, close_quantity=0)


def test_remaining_not_above_initial() -> None:
    with pytest.raises(ValidationError, match="exceed"):
        _facts(initial_quantity=1, remaining_quantity=2)


def test_metric_integrity() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        _facts(metrics=(_metric(), _metric()))
    with pytest.raises(ValidationError, match="formula_versions"):
        _facts(metrics=(_metric(name="x", formula="unknown"),))
    with pytest.raises(ValidationError, match="input evidence"):
        _facts(input_evidence_ids=())
    with pytest.raises(ValidationError, match="derivation"):
        DerivedMetric(
            name="bid",
            unit="USD",
            value=SourcedValue[Dec](
                evidence_id=uuid4(),
                value=Decimal("1"),
                source="robinhood",
                as_of=T0,
                retrieved_at=T0,
                tool_call_id=uuid4(),
                quality=DataQuality.OK,
            ),
        )
    with pytest.raises(ValidationError):
        _facts(limit_price=1.25)
