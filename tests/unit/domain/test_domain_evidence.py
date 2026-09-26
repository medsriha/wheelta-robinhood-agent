from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import DataQuality, SourceTier
from wheelta_robinhood_agent.domain.evidence import Citation, Derivation, Gap, SourcedValue

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)


def _sv(**kw: object) -> SourcedValue[Decimal]:
    base: dict[str, object] = {
        "evidence_id": uuid4(),
        "value": Decimal("1.25"),
        "source": "robinhood:get_option_quotes",
        "as_of": T0,
        "retrieved_at": T0,
        "tool_call_id": uuid4(),
        "quality": DataQuality.OK,
    }
    base.update(kw)
    return SourcedValue[Decimal].model_validate(base)


def test_ok_value() -> None:
    sv = _sv()
    assert sv.usable
    assert sv.value == Decimal("1.25")


def test_utc_normalization_and_naive_rejected() -> None:
    est = timezone(timedelta(hours=-4))
    sv = _sv(as_of=datetime(2026, 9, 25, 11, 0, tzinfo=est))
    assert sv.as_of == T0
    assert sv.as_of is not None and sv.as_of.tzinfo is UTC
    with pytest.raises(ValidationError):
        _sv(as_of=datetime(2026, 9, 25, 11, 0))  # noqa: DTZ001


def test_float_rejected() -> None:
    with pytest.raises(ValidationError):
        _sv(value=1.25)


def test_frozen_and_extra_forbidden() -> None:
    sv = _sv()
    with pytest.raises(ValidationError):
        sv.quality = DataQuality.STALE  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _sv(extra_field=1)


def test_missing_value_never_ok() -> None:
    with pytest.raises(ValidationError, match="missing value"):
        _sv(value=None)
    sv = _sv(value=None, quality=DataQuality.MISSING, reason="no quote returned")
    assert not sv.usable


def test_unknown_as_of_never_ok() -> None:
    with pytest.raises(ValidationError, match="as_of"):
        _sv(as_of=None)


def test_tier3_never_ok() -> None:
    with pytest.raises(ValidationError, match="tier-3"):
        _sv(source_tier=SourceTier.TIER_3)


def test_non_ok_requires_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        _sv(quality=DataQuality.STALE)


def test_provenance_required() -> None:
    with pytest.raises(ValidationError, match="tool_call_id or a derivation"):
        _sv(tool_call_id=None)


def test_derived_value() -> None:
    inp = uuid4()
    sv = _sv(
        tool_call_id=None,
        source="derived",
        derivation=Derivation(formula="dte", formula_version="1", input_evidence_ids=(inp,)),
    )
    assert sv.derivation is not None
    with pytest.raises(ValidationError, match="own input"):
        _sv(
            evidence_id=inp,
            derivation=Derivation(formula="x", formula_version="1", input_evidence_ids=(inp,)),
        )


def test_derivation_needs_unique_inputs() -> None:
    with pytest.raises(ValidationError):
        Derivation(formula="x", formula_version="1", input_evidence_ids=())
    i = uuid4()
    with pytest.raises(ValidationError, match="duplicate"):
        Derivation(formula="x", formula_version="1", input_evidence_ids=(i, i))


def test_gap_not_ok() -> None:
    Gap(field="bid", kind=DataQuality.MISSING, detail="absent")
    with pytest.raises(ValidationError):
        Gap(field="bid", kind=DataQuality.OK, detail="absent")


def _citation(**kw: object) -> Citation:
    base: dict[str, object] = {
        "citation_id": uuid4(),
        "url": "https://www.sec.gov/x",
        "title": "10-Q",
        "publisher": "SEC",
        "published_at": None,
        "retrieved_at": T0,
        "tier": SourceTier.TIER_1,
        "excerpt": "Revenue grew.",
        "tool_call_id": uuid4(),
    }
    base.update(kw)
    return Citation.model_validate(base)


def test_citation() -> None:
    assert _citation().supports_decision
    assert not _citation(tier=SourceTier.TIER_3).supports_decision
    with pytest.raises(ValidationError, match="http"):
        _citation(url="ftp://x")
    with pytest.raises(ValidationError):
        _citation(excerpt="")
