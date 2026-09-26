from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.account import (
    AccountSnapshot,
    ReservationEvidence,
    ReservationKind,
)
from wheelta_robinhood_agent.domain.enums import DataQuality
from wheelta_robinhood_agent.domain.evidence import Gap

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)


def _snap(**kw: object) -> AccountSnapshot:
    base: dict[str, object] = {
        "snapshot_id": uuid4(),
        "as_of": T0,
        "retrieved_at": T0,
        "tool_call_ids": (uuid4(),),
        "account_ref": "****1234",
        "agentic_verified": True,
        "account_value_usd": Decimal("10000.00"),
        "available_settled_cash_usd": Decimal("4000.00"),
        "csp_reserved_cash_usd": Decimal("5000.00"),
        "csp_cash_base_usd": Decimal("9000.00"),
        "csp_cash_base_evidence_ids": (uuid4(), uuid4()),
        "positions_ref": uuid4(),
        "open_orders_ref": uuid4(),
        "tax_lots_ref": None,
        "quality": DataQuality.OK,
    }
    base.update(kw)
    return AccountSnapshot.model_validate(base)


def _gap(field: str) -> Gap:
    return Gap(field=field, kind=DataQuality.MISSING, detail="broker semantics unverified")


def test_valid_snapshot() -> None:
    s = _snap()
    assert s.csp_cash_base_usd == Decimal("9000.00")


def test_cash_base_must_equal_sum() -> None:
    with pytest.raises(ValidationError, match="csp_cash_base_usd"):
        _snap(csp_cash_base_usd=Decimal("9000.01"))


def test_cash_base_none_when_input_missing() -> None:
    gaps = (_gap("csp_reserved_cash_usd"), _gap("csp_cash_base_usd"))
    s = _snap(
        csp_reserved_cash_usd=None,
        csp_cash_base_usd=None,
        csp_cash_base_evidence_ids=(),
        gaps=gaps,
        quality=DataQuality.MISSING,
    )
    assert s.csp_cash_base_usd is None
    with pytest.raises(ValidationError, match="csp_cash_base_usd"):
        _snap(csp_reserved_cash_usd=None, gaps=gaps, quality=DataQuality.MISSING)


def test_none_field_needs_gap() -> None:
    with pytest.raises(ValidationError, match="no gap names it"):
        _snap(account_value_usd=None, quality=DataQuality.MISSING)


def test_ok_quality_forbids_gaps() -> None:
    with pytest.raises(ValidationError, match="quality 'ok'"):
        _snap(account_value_usd=None, gaps=(_gap("account_value_usd"),))


def test_cash_base_needs_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        _snap(csp_cash_base_evidence_ids=())


def test_account_ref_redacted() -> None:
    with pytest.raises(ValidationError, match="redacted"):
        _snap(account_ref="123456789")


def test_negative_cash_and_float_rejected() -> None:
    with pytest.raises(ValidationError):
        _snap(available_settled_cash_usd=Decimal("-1"), csp_cash_base_usd=Decimal("4999"))
    with pytest.raises(ValidationError):
        _snap(account_value_usd=10000.0)


def test_tool_calls_required_and_unique() -> None:
    with pytest.raises(ValidationError):
        _snap(tool_call_ids=())
    t = uuid4()
    with pytest.raises(ValidationError, match="duplicate"):
        _snap(tool_call_ids=(t, t))


def test_reservation_evidence() -> None:
    r = ReservationEvidence(
        kind=ReservationKind.WORKING_ORDER,
        broker_ref="ord-1",
        quantity=1,
        source_tool_call_ids=(uuid4(),),
    )
    assert _snap(reservation_evidence=(r,)).reservation_evidence == (r,)
    with pytest.raises(ValidationError):
        ReservationEvidence(
            kind=ReservationKind.POSITION, broker_ref="p", quantity=1, source_tool_call_ids=()
        )
