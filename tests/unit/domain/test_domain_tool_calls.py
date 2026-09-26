from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.tool_calls import (
    ToolCallDecision,
    ToolCallIdentity,
    ToolCallRecord,
)

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)


def _ident() -> ToolCallIdentity:
    return ToolCallIdentity(
        tool_call_id=uuid4(),
        sdk_tool_use_id="toolu_1",
        run_id=uuid4(),
        stage="agent",
        server="robinhood",
        tool="place_option_order",
        tier=ToolTier.X,
        requested_at=T0,
        arguments_redacted={"quantity": 1, "limit_price": "1.25", "legs": [{"a": None}]},
    )


def _rec(**kw: object) -> ToolCallRecord:
    base: dict[str, object] = {
        "identity": _ident(),
        "effective_arguments_redacted": None,
        "decision": ToolCallDecision.ALLOWED,
        "status": ToolCallStatus.SUCCEEDED,
        "dispatched_at": T1,
        "completed_at": T2,
        "result_ref": uuid4(),
    }
    base.update(kw)
    return ToolCallRecord.model_validate(base)


def test_succeeded() -> None:
    assert not _rec().outcome_unresolved


def test_denied() -> None:
    r = _rec(
        decision=ToolCallDecision.DENIED,
        deny_reason="not allowlisted",
        status=ToolCallStatus.DENIED,
        dispatched_at=None,
        completed_at=None,
        result_ref=None,
    )
    assert not r.outcome_unresolved
    with pytest.raises(ValidationError, match="iff"):
        _rec(decision=ToolCallDecision.DENIED, deny_reason="x")
    with pytest.raises(ValidationError, match="never dispatched"):
        _rec(decision=ToolCallDecision.DENIED, deny_reason="x", status=ToolCallStatus.DENIED)
    with pytest.raises(ValidationError, match="never dispatched"):
        _rec(
            decision=ToolCallDecision.DENIED,
            status=ToolCallStatus.DENIED,
            dispatched_at=None,
            completed_at=None,
        )


def test_unresolved_financial_action() -> None:
    r = _rec(status=ToolCallStatus.UNKNOWN, completed_at=None, result_ref=None)
    assert r.outcome_unresolved
    assert _rec(
        status=ToolCallStatus.REQUESTED, completed_at=None, result_ref=None
    ).outcome_unresolved
    with pytest.raises(ValidationError, match="requires dispatch and completion"):
        _rec(status=ToolCallStatus.FAILED, completed_at=None)


def test_time_ordering() -> None:
    with pytest.raises(ValidationError, match="before requested"):
        _rec(dispatched_at=T0 - timedelta(seconds=1))
    with pytest.raises(ValidationError, match="completed before"):
        _rec(completed_at=T0)
    with pytest.raises(ValidationError, match="without dispatch"):
        _rec(status=ToolCallStatus.REQUESTED, dispatched_at=None, decision=None)
    with pytest.raises(ValidationError, match="only an allowed"):
        _rec(decision=None, status=ToolCallStatus.REQUESTED, completed_at=None)


def test_succeeded_needs_result() -> None:
    with pytest.raises(ValidationError, match="result_ref"):
        _rec(result_ref=None)


def test_reconciled_outcome() -> None:
    read = uuid4()
    rec = _rec(
        status=ToolCallStatus.FAILED,
        completed_at=None,
        result_ref=None,
        reconciled_by_tool_call_ids=(read,),
    )
    assert rec.reconciled_by_tool_call_ids == (read,)
    _rec(completed_at=None, result_ref=None, reconciled_by_tool_call_ids=(read,))
    with pytest.raises(ValidationError, match="requires dispatch and completion"):
        _rec(status=ToolCallStatus.FAILED, completed_at=None, result_ref=None)
    with pytest.raises(ValidationError, match="duplicate"):
        _rec(completed_at=None, reconciled_by_tool_call_ids=(read, read))
    with pytest.raises(ValidationError, match="only a dispatched"):
        _rec(
            decision=None,
            status=ToolCallStatus.UNKNOWN,
            dispatched_at=None,
            completed_at=None,
            reconciled_by_tool_call_ids=(read,),
        )
