"""LedgerToolEventRecorder forwards to ledger/tool_calls.py (no database: functions patched)."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from wheelta_robinhood_agent.agent import recorder as recorder_module
from wheelta_robinhood_agent.agent.recorder import (
    LedgerToolEventRecorder,
    ResultKind,
    ToolEventRecorder,
)
from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier
from wheelta_robinhood_agent.ledger.tool_calls import ToolCallRef

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
RUN_ID = uuid.uuid4()
CALL_ID = uuid.uuid4()
CONN: Any = object()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    seen: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    ledger = recorder_module.ledger_tool_calls

    def fake(name: str, ret: Any) -> Any:
        def inner(*args: Any, **kwargs: Any) -> Any:
            seen.append((name, args, kwargs))
            return ret

        return inner

    ref = ToolCallRef(CALL_ID, RUN_ID, "toolu_1", created=True)
    monkeypatch.setattr(ledger, "record_tool_call_requested", fake("requested", ref))
    monkeypatch.setattr(ledger, "record_tool_call_dispatched", fake("dispatched", None))
    monkeypatch.setattr(ledger, "append_tool_call_outcome", fake("outcome", None))
    monkeypatch.setattr(ledger, "record_tool_call_delivered", fake("delivered", None))
    return seen


def test_forwards_every_event(calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
    written: list[dict[str, Any]] = []
    result_id = uuid.uuid4()

    def writer(conn: Any, **kwargs: Any) -> uuid.UUID:
        assert conn is CONN
        written.append(kwargs)
        return result_id

    rec: ToolEventRecorder = LedgerToolEventRecorder(CONN, run_id=RUN_ID, result_writer=writer)
    tool_call_id = rec.requested(
        sdk_tool_use_id="toolu_1",
        server="wheelta",
        tool="wheelta_board_query",
        tier=ToolTier.R,
        arguments_redacted={"limit": 5},
        requested_at=NOW,
    )
    assert tool_call_id == CALL_ID
    rec.dispatched(CALL_ID, effective_arguments_redacted={"limit": 5}, dispatched_at=NOW)
    ref = rec.store_result(CALL_ID, ResultKind.DELIVERED, {"a": 1})
    rec.outcome(CALL_ID, ToolCallStatus.SUCCEEDED, observed_at=NOW, result_ref=ref)
    rec.delivered(CALL_ID, delivered_result_ref=ref, observed_at=NOW)

    assert [c[0] for c in calls] == ["requested", "dispatched", "outcome", "delivered"]
    requested = calls[0][2]
    assert requested["run_id"] == RUN_ID and requested["stage"] == "agent"
    assert requested["tier"] is ToolTier.R and requested["arguments_redacted"] == {"limit": 5}
    assert calls[0][1] == (CONN,)
    assert calls[1][1] == (CONN, CALL_ID)
    assert calls[2][1] == (CONN, CALL_ID, ToolCallStatus.SUCCEEDED)
    assert calls[2][2]["result_ref"] == result_id
    assert calls[3][2]["delivered_result_ref"] == result_id
    assert written == [
        {"run_id": RUN_ID, "tool_call_id": CALL_ID, "kind": "delivered", "payload": {"a": 1}}
    ]
