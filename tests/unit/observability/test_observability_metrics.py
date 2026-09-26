"""Per-run metrics accumulator (CLAUDE.md §16)."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import AttemptStatus, DecisionAction
from wheelta_robinhood_agent.observability.metrics import MetricsError, RunMetrics

T0 = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)


def test_snapshot_accumulates_everything_and_is_json_serializable() -> None:
    m = RunMetrics("run-1")
    m.stage_started("preflight", T0)
    m.stage_finished("preflight", T0 + timedelta(milliseconds=1500))
    m.stage_started("preflight", T0 + timedelta(seconds=10))
    m.stage_finished("preflight", T0 + timedelta(seconds=10, milliseconds=500))
    m.stage_started("agent", T0 + timedelta(seconds=11))
    m.tool_call("robinhood")
    m.tool_call("robinhood")
    m.tool_call("wheelta")
    m.error("SchemaViolation")
    m.decision(DecisionAction.HOLD)
    m.order(AttemptStatus.NOT_PLACED)
    m.llm_usage(input_tokens=100, output_tokens=20, cost_usd=Decimal("0.0125"))
    m.llm_usage(input_tokens=50, output_tokens=5, cost_usd=Decimal("0.0100"))
    m.staleness_event(source="wheelta", field="board", as_of=T0, observed_at=T0)

    snap = m.snapshot()
    assert snap.stages[0].stage == "preflight"
    assert snap.stages[0].duration_ms == 2000
    assert snap.stages[0].completed_count == 2
    assert snap.unfinished_stages == ("agent",)
    assert snap.tool_calls_by_server == {"robinhood": 2, "wheelta": 1}
    assert snap.errors_by_type == {"SchemaViolation": 1}
    assert snap.decisions_by_action == {DecisionAction.HOLD: 1}
    assert snap.orders_by_status == {AttemptStatus.NOT_PLACED: 1}
    assert (snap.llm_input_tokens, snap.llm_output_tokens) == (150, 25)
    assert snap.llm_cost_usd == Decimal("0.0225")
    data = json.loads(snap.model_dump_json())
    assert data["llm_cost_usd"] == "0.0225"
    assert data["decisions_by_action"] == {"HOLD": 1}


def test_snapshot_is_frozen_and_independent() -> None:
    m = RunMetrics("r")
    m.tool_call("wheelta")
    snap = m.snapshot()
    with pytest.raises(ValidationError):
        snap.run_id = "other"  # type: ignore[misc]
    m.tool_call("wheelta")
    assert snap.tool_calls_by_server == {"wheelta": 1}


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.stage_started("s", datetime(2026, 1, 1)),  # noqa: DTZ001 - naive on purpose
        lambda m: m.stage_finished("never", T0),
        lambda m: m.llm_usage(input_tokens=-1, output_tokens=0, cost_usd=Decimal(0)),
        lambda m: m.llm_usage(input_tokens=0, output_tokens=0, cost_usd=0.1),
        lambda m: m.llm_usage(input_tokens=0, output_tokens=0, cost_usd=Decimal("NaN")),
        lambda m: m.staleness_event(
            source="s",
            field="f",
            as_of=datetime(2026, 1, 1),  # noqa: DTZ001 - naive on purpose
            observed_at=T0,
        ),
    ],
)
def test_inconsistent_calls_rejected(call: object) -> None:
    with pytest.raises(MetricsError):
        call(RunMetrics("r"))  # type: ignore[operator]


def test_stage_errors() -> None:
    m = RunMetrics("r")
    m.stage_started("s", T0)
    with pytest.raises(MetricsError):
        m.stage_started("s", T0)
    with pytest.raises(MetricsError):
        m.stage_finished("s", T0 - timedelta(seconds=1))
