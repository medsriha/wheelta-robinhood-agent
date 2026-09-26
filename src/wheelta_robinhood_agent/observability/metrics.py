"""Per-run metrics accumulator (CLAUDE.md §16).

An explicitly passed object, one per run: stage durations, tool calls per server, errors by
type, decisions by action, orders by status, LLM tokens and cost, data-staleness events.
Pure: every time is passed in; nothing reads the clock. ``snapshot()`` returns a frozen,
JSON-serializable model for logging and the ledger.
"""

from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from wheelta_robinhood_agent.domain.enums import AttemptStatus, DecisionAction

_ONE_MS = timedelta(milliseconds=1)


class MetricsError(ValueError):
    """A metrics call was inconsistent (naive time, stage finished before it started, ...)."""


class StageTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    duration_ms: int = Field(ge=0)
    completed_count: int = Field(ge=0)


class StalenessEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    field: str
    as_of: datetime | None
    observed_at: datetime


class MetricsSnapshot(BaseModel):
    """Frozen copy of a run's metrics. Mappings are copies; mutating them changes nothing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    stages: tuple[StageTiming, ...]
    unfinished_stages: tuple[str, ...]
    tool_calls_by_server: dict[str, int]
    errors_by_type: dict[str, int]
    decisions_by_action: dict[DecisionAction, int]
    orders_by_status: dict[AttemptStatus, int]
    llm_input_tokens: int = Field(ge=0)
    llm_output_tokens: int = Field(ge=0)
    llm_cost_usd: Decimal = Field(ge=0)
    staleness_events: tuple[StalenessEvent, ...]


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise MetricsError(f"{name} must be timezone-aware")


class RunMetrics:
    """Mutable accumulator for one run. Not thread-safe; owned by the orchestrator."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._open_stages: dict[str, datetime] = {}
        self._stage_ms: dict[str, int] = {}
        self._stage_counts: Counter[str] = Counter()
        self._tool_calls: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._decisions: Counter[DecisionAction] = Counter()
        self._orders: Counter[AttemptStatus] = Counter()
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = Decimal("0")
        self._staleness: list[StalenessEvent] = []

    def stage_started(self, stage: str, at: datetime) -> None:
        """Start timing ``stage``. A stage may not be started twice without finishing."""
        _require_aware(at, "at")
        if stage in self._open_stages:
            raise MetricsError(f"stage {stage!r} already started")
        self._open_stages[stage] = at

    def stage_finished(self, stage: str, at: datetime) -> None:
        """Finish timing ``stage``; repeated runs of a stage accumulate."""
        _require_aware(at, "at")
        started = self._open_stages.pop(stage, None)
        if started is None:
            raise MetricsError(f"stage {stage!r} was not started")
        elapsed = at - started
        if elapsed.total_seconds() < 0:
            raise MetricsError(f"stage {stage!r} finished before it started")
        self._stage_ms[stage] = self._stage_ms.get(stage, 0) + elapsed // _ONE_MS
        self._stage_counts[stage] += 1

    def tool_call(self, server: str) -> None:
        self._tool_calls[server] += 1

    def error(self, error_type: str) -> None:
        self._errors[error_type] += 1

    def decision(self, action: DecisionAction) -> None:
        self._decisions[action] += 1

    def order(self, status: AttemptStatus) -> None:
        self._orders[status] += 1

    def llm_usage(self, *, input_tokens: int, output_tokens: int, cost_usd: Decimal) -> None:
        """Add one LLM turn's (or session's) usage. Cost is Decimal USD, never float."""
        if input_tokens < 0 or output_tokens < 0:
            raise MetricsError("token counts must be non-negative")
        if not isinstance(cost_usd, Decimal) or not cost_usd.is_finite() or cost_usd < 0:
            raise MetricsError("cost_usd must be a finite, non-negative Decimal")
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cost_usd += cost_usd

    def staleness_event(
        self, *, source: str, field: str, as_of: datetime | None, observed_at: datetime
    ) -> None:
        """Record that ``field`` from ``source`` was stale (or had no ``as_of``) when observed."""
        _require_aware(observed_at, "observed_at")
        if as_of is not None:
            _require_aware(as_of, "as_of")
        self._staleness.append(
            StalenessEvent(source=source, field=field, as_of=as_of, observed_at=observed_at)
        )

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            run_id=self._run_id,
            stages=tuple(
                StageTiming(stage=s, duration_ms=ms, completed_count=self._stage_counts[s])
                for s, ms in sorted(self._stage_ms.items())
            ),
            unfinished_stages=tuple(sorted(self._open_stages)),
            tool_calls_by_server=dict(sorted(self._tool_calls.items())),
            errors_by_type=dict(sorted(self._errors.items())),
            decisions_by_action=dict(sorted(self._decisions.items())),
            orders_by_status=dict(sorted(self._orders.items())),
            llm_input_tokens=self._input_tokens,
            llm_output_tokens=self._output_tokens,
            llm_cost_usd=self._cost_usd,
            staleness_events=tuple(self._staleness),
        )
