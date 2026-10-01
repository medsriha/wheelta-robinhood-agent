"""The code-run order walk (ADR-0066): code works each option order the agent asks for.

The agent calls `mcp__wra_orders__work_option_order(option_id, side, quantity, start_price,
worst_price)`; the hooks admit it like any order tool (venue, kill switch and stop latch, role,
concurrency, pre-trade validation at `start_price`, no late start) and `OrderWorkRunner.start`
starts one `WalkJob` in the session's task group. The agent waits with
`mcp__wra_orders__await_order_work(work_ref, wait_seconds)`. The server for both tools is
`agent/order_work_server.py`.

A job walks `domain.order_walk.step_prices(start_price → worst_price)` inside
`orders.walk.window_seconds`, measured from acceptance. Per step:

1. `get_option_quotes` for the contract (and `get_equity_quotes` for a sell to open: the
   pre-trade gate needs the underlying). A failed read stops the job; the step's price must
   lie within the fresh [bid, ask] (`orders.limit_price_bounds`), else the job ends
   `QUOTE_MOVED` without placing.
2. `review_option_order` with the exact order; any broker alert or mismatch stops the job.
3. `place_option_order` with the reviewed parameters, for the target less confirmed fills.
   The hooks' gate runs the pre-trade validation and concurrency checks on it, as for any
   placement; a denial stops the job with its reason.
4. Wait up to the step's deadline, reading the order by ID every `poll_seconds`.
5. Unfilled: `cancel_option_order`, then read until the order is terminal, counting every
   fill, including fills during the cancel.

Every call goes through `ExecutorGate.admit` (the hooks' checks and recording, with
`parent_tool_call_id` = the job's `work_option_order` call) and `ValidatingProxy.execute`
(intent before dispatch, upstream, validation, recording), so it is recorded and checked
exactly like a model call; only its result is never delivered to the model.

Stop rules (CLAUDE.md §14): no call is retried. A place or cancel that errors, times out, or
returns an unusable result ends the job `unknown` and sets `placement_ended`, after which the
hooks deny every new `work_option_order` of the run. A job never starts a step while a cancel
is pending or a final quantity is unknown. On the stop latch a job places nothing more; it
sends one cancel for its working step, bounded by `LATCH_CANCEL_SECONDS` (owner decision,
ADR-0066 open question 1), and confirms it by reads. The executor never picks another contract.
"""

import math
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from types import TracebackType
from typing import Any, Final, Protocol

import anyio
from anyio.abc import TaskGroup
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator

from wheelta_robinhood_agent.agent.account_scope import AGENTIC_ACCOUNT_PLACEHOLDER
from wheelta_robinhood_agent.agent.mapped_evidence import (
    BrokerOrderObservation,
    MappedEvidence,
    OrderReviewObservation,
)
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.config.rules import PartialFill
from wheelta_robinhood_agent.domain.enums import AttemptStatus, OrderSide, ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument
from wheelta_robinhood_agent.domain.order_walk import (
    STEP_OVERHEAD_SECONDS,
    TickSchedule,
    WalkStatus,
    WalkTiming,
    remaining_quantity,
    step_prices,
    within_bounds,
)
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, make_registry

ORDER_WORK_SERVER: Final = "wra_orders"
WORK_TOOL: Final = "work_option_order"
AWAIT_TOOL: Final = "await_order_work"
QUALIFIED_WORK_TOOL: Final = f"mcp__{ORDER_WORK_SERVER}__{WORK_TOOL}"
QUALIFIED_AWAIT_TOOL: Final = f"mcp__{ORDER_WORK_SERVER}__{AWAIT_TOOL}"
ORDER_WORK_REF_PREFIX: Final = "order_work:"

ORDER_WORK_REGISTRY: ToolRegistry = make_registry(
    ORDER_WORK_SERVER,
    {ToolTier.R: (AWAIT_TOOL,), ToolTier.X: (WORK_TOOL,)},
    verified=True,
    live_order_tools=(WORK_TOOL,),
)

# Robinhood tools the executor calls; no other tool is admitted for it.
QUOTES_TOOL: Final = "get_option_quotes"
EQUITY_QUOTES_TOOL: Final = "get_equity_quotes"
ORDERS_TOOL: Final = "get_option_orders"
REVIEW_TOOL: Final = "review_option_order"
PLACE_TOOL: Final = "place_option_order"
CANCEL_TOOL: Final = "cancel_option_order"
ACCOUNT_TOOL: Final = "get_portfolio"
EXECUTOR_TOOLS: Final = frozenset(
    {
        QUOTES_TOOL,
        EQUITY_QUOTES_TOOL,
        ORDERS_TOOL,
        ACCOUNT_TOOL,
        REVIEW_TOOL,
        PLACE_TOOL,
        CANCEL_TOOL,
    }
)

# Operational bounds, not trading values (ADR-0066).
# A step's wait ends this long before the window ends, leaving time to cancel and confirm.
CANCEL_RESERVE_SECONDS: Final = 5.0
# How long the executor reads an order after its cancel was accepted before it gives up and
# records the outcome as unknown.
CONFIRM_SECONDS: Final = 30.0
# The one cancel after the stop latch: its upstream deadline and its whole confirmation.
LATCH_CANCEL_SECONDS: Final = 30.0
# `await_order_work` waits at most this long per call (below the proxy's answer time).
MAX_AWAIT_SECONDS: Final = 45

_TERMINAL_ORDER = frozenset(
    {
        AttemptStatus.FILLED,
        AttemptStatus.CANCELLED,
        AttemptStatus.REJECTED,
        AttemptStatus.EXPIRED,
    }
)


def order_work_ref_for(work_call_id: uuid.UUID) -> str:
    """The code-issued reference of one job: its `work_option_order` call (ADR-0066)."""
    return f"{ORDER_WORK_REF_PREFIX}{work_call_id}"


def work_id_of(ref: object) -> uuid.UUID | None:
    """The job id inside an `order_work:` reference, or None."""
    if not isinstance(ref, str) or not ref.startswith(ORDER_WORK_REF_PREFIX):
        return None
    try:
        return uuid.UUID(ref.removeprefix(ORDER_WORK_REF_PREFIX))
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------------------------


def _price(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ValueError('prices are decimal strings, e.g. "0.85"')
    try:
        price = Decimal(value.strip())
    except InvalidOperation:
        raise ValueError(f"malformed price {value!r}") from None
    if not price.is_finite() or price <= 0:
        raise ValueError(f"price must be positive, got {value!r}")
    return price


class WorkRequest(BaseModel):
    """`work_option_order` arguments, strictly parsed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: StrictStr = Field(min_length=1)
    side: OrderSide
    quantity: int = Field(strict=True, gt=0)
    start_price: Decimal
    worst_price: Decimal

    @field_validator("option_id")
    @classmethod
    def _canonical_id(cls, value: str) -> str:
        """The broker instrument UUID in canonical form (evidence keys use it)."""
        try:
            return str(uuid.UUID(value.strip()))
        except ValueError:
            raise ValueError("option_id must be the contract's instrument UUID") from None

    @field_validator("start_price", "worst_price", mode="before")
    @classmethod
    def _decimal(cls, value: object) -> Decimal:
        return _price(value)

    @property
    def opening(self) -> bool:
        return self.side is OrderSide.SELL_TO_OPEN

    def order_input(self, price: Decimal, quantity: int) -> dict[str, Any]:
        """The review/place arguments of one step: one leg, limit, day order."""
        return {
            "account_number": AGENTIC_ACCOUNT_PLACEHOLDER,
            "legs": [
                {
                    "option_id": self.option_id,
                    "side": "sell" if self.opening else "buy",
                    "position_effect": "open" if self.opening else "close",
                    "ratio_quantity": 1,
                }
            ],
            "quantity": str(quantity),
            "price": str(price),
            "type": "limit",
            "time_in_force": "gfd",
            "direction": "credit" if self.opening else "debit",
        }


def parse_work_request(tool_input: Mapping[str, object]) -> WorkRequest | str:
    """The request, or the issues as one line for the agent."""
    try:
        return WorkRequest.model_validate(dict(tool_input))
    except ValidationError as exc:
        issues = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        return f"work_option_order input is invalid: {issues}"


def tick_schedule_of(instrument: OptionInstrument) -> TickSchedule | None:
    """The contract's ticks: the broker's `min_ticks`, else a uniform `tick_increment`."""
    if instrument.tick_schedule is not None:
        return instrument.tick_schedule
    tick = instrument.tick_increment
    if tick is None:
        return None
    return TickSchedule(above_tick=tick, below_tick=tick, cutoff_price=Decimal(0))


def planned_prices(
    request: WorkRequest, instrument: OptionInstrument | None, max_steps: int
) -> tuple[Decimal, ...] | str:
    """The walk's prices, or why none can be planned (shown to the agent)."""
    if instrument is None:
        return (
            f"no validated instrument for option_id {request.option_id} in this run; read it "
            "with get_option_instruments first"
        )
    ticks = tick_schedule_of(instrument)
    if ticks is None:
        return "the contract's price ticks are unknown; re-read get_option_instruments"
    try:
        return step_prices(request.start_price, request.worst_price, max_steps, request.side, ticks)
    except ValueError as exc:
        return str(exc)


# --------------------------------------------------------------------------------------------
# What the executor needs from the hooks and the proxy
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Admitted:
    """The hooks recorded and allowed the call; `use_id` names its proxy registration."""

    use_id: str
    tool_call_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class Refused:
    """The hooks denied the call (recorded as denied); nothing was sent."""

    reason: str


class ExecutorGate(Protocol):
    """The hooks' checks and recording for one executor call (`hooks.build_hooks_with_gate`)."""

    def admit(
        self,
        tool: str,
        tool_input: dict[str, Any],
        *,
        job_id: uuid.UUID,
        after_stop_cancel: bool = False,
    ) -> Admitted | Refused: ...


@dataclass(frozen=True, slots=True)
class CallOutcome:
    status: ToolCallStatus
    evidence: MappedEvidence | None


class ExecutorTransport(Protocol):
    """`ValidatingProxy.execute` for the Robinhood proxy, with the recorded envelope decoded
    into its typed evidence (None unless validated); a fake in tests."""

    async def __call__(
        self, use_id: str, *, timeout_seconds: float | None = None
    ) -> CallOutcome: ...


# A job's lifecycle as run events (ADR-0066 item 4): name, payload, the calls behind it.
JobEventSink = Callable[[str, Mapping[str, object], tuple[uuid.UUID, ...]], None]


# --------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------


@dataclass
class WalkStep:
    index: int
    price: Decimal
    quantity: int
    place_tool_call_id: uuid.UUID | None = None
    broker_order_id: str | None = None
    order_status: AttemptStatus | None = None
    filled_quantity: int = 0
    cancel_tool_call_id: uuid.UUID | None = None
    # Set before a cancel is sent: a step's order is cancelled at most once (CLAUDE.md §14).
    cancel_sent: bool = False
    placed_order: BrokerOrderObservation | None = None


@dataclass
class WalkJob:
    """One trade being worked. Mutated only by its own task."""

    job_id: uuid.UUID
    request: WorkRequest
    prices: tuple[Decimal, ...]
    accepted_at: datetime
    window_end: datetime
    status: WalkStatus = WalkStatus.WORKING
    reason: str | None = None
    steps: list[WalkStep] = field(default_factory=list)
    filled_quantity: int = 0
    ended_at: datetime | None = None
    done: anyio.Event = field(default_factory=anyio.Event)

    @property
    def work_ref(self) -> str:
        return order_work_ref_for(self.job_id)

    @property
    def working_order_id(self) -> str | None:
        """The broker order of the current step while it may still be working."""
        if not self.steps:
            return None
        step = self.steps[-1]
        if step.broker_order_id is None or step.order_status in _TERMINAL_ORDER:
            return None
        return step.broker_order_id

    def view(self) -> dict[str, Any]:
        """What `work_option_order` / `await_order_work` deliver to the agent."""
        return {
            "work_ref": self.work_ref,
            "status": self.status.value,
            "reason": self.reason,
            "option_id": self.request.option_id,
            "side": self.request.side.value,
            "quantity": self.request.quantity,
            "filled_quantity": self.filled_quantity,
            "start_price": str(self.request.start_price),
            "worst_price": str(self.request.worst_price),
            "planned_prices": [str(p) for p in self.prices],
            "accepted_at": self.accepted_at.isoformat(),
            "window_ends_at": self.window_end.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "steps": [
                {
                    "step": s.index,
                    "price": str(s.price),
                    "quantity": s.quantity,
                    "broker_order_id": s.broker_order_id,
                    "order_status": s.order_status.value if s.order_status else None,
                    "filled_quantity": s.filled_quantity,
                }
                for s in self.steps
            ],
        }


class _JobEnded(Exception):
    def __init__(self, status: WalkStatus, reason: str, *, ends_placement: bool = False) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.ends_placement = ends_placement


@dataclass(frozen=True)
class ActiveJob:
    """An unfinished job as the concurrency check sees it (ADR-0051, ADR-0066 item 5)."""

    job_id: uuid.UUID
    option_id: str
    closing: bool
    remaining_quantity: int
    worst_price: Decimal


class OrderWorkRunner:
    """Hosts the session's walk jobs in one task group (entered by the session)."""

    def __init__(
        self,
        *,
        timing: WalkTiming,
        partial_fill: PartialFill,
        instruments: Callable[[str], OptionInstrument | None],
        run_control: RunControl,
        clock: Callable[[], datetime],
        events: JobEventSink | None = None,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
        quote_max_age_seconds: int | None = None,
        max_await_seconds: float = MAX_AWAIT_SECONDS,
    ) -> None:
        """`quote_max_age_seconds`: `data_quality.freshness.option_quote_max_age_seconds`;
        a quote older than this stops the job (None: the rule is `none`)."""
        self.timing = timing
        self.quote_max_age_seconds = quote_max_age_seconds
        self.max_await_seconds = max(0.0, min(max_await_seconds, MAX_AWAIT_SECONDS))
        # Set when the session ends while jobs still run (its agent is gone): every job then
        # stops as on the stop latch. Runner-local, so it never stops another run.
        self._halted = False
        # One executor placement at a time (ADR-0051: its concurrency check counts the
        # placements in flight), so concurrent close jobs never refuse each other.
        self._place_lock = anyio.Lock()
        self.partial_fill = partial_fill
        self._transport: ExecutorTransport | None = None
        self._instruments = instruments
        self._run_control = run_control
        self._clock = clock
        self._events = events
        self._sleep = sleep
        self._gate: ExecutorGate | None = None
        self._jobs: dict[uuid.UUID, WalkJob] = {}
        # Admitted work_option_order calls whose job has not started yet (ADR-0051): the
        # model can send two calls at once, so an admission counts as working at once.
        self._reserved: dict[uuid.UUID, WorkRequest] = {}
        self._group: TaskGroup | None = None
        # Set once any order action of a job has an unknown outcome; the hooks then deny
        # every new work_option_order of the run.
        self.placement_ended = False

    def bind(self, gate: ExecutorGate, transport: ExecutorTransport) -> None:
        """The hooks' gate and the Robinhood proxy's executor entry. Both are built after the
        runner: the hooks read the runner's view, and the proxy is built with the options."""
        self._gate = gate
        self._transport = transport

    async def __aenter__(self) -> "OrderWorkRunner":
        self._group = anyio.create_task_group()
        await self._group.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Wait for every job: each places nothing more after the latch and cancels its
        working step once (module docstring)."""
        group, self._group = self._group, None
        if group is None:
            return None
        return await group.__aexit__(exc_type, exc, tb)

    # -- views the hooks and the session use ----------------------------------------------

    def job(self, job_id: uuid.UUID) -> WalkJob | None:
        return self._jobs.get(job_id)

    def active(self) -> tuple[WalkJob, ...]:
        return tuple(j for j in self._jobs.values() if j.status is WalkStatus.WORKING)

    def holds(self, broker_order_id: str) -> bool:
        """Whether an unfinished job owns this broker order (the agent may not cancel it)."""
        try:
            wanted = str(uuid.UUID(broker_order_id.strip()))
        except ValueError:
            wanted = broker_order_id
        return any(
            s.broker_order_id is not None and s.broker_order_id.lower() == wanted.lower()
            for j in self.active()
            for s in j.steps
        )

    def active_jobs(self, exclude: uuid.UUID | None = None) -> tuple[ActiveJob, ...]:
        """Unfinished jobs and admitted, not yet started ones, as the concurrency check sees
        them (`exclude`: the job asking)."""
        running = tuple(
            ActiveJob(
                job_id=j.job_id,
                option_id=j.request.option_id,
                closing=not j.request.opening,
                remaining_quantity=remaining_quantity(j.request.quantity, j.filled_quantity),
                worst_price=j.request.worst_price,
            )
            for j in self.active()
            if j.job_id != exclude
        )
        reserved = tuple(
            ActiveJob(
                job_id=job_id,
                option_id=r.option_id,
                closing=not r.opening,
                remaining_quantity=r.quantity,
                worst_price=r.worst_price,
            )
            for job_id, r in self._reserved.items()
            if job_id != exclude
        )
        return running + reserved

    def reserve(self, job_id: uuid.UUID, request: WorkRequest) -> None:
        """Hold an admitted call's place until its job starts or the call fails."""
        self._reserved[job_id] = request

    def release(self, job_id: uuid.UUID) -> None:
        """Drop a reservation whose call never started a job."""
        self._reserved.pop(job_id, None)

    def pending(self) -> bool:
        """Whether an admitted call has not started (or failed to start) its job yet."""
        return bool(self._reserved)

    async def wait_all(self) -> None:
        for job in tuple(self._jobs.values()):
            await job.done.wait()

    async def wait(self, job_id: uuid.UUID, seconds: float) -> WalkJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        with anyio.move_on_after(max(0.0, min(seconds, self.max_await_seconds))):
            await job.done.wait()
        return job

    def halt(self) -> None:
        """The session ended (normally or not): no job places anything more; each sends its
        working step's one bounded cancel and ends. Called before the runner is left."""
        self._halted = True

    def placing(self) -> bool:
        """Whether an executor placement is in flight (its broker order ID is not known yet,
        so the agent's cancels wait; ADR-0066)."""
        return self._place_lock.locked()

    # -- starting a job -------------------------------------------------------------------

    def plan(self, request: WorkRequest) -> tuple[Decimal, ...] | str:
        return planned_prices(request, self._instruments(request.option_id), self.timing.max_steps)

    def start(self, job_id: uuid.UUID, request: WorkRequest) -> WalkJob | str:
        """Start a job for an admitted `work_option_order` call, or say why not."""
        reserved = self._reserved.pop(job_id, None)
        if self._group is None or self._gate is None or self._transport is None:
            return "the order executor is not running"
        if job_id in self._jobs:
            return "this work_option_order call already started a job"
        if reserved != request:
            return "this work_option_order call was not admitted with these arguments"
        prices = self.plan(request)
        if isinstance(prices, str):
            return prices
        now = self._clock()
        job = WalkJob(
            job_id=job_id,
            request=request,
            prices=prices,
            accepted_at=now,
            window_end=now + timedelta(seconds=self.timing.window_seconds),
        )
        self._jobs[job_id] = job
        self._event(job, "started", ())
        self._group.start_soon(self._run, job)
        return job

    # -- the walk -------------------------------------------------------------------------

    def _event(self, job: WalkJob, name: str, calls: tuple[uuid.UUID, ...]) -> None:
        if self._events is None:
            return
        try:
            self._events(f"{job.work_ref}:{name}", {"order_work": job.view()}, calls)
        except Exception:  # noqa: BLE001 - recording failed: stop the run, finish the job safely
            self._run_control.request_stop(StopReason.INFRASTRUCTURE_FAILURE, self._clock())

    def _stopping(self) -> bool:
        return self._halted or self._run_control.stop_requested

    def _seconds_left(self, job: WalkJob) -> float:
        return (job.window_end - self._clock()).total_seconds()

    async def _run(self, job: WalkJob) -> None:
        try:
            await self._walk(job)
            self._end(
                job, self._final(job), job.reason or "the window or the last price step ended"
            )
        except _JobEnded as ended:
            status = ended.status
            if status is WalkStatus.CANCELLED and job.filled_quantity > 0:
                status = WalkStatus.PARTIALLY_FILLED
            if status is WalkStatus.UNKNOWN or ended.ends_placement:
                self.placement_ended = True
            self._end(job, status, ended.reason)
        except BaseException as exc:
            # A failure, or the session's task group cancelled us. Either way: placement ends,
            # one bounded cancel of a working step that has none (shielded so it is sent and
            # confirmed), the job ends unknown, and a cancellation propagates.
            self.placement_ended = True
            failure = isinstance(exc, Exception)
            with anyio.CancelScope(shield=True):
                if failure:
                    self._run_control.request_stop(StopReason.INFRASTRUCTURE_FAILURE, self._clock())
                await self._salvage(job)
                self._end(
                    job,
                    WalkStatus.UNKNOWN,
                    f"executor failure ({type(exc).__name__})"
                    if failure
                    else "the executor was cancelled",
                )
            if not failure:
                raise

    def _final(self, job: WalkJob) -> WalkStatus:
        if job.filled_quantity >= job.request.quantity:
            return WalkStatus.FILLED
        return WalkStatus.PARTIALLY_FILLED if job.filled_quantity > 0 else WalkStatus.CANCELLED

    async def _salvage(self, job: WalkJob) -> None:
        """After a failure: one cancel of the working step if none was sent yet, then bounded
        reads to confirm it, all within `LATCH_CANCEL_SECONDS`. Counts a terminal order's fills.
        Never raises: a leftover order is reported by the session (R19)."""
        step = job.steps[-1] if job.steps else None
        order_id = job.working_order_id
        if step is None or order_id is None or step.cancel_sent:
            return
        try:
            with anyio.move_on_after(LATCH_CANCEL_SECONDS):
                step.cancel_sent = True
                result = await self._call(
                    job,
                    CANCEL_TOOL,
                    {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER, "order_id": order_id},
                    after_stop_cancel=True,
                    timeout_seconds=LATCH_CANCEL_SECONDS,
                )
                if isinstance(result, Refused):
                    return
                step.cancel_tool_call_id = result[0]
                order = await self._confirm(job, order_id, LATCH_CANCEL_SECONDS)
                if order is not None:
                    self._settle(job, step, order)
        except Exception:  # noqa: BLE001 - best effort; the leftover order is reported (R19)
            return

    def _end(self, job: WalkJob, status: WalkStatus, reason: str) -> None:
        job.status = status
        job.reason = reason
        job.ended_at = self._clock()
        calls = tuple(
            c
            for s in job.steps
            for c in (s.place_tool_call_id, s.cancel_tool_call_id)
            if c is not None
        )
        self._event(job, "ended", calls)
        job.done.set()

    async def _nap(self, seconds: float) -> None:
        """Sleep, waking within a second of a stop request."""
        left = max(0.0, seconds)
        while left > 0 and not self._stopping():
            chunk = min(1.0, left)
            await self._sleep(chunk)
            left -= chunk

    async def _call(
        self,
        job: WalkJob,
        tool: str,
        tool_input: dict[str, Any],
        *,
        after_stop_cancel: bool = False,
        timeout_seconds: float | None = None,
    ) -> tuple[uuid.UUID, CallOutcome] | Refused:
        gate, transport = self._gate, self._transport
        if gate is None or transport is None:
            raise RuntimeError("the order executor is not bound")
        admitted = gate.admit(
            tool, tool_input, job_id=job.job_id, after_stop_cancel=after_stop_cancel
        )
        if isinstance(admitted, Refused):
            return admitted
        if tool in (PLACE_TOOL, CANCEL_TOOL):
            # A sent order action always runs to its recorded outcome (bounded by the proxy's
            # upstream deadline): abandoning it would leave an order whose ID nobody knows.
            with anyio.CancelScope(shield=True):
                outcome = await transport(admitted.use_id, timeout_seconds=timeout_seconds)
        else:
            outcome = await transport(admitted.use_id, timeout_seconds=timeout_seconds)
        return admitted.tool_call_id, outcome

    def _check_placement_open(self) -> None:
        """No new step once any order action of the run has an unknown outcome (§14)."""
        if self.placement_ended:
            raise _JobEnded(
                WalkStatus.STOPPED,
                "another order action of this run has an unknown outcome; placement has ended",
            )

    async def _quote(self, job: WalkJob) -> Quote:
        result = await self._call(job, QUOTES_TOOL, {"instrument_ids": [job.request.option_id]})
        if isinstance(result, Refused):
            raise _JobEnded(WalkStatus.STOPPED, f"quote read denied: {result.reason}")
        _, outcome = result
        quotes = [
            q
            for q in (outcome.evidence.option_quotes if outcome.evidence else ())
            if q.broker_instrument_id == job.request.option_id
        ]
        if not quotes:
            raise _JobEnded(WalkStatus.STOPPED, "no valid live quote for the contract")
        quote = quotes[-1]
        if self.quote_max_age_seconds is not None:
            age = (self._clock() - quote.as_of).total_seconds()
            if age > self.quote_max_age_seconds:
                raise _JobEnded(
                    WalkStatus.STOPPED,
                    f"the live quote is {age:.0f}s old (data_quality.freshness."
                    f"option_quote_max_age_seconds {self.quote_max_age_seconds})",
                )
        if job.request.opening:
            instrument = self._instruments(job.request.option_id)
            symbol = instrument.underlying if instrument is not None else None
            if symbol is None:
                raise _JobEnded(WalkStatus.STOPPED, "the contract's underlying is unknown")
            underlying = await self._call(job, EQUITY_QUOTES_TOOL, {"symbols": [symbol]})
            if isinstance(underlying, Refused) or not (
                underlying[1].evidence and underlying[1].evidence.underlying_quotes
            ):
                raise _JobEnded(WalkStatus.STOPPED, "no valid live quote for the underlying")
        return quote

    def _review_matches(
        self, review: OrderReviewObservation, request: WorkRequest, price: Decimal, quantity: int
    ) -> str | None:
        """Why the review does not match the intended order (None: it matches and is clean)."""
        if not review.clean:
            return f"the broker's review raised an alert ({review.alert_type or 'unnamed'})"
        legs = [(leg.broker_instrument_id, leg.side_raw) for leg in review.legs]
        if legs != [(request.option_id, request.side.value)]:
            return "the review's legs differ from the intended order"
        if review.quantity != quantity:
            return f"the review's quantity {review.quantity} differs from {quantity}"
        if review.order_type_raw.lower() != "limit":
            return f"the review's order type is {review.order_type_raw}, not limit"
        if (review.time_in_force_raw or "gfd").lower() != "gfd":
            return f"the review's time in force is {review.time_in_force_raw}, not gfd"
        if review.limit_price != price:
            return f"the review's limit price {review.limit_price} differs from {price}"
        return None

    @staticmethod
    def _placed_matches(
        order: BrokerOrderObservation, request: WorkRequest, price: Decimal, quantity: int
    ) -> str | None:
        """Why the broker's order differs from what was sent (None: it matches)."""
        legs = [(leg.broker_instrument_id, leg.side_raw) for leg in order.legs]
        if legs != [(request.option_id, request.side.value)]:
            return "the placed order's leg differs from the request"
        if order.quantity != quantity:
            return f"the placed order's quantity {order.quantity} differs from {quantity}"
        if order.limit_price is not None and order.limit_price != price:
            return f"the placed order's limit price {order.limit_price} differs from {price}"
        return None

    def _order_in(self, outcome: CallOutcome, order_id: str) -> BrokerOrderObservation | None:
        if outcome.evidence is None:
            return None
        found = [o for o in outcome.evidence.broker_orders if o.broker_order_id == order_id]
        return found[-1] if found else None

    async def _read_order(
        self, job: WalkJob, order_id: str, timeout_seconds: float | None = None
    ) -> tuple[uuid.UUID, BrokerOrderObservation] | None:
        result = await self._call(
            job,
            ORDERS_TOOL,
            {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER, "order_id": order_id},
            timeout_seconds=timeout_seconds,
        )
        if isinstance(result, Refused):
            return None
        call_id, outcome = result
        order = self._order_in(outcome, order_id)
        return (call_id, order) if order is not None else None

    async def _walk(self, job: WalkJob) -> None:
        request = job.request
        timing = self.timing
        for index, price in enumerate(job.prices, start=1):
            if self._stopping():
                raise _JobEnded(WalkStatus.STOPPED, "run stop requested")
            self._check_placement_open()
            if self._seconds_left(job) < timing.poll_seconds + CANCEL_RESERVE_SECONDS:
                return  # no time for another step: the window ends the job
            quantity = remaining_quantity(request.quantity, job.filled_quantity)
            if quantity == 0:
                return
            quote = await self._quote(job)
            if not within_bounds(price, quote.bid, quote.ask):
                raise _JobEnded(
                    WalkStatus.CANCELLED,
                    f"QUOTE_MOVED: step {index} price {price} is outside the live "
                    f"[{quote.bid}, {quote.ask}]",
                )
            order_input = request.order_input(price, quantity)
            step_started = self._clock()
            reviewed = await self._call(job, REVIEW_TOOL, order_input)
            if isinstance(reviewed, Refused):
                raise _JobEnded(WalkStatus.STOPPED, f"review denied: {reviewed.reason}")
            _, review_outcome = reviewed
            reviews = review_outcome.evidence.order_reviews if review_outcome.evidence else ()
            if review_outcome.status is not ToolCallStatus.SUCCEEDED or len(reviews) != 1:
                raise _JobEnded(WalkStatus.STOPPED, "the order review failed or was unusable")
            mismatch = self._review_matches(reviews[0], request, price, quantity)
            if mismatch is not None:
                raise _JobEnded(WalkStatus.STOPPED, mismatch)
            step = await self._place(job, index, price, quantity, order_input)
            if step is None:
                return  # slow reads used the step's time: never place past the window
            order = await self._wait_for_fill(job, step, step_started)
            if order.status not in _TERMINAL_ORDER:
                order = await self._cancel(job, step)
            self._settle(job, step, order)
            self._event(
                job,
                f"step:{index}:ended",
                tuple(c for c in (step.place_tool_call_id, step.cancel_tool_call_id) if c),
            )
            if job.filled_quantity >= request.quantity:
                return
            if order.status is AttemptStatus.REJECTED:
                raise _JobEnded(WalkStatus.STOPPED, "the broker rejected the order")
            if self._stopping():
                raise _JobEnded(WalkStatus.STOPPED, "run stop requested")
            if job.filled_quantity > 0 and self.partial_fill is PartialFill.STOP:
                raise _JobEnded(WalkStatus.PARTIALLY_FILLED, "partial fill (partial_fill = stop)")

    async def _place(
        self,
        job: WalkJob,
        index: int,
        price: Decimal,
        quantity: int,
        order_input: dict[str, Any],
    ) -> WalkStep | None:
        """Place one step, one executor placement at a time; None when no time is left."""
        async with self._place_lock:
            if self._stopping():
                raise _JobEnded(WalkStatus.STOPPED, "run stop requested")
            self._check_placement_open()
            if self._seconds_left(job) < self.timing.poll_seconds + CANCEL_RESERVE_SECONDS:
                return None
            if self.active_jobs(exclude=job.job_id):
                # ADR-0051: placing beside other work needs a fresh Agentic account snapshot
                # for the combined-debit check; a failed read leaves the check to deny.
                await self._call(job, ACCOUNT_TOOL, {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER})
            step = WalkStep(index=index, price=price, quantity=quantity)
            placed = await self._call(job, PLACE_TOOL, order_input)
            if isinstance(placed, Refused):
                raise _JobEnded(WalkStatus.STOPPED, f"placement denied: {placed.reason}")
            step.place_tool_call_id, outcome = placed
            orders = outcome.evidence.broker_orders if outcome.evidence else ()
            if outcome.status is ToolCallStatus.FAILED:
                # FAILED is recorded only when nothing reached the broker (not forwarded).
                raise _JobEnded(WalkStatus.STOPPED, "the placement was not sent")
            job.steps.append(step)
            if outcome.status is not ToolCallStatus.SUCCEEDED or len(orders) != 1:
                raise _JobEnded(
                    WalkStatus.UNKNOWN,
                    "the placement's outcome is unknown; no further order action this run",
                )
            order = orders[0]
            step.broker_order_id = order.broker_order_id
            step.order_status = order.status
            step.placed_order = order
        self._event(job, f"step:{index}:placed", (step.place_tool_call_id,))
        differs = self._placed_matches(order, job.request, price, quantity)
        if differs is not None:
            # The broker holds an order we did not ask for: cancel it once, count any fills,
            # and end placement for the run.
            if order.status not in _TERMINAL_ORDER:
                order = await self._cancel(job, step)
            self._settle(job, step, order, strict=False)
            raise _JobEnded(WalkStatus.UNKNOWN, differs)
        return step

    async def _wait_for_fill(
        self, job: WalkJob, step: WalkStep, step_started: datetime
    ) -> BrokerOrderObservation:
        """Read the order by ID until it is terminal, the step's wait ends, or a stop."""
        order = step.placed_order
        assert order is not None  # noqa: S101 - set by _place
        step_end = min(
            step_started + timedelta(seconds=self.timing.step_wait_seconds),
            job.window_end - timedelta(seconds=CANCEL_RESERVE_SECONDS),
        )
        order_id = order.broker_order_id
        # Bounded by count as well as by the clock, so a step ends whatever the clock does.
        reads = math.ceil(self.timing.step_wait_seconds / self.timing.poll_seconds)
        while order.status not in _TERMINAL_ORDER and not self._stopping() and reads > 0:
            left = (step_end - self._clock()).total_seconds()
            if left <= 0:
                break
            reads -= 1
            await self._nap(min(float(self.timing.poll_seconds), left))
            if self._stopping():
                break
            read = await self._read_order(
                job, order_id, timeout_seconds=max(left, float(self.timing.poll_seconds))
            )
            if read is not None:
                order = read[1]
                step.order_status = order.status
        return order

    async def _confirm(
        self, job: WalkJob, order_id: str, seconds: float
    ) -> BrokerOrderObservation | None:
        """Read the order until it is terminal, within `seconds` (count-bounded too)."""
        deadline = self._clock() + timedelta(seconds=seconds)
        reads = math.ceil(seconds / self.timing.poll_seconds) + 1
        while reads > 0:
            reads -= 1
            left = (deadline - self._clock()).total_seconds()
            if left <= 0:
                break
            read = await self._read_order(job, order_id, timeout_seconds=left)
            if read is not None and read[1].status in _TERMINAL_ORDER:
                return read[1]
            left = (deadline - self._clock()).total_seconds()
            if left <= 0:
                break
            await self._sleep(min(float(self.timing.poll_seconds), left))
        return None

    async def _cancel(self, job: WalkJob, step: WalkStep) -> BrokerOrderObservation:
        """Cancel the step's working order once, then read until it is terminal.

        A cancel that errored, timed out, or was refused still gets the bounded reads: a
        terminal state they find is counted (an order often fills just as it is cancelled).
        Placement for the run ends either way (CLAUDE.md §14); an unresolved order ends the
        job unknown."""
        order_id = step.broker_order_id
        if order_id is None:
            raise _JobEnded(WalkStatus.UNKNOWN, "a working order has no broker ID")
        if step.cancel_sent:
            raise _JobEnded(WalkStatus.UNKNOWN, "the step's cancel was already sent")
        stopping = self._stopping()
        cancel_input = {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER, "order_id": order_id}
        step.cancel_sent = True
        cancelled = await self._call(
            job,
            CANCEL_TOOL,
            cancel_input,
            after_stop_cancel=stopping,
            timeout_seconds=LATCH_CANCEL_SECONDS if stopping else None,
        )
        unsent = isinstance(cancelled, Refused) or cancelled[1].status is ToolCallStatus.FAILED
        if unsent and not stopping and self._stopping():
            # The latch fell between the check and the send, so nothing was sent: send the
            # job's one after-stop cancel now. Not a retry: the first never left.
            stopping = True
            cancelled = await self._call(
                job,
                CANCEL_TOOL,
                cancel_input,
                after_stop_cancel=True,
                timeout_seconds=LATCH_CANCEL_SECONDS,
            )
        if isinstance(cancelled, Refused):
            step.cancel_sent = False  # nothing was sent; a salvage may still cancel it
            raise _JobEnded(
                WalkStatus.UNKNOWN, f"the working order could not be cancelled: {cancelled.reason}"
            )
        step.cancel_tool_call_id, outcome = cancelled
        requests = outcome.evidence.cancel_requests if outcome.evidence else ()
        accepted = outcome.status is ToolCallStatus.SUCCEEDED and any(
            r.broker_order_id == order_id and r.accepted for r in requests
        )
        if outcome.status is ToolCallStatus.FAILED:
            step.cancel_sent = False  # not forwarded: nothing reached the broker
        order = await self._confirm(
            job, order_id, LATCH_CANCEL_SECONDS if stopping else CONFIRM_SECONDS
        )
        if accepted and order is not None:
            step.order_status = order.status
            return order
        if order is not None:
            # Resolved by reads after a cancel error or refusal: count its fills, end placement.
            self._settle(job, step, order)
            raise _JobEnded(
                self._final(job),
                f"the cancel was not accepted; the order is {order.status.value}",
                ends_placement=True,
            )
        raise _JobEnded(
            WalkStatus.UNKNOWN,
            "the cancel was not confirmed by an order read; outcome unknown"
            if accepted
            else "the cancel's outcome is unknown or it was refused; no further order action "
            "this run",
        )

    def _settle(
        self,
        job: WalkJob,
        step: WalkStep,
        order: BrokerOrderObservation,
        *,
        strict: bool = True,
    ) -> None:
        """Count the terminal order's fills (the broker's cumulative processed quantity).
        `strict` also requires the order's own quantity to be the step's."""
        filled = order.processed_quantity
        if (
            filled < 0
            or filled > step.quantity
            or (strict and order.quantity != step.quantity)
            or (order.status is AttemptStatus.FILLED and filled != order.quantity)
            or (order.status is AttemptStatus.CANCELLED and filled == order.quantity)
        ):
            raise _JobEnded(WalkStatus.UNKNOWN, "the order's filled quantity is inconsistent")
        if step.filled_quantity:
            return  # already counted
        step.order_status = order.status
        step.filled_quantity = filled
        job.filled_quantity += filled


__all__ = [
    "AWAIT_TOOL",
    "EXECUTOR_TOOLS",
    "MAX_AWAIT_SECONDS",
    "ORDER_WORK_REF_PREFIX",
    "ORDER_WORK_REGISTRY",
    "ORDER_WORK_SERVER",
    "QUALIFIED_AWAIT_TOOL",
    "QUALIFIED_WORK_TOOL",
    "STEP_OVERHEAD_SECONDS",
    "WORK_TOOL",
    "ActiveJob",
    "Admitted",
    "CallOutcome",
    "ExecutorGate",
    "ExecutorTransport",
    "OrderWorkRunner",
    "Refused",
    "WalkJob",
    "WorkRequest",
    "order_work_ref_for",
    "parse_work_request",
    "planned_prices",
    "work_id_of",
]
