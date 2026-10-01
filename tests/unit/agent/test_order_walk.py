"""The order-walk executor (ADR-0066, agent/order_walk.py) against a fake gate and broker.

No network, no database, no wall clock: the runner's clock is a fake that advances only on
the injected sleep (and, optionally, by a fixed amount per broker call).
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import anyio
import anyio.lowlevel
import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.agent.mapped_evidence import (
    BrokerOrderObservation,
    CancelRequestObservation,
    MappedEvidence,
    OrderLeg,
    OrderReviewObservation,
)
from wheelta_robinhood_agent.agent.order_walk import (
    LATCH_CANCEL_SECONDS,
    Admitted,
    CallOutcome,
    OrderWorkRunner,
    Refused,
    WalkJob,
    WorkRequest,
    order_work_ref_for,
    parse_work_request,
    planned_prices,
    tick_schedule_of,
    work_id_of,
)
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.config.rules import PartialFill
from wheelta_robinhood_agent.domain.enums import AttemptStatus, ToolCallStatus
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument, UnderlyingQuote
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.order_walk import TickSchedule, WalkStatus, WalkTiming
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.domain.run_record import Quote

D = Decimal
T0 = datetime(2026, 10, 1, 15, 0, tzinfo=UTC)
JOB = uuid.UUID(int=7000)
SRC = (uuid.UUID(int=1),)
TIMING = WalkTiming(
    window_seconds=300,
    max_steps=5,
    step_wait_seconds=50,
    poll_seconds=10,
    step_overhead_seconds=10,
)
INSTRUMENT = OptionInstrument(
    evidence_id=uuid.UUID(int=2),
    as_of=T0,
    source_tool_call_ids=SRC,
    occ_symbol=OccSymbol.parse("XYZ   261015P00073000"),
    broker_instrument_id="0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
    underlying="XYZ",
    multiplier=100,
    tick_increment=D("0.01"),
)
STO = WorkRequest.model_validate(
    {
        "option_id": "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
        "side": "sell_to_open",
        "quantity": 2,
        "start_price": "1.10",
        "worst_price": "1.00",
    }
)
STO_PRICES = (D("1.10"), D("1.08"), D("1.05"), D("1.03"), D("1.00"))
BTC = WorkRequest.model_validate(
    {
        "option_id": "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
        "side": "buy_to_close",
        "quantity": 1,
        "start_price": "0.40",
        "worst_price": "0.50",
    }
)


class FakeClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        await anyio.lowlevel.checkpoint()


@dataclass
class Sent:
    tool: str
    tool_input: dict[str, Any]
    after_stop_cancel: bool
    timeout_seconds: float | None = None
    at: datetime | None = None


class FakeGate:
    """Admits every call unless the tool is in `refuse`; remembers each admitted call."""

    def __init__(self, refuse: dict[str, str] | None = None) -> None:
        self.refuse = refuse or {}
        self.admitted: dict[str, Sent] = {}
        self.calls: list[Sent] = []
        self.job_ids: list[uuid.UUID] = []

    def admit(
        self,
        tool: str,
        tool_input: dict[str, Any],
        *,
        job_id: uuid.UUID,
        after_stop_cancel: bool = False,
    ) -> Admitted | Refused:
        self.job_ids.append(job_id)
        sent = Sent(tool, dict(tool_input), after_stop_cancel)
        self.calls.append(sent)
        if tool in self.refuse:
            return Refused(self.refuse[tool])
        use_id = f"use-{len(self.calls)}"
        self.admitted[use_id] = sent
        return Admitted(use_id=use_id, tool_call_id=uuid.uuid5(uuid.NAMESPACE_URL, use_id))


@dataclass
class _Order:
    order_id: str
    quantity: int
    price: Decimal
    step: int
    processed: int = 0
    status: AttemptStatus = AttemptStatus.PLACED
    reads: int = 0
    cancel_requested: bool = False


@dataclass
class FakeBroker:
    """The executor's transport. Per step (1-based):

    - `fill_on_poll[k] = n`: the first poll read of step k's order shows n contracts filled
      (all of them: FILLED; fewer: still working);
    - `fill_on_cancel[k] = n`: n more contracts fill while step k's cancel is pending.
    """

    clock: FakeClock
    gate: FakeGate
    bid: Decimal = D("0.95")
    ask: Decimal = D("1.15")
    fill_on_poll: dict[int, int] = field(default_factory=dict)
    fill_on_cancel: dict[int, int] = field(default_factory=dict)
    review: str = "clean"  # clean | alert | mismatch | failed
    place: str = "ok"  # ok | unknown | not_sent
    cancel: str = "ok"  # ok | refused | unknown
    cancel_confirms: bool = True
    call_seconds: float = 0.0
    on_read: Callable[["FakeBroker", _Order], None] | None = None
    raise_on: str | None = None
    quote_age_seconds: float = 0.0
    placed_price_offset: Decimal = D("0")
    placing: int = 0
    max_placing: int = 0
    orders: list[_Order] = field(default_factory=list)
    sent: list[Sent] = field(default_factory=list)

    def _obs(self) -> dict[str, Any]:
        return {"evidence_id": uuid.uuid4(), "as_of": self.clock(), "source_tool_call_ids": SRC}

    def _order_obs(self, order: _Order, request_side: str) -> BrokerOrderObservation:
        return BrokerOrderObservation(
            **self._obs(),
            broker_order_id=order.order_id,
            state_raw=order.status.value,
            status=order.status,
            underlying="XYZ",
            order_type_raw="limit",
            trigger_raw="immediate",
            time_in_force_raw="gfd",
            quantity=order.quantity,
            processed_quantity=order.processed,
            pending_quantity=order.quantity - order.processed,
            canceled_quantity=0,
            limit_price=order.price,
            multiplier=100,
            placed_agent="agent",
            created_at=self.clock(),
            legs=(
                OrderLeg(
                    broker_instrument_id="0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
                    side_raw=request_side,
                ),
            ),
        )

    @staticmethod
    def _side(tool_input: dict[str, Any]) -> str:
        leg = tool_input["legs"][0]
        return f"{leg['side']}_to_{leg['position_effect']}"

    def _find(self, order_id: str) -> _Order:
        return next(o for o in self.orders if o.order_id == order_id)

    async def __call__(self, use_id: str, *, timeout_seconds: float | None = None) -> CallOutcome:
        sent = self.gate.admitted.pop(use_id)  # each admission is used exactly once
        sent.timeout_seconds = timeout_seconds
        sent.at = self.clock()
        self.sent.append(sent)
        placing = sent.tool == "place_option_order"
        if placing:
            self.placing += 1
            self.max_placing = max(self.max_placing, self.placing)
        self.clock.advance(self.call_seconds)
        await anyio.lowlevel.checkpoint()
        await anyio.lowlevel.checkpoint()
        if placing:
            self.placing -= 1
        if self.raise_on == sent.tool:
            raise RuntimeError("transport broke")
        handler = getattr(self, f"_{sent.tool}")
        result: CallOutcome = handler(sent.tool_input)
        return result

    def _ok(self, evidence: MappedEvidence) -> CallOutcome:
        return CallOutcome(ToolCallStatus.SUCCEEDED, evidence)

    def _get_option_quotes(self, tool_input: dict[str, Any]) -> CallOutcome:
        quote = Quote(
            quote_id=uuid.uuid4(),
            broker_instrument_id=tool_input["instrument_ids"][0],
            bid=self.bid,
            ask=self.ask,
            as_of=self.clock() - timedelta(seconds=self.quote_age_seconds),
            source_tool_call_ids=SRC,
        )
        return self._ok(MappedEvidence(option_quotes=(quote,)))

    def _get_portfolio(self, tool_input: dict[str, Any]) -> CallOutcome:
        return self._ok(MappedEvidence())  # the gate (faked) owns the cash check

    def _get_equity_quotes(self, tool_input: dict[str, Any]) -> CallOutcome:
        spot = UnderlyingQuote(**self._obs(), symbol=tool_input["symbols"][0], price=D("80"))
        return self._ok(MappedEvidence(underlying_quotes=(spot,)))

    def _review_option_order(self, tool_input: dict[str, Any]) -> CallOutcome:
        if self.review == "failed":
            return CallOutcome(ToolCallStatus.FAILED, None)
        price = D(tool_input["price"])
        review = OrderReviewObservation(
            **self._obs(),
            legs=(
                OrderLeg(
                    broker_instrument_id="0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
                    side_raw=self._side(tool_input),
                ),
            ),
            quantity=int(tool_input["quantity"]),
            order_type_raw=tool_input["type"],
            time_in_force_raw=tool_input["time_in_force"],
            limit_price=price + D("0.01") if self.review == "mismatch" else price,
            clean=self.review != "alert",
            alert_type="price_far_from_mark" if self.review == "alert" else None,
        )
        return self._ok(MappedEvidence(order_reviews=(review,)))

    def _place_option_order(self, tool_input: dict[str, Any]) -> CallOutcome:
        step = len(self.orders) + 1
        price = D(tool_input["price"]) + self.placed_price_offset
        order = _Order(f"ord-{step}", int(tool_input["quantity"]), price, step)
        self.orders.append(order)
        if self.place == "unknown":
            return CallOutcome(ToolCallStatus.UNKNOWN, None)
        if self.place == "not_sent":
            self.orders.pop()
            return CallOutcome(ToolCallStatus.FAILED, None)
        side = self._side(tool_input)
        return self._ok(MappedEvidence(broker_orders=(self._order_obs(order, side),)))

    def _get_option_orders(self, tool_input: dict[str, Any]) -> CallOutcome:
        order = self._find(tool_input["order_id"])
        order.reads += 1
        if self.on_read is not None:
            self.on_read(self, order)
        if not order.cancel_requested and order.reads == 1 and order.step in self.fill_on_poll:
            order.processed = self.fill_on_poll[order.step]
            order.status = (
                AttemptStatus.FILLED
                if order.processed == order.quantity
                else AttemptStatus.PARTIALLY_FILLED
            )
        if order.cancel_requested and self.cancel_confirms:
            order.status = (
                AttemptStatus.FILLED
                if order.processed == order.quantity
                else AttemptStatus.CANCELLED
            )
        return self._ok(MappedEvidence(broker_orders=(self._order_obs(order, "x"),)))

    def _cancel_option_order(self, tool_input: dict[str, Any]) -> CallOutcome:
        order = self._find(tool_input["order_id"])
        if self.cancel == "unknown":
            return CallOutcome(ToolCallStatus.UNKNOWN, None)
        order.cancel_requested = self.cancel == "ok"
        order.processed += self.fill_on_cancel.get(order.step, 0)
        request = CancelRequestObservation(
            **self._obs(), broker_order_id=order.order_id, accepted=self.cancel == "ok"
        )
        return self._ok(MappedEvidence(cancel_requests=(request,)))

    def tools(self) -> list[str]:
        return [s.tool for s in self.sent]

    def placed(self) -> list[Sent]:
        return [s for s in self.sent if s.tool == "place_option_order"]

    def cancels(self) -> list[Sent]:
        return [s for s in self.sent if s.tool == "cancel_option_order"]


@dataclass
class Rig:
    clock: FakeClock
    gate: FakeGate
    broker: FakeBroker
    runner: OrderWorkRunner
    run_control: RunControl
    events: list[tuple[str, tuple[uuid.UUID, ...]]]


def make_rig(
    partial_fill: PartialFill = PartialFill.CONTINUE,
    refuse: dict[str, str] | None = None,
    instrument: OptionInstrument | None = INSTRUMENT,
    sleep: Callable[[float], Any] | None = None,
    quote_max_age_seconds: int | None = None,
    **broker: Any,
) -> Rig:
    clock = FakeClock()
    gate = FakeGate(refuse)
    fake = FakeBroker(clock=clock, gate=gate, **broker)
    run_control = RunControl()
    events: list[tuple[str, tuple[uuid.UUID, ...]]] = []
    runner = OrderWorkRunner(
        timing=TIMING,
        partial_fill=partial_fill,
        instruments=lambda iid: (
            instrument if iid == "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01" else None
        ),
        run_control=run_control,
        clock=clock,
        events=lambda name, payload, calls: events.append((name, calls)),
        sleep=sleep or clock.sleep,
        quote_max_age_seconds=quote_max_age_seconds,
    )
    runner.bind(gate, fake)
    return Rig(clock, gate, fake, runner, run_control, events)


def admit_start(runner: OrderWorkRunner, job_id: uuid.UUID, request: WorkRequest) -> Any:
    """Reserve, as the hooks do on admission, then start (ADR-0066)."""
    runner.reserve(job_id, request)
    return runner.start(job_id, request)


def work(rig: Rig, request: WorkRequest = STO) -> WalkJob:
    async def main() -> WalkJob:
        async with rig.runner:
            job = admit_start(rig.runner, JOB, request)
            assert isinstance(job, WalkJob), job
            assert rig.runner.active() == (job,)
            await job.done.wait()
        return job

    return anyio.run(main)


def prices(rig: Rig) -> list[Decimal]:
    return [D(s.tool_input["price"]) for s in rig.broker.placed()]


# ---- the request and the plan ---------------------------------------------------------------


def test_refs_round_trip() -> None:
    ref = order_work_ref_for(JOB)
    assert ref == f"order_work:{JOB}" and work_id_of(ref) == JOB
    for bad in (None, 7, "order_work:not-a-uuid", f"work:{JOB}"):
        assert work_id_of(bad) is None


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"quantity": 0}, "quantity"),
        ({"quantity": True}, "quantity"),
        ({"side": "sell_to_close"}, "side"),
        ({"start_price": 1.1}, "decimal strings"),
        ({"start_price": "abc"}, "malformed price"),
        ({"worst_price": "0"}, "positive"),
        ({"worst_price": "NaN"}, "positive"),
        ({"option_id": ""}, "option_id"),
        ({"note": "x"}, "note"),
    ],
)
def test_work_request_validation(override: dict[str, Any], fragment: str) -> None:
    base = {
        "option_id": "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
        "side": "sell_to_open",
        "quantity": 1,
        "start_price": "1.10",
        "worst_price": "1.00",
    }
    issues = parse_work_request({**base, **override})
    assert isinstance(issues, str) and issues.startswith("work_option_order input is invalid")
    assert fragment in issues


def test_work_request_order_input_is_one_limit_day_leg() -> None:
    assert STO.order_input(D("1.05"), 2) == {
        "account_number": "AGENTIC_ACCOUNT",
        "legs": [
            {
                "option_id": "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
                "side": "sell",
                "position_effect": "open",
                "ratio_quantity": 1,
            }
        ],
        "quantity": "2",
        "price": "1.05",
        "type": "limit",
        "time_in_force": "gfd",
        "direction": "credit",
    }
    closing = BTC.order_input(D("0.45"), 1)
    assert closing["legs"][0]["side"] == "buy" and closing["legs"][0]["position_effect"] == "close"
    assert closing["direction"] == "debit"


def test_planned_prices() -> None:
    assert planned_prices(STO, INSTRUMENT, 5) == STO_PRICES
    missing = planned_prices(STO, None, 5)
    assert isinstance(missing, str) and "no validated instrument" in missing
    no_ticks = INSTRUMENT.model_copy(update={"tick_increment": None})
    unknown = planned_prices(STO, no_ticks, 5)
    assert isinstance(unknown, str) and "ticks are unknown" in unknown
    off_tick = STO.model_copy(update={"start_price": D("1.105")})
    bad = planned_prices(off_tick, INSTRUMENT, 5)
    assert isinstance(bad, str) and "tick-valid" in bad
    backwards = STO.model_copy(update={"worst_price": D("1.20")})
    wrong = planned_prices(backwards, INSTRUMENT, 5)
    assert isinstance(wrong, str) and "worst_price" in wrong


def test_tick_schedule_prefers_the_broker_min_ticks() -> None:
    schedule = TickSchedule(above_tick=D("0.05"), below_tick=D("0.01"), cutoff_price=D("3"))
    assert tick_schedule_of(INSTRUMENT.model_copy(update={"tick_schedule": schedule})) == schedule
    uniform = tick_schedule_of(INSTRUMENT)
    assert uniform is not None and uniform.tick_at(D("5")) == D("0.01")


def test_start_refuses_outside_the_task_group_twice_and_without_a_plan() -> None:
    rig = make_rig()
    assert admit_start(rig.runner, JOB, STO) == "the order executor is not running"

    async def main() -> None:
        async with rig.runner:
            unplanned = admit_start(
                rig.runner, uuid.uuid4(), STO.model_copy(update={"option_id": "x"})
            )
            assert isinstance(unplanned, str) and "no validated instrument" in unplanned
            job = admit_start(rig.runner, JOB, STO)
            assert isinstance(job, WalkJob)
            assert (
                admit_start(rig.runner, JOB, STO)
                == "this work_option_order call already started a job"
            )
            await job.done.wait()
            assert await rig.runner.wait(uuid.uuid4(), 1) is None
            assert await rig.runner.wait(JOB, 0) is job

    anyio.run(main)


def test_an_unbound_runner_does_not_start() -> None:
    runner = OrderWorkRunner(
        timing=TIMING,
        partial_fill=PartialFill.CONTINUE,
        instruments=lambda iid: INSTRUMENT,
        run_control=RunControl(),
        clock=FakeClock(),
    )

    async def main() -> None:
        async with runner:
            assert admit_start(runner, JOB, STO) == "the order executor is not running"

    anyio.run(main)


# ---- fills ----------------------------------------------------------------------------------


def test_fill_at_step_one() -> None:
    rig = make_rig(fill_on_poll={1: 2})
    job = work(rig)
    assert job.status is WalkStatus.FILLED and job.filled_quantity == 2
    assert rig.broker.tools() == [
        "get_option_quotes",
        "get_equity_quotes",
        "review_option_order",
        "place_option_order",
        "get_option_orders",
    ]
    assert prices(rig) == [D("1.10")]
    assert rig.broker.cancels() == []
    assert set(rig.gate.job_ids) == {JOB}
    # Reviewed = placed: the exact same arguments.
    review = next(s for s in rig.broker.sent if s.tool == "review_option_order")
    assert review.tool_input == rig.broker.placed()[0].tool_input
    names = [n.removeprefix(f"{job.work_ref}:") for n, _ in rig.events]
    assert names == ["started", "step:1:placed", "step:1:ended", "ended"]
    assert rig.events[-1][1] == (uuid.uuid5(uuid.NAMESPACE_URL, "use-4"),)  # the place call
    assert rig.runner.active() == () and not rig.runner.placement_ended
    view = job.view()
    assert view["status"] == "filled" and view["work_ref"] == f"order_work:{JOB}"
    assert view["planned_prices"] == [str(p) for p in STO_PRICES]
    assert view["steps"][0]["broker_order_id"] == "ord-1"


def test_fill_at_step_k_after_cancels() -> None:
    rig = make_rig(fill_on_poll={3: 2})
    job = work(rig)
    assert job.status is WalkStatus.FILLED and job.filled_quantity == 2
    assert prices(rig) == list(STO_PRICES[:3])
    assert [c.tool_input["order_id"] for c in rig.broker.cancels()] == ["ord-1", "ord-2"]
    # Never a next step while a cancel is pending: each cancel is confirmed by a read first.
    tools = rig.broker.tools()
    for index, tool in enumerate(tools):
        if tool == "cancel_option_order":
            assert tools[index + 1] == "get_option_orders"
    assert all(not c.after_stop_cancel for c in rig.broker.cancels())
    assert [s.order_status for s in job.steps] == [
        AttemptStatus.CANCELLED,
        AttemptStatus.CANCELLED,
        AttemptStatus.FILLED,
    ]


def test_buy_to_close_walks_up_without_an_underlying_quote() -> None:
    rig = make_rig(bid=D("0.35"), ask=D("0.55"), fill_on_poll={2: 1})
    job = work(rig, BTC)
    assert job.status is WalkStatus.FILLED
    assert "get_equity_quotes" not in rig.broker.tools()
    assert prices(rig) == [D("0.40"), D("0.42")]
    assert rig.broker.placed()[0].tool_input["direction"] == "debit"


def test_partial_fill_continue_walks_the_remainder() -> None:
    rig = make_rig(fill_on_poll={1: 1, 2: 1})
    job = work(rig)
    assert job.status is WalkStatus.FILLED and job.filled_quantity == 2
    assert [s.tool_input["quantity"] for s in rig.broker.placed()] == ["2", "1"]
    assert [s.filled_quantity for s in job.steps] == [1, 1]


def test_partial_fill_stop_ends_the_trade() -> None:
    rig = make_rig(partial_fill=PartialFill.STOP, fill_on_poll={1: 1})
    job = work(rig)
    assert job.status is WalkStatus.PARTIALLY_FILLED and job.filled_quantity == 1
    assert len(rig.broker.placed()) == 1 and len(rig.broker.cancels()) == 1
    assert "partial_fill = stop" in (job.reason or "")


def test_a_fill_during_the_cancel_is_counted() -> None:
    rig = make_rig(fill_on_cancel={1: 2})
    job = work(rig)
    assert job.status is WalkStatus.FILLED and job.filled_quantity == 2
    assert len(rig.broker.placed()) == 1


def test_a_partial_fill_during_the_cancel_shrinks_the_next_step() -> None:
    rig = make_rig(fill_on_cancel={1: 1}, fill_on_poll={2: 1})
    job = work(rig)
    assert job.status is WalkStatus.FILLED
    assert [s.tool_input["quantity"] for s in rig.broker.placed()] == ["2", "1"]


def test_no_fill_ends_cancelled_after_the_last_step() -> None:
    rig = make_rig()
    job = work(rig)
    assert job.status is WalkStatus.CANCELLED and job.filled_quantity == 0
    assert prices(rig) == list(STO_PRICES)
    assert len(rig.broker.cancels()) == 5
    assert rig.clock.now <= job.window_end


def test_window_expiry_ends_cancelled_inside_the_window() -> None:
    """Slow broker calls (3 s each): the window, not the step wait, ends the last step, and
    every placement and cancel is sent inside the window."""
    rig = make_rig(call_seconds=3.0)
    job = work(rig)
    assert job.status is WalkStatus.CANCELLED and job.filled_quantity == 0
    placed, cancels = rig.broker.placed(), rig.broker.cancels()
    assert placed and len(cancels) == len(placed)  # nothing left working
    for sent in (*placed, *cancels):
        assert sent.at is not None and sent.at < job.window_end
    last_place, last_cancel = placed[-1].at, cancels[-1].at
    assert last_place is not None and last_cancel is not None
    assert (last_cancel - last_place).total_seconds() < TIMING.step_wait_seconds  # cut short
    assert all(s.order_status is AttemptStatus.CANCELLED for s in job.steps)


def test_slow_calls_never_place_past_the_window() -> None:
    """Very slow broker calls (8 s each): a step whose reads used up its time is not placed,
    so no placement is sent within poll_seconds + 5 s of the window's end; the last cancel
    follows at once (ADR-0066 implementation notes)."""
    rig = make_rig(call_seconds=8.0)
    job = work(rig)
    assert job.status is WalkStatus.CANCELLED
    placed, cancels = rig.broker.placed(), rig.broker.cancels()
    assert placed and len(cancels) == len(placed)
    latest = job.window_end - timedelta(seconds=TIMING.poll_seconds + 5)
    assert all(s.at is not None and s.at <= latest for s in placed)
    # The last cancel may land past the window, but only by the read in flight at its end.
    last = cancels[-1].at
    assert last is not None and last <= job.window_end + timedelta(seconds=2 * 8.0)


# ---- stops ----------------------------------------------------------------------------------


def test_quote_moved_places_nothing() -> None:
    rig = make_rig(bid=D("1.20"), ask=D("1.30"))
    job = work(rig)
    assert job.status is WalkStatus.CANCELLED and "QUOTE_MOVED" in (job.reason or "")
    assert "review_option_order" not in rig.broker.tools()
    assert rig.broker.placed() == [] and not rig.runner.placement_ended


@pytest.mark.parametrize(
    ("mode", "fragment"),
    [
        ("alert", "raised an alert (price_far_from_mark)"),
        ("mismatch", "limit price"),
        ("failed", "review failed"),
    ],
)
def test_review_alert_or_mismatch_stops(mode: str, fragment: str) -> None:
    rig = make_rig(review=mode)
    job = work(rig)
    assert job.status is WalkStatus.STOPPED and fragment in (job.reason or "")
    assert rig.broker.placed() == [] and not rig.runner.placement_ended


def test_place_unknown_ends_unknown_and_placement_for_the_run() -> None:
    rig = make_rig(place="unknown")
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and rig.runner.placement_ended
    assert len(rig.broker.placed()) == 1  # never retried
    assert rig.broker.cancels() == []
    assert rig.broker.tools()[-1] == "place_option_order"


@pytest.mark.parametrize("mode", ["refused", "unknown"])
def test_cancel_refused_or_unknown_ends_unknown(mode: str) -> None:
    rig = make_rig(cancel=mode)
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and rig.runner.placement_ended
    assert len(rig.broker.placed()) == 1 and len(rig.broker.cancels()) == 1
    # Reads still try to resolve the order; nothing is placed or cancelled after the cancel.
    after = rig.broker.tools()[rig.broker.tools().index("cancel_option_order") + 1 :]
    assert after and set(after) == {"get_option_orders"}


def test_unconfirmed_cancel_ends_unknown() -> None:
    rig = make_rig(cancel_confirms=False)
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and rig.runner.placement_ended
    assert len(rig.broker.placed()) == 1


def test_gate_refusal_of_the_place_stops() -> None:
    rig = make_rig(refuse={"place_option_order": "Pre-trade validation failed; delta"})
    job = work(rig)
    assert job.status is WalkStatus.STOPPED
    assert job.reason == "placement denied: Pre-trade validation failed; delta"
    assert rig.broker.placed() == [] and not rig.runner.placement_ended


def test_gate_refusal_of_the_cancel_ends_unknown() -> None:
    rig = make_rig(refuse={"cancel_option_order": "kill switch engaged"})
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and rig.runner.placement_ended


def test_refused_quote_read_stops() -> None:
    rig = make_rig(refuse={"get_option_quotes": "withheld"})
    job = work(rig)
    assert job.status is WalkStatus.STOPPED and "quote read denied" in (job.reason or "")


def test_no_instrument_starts_no_job() -> None:
    rig = make_rig(instrument=None)

    async def main() -> str | WalkJob:
        async with rig.runner:
            return admit_start(rig.runner, JOB, STO)

    assert isinstance(anyio.run(main), str)  # no plan without the instrument


def test_transport_failure_ends_unknown_and_stops_the_run() -> None:
    rig = make_rig(raise_on="review_option_order")
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and "executor failure" in (job.reason or "")
    assert rig.runner.placement_ended and rig.run_control.stop_requested


def test_stop_latch_sends_exactly_one_cancel_and_places_nothing_more() -> None:
    def stop_on_second_read(broker: FakeBroker, order: _Order) -> None:
        if order.reads == 2 and not order.cancel_requested:
            rig.run_control.request_stop(StopReason.SIGTERM, broker.clock())

    rig = make_rig(on_read=stop_on_second_read)
    job = work(rig)
    assert job.status is WalkStatus.STOPPED and job.reason == "run stop requested"
    assert len(rig.broker.placed()) == 1
    (cancel,) = rig.broker.cancels()
    assert cancel.after_stop_cancel is True
    assert cancel.timeout_seconds == LATCH_CANCEL_SECONDS
    # Only that cancel carried the latch flag; it was confirmed by a read.
    assert [c.after_stop_cancel for c in rig.gate.calls].count(True) == 1
    assert rig.broker.tools()[-1] == "get_option_orders"
    assert job.steps[0].order_status is AttemptStatus.CANCELLED


def test_stop_before_a_step_places_nothing() -> None:
    rig = make_rig()
    rig.run_control.request_stop(StopReason.DEADLINE, T0)
    job = work(rig)
    assert job.status is WalkStatus.STOPPED and rig.broker.sent == []


def test_holds_and_active_jobs_while_a_step_works() -> None:
    seen: list[tuple[bool, int, int, bool]] = []

    def look(broker: FakeBroker, order: _Order) -> None:
        if order.reads == 1:
            seen.append(
                (
                    rig.runner.holds(order.order_id),
                    len(rig.runner.active_jobs()),
                    len(rig.runner.active_jobs(exclude=JOB)),
                    rig.runner.holds("ord-other"),
                )
            )

    rig = make_rig(on_read=look, fill_on_poll={1: 2})
    work(rig)
    assert seen == [(True, 1, 0, False)]
    assert not rig.runner.holds("ord-1") and rig.runner.active_jobs() == ()


def test_active_job_reports_its_remaining_quantity_and_worst_price() -> None:
    seen = []

    def look(broker: FakeBroker, order: _Order) -> None:
        if order.step == 2 and order.reads == 1:
            seen.extend(rig.runner.active_jobs())

    rig = make_rig(on_read=look, fill_on_poll={1: 1, 2: 1})
    work(rig)
    (active,) = seen
    assert (
        active.job_id == JOB
        and active.option_id == "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01"
        and not active.closing
    )
    assert active.remaining_quantity == 1 and active.worst_price == D("1.00")


def test_an_event_sink_failure_stops_the_run_but_the_job_ends() -> None:
    clock = FakeClock()
    gate = FakeGate()
    broker = FakeBroker(clock=clock, gate=gate, fill_on_poll={1: 2})
    run_control = RunControl()

    def broken(name: str, payload: Any, calls: Any) -> None:
        raise RuntimeError("ledger down")

    runner = OrderWorkRunner(
        timing=TIMING,
        partial_fill=PartialFill.CONTINUE,
        instruments=lambda iid: INSTRUMENT,
        run_control=run_control,
        clock=clock,
        events=broken,
        sleep=clock.sleep,
    )
    runner.bind(gate, broker)
    rig = Rig(clock, gate, broker, runner, run_control, [])
    job = work(rig)
    assert run_control.stop_requested
    assert job.done.is_set() and job.status is not WalkStatus.WORKING
    assert broker.placed() == []  # the latch was set at start: nothing placed


def test_a_frozen_clock_still_ends_each_wait_and_the_cancel_confirmation() -> None:
    """Reads are bounded by count as well as by the clock: a sleep that never advances the
    clock cannot keep a step (or an unconfirmed cancel) polling forever."""

    async def frozen(seconds: float) -> None:
        await anyio.lowlevel.checkpoint()

    rig = make_rig(cancel_confirms=False, sleep=frozen)
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and rig.runner.placement_ended
    tools = rig.broker.tools()
    cancel_at = tools.index("cancel_option_order")
    wait_reads = tools[tools.index("place_option_order") + 1 : cancel_at]
    assert wait_reads == ["get_option_orders"] * 5  # ceil(step_wait 50 / poll 10)
    assert tools[cancel_at + 1 :] == ["get_option_orders"] * 4  # ceil(30 / 10) + 1


# ---- hardening (review of 2026-10-01) -------------------------------------------------------


def test_an_admitted_call_counts_as_working_until_its_job_starts() -> None:
    """Two work calls sent at once: the first one's reservation is visible to the second's
    concurrency check; a start without its admission, or with other arguments, is refused."""
    rig = make_rig()
    rig.runner.reserve(JOB, STO)
    (held,) = rig.runner.active_jobs()
    assert (held.job_id, held.option_id, held.closing) == (
        JOB,
        "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
        False,
    )
    assert rig.runner.pending() and rig.runner.active_jobs(exclude=JOB) == ()
    rig.runner.release(JOB)
    assert not rig.runner.pending() and rig.runner.active_jobs() == ()

    async def main() -> tuple[Any, Any]:
        async with rig.runner:
            unadmitted = rig.runner.start(JOB, STO)
            rig.runner.reserve(JOB, STO)
            changed = rig.runner.start(JOB, STO.model_copy(update={"quantity": 1}))
            return unadmitted, changed

    unadmitted, changed = anyio.run(main)
    assert "not admitted" in unadmitted and "not admitted" in changed


def test_a_transport_failure_after_a_placement_cancels_the_working_order_once() -> None:
    """The order is at the broker when the transport breaks on the first read: the job ends
    unknown, stops the run, and sends its one bounded after-stop cancel."""
    rig = make_rig(raise_on="get_option_orders")
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and rig.run_control.stop_requested
    (cancel,) = rig.broker.cancels()
    assert cancel.after_stop_cancel and cancel.timeout_seconds == 30.0
    assert job.steps[-1].cancel_tool_call_id is not None


def test_a_cancelled_executor_cancels_its_working_step_and_ends_the_job() -> None:
    """The session's task group is cancelled mid-step: the shielded salvage sends one cancel,
    the job ends unknown with `done` set, and the cancellation propagates."""
    rig = make_rig()

    async def main() -> WalkJob:
        with anyio.CancelScope() as scope:
            async with rig.runner:
                job = admit_start(rig.runner, JOB, STO)
                assert isinstance(job, WalkJob)
                while not rig.broker.placed():
                    await anyio.lowlevel.checkpoint()
                scope.cancel()
        return job

    job = anyio.run(main)
    assert job.done.is_set() and job.status is WalkStatus.UNKNOWN
    assert rig.runner.placement_ended
    (cancel,) = rig.broker.cancels()
    assert cancel.after_stop_cancel


def test_the_latch_falling_before_the_cancel_is_sent_still_cancels_once() -> None:
    """The gate refuses the plain cancel because the stop latch fell in between: nothing was
    sent, so the job sends its one after-stop cancel instead of leaving the order working."""
    rig = make_rig()
    gate = rig.gate
    plain_admit = gate.admit

    def admit(tool: str, tool_input: dict[str, Any], **kw: Any) -> Admitted | Refused:
        if tool == "cancel_option_order" and not kw.get("after_stop_cancel"):
            rig.run_control.request_stop(StopReason.SIGTERM, rig.clock())
            gate.calls.append(Sent(tool, dict(tool_input), False))
            return Refused("run stop requested")
        return plain_admit(tool, tool_input, **kw)

    gate.admit = admit  # type: ignore[method-assign]
    job = work(rig)
    (cancel,) = rig.broker.cancels()  # only the after-stop one reached the broker
    assert cancel.after_stop_cancel and cancel.timeout_seconds == 30.0
    assert job.status is WalkStatus.STOPPED
    assert job.steps[-1].order_status is AttemptStatus.CANCELLED


def test_a_placement_that_was_never_sent_stops_without_unknown() -> None:
    rig = make_rig(place="not_sent")
    job = work(rig)
    assert job.status is WalkStatus.STOPPED and not rig.runner.placement_ended
    assert job.steps == []


def test_a_filled_order_with_a_short_filled_quantity_is_unknown() -> None:
    """A read that says filled but reports fewer contracts processed must not let the walk
    place again: the filled quantity is inconsistent, so the job fails closed."""

    def bad(broker: FakeBroker, order: Any) -> None:
        order.status, order.processed = AttemptStatus.FILLED, 0

    rig = make_rig(on_read=bad)
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and rig.runner.placement_ended
    assert len(rig.broker.placed()) == 1


# ---- review findings (2026-10-01, second pass) ----------------------------------------------


def test_an_unknown_outcome_in_one_job_stops_every_other_job_from_placing() -> None:
    """CLAUDE.md §14: once any order action of the run is unknown, placement ends for all."""

    def flag(broker: FakeBroker, order: Any) -> None:
        rig.runner.placement_ended = True  # another job's place just went unknown

    rig = make_rig(on_read=flag)
    job = work(rig)
    assert job.status is WalkStatus.STOPPED and "placement has ended" in (job.reason or "")
    assert len(rig.broker.placed()) == 1  # its working step was cancelled, nothing more placed
    assert len(rig.broker.cancels()) == 1


def test_a_failure_after_the_cancel_was_sent_never_sends_a_second_cancel() -> None:
    rig = make_rig(raise_on="cancel_option_order")
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN
    assert len(rig.broker.cancels()) == 1  # the salvage saw it was already sent


def test_a_refused_cancel_on_a_filled_order_counts_the_fill_and_ends_placement() -> None:
    """The order filled just as its step's wait ended: the cancel is refused, the reads show
    it filled, the fill counts, and placement for the run still ends (§14)."""

    def fill_after_cancel(broker: FakeBroker, order: Any) -> None:
        if broker.cancels():
            order.status, order.processed = AttemptStatus.FILLED, order.quantity

    rig = make_rig(cancel="refused", on_read=fill_after_cancel)
    job = work(rig)
    assert job.status is WalkStatus.FILLED and job.filled_quantity == STO.quantity
    assert rig.runner.placement_ended
    assert len(rig.broker.placed()) == 1


def test_a_placed_order_that_differs_from_the_request_is_cancelled_and_unknown() -> None:
    rig = make_rig(placed_price_offset=D("0.05"))
    job = work(rig)
    assert job.status is WalkStatus.UNKNOWN and "limit price" in (job.reason or "")
    assert len(rig.broker.placed()) == 1 and len(rig.broker.cancels()) == 1
    assert rig.runner.placement_ended


def test_a_stale_quote_stops_before_anything_is_sent() -> None:
    rig = make_rig(quote_max_age_seconds=60, quote_age_seconds=61)
    job = work(rig)
    assert job.status is WalkStatus.STOPPED and "61s old" in (job.reason or "")
    assert rig.broker.placed() == [] and "review_option_order" not in rig.broker.tools()


def test_option_ids_are_canonical_and_held_orders_match_in_any_case() -> None:
    upper = WorkRequest.model_validate(
        {**STO.model_dump(mode="json"), "option_id": STO.option_id.upper()}
    )
    assert upper.option_id == STO.option_id
    with pytest.raises(ValidationError):
        WorkRequest.model_validate({**STO.model_dump(mode="json"), "option_id": "inst-1"})


def test_executor_placements_never_overlap() -> None:
    """Two concurrent jobs: their placements are serialized (ADR-0051 counts placements in
    flight), so neither is refused as "another placement in flight"."""
    rig = make_rig(fill_on_poll={1: STO.quantity, 2: STO.quantity})
    other = uuid.UUID(int=7001)

    async def main() -> tuple[WalkJob, WalkJob]:
        async with rig.runner:
            first = admit_start(rig.runner, JOB, STO)
            second = admit_start(rig.runner, other, STO)
            assert isinstance(first, WalkJob) and isinstance(second, WalkJob)
            await first.done.wait()
            await second.done.wait()
            return first, second

    first, second = anyio.run(main)
    assert rig.broker.max_placing == 1
    assert first.status is WalkStatus.FILLED and second.status is WalkStatus.FILLED
