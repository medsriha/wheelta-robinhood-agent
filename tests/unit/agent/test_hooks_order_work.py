"""PreToolUse admission of the order-work tools and job-held cancels (ADR-0066).

Same fakes as test_hooks.py; no network, no database.
"""

import dataclasses
import uuid
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pytest
from test_hooks import (
    ACCOUNT,
    NOW,
    RH,
    Session,
    assert_allowed,
    assert_denied,
    executor_session,
)

from wheelta_robinhood_agent.agent.hooks import (
    JOB_HELD_CANCEL_DENIAL,
    NO_ORDER_WORK,
    PLACEMENT_ENDED_DENIAL,
)
from wheelta_robinhood_agent.agent.order_walk import (
    ORDER_WORK_REGISTRY,
    QUALIFIED_AWAIT_TOOL,
    QUALIFIED_WORK_TOOL,
    WorkRequest,
)
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyDispatch
from wheelta_robinhood_agent.domain.enums import ExecutionMode
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

WORK_ARGS: dict[str, Any] = {
    "option_id": "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
    "side": "sell_to_open",
    "quantity": 2,
    "start_price": "1.10",
    "worst_price": "1.00",
}
WINDOW = 300  # rules v17 orders.walk.window_seconds
WIND_DOWN = 180.0


@dataclasses.dataclass
class FakeOrderWork:
    placement_ended: bool = False
    held: frozenset[str] = frozenset()
    placing_now: bool = False
    planned: tuple[Decimal, ...] | str = (Decimal("1.10"), Decimal("1.00"))
    plans: list[WorkRequest] = dataclasses.field(default_factory=list)
    reserved: dict[uuid.UUID, WorkRequest] = dataclasses.field(default_factory=dict)

    def holds(self, broker_order_id: str) -> bool:
        return broker_order_id in self.held

    def placing(self) -> bool:
        return self.placing_now

    def plan(self, request: WorkRequest) -> tuple[Any, ...] | str:
        self.plans.append(request)
        return self.planned

    def reserve(self, job_id: uuid.UUID, request: WorkRequest) -> None:
        self.reserved[job_id] = request

    def release(self, job_id: uuid.UUID) -> None:
        self.reserved.pop(job_id, None)


class PretradeSpy:
    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        self.calls: list[tuple[Mapping[str, object], dict[str, Any]]] = []

    def __call__(self, tool_input: Mapping[str, object], **kw: Any) -> str | None:
        self.calls.append((dict(tool_input), kw))
        return self.reason


def work_session(**overrides: Any) -> Session:
    base: dict[str, Any] = {
        "registries": (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY, ORDER_WORK_REGISTRY),
        "proxy_dispatch": ProxyDispatch(frozenset({"robinhood", "wra_orders"})),
        "order_work": FakeOrderWork(),
        "session_remaining": lambda: 1000.0,
        "wind_down_seconds": WIND_DOWN,
        "pretrade_gate": PretradeSpy(),
    }
    return executor_session(**{**base, **overrides})


def test_work_is_admitted_and_handed_to_the_order_work_server() -> None:
    spy = PretradeSpy()
    s = work_session(pretrade_gate=spy)
    assert_allowed(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS))
    assert s.deps.proxy_dispatch is not None
    call = s.deps.proxy_dispatch.claim("toolu_1")
    assert call is not None and call.server == "wra_orders" and call.tool == "work_option_order"
    assert not call.by_executor and call.effective_input == WORK_ARGS
    # Pre-trade validation at start_price for the full quantity; the admission is not itself
    # a place call in flight.
    ((order, kw),) = spy.calls
    assert kw == {"self_in_flight": False}
    assert order["price"] == "1.10" and order["quantity"] == "2"
    assert order["legs"] == [
        {
            "option_id": "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
            "side": "sell",
            "position_effect": "open",
            "ratio_quantity": 1,
        }
    ]


def test_no_runner_denies_work() -> None:
    s = work_session(order_work=None)
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), NO_ORDER_WORK)
    s = work_session(session_remaining=None)
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), NO_ORDER_WORK)


def test_placement_ended_denies_new_work() -> None:
    s = work_session(order_work=FakeOrderWork(placement_ended=True))
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), PLACEMENT_ENDED_DENIAL)


@pytest.mark.parametrize(
    ("remaining", "allowed"),
    [(WINDOW + WIND_DOWN - 1, False), (WINDOW + WIND_DOWN, True), (WINDOW + WIND_DOWN + 1, True)],
)
def test_no_late_start(remaining: float, allowed: bool) -> None:
    """ADR-0066 item 6: a window starts only if it and the wind-down still fit."""
    s = work_session(session_remaining=lambda: remaining)
    out = s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS)
    if allowed:
        assert_allowed(s, out)
    else:
        assert_denied(s, out, "too little session time")


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"quantity": 0}, "quantity"),
        ({"quantity": "2"}, "quantity"),
        ({"side": "buy_to_open"}, "side"),
        ({"start_price": 1.1}, "decimal strings"),
        ({"worst_price": "-1"}, "positive"),
        ({"option_id": ""}, "option_id"),
        ({"extra": 1}, "extra"),
    ],
)
def test_invalid_request_denied_with_its_issues(override: dict[str, Any], fragment: str) -> None:
    s = work_session()
    out = s.pre(QUALIFIED_WORK_TOOL, {**WORK_ARGS, **override})
    assert_denied(s, out, "work_option_order input is invalid")
    assert_denied(s, out, fragment)


def test_planned_price_failure_denied_after_pretrade_passes() -> None:
    spy = PretradeSpy()
    view = FakeOrderWork(planned="start_price 1.13 is not a positive tick-valid price")
    s = work_session(order_work=view, pretrade_gate=spy)
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), "not a positive tick-valid price")
    assert len(spy.calls) == 1
    assert view.plans and view.plans[0].option_id == "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01"


def test_pretrade_reason_wins_over_a_plan_failure() -> None:
    """The trade may not be made at all: the agent gets the gate's reason, and no plan."""
    view = FakeOrderWork(planned="start_price 1.13 is not a positive tick-valid price")
    s = work_session(order_work=view, pretrade_gate=PretradeSpy("Pre-trade validation failed"))
    out = s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS)
    assert_denied(s, out, "Pre-trade validation failed")
    assert "tick-valid" not in str(out)
    assert view.plans == []


def test_pretrade_denial_of_work_is_feedback() -> None:
    s = work_session(pretrade_gate=PretradeSpy("Pre-trade validation failed; delta"))
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), "Pre-trade validation failed")
    assert not s.deps.run_control.stop_requested


def test_no_pretrade_gate_denies_work() -> None:
    s = work_session(pretrade_gate=None)
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), "pre-trade validation")


def test_work_denied_without_a_venue_or_the_robinhood_proxy() -> None:
    s = work_session(effective_mode=ExecutionMode.OFF)
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), "not available in this run")
    s = work_session(proxy_dispatch=ProxyDispatch(frozenset({"wra_orders"})))
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), "validating proxy")


def test_work_denied_after_the_latch_and_by_the_kill_switch() -> None:
    s = work_session()
    s.deps.run_control.request_stop(StopReason.SIGTERM, NOW)
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), "run stop requested")
    s = work_session(kill_switch=True)
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS), "kill switch engaged")


def test_work_denied_to_a_mignon() -> None:
    from test_hooks import MARKET

    s = work_session()
    assert_denied(s, s.pre(QUALIFIED_WORK_TOOL, WORK_ARGS, **MARKET), "mignon-market")


def test_await_is_a_read_allowed_after_the_latch() -> None:
    s = work_session()
    s.deps.run_control.request_stop(StopReason.DEADLINE, NOW)
    assert_allowed(s, s.pre(QUALIFIED_AWAIT_TOOL, {"work_ref": "order_work:x"}))


def test_model_cancel_of_a_job_held_order_denied() -> None:
    view = FakeOrderWork(held=frozenset({"ord-held"}))
    s = work_session(order_work=view)
    cancel = RH + "cancel_option_order"
    out = s.pre(cancel, {"account_number": ACCOUNT, "order_id": "ord-held"})
    assert_denied(s, out, JOB_HELD_CANCEL_DENIAL)
    s = work_session(order_work=view)
    assert_allowed(s, s.pre(cancel, {"account_number": ACCOUNT, "order_id": "ord-leftover"}))


def test_executor_cancel_of_its_own_held_order_allowed() -> None:
    from test_hooks import CANCEL_ARGS, admit, assert_admitted

    s = work_session(order_work=FakeOrderWork(held=frozenset({"ord-1"})))
    assert_admitted(s, admit(s, "cancel_option_order", CANCEL_ARGS))
