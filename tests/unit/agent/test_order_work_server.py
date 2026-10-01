"""The in-process `wra_orders` server (ADR-0066, agent/order_work_server.py).

The hooks' dispatch registration is built by hand; the runner is the real one over the fake
gate and broker of test_order_walk.py. No network, no database.
"""

import uuid
from typing import Any

import anyio
import mcp_types as types
from test_hooks import NOW, FakeRecorder, wire
from test_order_walk import JOB, Rig, make_rig

from wheelta_robinhood_agent.agent.order_walk import ORDER_WORK_SERVER, parse_work_request
from wheelta_robinhood_agent.agent.order_work_server import (
    ORDER_WORK_DEDUP_KEY,
    TOOLS,
    OrderWorkServer,
    build_order_work_server,
)
from wheelta_robinhood_agent.agent.proxy import TOOL_USE_ID_META
from wheelta_robinhood_agent.agent.proxy_dispatch import CallState, ProxyCall, ProxyDispatch
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier

WORK_ARGS: dict[str, Any] = {
    "option_id": "0e1b5c3a-9d2f-4b7a-8c61-2f4e6a8b9c01",
    "side": "sell_to_open",
    "quantity": 2,
    "start_price": "1.10",
    "worst_price": "1.00",
}


class ServerRig:
    def __init__(self, rig: Rig | None = None) -> None:
        self.walk = rig or make_rig(fill_on_poll={1: 2})
        self.dispatch = ProxyDispatch(frozenset({ORDER_WORK_SERVER}))
        self.recorder = FakeRecorder()
        self.run_control = RunControl()
        self.server = OrderWorkServer(
            runner=self.walk.runner,
            dispatch=self.dispatch,
            recorder=self.recorder,  # type: ignore[arg-type]
            run_control=self.run_control,
            clock=lambda: NOW,
        )

    def register(
        self,
        tool: str,
        args: dict[str, Any],
        use_id: str = "toolu_1",
        tool_call_id: uuid.UUID = JOB,
        by_executor: bool = False,
    ) -> None:
        request = parse_work_request(args)
        if tool == "work_option_order" and not by_executor and not isinstance(request, str):
            self.walk.runner.reserve(tool_call_id, request)  # as PreToolUse does on admission
        self.dispatch.register(
            use_id,
            ProxyCall(
                tool_call_id=tool_call_id,
                server=ORDER_WORK_SERVER,
                tool=tool,
                tier=ToolTier.X if tool == "work_option_order" else ToolTier.R,
                effective_input=args,
                by_executor=by_executor,
            ),
        )

    async def call(self, tool: str, args: dict[str, Any], use_id: Any = "toolu_1") -> Any:
        params = types.CallToolRequestParams.model_validate(
            {"name": tool, "arguments": args}
            | ({"_meta": {TOOL_USE_ID_META: use_id}} if use_id is not None else {})
        )
        result = await self.server.call_tool(params)
        assert result.is_error is False
        return wire([{"type": "text", "text": c.text} for c in result.content])  # type: ignore[union-attr]

    def outcomes(self) -> list[dict[str, Any]]:
        return [kw for n, kw in self.recorder.events if n == "outcome"]


def run(fn: Any) -> Any:
    return anyio.run(fn)


def test_the_server_lists_both_tools() -> None:
    assert {t.name for t in TOOLS} == {"work_option_order", "await_order_work"}
    assert build_order_work_server(ServerRig().server) is not None


def test_work_starts_a_job_and_records_the_outcome() -> None:
    r = ServerRig()

    async def main() -> Any:
        async with r.walk.runner:
            r.register("work_option_order", WORK_ARGS)
            envelope = await r.call("work_option_order", WORK_ARGS)
            job = r.walk.runner.job(JOB)
            assert job is not None
            await job.done.wait()
            return envelope

    envelope = run(main)
    assert envelope["kind"] == "validated" and envelope["server"] == ORDER_WORK_SERVER
    view = envelope["data"]["order_work"]
    assert view["work_ref"] == f"order_work:{JOB}" and view["status"] == "working"
    (outcome,) = r.outcomes()
    assert outcome["status"] is ToolCallStatus.SUCCEEDED
    assert outcome["dedup_key"] == ORDER_WORK_DEDUP_KEY and outcome["result_ref"] is not None
    assert r.dispatch.state("toolu_1") is CallState.COMPLETED
    assert wire(r.dispatch.delivered("toolu_1")) == envelope  # PostToolUse checks this output
    assert not r.run_control.stop_requested


def test_a_call_is_claimed_only_once() -> None:
    r = ServerRig()

    async def main() -> tuple[Any, Any]:
        async with r.walk.runner:
            r.register("work_option_order", WORK_ARGS)
            first = await r.call("work_option_order", WORK_ARGS)
            replay = await r.call("work_option_order", WORK_ARGS)
            job = r.walk.runner.job(JOB)
            assert job is not None
            await job.done.wait()
            return first, replay

    first, replay = run(main)
    assert first["kind"] == "validated"
    assert replay["kind"] == "error" and replay["tool_call_id"] is None
    assert "no dispatched call" in replay["gaps"][0]
    assert r.run_control.stop_requested
    assert len(r.walk.broker.placed()) == 1  # one job, one walk


def test_a_call_without_tool_use_id_is_refused_and_stops() -> None:
    r = ServerRig()
    r.register("work_option_order", WORK_ARGS)
    envelope = run(lambda: r.call("work_option_order", WORK_ARGS, use_id=None))
    assert envelope["kind"] == "error" and "no tool_use_id" in envelope["gaps"][0]
    assert r.run_control.stop_requested
    assert r.dispatch.state("toolu_1") is CallState.PENDING  # never claimed


def test_an_executor_registration_is_never_served_to_the_cli() -> None:
    r = ServerRig()
    r.register("work_option_order", WORK_ARGS, by_executor=True)
    envelope = run(lambda: r.call("work_option_order", WORK_ARGS))
    assert envelope["kind"] == "error" and r.run_control.stop_requested
    assert r.outcomes() == []


def test_arguments_differing_from_the_dispatch_stop() -> None:
    r = ServerRig()
    r.register("work_option_order", WORK_ARGS)
    envelope = run(lambda: r.call("work_option_order", {**WORK_ARGS, "quantity": 9}))
    assert envelope["kind"] == "error" and "differ" in envelope["gaps"][0]
    assert r.run_control.stop_requested
    (outcome,) = r.outcomes()
    assert outcome["status"] is ToolCallStatus.FAILED and "differ" in outcome["reason"]
    assert r.walk.broker.sent == []


def test_a_job_that_cannot_start_is_a_failed_call() -> None:
    r = ServerRig()  # the runner's task group is not entered
    r.register("work_option_order", WORK_ARGS)
    envelope = run(lambda: r.call("work_option_order", WORK_ARGS))
    assert envelope["kind"] == "error"
    assert envelope["gaps"] == ["order work not started: the order executor is not running"]
    (outcome,) = r.outcomes()
    assert outcome["status"] is ToolCallStatus.FAILED and outcome["error_ref"] is not None
    assert not r.run_control.stop_requested  # nothing reached the broker


def test_an_invalid_request_is_a_failed_call() -> None:
    r = ServerRig()
    bad = {**WORK_ARGS, "quantity": 0}
    r.register("work_option_order", bad)
    envelope = run(lambda: r.call("work_option_order", bad))
    assert envelope["kind"] == "error" and "input is invalid" in envelope["gaps"][0]
    assert r.outcomes()[0]["status"] is ToolCallStatus.FAILED


def test_await_returns_the_job_view() -> None:
    r = ServerRig()
    ref = {"work_ref": f"order_work:{JOB}", "wait_seconds": 45}

    async def main() -> Any:
        async with r.walk.runner:
            r.register("work_option_order", WORK_ARGS)
            await r.call("work_option_order", WORK_ARGS)
            r.register("await_order_work", ref, use_id="toolu_2", tool_call_id=uuid.uuid4())
            return await r.call("await_order_work", ref, use_id="toolu_2")

    envelope = run(main)
    assert envelope["kind"] == "validated"
    assert envelope["data"]["order_work"]["status"] == "filled"
    assert envelope["data"]["order_work"]["filled_quantity"] == 2


def test_await_with_an_unknown_or_malformed_ref_fails() -> None:
    r = ServerRig()
    unknown = {"work_ref": f"order_work:{uuid.uuid4()}", "wait_seconds": 1}
    malformed = {"work_ref": "ord-1"}
    r.register("await_order_work", unknown, use_id="u1", tool_call_id=uuid.uuid4())
    r.register("await_order_work", malformed, use_id="u2", tool_call_id=uuid.uuid4())
    first = run(lambda: r.call("await_order_work", unknown, use_id="u1"))
    second = run(lambda: r.call("await_order_work", malformed, use_id="u2"))
    assert first["kind"] == "error" and "no order-work job" in first["gaps"][0]
    assert second["kind"] == "error" and "order_work:" in second["gaps"][0]
    assert [o["status"] for o in r.outcomes()] == [ToolCallStatus.FAILED] * 2
    assert not r.run_control.stop_requested


def test_a_recording_failure_returns_a_fallback_and_stops() -> None:
    r = ServerRig()
    r.recorder = FakeRecorder(frozenset({"store_error"}))
    r.server = OrderWorkServer(
        runner=r.walk.runner,
        dispatch=r.dispatch,
        recorder=r.recorder,  # type: ignore[arg-type]
        run_control=r.run_control,
        clock=lambda: NOW,
    )
    r.register("work_option_order", WORK_ARGS)
    envelope = run(lambda: r.call("work_option_order", WORK_ARGS))
    assert envelope["kind"] == "error" and "order work failure" in envelope["gaps"][0]
    assert r.run_control.stop_requested
