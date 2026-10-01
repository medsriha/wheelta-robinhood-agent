"""The in-process `wra_orders` MCP server: `work_option_order` and `await_order_work` (ADR-0066).

Served like a validating proxy (ADR-0023): the PreToolUse hook admits and records each call
and registers it in the session's `ProxyDispatch` under its SDK `tool_use_id`; this server
claims it by `_meta["claudecode/toolUseId"]` (sent by the pinned CLI on every MCP
`tools/call`), refuses a call whose name or arguments differ from the dispatched ones, records
the result and the outcome, and completes the call with the exact output it returns, so
PostToolUse records the delivery as for any proxied call.

- `work_option_order` starts one order-walk job (`OrderWorkRunner.start`) and returns the
  job's view with its `work_ref` (`order_work:<this call's tool_call_id>`). A job that cannot
  start returns an error envelope, recorded `failed`: nothing was sent to the broker.
- `await_order_work` waits up to `MAX_AWAIT_SECONDS` for the job to end and returns its view.

The server exists only in a run with an order venue (`session.build_session_options`), so the
model never sees these tools otherwise (CLAUDE.md §8 layer 1).
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Final

import mcp_types as types
from mcp.server.lowlevel import Server

from wheelta_robinhood_agent.agent.hooks import EnvelopeKind, ResultEnvelope, mcp_tool_output
from wheelta_robinhood_agent.agent.order_walk import (
    AWAIT_TOOL,
    MAX_AWAIT_SECONDS,
    ORDER_WORK_SERVER,
    WORK_TOOL,
    OrderWorkRunner,
    parse_work_request,
    work_id_of,
)
from wheelta_robinhood_agent.agent.proxy import TOOL_USE_ID_META
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyCall, ProxyDispatch
from wheelta_robinhood_agent.agent.recorder import ResultKind, ToolEventRecorder
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.domain.enums import ToolCallStatus
from wheelta_robinhood_agent.domain.run import StopReason

ORDER_WORK_SERVER_VERSION: Final = "1"
ORDER_WORK_DEDUP_KEY: Final = "order_work"

_PRICE = {"type": "string", "pattern": r"^\d+(\.\d+)?$"}
WORK_INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["option_id", "side", "quantity", "start_price", "worst_price"],
    "properties": {
        "option_id": {"type": "string", "description": "The contract's Robinhood instrument ID"},
        "side": {"type": "string", "enum": ["sell_to_open", "buy_to_close"]},
        "quantity": {"type": "integer", "minimum": 1, "description": "Contracts to trade"},
        "start_price": {
            **_PRICE,
            "description": "First limit price, tick-valid, within the live [bid, ask]",
        },
        "worst_price": {
            **_PRICE,
            "description": (
                "Last limit price you accept: not above start_price for a sell to open, not "
                "below it for a buy to close; tick-valid"
            ),
        },
    },
}
AWAIT_INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["work_ref"],
    "properties": {
        "work_ref": {"type": "string", "description": "The order_work:... ref of the job"},
        "wait_seconds": {"type": "integer", "minimum": 0, "maximum": MAX_AWAIT_SECONDS},
    },
}
TOOLS: Final = (
    types.Tool(
        name=WORK_TOOL,
        description=(
            "Work one option order inside the order window (orders.walk): code re-quotes, "
            "checks, reviews, places, waits, cancels, and confirms, one price step at a time "
            "from start_price to worst_price. Returns at once with the job's work_ref."
        ),
        input_schema=WORK_INPUT_SCHEMA,
    ),
    types.Tool(
        name=AWAIT_TOOL,
        description=(
            f"Wait up to {MAX_AWAIT_SECONDS}s for an order-work job to end; returns its status, "
            "steps, and filled quantity."
        ),
        input_schema=AWAIT_INPUT_SCHEMA,
    ),
)


class OrderWorkServer:
    """The `tools/call` logic of `wra_orders` (module docstring)."""

    def __init__(
        self,
        *,
        runner: OrderWorkRunner,
        dispatch: ProxyDispatch,
        recorder: ToolEventRecorder,
        run_control: RunControl,
        clock: Callable[[], datetime],
    ) -> None:
        self._runner = runner
        self._dispatch = dispatch
        self._recorder = recorder
        self._run_control = run_control
        self._clock = clock

    def _stop(self) -> None:
        try:
            now = self._clock()
        except Exception:  # noqa: BLE001 - the latch must still be set without the clock
            now = datetime.now(UTC)
        self._run_control.request_stop(StopReason.INFRASTRUCTURE_FAILURE, now)

    def _fallback(self, tool: str, gap: str) -> types.CallToolResult:
        text = json.dumps(
            {
                "data": None,
                "gaps": [gap],
                "kind": EnvelopeKind.ERROR.value,
                "retrieved_at": None,
                "server": ORDER_WORK_SERVER,
                "source_as_of": None,
                "tool": tool,
                "tool_call_id": None,
            },
            sort_keys=True,
        )
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    async def call_tool(self, params: types.CallToolRequestParams) -> types.CallToolResult:
        use_id = (params.meta or {}).get(TOOL_USE_ID_META)
        if not isinstance(use_id, str) or not use_id:
            self._stop()
            return self._fallback(params.name, "call has no tool_use_id")
        call = self._dispatch.claim(use_id)
        if call is None or call.by_executor:
            self._stop()
            return self._fallback(params.name, "no dispatched call for this tool_use_id")
        try:
            output = await self._handle(call, params)
            self._dispatch.complete(use_id, output)
        except Exception as exc:  # noqa: BLE001 - nothing unrecorded may reach the CLI
            self._stop()
            if self._runner.job(call.tool_call_id) is None:
                self._runner.release(call.tool_call_id)
            return self._fallback(params.name, f"order work failure ({type(exc).__name__})")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=b["text"]) for b in output]
        )

    async def _handle(
        self, call: ProxyCall, params: types.CallToolRequestParams
    ) -> list[dict[str, str]]:
        arguments = dict(params.arguments or {})
        if params.name != call.tool or arguments != call.effective_input:
            self._stop()
            self._runner.release(call.tool_call_id)
            return self._record(call, None, "tool or arguments differ from the dispatched call")
        if call.tool == WORK_TOOL:
            request = parse_work_request(arguments)
            if isinstance(request, str):
                return self._record(call, None, request)
            job = self._runner.start(call.tool_call_id, request)
            if isinstance(job, str):
                return self._record(call, None, f"order work not started: {job}")
            return self._record(call, {"order_work": job.view()}, None)
        if call.tool == AWAIT_TOOL:
            job_id = work_id_of(arguments.get("work_ref"))
            wait = arguments.get("wait_seconds", MAX_AWAIT_SECONDS)
            if job_id is None or not isinstance(wait, int) or isinstance(wait, bool):
                return self._record(call, None, "work_ref must be an order_work:... ref")
            awaited = await self._runner.wait(job_id, float(wait))
            if awaited is None:
                return self._record(call, None, "no order-work job with this work_ref")
            return self._record(call, {"order_work": awaited.view()}, None)
        return self._record(call, None, f"unknown tool {call.tool}")

    def _record(
        self, call: ProxyCall, data: dict[str, Any] | None, gap: str | None
    ) -> list[dict[str, str]]:
        """Persist the envelope and the outcome; the output the CLI receives."""
        now = self._clock()
        valid = gap is None
        envelope = ResultEnvelope(
            tool_call_id=call.tool_call_id,
            server=ORDER_WORK_SERVER,
            tool=call.tool,
            kind=EnvelopeKind.VALIDATED if valid else EnvelopeKind.ERROR,
            data=data,
            gaps=() if gap is None else (gap,),
            retrieved_at=now,
        )
        payload = envelope.model_dump(mode="json")
        ref = self._recorder.store_result(
            call.tool_call_id, ResultKind.VALIDATED if valid else ResultKind.ERROR, payload
        )
        if valid:
            self._recorder.outcome(
                call.tool_call_id,
                ToolCallStatus.SUCCEEDED,
                observed_at=now,
                dedup_key=ORDER_WORK_DEDUP_KEY,
                result_ref=ref,
            )
        else:
            # Nothing reached the broker: the outcome is known.
            self._recorder.outcome(
                call.tool_call_id,
                ToolCallStatus.FAILED,
                observed_at=now,
                dedup_key=ORDER_WORK_DEDUP_KEY,
                reason=gap,
                error_ref=ref,
            )
        return mcp_tool_output(payload)


def build_order_work_server(server: OrderWorkServer) -> Server[Any]:
    """The in-process MCP server serving both order-work tools."""

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(TOOLS))

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return await server.call_tool(params)

    return Server(
        ORDER_WORK_SERVER,
        version=ORDER_WORK_SERVER_VERSION,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


__all__ = [
    "AWAIT_INPUT_SCHEMA",
    "TOOLS",
    "WORK_INPUT_SCHEMA",
    "OrderWorkServer",
    "build_order_work_server",
]
