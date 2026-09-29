"""The validating proxy: remote MCP tools served in-process (ADR-0023; DATA_QUALITY.md).

The real-CLI acceptance tests showed that PostToolUse hooks alone cannot keep raw remote
output from the model: MCP `isError` results, transport errors, oversized results (sent to
`count_tokens` before the hook), and a hook exception or timeout all let the CLI forward the
raw result. The proxy removes the raw result from the CLI entirely. For each proxied server
(`robinhood`, `wheelta`) the session carries an in-process `mcp.server.lowlevel.Server` under
the same name, so qualified tool names, registries, tool access, and the PreToolUse hook are
unchanged. Its `tools/call` handler:

1. Correlates the call with its PreToolUse record through `_meta["claudecode/toolUseId"]`,
   which the pinned CLI sends on every MCP `tools/call` (verified in the bundled CLI 2.1.283
   and by the real-CLI tests). A call without a matching, dispatched PreToolUse record, or
   whose name or arguments differ from the dispatched ones, is never forwarded.
2. Forwards the dispatched arguments (with the account placeholder replaced by the configured
   number when the hook resolved one, ADR-0030) through `integrations/mcp_upstream.py` within
   `upstream_timeout_seconds` (below the CLI's `MCP_TOOL_TIMEOUT`, so the proxy answers
   first). No retry, in any tier (CLAUDE.md §14).
3. Validates the result with the injected `ResultValidator` (the same `BoundaryValidator`
   used for direct delivery), records `raw_invalid` / `validated` or `error` and the
   outcome, and returns the envelope's model view (`model_view`, ADR-0037) as one text block
   (`mcp_tool_output`). The ledger keeps the full envelope. A view longer than
   `MAX_DELIVERED_CHARS` is not delivered: the CLI would rewrite it and the delivery check
   would stop the run. It becomes an `error` envelope naming its size, and
   is not recorded as validated evidence, because the model never saw it.

Whatever happens, the CLI receives only text our code produced: an upstream failure becomes
an `error` envelope with a fixed gap text, and any exception in the handler becomes a static
fallback envelope built without the clock or the failing dependency. A later hook failure or
timeout can therefore only fall back to our envelope. PostToolUse (hooks.py) records the
delivery of the envelope; it no longer validates proxied results.

Tier S/X calls whose result is unusable are `unknown`, as in the hooks. A cancelled handler
(CLI interrupt) propagates cancellation; `session.close_unresolved_calls` records `unknown`.
"""

import json
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import mcp_types as types
from mcp.server.lowlevel import Server

from wheelta_robinhood_agent.agent.hooks import (
    EnvelopeKind,
    ResultEnvelope,
    ResultValidator,
    ValidationRequest,
    mcp_tool_output,
)
from wheelta_robinhood_agent.agent.model_view import model_view
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyCall, ProxyDispatch
from wheelta_robinhood_agent.agent.recorder import ResultKind, ToolEventRecorder
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.mcp_upstream import McpUpstream, UpstreamError
from wheelta_robinhood_agent.integrations.registry import ToolRegistry

TOOL_USE_ID_META: Final = "claudecode/toolUseId"
# The proxy's upstream deadline is the CLI's per-call timeout minus this margin, leaving time
# to validate and record before the CLI gives up on the call. Not a trading value.
PROXY_TIMEOUT_MARGIN_SECONDS: Final = 10.0
PROXY_SERVER_VERSION: Final = "1"
PROXY_DEDUP_KEY: Final = "proxy"
# The CLI rewrites MCP output above its token limit (MAX_MCP_OUTPUT_TOKENS, default 25,000),
# and the PostToolUse delivery check then stops the run. Dry runs on 2026-09-28 hit this with
# 102 KB and 57 KB scan envelopes: JSON full of UUIDs and long decimals runs near 2 characters
# per token, so this stays under the limit at that density.
MAX_DELIVERED_CHARS: Final = 30_000

__all__ = [
    "MAX_DELIVERED_CHARS",
    "PROXY_DEDUP_KEY",
    "PROXY_TIMEOUT_MARGIN_SECONDS",
    "TOOL_USE_ID_META",
    "ValidatingProxy",
    "build_proxy_server",
    "upstream_timeout_seconds",
]


class OrderRecorder(Protocol):
    """Ledger writes around live broker calls (agent/broker_ledger.py, ADR-0034)."""

    def before_dispatch(self, call: ProxyCall) -> None: ...

    def after_validated(self, call: ProxyCall, envelope: dict[str, Any]) -> None: ...


def upstream_timeout_seconds(mcp_tool_timeout_ms: int) -> float:
    """The proxy's per-call upstream deadline. Raises ValueError if no time would be left."""
    seconds = mcp_tool_timeout_ms / 1000 - PROXY_TIMEOUT_MARGIN_SECONDS
    if seconds <= 0:
        raise ValueError(
            f"MCP_TOOL_TIMEOUT must exceed {PROXY_TIMEOUT_MARGIN_SECONDS:g}s for the proxy"
        )
    return seconds


def _fallback_output(server: str, tool: str, tool_call_id: uuid.UUID | None, gap: str) -> str:
    """An error envelope that needs no clock, validator, or recorder (last resort)."""
    return json.dumps(
        {
            "data": None,
            "gaps": [gap],
            "kind": EnvelopeKind.ERROR.value,
            "retrieved_at": None,
            "server": server,
            "source_as_of": None,
            "tool": tool,
            "tool_call_id": str(tool_call_id) if tool_call_id else None,
        },
        sort_keys=True,
    )


def _text_result(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])


def _unresolved(tier: ToolTier) -> ToolCallStatus:
    """A Tier S/X action whose result is unusable has an unknown outcome (§14)."""
    return ToolCallStatus.UNKNOWN if tier in (ToolTier.S, ToolTier.X) else ToolCallStatus.FAILED


@dataclass(frozen=True)
class ValidatingProxy:
    """The `tools/call` logic for one proxied server (see module docstring)."""

    server: str
    upstream: McpUpstream
    dispatch: ProxyDispatch
    recorder: ToolEventRecorder
    validator: ResultValidator
    run_control: RunControl
    clock: Callable[[], datetime]
    upstream_timeout_seconds: float
    # ADR-0034: set in live mode. A failed intent write means the call is not forwarded; a
    # failed write after a result stops the run but still delivers the envelope.
    order_recorder: OrderRecorder | None = None

    def _stop(self) -> None:
        try:
            now = self.clock()
        except Exception:  # noqa: BLE001 - the latch must still be set without the clock
            now = datetime.now(UTC)
        self.run_control.request_stop(StopReason.INFRASTRUCTURE_FAILURE, now)

    async def call_tool(self, params: types.CallToolRequestParams) -> types.CallToolResult:
        tool_call_id: uuid.UUID | None = None
        try:
            use_id = (params.meta or {}).get(TOOL_USE_ID_META)
            if not isinstance(use_id, str) or not use_id:
                self._stop()
                return _text_result(
                    _fallback_output(self.server, params.name, None, "call has no tool_use_id")
                )
            call = self.dispatch.claim(use_id)
            if call is None:
                self._stop()
                return _text_result(
                    _fallback_output(
                        self.server, params.name, None, "no dispatched call for this tool_use_id"
                    )
                )
            tool_call_id = call.tool_call_id
            output = await self._handle(call, params)
            self.dispatch.complete(use_id, output)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=b["text"]) for b in output]
            )
        except Exception as exc:  # noqa: BLE001 - nothing raw may reach the CLI
            self._stop()
            return _text_result(
                _fallback_output(
                    self.server,
                    params.name,
                    tool_call_id,
                    f"proxy failure ({type(exc).__name__})",
                )
            )

    async def _handle(
        self, call: ProxyCall, params: types.CallToolRequestParams
    ) -> list[dict[str, str]]:
        arguments = dict(params.arguments or {})
        if params.name != call.tool or arguments != call.effective_input:
            self._stop()
            return self._not_forwarded(call, "tool or arguments differ from the dispatched call")
        if call.tier in (ToolTier.S, ToolTier.X) and self.run_control.stop_requested:
            return self._not_forwarded(call, "run stop requested")
        sent = call.upstream_input if call.upstream_input is not None else arguments
        if self.order_recorder is not None:
            try:
                self.order_recorder.before_dispatch(call)
            except Exception as exc:  # noqa: BLE001 - never send an unrecorded order
                self._stop()
                return self._not_forwarded(
                    call, f"order intent not recorded ({type(exc).__name__})"
                )
        try:
            result = await self.upstream.call_tool(
                call.tool, sent, timeout_seconds=self.upstream_timeout_seconds
            )
        except UpstreamError as exc:
            return self._failed(call, str(exc))
        now = self.clock()
        outcome = self.validator(
            ValidationRequest(
                tool_call_id=call.tool_call_id,
                server=call.server,
                tool=call.tool,
                tier=call.tier,
                effective_input=sent,
                tool_response=result.response,
                retrieved_at=now,
            )
        )
        envelope = outcome.envelope
        if (envelope.tool_call_id, envelope.server, envelope.tool) != (
            call.tool_call_id,
            call.server,
            call.tool,
        ):
            raise ValueError("validator returned an envelope for another call")
        if outcome.raw_redacted is not None:
            self.recorder.store_result(
                call.tool_call_id, ResultKind.RAW_INVALID, outcome.raw_redacted
            )
        payload = envelope.model_dump(mode="json")
        # ADR-0037: the ledger keeps the full envelope; the model gets its view.
        output = mcp_tool_output(model_view(payload))
        size = len(output[0]["text"])
        if size > MAX_DELIVERED_CHARS:
            return self._failed(
                call,
                f"result too large to deliver ({size} characters; limit {MAX_DELIVERED_CHARS})"
                "; request less data",
            )
        valid = envelope.kind is EnvelopeKind.VALIDATED
        ref = self.recorder.store_result(
            call.tool_call_id, ResultKind.VALIDATED if valid else ResultKind.ERROR, payload
        )
        if valid:
            self.recorder.outcome(
                call.tool_call_id,
                ToolCallStatus.SUCCEEDED,
                observed_at=now,
                dedup_key=PROXY_DEDUP_KEY,
                result_ref=ref,
            )
        else:
            self.recorder.outcome(
                call.tool_call_id,
                _unresolved(call.tier),
                observed_at=now,
                dedup_key=PROXY_DEDUP_KEY,
                reason=f"result {envelope.kind.value}",
                error_ref=ref,
            )
        if valid and self.order_recorder is not None:
            try:
                self.order_recorder.after_validated(call, payload)
            except Exception:  # noqa: BLE001 - deliver the broker's answer; stop further actions
                self._stop()
        return output

    def _error_output(self, call: ProxyCall, gap: str, now: datetime) -> list[dict[str, str]]:
        envelope = ResultEnvelope(
            tool_call_id=call.tool_call_id,
            server=call.server,
            tool=call.tool,
            kind=EnvelopeKind.ERROR,
            gaps=(gap,),
            retrieved_at=now,
        ).model_dump(mode="json")
        return mcp_tool_output(envelope)

    def _failed(self, call: ProxyCall, gap: str) -> list[dict[str, str]]:
        """The upstream exchange failed, or its result cannot be delivered, after dispatch:
        S/X outcomes are unknown."""
        now = self.clock()
        output = self._error_output(call, gap, now)
        ref = self.recorder.store_result(
            call.tool_call_id, ResultKind.ERROR, json.loads(output[0]["text"])
        )
        self.recorder.outcome(
            call.tool_call_id,
            _unresolved(call.tier),
            observed_at=now,
            dedup_key=PROXY_DEDUP_KEY,
            reason=gap,
            error_ref=ref,
        )
        return output

    def _not_forwarded(self, call: ProxyCall, reason: str) -> list[dict[str, str]]:
        """Nothing reached the server, so the outcome is known: failed, in every tier."""
        now = self.clock()
        output = self._error_output(call, f"not forwarded: {reason}", now)
        ref = self.recorder.store_result(
            call.tool_call_id, ResultKind.ERROR, json.loads(output[0]["text"])
        )
        self.recorder.outcome(
            call.tool_call_id,
            ToolCallStatus.FAILED,
            observed_at=now,
            dedup_key=PROXY_DEDUP_KEY,
            reason=f"not forwarded: {reason}",
            error_ref=ref,
        )
        return output


def _listed_tools(
    registry: ToolRegistry, upstream: McpUpstream, allowed_tools: Iterable[str]
) -> list[types.Tool]:
    """Tools the session may call this run (qualified names in `allowed_tools`) that are
    registered and that the server listed, with the server's own input schema.

    Only allowed tools are served: the real CLI shows an in-process server's tools to the
    model even when they are in `disallowed_tools` (real-CLI test, 2026-09-27), so a
    disallowed tool, order tools included, must not exist on the proxy at all (CLAUDE.md §8
    layer 1). Discovery uses the upstream's full list, not this one.
    """
    allowed = frozenset(allowed_tools)
    return [
        types.Tool(name=t.name, description=t.description, input_schema=t.input_schema)
        for t in upstream.tools
        if registry.get(t.name) is not None and registry.qualified(t.name) in allowed
    ]


def build_proxy_server(
    proxy: ValidatingProxy, registry: ToolRegistry, allowed_tools: Iterable[str]
) -> Server[Any]:
    """The in-process MCP server for one proxied source (named like the source), serving
    only the tools allowed this run."""
    if registry.server != proxy.server or proxy.upstream.server != proxy.server:
        raise ValueError("proxy, registry, and upstream must name the same server")
    tools = _listed_tools(registry, proxy.upstream, allowed_tools)

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return await proxy.call_tool(params)

    return Server(
        proxy.server,
        version=PROXY_SERVER_VERSION,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
