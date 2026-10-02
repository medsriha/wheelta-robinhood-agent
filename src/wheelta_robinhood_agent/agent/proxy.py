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

ADR-0052: every envelope a Tier X call (the three option-order tools) produces here carries
`order_call_ref` (`order_call:<tool call id>`), whatever the outcome: validated, error,
failed upstream, or not forwarded. The stored envelope and the delivered view both hold it,
so `run_loader` can confirm the model received the ref it cites in `execution_refs`. The
static fallback envelope carries none: its delivery stops the session.

ADR-0063: the orchestrator sees only its own tools. The session's in-process server lists the
orchestrator's tools of a source (`build_proxy_server` with that allowlist). Each Mignon role
reaches the same `ValidatingProxy` through a loopback streamable-HTTP server inlined in its
`AgentDefinition.mcpServers`, the only mechanism the real CLI 2.1.283 offers for tools a
subagent sees and its parent does not (tests/e2e/test_e2e_orchestrator_tool_visibility_cli.py):

- `build_role_proxy_servers`: one `Server` per (Mignon role, source) listing the role's
  allowed tools of that source, under the path `role_path(role, source)`. The CLI offers a
  Mignon every tool of an inline server whatever its `AgentDefinition.tools`, so the listing
  is the role's set and never holds a Tier X tool.
- `LoopbackProxyApp`: the ASGI app serving them (stateless JSON responses), refusing a
  request without the session's bearer token or with a non-loopback Host/Origin (DNS
  rebinding). It opens no socket: the listener belongs in `integrations/` (CLAUDE.md §3) and
  is injected into the session (`session.SessionDeps.loopback_server`).

Calls through it carry the same `_meta["claudecode/toolUseId"]` and pass the same parent
PreToolUse hook (with the Mignon's `agent_id`/`agent_type`), so correlation, recording, and
validation are unchanged.
"""

import hmac
import json
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, MutableMapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings

from wheelta_robinhood_agent.agent.hooks import (
    EnvelopeKind,
    ResultEnvelope,
    ResultValidator,
    ValidationRequest,
    mcp_tool_output,
)
from wheelta_robinhood_agent.agent.mignons import Role, role_allowed
from wheelta_robinhood_agent.agent.model_view import ORDER_CALL_REF_KEY, model_view
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyCall, ProxyDispatch
from wheelta_robinhood_agent.agent.recorder import ResultKind, ToolEventRecorder
from wheelta_robinhood_agent.agent.result_boundary import (
    SEC_FILING_TOOL,
    SEC_FILING_UNAVAILABLE_GAP,
)
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.domain.assembly_context import order_call_ref_for
from wheelta_robinhood_agent.domain.enums import ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.loopback_http import LOOPBACK_HOST
from wheelta_robinhood_agent.integrations.mcp_upstream import McpUpstream, UpstreamError
from wheelta_robinhood_agent.integrations.registry import ToolRegistry
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    SERVER_NAME as ROBINHOOD_SERVER,
)

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
# ADR-0063: the Mignons' loopback servers. Only this host is ever bound or accepted.
MCP_PATH: Final = "mcp"

__all__ = [
    "LOOPBACK_HOST",
    "ExecutorOutcome",
    "MAX_DELIVERED_CHARS",
    "PROXY_DEDUP_KEY",
    "PROXY_TIMEOUT_MARGIN_SECONDS",
    "TOOL_USE_ID_META",
    "LoopbackProxyApp",
    "ValidatingProxy",
    "build_proxy_server",
    "build_role_proxy_servers",
    "role_path",
    "upstream_timeout_seconds",
]


class OrderRecorder(Protocol):
    """Ledger writes around live broker calls (agent/broker_ledger.py, ADR-0034)."""

    def before_dispatch(self, call: ProxyCall) -> None: ...

    def after_validated(self, call: ProxyCall, envelope: dict[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutorOutcome:
    """What an order-walk call returned (ADR-0066): the recorded, labeled envelope and the
    call's recorded status (`succeeded` only for a validated result)."""

    payload: dict[str, Any]
    status: ToolCallStatus


@dataclass(frozen=True, slots=True)
class _Handled:
    output: list[dict[str, str]]
    payload: dict[str, Any]
    status: ToolCallStatus


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


def _sec_filing_id(call: ProxyCall) -> str | None:
    """The filing_id of a Robinhood get_sec_filing call, else None."""
    if (call.server, call.tool) != (ROBINHOOD_SERVER, SEC_FILING_TOOL):
        return None
    filing_id = call.effective_input.get("filing_id")
    return filing_id if isinstance(filing_id, str) else None


def _labeled(call: ProxyCall, envelope: dict[str, Any]) -> dict[str, Any]:
    """The envelope with its order-call ref when the call is an order action (ADR-0052)."""
    if call.tier is not ToolTier.X:
        return envelope
    return {**envelope, ORDER_CALL_REF_KEY: order_call_ref_for(call.tool_call_id)}


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
    # filing_ids Robinhood answered "content not available" for in this run: never forwarded
    # again (one proxy per source serves the orchestrator and every Mignon).
    unavailable_filings: set[str] = field(default_factory=set)

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
            if call.by_executor:
                self._stop()
                return _text_result(
                    _fallback_output(
                        self.server, params.name, None, "an executor call came from the CLI"
                    )
                )
            output = (await self._handle(call, params)).output
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

    async def execute(
        self, use_id: str, *, timeout_seconds: float | None = None
    ) -> ExecutorOutcome:
        """Forward one order-walk call the hooks' gate admitted (ADR-0066), in-process.

        Same path as a CLI call: claim, intent before dispatch, upstream within the deadline,
        validation, recording, `order_call_ref` label. Nothing is delivered to the model, so
        the delivery size limit does not apply. `timeout_seconds` narrows the upstream
        deadline (the bounded cancel after the stop latch). Never raises for an upstream or
        validation failure: the outcome says what was recorded; a recording failure sets the
        stop latch and raises.
        """
        call = self.dispatch.claim(use_id)
        if call is None or not call.by_executor:
            self._stop()
            raise ValueError("no admitted executor call for this id")
        params = types.CallToolRequestParams(name=call.tool, arguments=dict(call.effective_input))
        try:
            handled = await self._handle(call, params, timeout_seconds=timeout_seconds)
        except Exception:
            self._stop()
            raise
        self.dispatch.complete(use_id, handled.output)
        return ExecutorOutcome(payload=handled.payload, status=handled.status)

    async def _handle(
        self,
        call: ProxyCall,
        params: types.CallToolRequestParams,
        *,
        timeout_seconds: float | None = None,
    ) -> _Handled:
        arguments = dict(params.arguments or {})
        if params.name != call.tool or arguments != call.effective_input:
            self._stop()
            return self._not_forwarded(call, "tool or arguments differ from the dispatched call")
        stopped = call.tier in (ToolTier.S, ToolTier.X) and self.run_control.stop_requested
        if stopped and not call.after_stop_cancel:
            return self._not_forwarded(call, "run stop requested")
        sent = call.upstream_input if call.upstream_input is not None else arguments
        filing_id = _sec_filing_id(call)
        if filing_id is not None and filing_id in self.unavailable_filings:
            return self._not_forwarded(call, SEC_FILING_UNAVAILABLE_GAP)
        if self.order_recorder is not None:
            try:
                self.order_recorder.before_dispatch(call)
            except Exception as exc:  # noqa: BLE001 - never send an unrecorded order
                self._stop()
                return self._not_forwarded(
                    call, f"order intent not recorded ({type(exc).__name__})"
                )
        deadline = self.upstream_timeout_seconds
        if timeout_seconds is not None:
            deadline = min(deadline, timeout_seconds)
        try:
            result = await self.upstream.call_tool(call.tool, sent, timeout_seconds=deadline)
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
        payload = _labeled(call, envelope.model_dump(mode="json"))
        # ADR-0037: the ledger keeps the full envelope; the model gets its view.
        output = mcp_tool_output(model_view(payload))
        size = len(output[0]["text"])
        if size > MAX_DELIVERED_CHARS and not call.by_executor:
            return self._failed(
                call,
                f"result too large to deliver ({size} characters; limit {MAX_DELIVERED_CHARS})"
                "; request less data",
            )
        valid = envelope.kind is EnvelopeKind.VALIDATED
        if filing_id is not None and SEC_FILING_UNAVAILABLE_GAP in envelope.gaps:
            self.unavailable_filings.add(filing_id)
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
        status = ToolCallStatus.SUCCEEDED if valid else _unresolved(call.tier)
        return _Handled(output, payload, status)

    def _error_output(self, call: ProxyCall, gap: str, now: datetime) -> list[dict[str, str]]:
        envelope = ResultEnvelope(
            tool_call_id=call.tool_call_id,
            server=call.server,
            tool=call.tool,
            kind=EnvelopeKind.ERROR,
            gaps=(gap,),
            retrieved_at=now,
        ).model_dump(mode="json")
        return mcp_tool_output(_labeled(call, envelope))

    def _failed(self, call: ProxyCall, gap: str) -> _Handled:
        """The upstream exchange failed, or its result cannot be delivered, after dispatch:
        S/X outcomes are unknown."""
        now = self.clock()
        output = self._error_output(call, gap, now)
        payload = json.loads(output[0]["text"])
        ref = self.recorder.store_result(call.tool_call_id, ResultKind.ERROR, payload)
        status = _unresolved(call.tier)
        self.recorder.outcome(
            call.tool_call_id,
            status,
            observed_at=now,
            dedup_key=PROXY_DEDUP_KEY,
            reason=gap,
            error_ref=ref,
        )
        return _Handled(output, payload, status)

    def _not_forwarded(self, call: ProxyCall, reason: str) -> _Handled:
        """Nothing reached the server, so the outcome is known: failed, in every tier."""
        now = self.clock()
        output = self._error_output(call, f"not forwarded: {reason}", now)
        payload = json.loads(output[0]["text"])
        ref = self.recorder.store_result(call.tool_call_id, ResultKind.ERROR, payload)
        self.recorder.outcome(
            call.tool_call_id,
            ToolCallStatus.FAILED,
            observed_at=now,
            dedup_key=PROXY_DEDUP_KEY,
            reason=f"not forwarded: {reason}",
            error_ref=ref,
        )
        return _Handled(output, payload, ToolCallStatus.FAILED)


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


def role_path(role: Role, server: str) -> str:
    """The loopback URL path of one Mignon role's server for one source (ADR-0063)."""
    return f"/{role.value}/{server}/{MCP_PATH}"


def build_role_proxy_servers(
    proxies: Mapping[str, tuple[ValidatingProxy, ToolRegistry]],
    allowed_tools: Iterable[str],
    roles: Iterable[Role],
) -> dict[Role, dict[str, Server[Any]]]:
    """For each Mignon role, one server per source (`proxies`: name -> proxy, registry) that
    lists the role's allowed tools of that source; a source with none is left out.

    The CLI offers a Mignon every tool an inline server lists (real CLI 2.1.283), so a
    listing never holds the orchestrator's role or a Tier X tool."""
    allowed = tuple(allowed_tools)
    out: dict[Role, dict[str, Server[Any]]] = {}
    for role in roles:
        if role is Role.ORCHESTRATOR:
            raise ValueError("the orchestrator is served in-process, never over loopback")
        tools = role_allowed(role, allowed)
        servers: dict[str, Server[Any]] = {}
        for name, (proxy, registry) in proxies.items():
            names = [t for t in tools if t.startswith(f"mcp__{name}__")]
            for qualified in names:
                spec = registry.get(qualified.removeprefix(f"mcp__{name}__"))
                if spec is None or spec.tier is ToolTier.X:
                    raise ValueError(f"{qualified} must never be served to a Mignon")
            if names and _listed_tools(registry, proxy.upstream, names):
                servers[name] = build_proxy_server(proxy, registry, names)
        if servers:
            out[role] = servers
    return out


Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class LoopbackProxyApp:
    """ASGI app serving `build_role_proxy_servers` at `role_path` (ADR-0063).

    The listener can start before the servers exist (the Mignons' inline configs need its
    port, and the proxies are built with the options): `mount` adds them once, then
    `running()` must be entered before any Mignon starts. Every request needs
    `authorization` (the bearer header, compared in constant time); the MCP transport also
    refuses a Host or Origin that is not this loopback address (DNS rebinding). Stateless JSON
    responses: each `tools/call` is one POST."""

    def __init__(self, token: str | None = None) -> None:
        token = token if token is not None else secrets.token_urlsafe(32)
        if len(token) < 32:
            raise ValueError("the loopback token must be at least 32 characters")
        self.authorization: Final = f"Bearer {token}"
        self._managers: dict[str, StreamableHTTPSessionManager] | None = None

    def mount(self, servers: Mapping[Role, Mapping[str, Server[Any]]]) -> None:
        if self._managers is not None:
            raise ValueError("the loopback servers are mounted once")
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"{LOOPBACK_HOST}:*"],
            allowed_origins=[f"http://{LOOPBACK_HOST}:*"],
        )
        self._managers = {
            role_path(role, name): StreamableHTTPSessionManager(
                app=server, json_response=True, stateless=True, security_settings=security
            )
            for role, by_name in servers.items()
            for name, server in by_name.items()
        }

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(self._managers or {})

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for manager in (self._managers or {}).values():
                await stack.enter_async_context(manager.run())
            yield

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            return
        headers = dict(scope.get("headers") or [])
        presented = headers.get(b"authorization", b"")
        if not hmac.compare_digest(presented, self.authorization.encode()):
            await _refuse(send, 401, "unauthorized")
            return
        manager = (self._managers or {}).get(str(scope.get("path", "")))
        if manager is None:
            await _refuse(send, 404, "not found")
            return
        await manager.handle_request(scope, receive, send)


async def _refuse(send: Send, status: int, text: str) -> None:
    body = json.dumps({"error": text}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})
