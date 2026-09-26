"""A scripted stand-in for the Claude Code CLI, speaking the SDK control protocol.

`FakeCli` is a `claude_agent_sdk.Transport`. The real `ClaudeSDKClient` and its `Query` run
unchanged on top of it, so our hooks are invoked through the SDK's own `hook_callback`
dispatch and the local `wra_local` tools through its in-process MCP bridge (`mcp_message`).
No model, no subprocess, no network.

The "model" is a script: an async function receiving a `FakeModel`, calling tools, and
returning the final text. Everything the model would receive is appended to
`FakeCli.model_inputs`, which is the result-boundary observation point (docs/TESTING.md:
observe at the fake model/transport boundary, not only the hook return value).

CLI behaviour mirrored here (not verified against the real CLI): PreToolUse hooks run before
the `allowed_tools`/`disallowed_tools` permission check (`dontAsk` denies anything not
allowed); a hook deny or `continue: false` ends that call; PostToolUse `updatedToolOutput`
replaces what the model sees; built-in outputs are not replaced.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, Transport

Handler = Callable[[dict[str, Any]], dict[str, Any]]


class FakeToolFailure(Exception):
    """Raised by a fake remote tool to simulate a transport failure (PostToolUseFailure)."""


class ScriptStopped(Exception):
    """The CLI stopped the turn (interrupt or a hook's `continue: false`)."""


@dataclass
class FakeWorld:
    """Fake remote MCP servers: per server, a status and tool handlers."""

    handlers: dict[str, dict[str, Handler]] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    init_statuses: dict[str, str] = field(default_factory=dict)
    web_results: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


@dataclass
class ToolTurn:
    """What the model received for one tool call."""

    name: str
    denied: bool
    reason: str | None = None
    output: Any = None
    context: str | None = None

    @property
    def data(self) -> Any:
        """`data` of a replaced envelope (dict output), else None."""
        return self.output.get("data") if isinstance(self.output, dict) else None


Script = Callable[["FakeModel"], Awaitable[str | None]]


class FakeModel:
    def __init__(self, cli: FakeCli) -> None:
        self._cli = cli

    async def call(self, name: str, tool_input: dict[str, Any]) -> ToolTurn:
        return await self._cli.tool_call(name, tool_input)

    @property
    def interrupted(self) -> bool:
        return self._cli.interrupted.is_set()


class FakeCli(Transport):
    """The scripted CLI. One instance per session (one `ClaudeSDKClient`)."""

    def __init__(self, options: ClaudeAgentOptions, world: FakeWorld, script: Script) -> None:
        self.options = options
        self.world = world
        self.script = script
        self.model_inputs: list[Any] = []
        self.turns: list[ToolTurn] = []
        self.user_messages: list[str] = []
        self.interrupted = asyncio.Event()
        self.local_tools: list[str] = []
        self._out: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._hooks: dict[str, list[tuple[str | None, list[str]]]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._counter = 0
        self._mcp_id = 100
        self._local_ready = False
        self._use_counter = 0

    # -- Transport ------------------------------------------------------------------------

    async def connect(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    async def end_input(self) -> None:
        return None

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self._out.put(None)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._out.get()
            if item is None:
                return
            yield item

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            if line.strip():
                self._spawn(self._route(json.loads(line)))

    # -- plumbing -------------------------------------------------------------------------

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _emit(self, message: dict[str, Any]) -> None:
        await self._out.put(message)

    async def _respond(self, request_id: str, response: dict[str, Any]) -> None:
        await self._emit(
            {
                "type": "control_response",
                "response": {"subtype": "success", "request_id": request_id, "response": response},
            }
        )

    async def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._counter += 1
        request_id = f"cli_{self._counter}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._emit({"type": "control_request", "request_id": request_id, "request": request})
        response = await asyncio.wait_for(future, 30)
        if response.get("subtype") == "error":
            raise RuntimeError(str(response.get("error")))
        body = response.get("response")
        return body if isinstance(body, dict) else {}

    async def _route(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "control_response":
            response = message.get("response", {})
            future = self._pending.pop(str(response.get("request_id")), None)
            if future is not None and not future.done():
                future.set_result(response)
            return
        if kind == "control_request":
            await self._control(message["request_id"], message["request"])
            return
        if kind == "user":
            content = message.get("message", {}).get("content")
            self.user_messages.append(content if isinstance(content, str) else json.dumps(content))
            self._spawn(self._run_script())

    async def _control(self, request_id: str, request: dict[str, Any]) -> None:
        subtype = request.get("subtype")
        if subtype == "initialize":
            for event, matchers in (request.get("hooks") or {}).items():
                self._hooks[event] = [(m.get("matcher"), m["hookCallbackIds"]) for m in matchers]
            await self._respond(request_id, {"commands": []})
        elif subtype == "mcp_status":
            await self._respond(request_id, await self._mcp_status())
        elif subtype == "interrupt":
            self.interrupted.set()
            await self._respond(request_id, {})
        else:
            await self._respond(request_id, {})

    # -- MCP --------------------------------------------------------------------------------

    def _sdk_servers(self) -> list[str]:
        servers = self.options.mcp_servers
        assert isinstance(servers, dict)
        return [name for name, cfg in servers.items() if cfg.get("type") == "sdk"]

    def _http_servers(self) -> list[str]:
        servers = self.options.mcp_servers
        assert isinstance(servers, dict)
        return [name for name, cfg in servers.items() if cfg.get("type") == "http"]

    async def _mcp(self, server: str, method: str, params: dict[str, Any] | None) -> Any:
        self._mcp_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._mcp_id, "method": method}
        if params is not None:
            message["params"] = params
        body = await self._request(
            {"subtype": "mcp_message", "server_name": server, "message": message}
        )
        return body.get("mcp_response", {}).get("result")

    async def _ensure_local(self) -> None:
        if self._local_ready:
            return
        for server in self._sdk_servers():
            await self._mcp(
                server,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "fake-cli", "version": "0"},
                },
            )
            await self._request(
                {
                    "subtype": "mcp_message",
                    "server_name": server,
                    "message": {"jsonrpc": "2.0", "method": "notifications/initialized"},
                }
            )
            listed = await self._mcp(server, "tools/list", {})
            self.local_tools = [t["name"] for t in (listed or {}).get("tools", [])]
        self._local_ready = True

    async def _mcp_status(self) -> dict[str, Any]:
        await self._ensure_local()
        entries = []
        for name in self._http_servers():
            status = self.world.statuses.get(name, "connected")
            entry: dict[str, Any] = {"name": name, "status": status}
            if status == "connected":
                entry["tools"] = [{"name": t} for t in self.world.handlers.get(name, {})]
            entries.append(entry)
        for name in self._sdk_servers():
            entries.append(
                {
                    "name": name,
                    "status": "connected",
                    "tools": [{"name": t} for t in self.local_tools],
                }
            )
        return {"mcpServers": entries}

    def visible_tools(self) -> set[str]:
        """Tools the model would have in context: offered built-ins and MCP tools, minus
        `disallowed_tools` (the SDK's removal list)."""
        denied = set(self.options.disallowed_tools)
        names = set(self.options.tools if isinstance(self.options.tools, list) else [])
        for server in self._http_servers():
            names |= {f"mcp__{server}__{t}" for t in self.world.handlers.get(server, {})}
        for server in self._sdk_servers():
            names |= {f"mcp__{server}__{t}" for t in self.local_tools}
        return names - denied

    # -- hooks ------------------------------------------------------------------------------

    async def _hook(
        self, event: str, tool_name: str, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        outputs = []
        for matcher, ids in self._hooks.get(event, []):
            if matcher not in (None, "", "*") and matcher != tool_name:
                continue
            for callback_id in ids:
                outputs.append(
                    await self._request(
                        {
                            "subtype": "hook_callback",
                            "callback_id": callback_id,
                            "input": {"hook_event_name": event, "session_id": "fake", **payload},
                            "tool_use_id": payload.get("tool_use_id"),
                        }
                    )
                )
        return outputs

    async def tool_call(self, name: str, tool_input: dict[str, Any]) -> ToolTurn:
        if self.interrupted.is_set():
            raise ScriptStopped("interrupted")
        self._use_counter += 1
        use_id = f"toolu_{self._use_counter:04d}"
        base = {"tool_name": name, "tool_input": tool_input, "tool_use_id": use_id}
        effective = tool_input
        for out in await self._hook("PreToolUse", name, base):
            specific = out.get("hookSpecificOutput") or {}
            if specific.get("permissionDecision") == "deny":
                turn = ToolTurn(name, True, specific.get("permissionDecisionReason"))
                return self._deliver(turn, stop=out.get("continue") is False)
            if out.get("continue") is False:
                return self._deliver(ToolTurn(name, True, out.get("stopReason")), stop=True)
            if isinstance(specific.get("updatedInput"), dict):
                effective = specific["updatedInput"]
        allowed = set(self.options.allowed_tools)
        if name in set(self.options.disallowed_tools) or name not in allowed:
            return self._deliver(ToolTurn(name, True, "permission denied (dontAsk)"))
        try:
            response = await self._execute(name, effective)
        except FakeToolFailure as exc:
            outs = await self._hook(
                "PostToolUseFailure", name, {**base, "error": str(exc), "is_interrupt": False}
            )
            stop = any(o.get("continue") is False for o in outs)
            return self._deliver(ToolTurn(name, False, output={"error": str(exc)}), stop=stop)
        visible: Any = response
        context: str | None = None
        stop = False
        for out in await self._hook("PostToolUse", name, {**base, "tool_response": response}):
            specific = out.get("hookSpecificOutput") or {}
            if "updatedToolOutput" in specific:
                visible = specific["updatedToolOutput"]
            context = specific.get("additionalContext") or context
            stop = stop or out.get("continue") is False
        return self._deliver(ToolTurn(name, False, output=visible, context=context), stop=stop)

    def _deliver(self, turn: ToolTurn, stop: bool = False) -> ToolTurn:
        self.turns.append(turn)
        self.model_inputs.append(turn.reason if turn.denied else turn.output)
        if turn.context:
            self.model_inputs.append(turn.context)
        if stop:
            raise ScriptStopped("a hook stopped the session")
        return turn

    async def _execute(self, name: str, tool_input: dict[str, Any]) -> Any:
        self.world.calls.append((name, tool_input))
        if name == "WebSearch":
            query = str(tool_input.get("query"))
            return {"query": query, "results": self.world.web_results.get(query, [])}
        if name == "WebFetch":
            return {"url": tool_input.get("url"), "content": "fetched page"}
        _, server, tool = name.split("__", 2)
        if server in self._sdk_servers():
            return await self._mcp(server, "tools/call", {"name": tool, "arguments": tool_input})
        handler = self.world.handlers.get(server, {}).get(tool)
        if handler is None:
            raise FakeToolFailure(f"unknown tool {name}")
        return handler(tool_input)

    # -- the turn ----------------------------------------------------------------------------

    async def _run_script(self) -> None:
        servers = [*self._http_servers(), *self._sdk_servers()]
        await self._emit(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "fake",
                "model": self.options.model,
                "tools": sorted(self.visible_tools()),
                "mcp_servers": [
                    {"name": n, "status": self.world.init_statuses.get(n, "connected")}
                    for n in servers
                ],
            }
        )
        text: str | None = None
        stopped = False
        try:
            text = await self.script(FakeModel(self))
        except ScriptStopped:
            stopped = True
        if self.interrupted.is_set() or stopped:
            # A real CLI keeps the stream open until it winds down; emit the aborted result.
            await self._emit(self._result(None, error=True))
            return
        if text is not None:
            await self._emit(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": self.options.model or "fake-model",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
        await self._emit(self._result(text, error=False))

    def _result(self, text: str | None, *, error: bool) -> dict[str, Any]:
        return {
            "type": "result",
            "subtype": "error_during_execution" if error else "success",
            "duration_ms": 10,
            "duration_api_ms": 5,
            "is_error": error,
            "num_turns": len(self.turns) + 1,
            "session_id": "fake",
            "total_cost_usd": 0.0125,
            "usage": {"input_tokens": 1200, "output_tokens": 300},
            "result": text,
            "terminal_reason": "aborted_tools" if error else "completed",
        }


def factory(
    world: FakeWorld, script: Script, created: list[FakeCli]
) -> Callable[[ClaudeAgentOptions], FakeCli]:
    """A transport factory for `OrchestratorDeps.transport_factory` that records instances."""

    def make(options: ClaudeAgentOptions) -> FakeCli:
        cli = FakeCli(options, world, script)
        created.append(cli)
        return cli

    return make


def dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True)
