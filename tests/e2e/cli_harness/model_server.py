"""A scripted, recording Anthropic Messages API endpoint on 127.0.0.1.

Every HTTP request the CLI sends is recorded (path, auth headers, JSON body). `POST
/v1/messages` answers from a script, streaming (SSE) or not, per the request's `stream` flag;
`count_tokens` gets a benign count; any other path a 404 JSON error.

The script drives the main agent loop, recognised by `SYSTEM_MARKER` in the system
prompt, and optionally a Mignon's loop (`mignon_steps`, recognised by `MIGNON_MARKER` in the
Mignon's system prompt, ADR-0025). Any auxiliary model call the CLI makes (titles,
summaries, ...) gets a short text reply and is recorded like the rest, so assertions cover
every request, not only the loop.
The step index is the number of assistant turns already in the request's `messages`.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

SYSTEM_MARKER = "WRA-HARNESS-SYSTEM-PROMPT"
MIGNON_MARKER = "WRA-HARNESS-MIGNON-PROMPT"


@dataclass(frozen=True)
class ToolUse:
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class FinalText:
    text: str


Reply = ToolUse | FinalText
Step = Reply | Callable[[dict[str, Any]], Reply]


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: Any

    @property
    def is_messages(self) -> bool:
        return self.method == "POST" and self.path.rstrip("/").endswith("/v1/messages")

    @property
    def is_main_loop(self) -> bool:
        return self.is_messages and SYSTEM_MARKER in json.dumps(self.body.get("system", ""))

    @property
    def is_mignon_loop(self) -> bool:
        return self.is_messages and MIGNON_MARKER in json.dumps(self.body.get("system", ""))

    def tool_names(self) -> list[str]:
        """The tools this request offered the model."""
        return [t.get("name") for t in self.body.get("tools", []) if isinstance(t, dict)]

    def text(self) -> str:
        """The whole request body as sent (JSON), for substring assertions."""
        return json.dumps(self.body, ensure_ascii=False)

    def tool_results(self) -> list[dict[str, Any]]:
        """Every tool_result block in the request's messages, in order."""
        out: list[dict[str, Any]] = []
        for message in self.body.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                out.extend(b for b in content if b.get("type") == "tool_result")
        return out


def _assistant_turns(body: dict[str, Any]) -> int:
    return sum(1 for m in body.get("messages", []) if m.get("role") == "assistant")


@dataclass
class RecordingModel:
    """The ASGI app. `steps` is the main-loop script; after it ends, a final text is sent."""

    steps: Sequence[Step] = ()
    mignon_steps: Sequence[Step] = ()
    requests: list[RecordedRequest] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ids: int = 0

    # -- inspection ------------------------------------------------------------------------

    def snapshot(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self.requests)

    def main_loop(self) -> list[RecordedRequest]:
        return [r for r in self.snapshot() if r.is_main_loop]

    def mignon_loop(self) -> list[RecordedRequest]:
        return [r for r in self.snapshot() if r.is_mignon_loop]

    # -- ASGI ------------------------------------------------------------------------------

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        raw = b""
        while True:
            message = await receive()
            raw += message.get("body", b"")
            if not message.get("more_body"):
                break
        try:
            body: Any = json.loads(raw) if raw else None
        except ValueError:
            body = raw.decode("utf-8", "replace")
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        request = RecordedRequest(
            method=scope["method"],
            path=scope["path"],
            headers={
                k: v
                for k, v in headers.items()
                if k in ("x-api-key", "authorization", "anthropic-beta", "user-agent")
            },
            body=body,
        )
        with self._lock:
            self.requests.append(request)
        if request.method == "POST" and request.path.rstrip("/").endswith("/count_tokens"):
            await self._json(send, 200, {"input_tokens": 100})
            return
        if not request.is_messages or not isinstance(body, dict):
            await self._json(
                send, 404, {"type": "error", "error": {"type": "not_found_error", "message": "x"}}
            )
            return
        reply = self._reply(request)
        model = str(body.get("model", "claude-harness"))
        if body.get("stream"):
            await self._sse(send, model, reply)
        else:
            await self._json(send, 200, self._message(model, reply))

    def _reply(self, request: RecordedRequest) -> Reply:
        if request.is_mignon_loop:
            steps = self.mignon_steps
        elif request.is_main_loop:
            steps = self.steps
        else:
            return FinalText("ok")
        index = _assistant_turns(request.body)
        if index >= len(steps):
            return FinalText("harness script finished")
        step = steps[index]
        return step if isinstance(step, ToolUse | FinalText) else step(request.body)

    def _next_id(self) -> str:
        with self._lock:
            self._ids += 1
            return f"toolu_harness_{self._ids:04d}"

    def _block(self, reply: Reply) -> dict[str, Any]:
        if isinstance(reply, ToolUse):
            return {
                "type": "tool_use",
                "id": self._next_id(),
                "name": reply.name,
                "input": reply.input,
            }
        return {"type": "text", "text": reply.text}

    def _message(self, model: str, reply: Reply) -> dict[str, Any]:
        return {
            # Unique per reply: the CLI merges assistant messages that share an id.
            "id": f"msg_harness_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [self._block(reply)],
            "stop_reason": "tool_use" if isinstance(reply, ToolUse) else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    @staticmethod
    async def _json(send: Any, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": data})

    async def _sse(self, send: Any, model: str, reply: Reply) -> None:
        message = self._message(model, reply)
        block = message["content"][0]
        start = dict(block)
        if block["type"] == "tool_use":
            start["input"] = {}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start["text"] = ""
            delta = {"type": "text_delta", "text": block["text"]}
        head = {**message, "content": [], "stop_reason": None}
        events: list[tuple[str, dict[str, Any]]] = [
            ("message_start", {"type": "message_start", "message": head}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": start},
            ),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"),
                ],
            }
        )
        for name, data in events:
            chunk = f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b""})
