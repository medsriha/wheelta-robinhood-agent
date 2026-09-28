"""Handoff of proxied calls between the hooks and the validating proxy (ADR-0023).

PreToolUse registers each allowed call to a proxied server under its SDK `tool_use_id`; the
proxy (agent/proxy.py) claims it when the CLI's `tools/call` arrives and completes it with the
exact output it returned; PostToolUse reads that output to record the delivery. Kept apart
from proxy.py so hooks.py can depend on it without importing the proxy.
"""

import json
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import JsonValue

from wheelta_robinhood_agent.domain.enums import ToolTier

__all__ = ["CallState", "ProxyCall", "ProxyDispatch", "delivered_matches", "delivered_payload"]


@dataclass(frozen=True, slots=True)
class ProxyCall:
    """A call the PreToolUse hook allowed and recorded as dispatched to a proxied server."""

    tool_call_id: uuid.UUID
    server: str
    tool: str
    tier: ToolTier
    effective_input: dict[str, Any]
    # ADR-0030: what the proxy sends upstream when it differs from `effective_input` (the
    # account placeholder replaced by the configured number). Never shown to the CLI.
    upstream_input: dict[str, Any] | None = None


class CallState(StrEnum):
    PENDING = "pending"  # dispatched by PreToolUse, not yet received by the proxy
    CLAIMED = "claimed"  # the proxy is handling it
    COMPLETED = "completed"  # the proxy returned an envelope


class ProxyDispatch:
    """Thread-safe handoff: PreToolUse registers, the proxy claims and completes, PostToolUse
    takes the delivered envelope. Each call is claimed at most once, so a replayed or
    duplicated `tools/call` is never forwarded twice."""

    def __init__(self, servers: frozenset[str]) -> None:
        self._servers = servers
        self._lock = threading.Lock()
        self._calls: dict[str, ProxyCall] = {}
        self._states: dict[str, CallState] = {}
        self._delivered: dict[str, list[dict[str, str]]] = {}

    def proxied(self, server: str) -> bool:
        return server in self._servers

    def register(self, use_id: str, call: ProxyCall) -> None:
        if not self.proxied(call.server):
            raise ValueError(f"{call.server} is not proxied")
        with self._lock:
            if use_id in self._states:
                raise ValueError("tool_use_id already registered")
            self._calls[use_id] = call
            self._states[use_id] = CallState.PENDING

    def claim(self, use_id: str) -> ProxyCall | None:
        with self._lock:
            if self._states.get(use_id) is not CallState.PENDING:
                return None
            self._states[use_id] = CallState.CLAIMED
            return self._calls[use_id]

    def complete(self, use_id: str, output: list[dict[str, str]]) -> None:
        with self._lock:
            if self._states.get(use_id) is not CallState.CLAIMED:
                raise ValueError("completing a call that was not claimed")
            self._states[use_id] = CallState.COMPLETED
            self._delivered[use_id] = output

    def state(self, use_id: str) -> CallState | None:
        with self._lock:
            return self._states.get(use_id)

    def delivered(self, use_id: str) -> list[dict[str, str]] | None:
        """The exact tool output the proxy returned for this call, if it completed."""
        with self._lock:
            return self._delivered.get(use_id)


def delivered_matches(tool_response: object, delivered: list[dict[str, str]]) -> bool:
    """Whether the CLI's PostToolUse `tool_response` is exactly the proxy's output (a bare
    list of text blocks for MCP tools, DATA_QUALITY.md real-CLI results)."""
    if not isinstance(tool_response, list) or len(tool_response) != len(delivered):
        return False
    for got, want in zip(tool_response, delivered, strict=True):
        if not isinstance(got, Mapping) or got.get("type") != "text":
            return False
        if got.get("text") != want["text"]:
            return False
    return True


def delivered_payload(delivered: list[dict[str, str]]) -> JsonValue:
    """The envelope JSON inside the proxy's single text block."""
    value: JsonValue = json.loads(delivered[0]["text"])
    return value
