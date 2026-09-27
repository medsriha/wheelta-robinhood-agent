"""The proxy's upstream client against a real streamable-HTTP MCP server (ADR-0023).

`integrations/mcp_upstream.py` over HTTP to the harness's fake server on 127.0.0.1 (no CLI):
listing, successful and `isError` results, and every failure class. Error texts must never
carry remote content. TCP is allowed to 127.0.0.1 only.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import anyio
import pytest
from cli_harness.mcp_server import FakeMcpServer, IsError, Json, Slow, TransportFailure
from cli_harness.servers import ThreadedServer
from pydantic import SecretStr

from wheelta_robinhood_agent.integrations.mcp_upstream import (
    UpstreamAuthError,
    UpstreamError,
    UpstreamTimeout,
    UpstreamTooLarge,
    UpstreamUnavailable,
    open_http_upstream,
)
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.status import McpHttpServer

pytestmark = pytest.mark.allow_hosts(["127.0.0.1"])

REMOTE_500_BODY = "upstream exploded"
BEHAVIORS = {
    "get_option_quotes": Json({"ok": True}),
    "get_equity_quotes": IsError("REMOTE-ERROR-TEXT"),
    "get_portfolio": TransportFailure(),
    "get_accounts": Slow(5, Json({})),
    "get_option_chains": Json({"rows": ["x" * 200] * 50}),
}


class Unauthorized:
    """Answers every request with 401 and a body that must never surface."""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b"REMOTE-401-BODY"})


@pytest.fixture
def fake() -> Iterator[tuple[FakeMcpServer, str]]:
    app = FakeMcpServer(BEHAVIORS)
    server = ThreadedServer(app, "upstream-fake")
    server.start()
    try:
        yield app, f"{server.base_url}/mcp"
    finally:
        server.stop()


def config(url: str, *, stored_login: bool = False) -> McpHttpServer:
    if stored_login:
        return McpHttpServer(name="robinhood", url=url, uses_stored_cli_login=True)  # type: ignore[arg-type]
    return McpHttpServer(name="robinhood", url=url, token=SecretStr("harness-token"))  # type: ignore[arg-type]


def test_lists_tools_and_returns_results_including_is_error(
    fake: tuple[FakeMcpServer, str],
) -> None:
    app, url = fake

    async def main() -> None:
        async with open_http_upstream(config(url), connect_timeout_seconds=10) as up:
            assert {t.name for t in up.tools} == {t.name for t in ROBINHOOD_REGISTRY.tools}
            ok = await up.call_tool("get_option_quotes", {"a": 1}, timeout_seconds=10)
            assert ok.response == {
                "content": [{"type": "text", "text": json.dumps({"ok": True})}],
                "isError": False,
            }
            assert ok.size_bytes > 0
            err = await up.call_tool("get_equity_quotes", {}, timeout_seconds=10)
            assert err.response["isError"] is True  # returned for the validator, not raised

    anyio.run(main)
    assert app.calls[0] == ("get_option_quotes", {"a": 1})


@pytest.mark.parametrize(
    ("tool", "timeout", "error"),
    [
        ("get_portfolio", 10.0, UpstreamUnavailable),
        ("get_accounts", 1.0, UpstreamTimeout),
    ],
)
def test_failures_are_typed_and_carry_no_remote_text(
    fake: tuple[FakeMcpServer, str], tool: str, timeout: float, error: type[UpstreamError]
) -> None:
    app, url = fake
    caught: list[UpstreamError] = []

    async def main() -> None:
        async with open_http_upstream(config(url), connect_timeout_seconds=10) as up:
            try:
                await up.call_tool(tool, {}, timeout_seconds=timeout)
            except UpstreamError as exc:
                caught.append(exc)
            # The connection stays usable after a failed call.
            ok = await up.call_tool("get_option_quotes", {}, timeout_seconds=10)
            assert ok.response["isError"] is False

    anyio.run(main)
    (exc,) = caught
    assert type(exc) is error
    assert REMOTE_500_BODY not in str(exc) and str(exc).startswith(f"{tool}: ")
    assert exc.__cause__ is None and exc.__suppress_context__
    assert app.called(tool) == 1  # never retried


def test_oversized_result_is_dropped(fake: tuple[FakeMcpServer, str]) -> None:
    _, url = fake

    async def main() -> None:
        async with open_http_upstream(
            config(url), connect_timeout_seconds=10, max_result_bytes=1_000
        ) as up:
            with pytest.raises(UpstreamTooLarge, match="exceeds 1000"):
                await up.call_tool("get_option_chains", {}, timeout_seconds=10)

    anyio.run(main)


def test_refused_credential_is_an_auth_error_without_the_body() -> None:
    server = ThreadedServer(Unauthorized(), "upstream-401")
    server.start()
    try:

        async def main() -> None:
            async with open_http_upstream(
                config(f"{server.base_url}/mcp"), connect_timeout_seconds=5
            ):
                pytest.fail("connected despite 401")

        with pytest.raises(UpstreamAuthError) as info:
            anyio.run(main)
    finally:
        server.stop()
    assert "REMOTE-401-BODY" not in str(info.value)


def test_unreachable_server_is_unavailable() -> None:
    async def main() -> None:
        async with open_http_upstream(config("http://127.0.0.1:9/mcp"), connect_timeout_seconds=2):
            pytest.fail("connected to a closed port")

    with pytest.raises(UpstreamUnavailable):
        anyio.run(main)


def test_stored_cli_login_cannot_be_proxied() -> None:
    async def main() -> None:
        async with open_http_upstream(
            config("http://127.0.0.1:9/mcp", stored_login=True), connect_timeout_seconds=1
        ):
            pass

    with pytest.raises(ValueError, match="bearer token"):
        anyio.run(main)
