"""The loopback listener for the Mignons' proxy servers (ADR-0063).

`serve_loopback(app)` is an `agent.session.LoopbackServer`: it binds 127.0.0.1 on a free port,
serves the ASGI app with uvicorn inside the caller's event loop (the proxies' upstream
sessions live there), and yields `http://127.0.0.1:<port>`. It lives here because a socket
listener is network I/O (CLAUDE.md §3). The app checks its own bearer token
(`agent.proxy.LoopbackProxyApp`); nothing listens beyond loopback.

uvicorn choices: `capture_signals` is a no-op (the orchestrator owns SIGTERM/SIGINT through
RunControl; uvicorn's `serve` would otherwise replace those handlers), `log_config=None`
(uvicorn's default dictConfig would reconfigure the process's logging), no access log, and
`lifespan="off"` (the app's `running()` is entered by the session).
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import AsyncIterator, Iterator
from typing import Any, Final

import anyio
import uvicorn

LOOPBACK_HOST: Final = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS: Final = 10.0


class _Server(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


@contextlib.asynccontextmanager
async def serve_loopback(app: Any) -> AsyncIterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LOOPBACK_HOST, 0))
    port = sock.getsockname()[1]
    server = _Server(
        uvicorn.Config(app, lifespan="off", log_config=None, access_log=False, log_level="warning")
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(server.serve, [sock])
        with anyio.fail_after(STARTUP_TIMEOUT_SECONDS):
            while not server.started:
                await anyio.sleep(0.01)
        try:
            yield f"http://{LOOPBACK_HOST}:{port}"
        finally:
            server.should_exit = True
    sock.close()
