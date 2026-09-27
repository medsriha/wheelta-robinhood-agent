"""Localhost-only server plumbing: uvicorn in a thread, and a recording blackhole proxy."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import uvicorn

LOCALHOST = "127.0.0.1"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOCALHOST, 0))
        port: int = s.getsockname()[1]
        return port


class ThreadedServer:
    """Serve an ASGI app with uvicorn on 127.0.0.1 in a daemon thread."""

    def __init__(self, app: Any, name: str) -> None:
        self.port = free_port()
        config = uvicorn.Config(
            app, host=LOCALHOST, port=self.port, log_level="warning", lifespan="on"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name=name, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{LOCALHOST}:{self.port}"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("local server did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


class BlackholeProxy:
    """An HTTP(S) proxy on 127.0.0.1 that records the first request line and refuses it.

    Pointing HTTP_PROXY/HTTPS_PROXY here (with NO_PROXY for 127.0.0.1) turns any attempt by
    the CLI to reach a non-local host into a recorded, failed request instead of traffic.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((LOCALHOST, 0))
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self.port: int = self._sock.getsockname()[1]
        self.attempts: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="blackhole", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{LOCALHOST}:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(2)
                try:
                    head = conn.recv(4096).split(b"\r\n", 1)[0]
                except OSError:
                    head = b""
                self.attempts.append(head.decode("latin-1", "replace") or "<empty>")
                try:
                    conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()
