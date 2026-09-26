"""Alert and heartbeat HTTP delivery (CLAUDE.md §16). httpx.MockTransport: no sockets."""

import io
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr

from wheelta_robinhood_agent.domain.enums import AppEnv, RunStatus
from wheelta_robinhood_agent.integrations.notifications import (
    DeliveryOutcome,
    DeliveryResult,
    deliver_alert,
    deliver_heartbeat,
)
from wheelta_robinhood_agent.observability.alerts import (
    AlertKind,
    build_alert,
    build_heartbeat,
)
from wheelta_robinhood_agent.observability.logging import configure_logging

T0 = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
URL = "https://hooks.example.com/services/T000/B000/sEcReTpAtH123"
SECRET_PART = "sEcReTpAtH123"  # noqa: S105 - test fixture

ALERT = build_alert(
    AlertKind.RUN_TIMEOUT, run_id="r", environment=AppEnv.STAGING, occurred_at=T0, message="m"
)
HEARTBEAT = build_heartbeat(
    RunStatus.FAILED, run_id="r", environment=AppEnv.STAGING, slot=T0, occurred_at=T0
)


class Recorder:
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses[min(len(self.requests), len(self.responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return item


def _run(
    responses: list[httpx.Response | Exception],
    *,
    heartbeat: bool = False,
    url: str | None = URL,
    now: Callable[[], datetime] = lambda: T0,
) -> tuple[DeliveryResult, Recorder, list[float]]:
    recorder = Recorder(responses)
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    kwargs = dict(
        client=client,
        url=SecretStr(url) if url is not None else None,
        timeout_seconds=5.0,
        sleep=sleeps.append,
        now=now,
        jitter=lambda: 0.5,
    )
    if heartbeat:
        result = deliver_heartbeat(HEARTBEAT, **kwargs)  # type: ignore[arg-type]
    else:
        result = deliver_alert(ALERT, **kwargs)  # type: ignore[arg-type]
    return result, recorder, sleeps


def test_delivered_first_try_posts_json_payload() -> None:
    result, rec, sleeps = _run([httpx.Response(200)])
    assert result.outcome is DeliveryOutcome.DELIVERED
    assert result.attempts == 1
    assert json.loads(rec.requests[0].content)["kind"] == "run_timeout"
    assert sleeps == []


def test_failure_heartbeat_posts_failure_status() -> None:
    result, rec, _ = _run([httpx.Response(204)], heartbeat=True)
    body = json.loads(rec.requests[0].content)
    assert body["status"] == "failure" and body["run_status"] == "failed"
    assert result.payload_kind == "heartbeat"


def test_not_configured_makes_no_request() -> None:
    result, rec, _ = _run([httpx.Response(200)], url=None)
    assert result.outcome is DeliveryOutcome.NOT_CONFIGURED
    assert rec.requests == []


def test_retries_bounded_on_5xx_and_transport_errors() -> None:
    result, rec, sleeps = _run([httpx.ConnectError("down"), httpx.Response(503)])
    assert len(rec.requests) == 3
    assert result.outcome is DeliveryOutcome.FAILED
    assert result.attempts == 3
    assert result.status_code == 503
    assert sleeps == [0.5, 1.0]  # full jitter (0.5) * base 1s * 2^(n-1); no sleep after last


def test_recovers_after_transient_error() -> None:
    result, rec, _ = _run([httpx.ReadTimeout("slow"), httpx.Response(200)])
    assert result.outcome is DeliveryOutcome.DELIVERED
    assert result.attempts == 2


def test_4xx_not_retried() -> None:
    result, rec, sleeps = _run([httpx.Response(400)])
    assert len(rec.requests) == 1 and sleeps == []
    assert result.error == "http_400"


def test_retry_after_seconds_honoured() -> None:
    result, _, sleeps = _run(
        [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)]
    )
    assert sleeps == [7.0]
    assert result.outcome is DeliveryOutcome.DELIVERED


def test_retry_after_http_date_honoured() -> None:
    headers = {"Retry-After": "Fri, 25 Sep 2026 14:00:12 GMT"}
    _, _, sleeps = _run([httpx.Response(503, headers=headers), httpx.Response(200)])
    assert sleeps == [12.0]


def test_retry_after_beyond_cap_gives_up_instead_of_retrying_early() -> None:
    result, rec, sleeps = _run([httpx.Response(503, headers={"Retry-After": "3600"})])
    assert len(rec.requests) == 1 and sleeps == []
    assert result.outcome is DeliveryOutcome.FAILED


def test_invalid_retry_after_falls_back_to_backoff() -> None:
    _, _, sleeps = _run([httpx.Response(503, headers={"Retry-After": "soon"}), httpx.Response(200)])
    assert sleeps == [0.5]


@pytest.mark.parametrize(
    "responses",
    [
        [httpx.ConnectError(f"cannot reach {URL}")],
        [httpx.Response(500, text=f"echo {URL}")],
        [httpx.UnsupportedProtocol(f"bad {URL}")],
    ],
)
def test_url_never_leaks_into_result_or_logs(
    responses: list[httpx.Response | Exception],
) -> None:
    stream = io.StringIO()
    root = logging.getLogger()
    previous_level = root.level
    handler = configure_logging("DEBUG", None, secrets=[SecretStr(URL)], stream=stream)
    try:
        result, _, _ = _run(responses)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
    assert SECRET_PART not in repr(result)
    assert SECRET_PART not in json.dumps(result.model_dump(mode="json"))
    assert SECRET_PART not in stream.getvalue()
    assert "notification_delivery_attempt_failed" in stream.getvalue()
    assert result.outcome is DeliveryOutcome.FAILED


def test_unexpected_client_failure_never_raises() -> None:
    result, rec, _ = _run([httpx.RemoteProtocolError("reset")] * 3)
    assert len(rec.requests) == 3
    assert result.error == "RemoteProtocolError"
