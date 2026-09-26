"""HTTP delivery of alert and heartbeat payloads (CLAUDE.md §16).

Notification posts are idempotent, so transport errors, 5xx and 429 are retried with bounded
full-jitter backoff (at most ``max_attempts``), honouring ``Retry-After`` exactly: if the server
asks to wait longer than ``max_retry_after_seconds``, delivery gives up instead of retrying
early. Delivery failures never raise into the caller; a typed :class:`DeliveryResult` is
returned for the caller to record in the ledger.

The destination URL is a secret: it is never logged and never appears in a result or an
exception message. Only exception type names and HTTP status codes are recorded.
"""

import logging
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from wheelta_robinhood_agent.observability.alerts import AlertPayload, HeartbeatPayload

_log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
_RETRYABLE_STATUS = 429


class DeliveryOutcome(StrEnum):
    DELIVERED = "delivered"
    FAILED = "failed"
    NOT_CONFIGURED = "not_configured"


class DeliveryResult(BaseModel):
    """Outcome of one notification delivery. Carries no URL and no response body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_kind: Literal["alert", "heartbeat"]
    outcome: DeliveryOutcome
    attempts: int = Field(ge=0, le=MAX_ATTEMPTS)
    status_code: int | None = None
    error: str | None = None


def _default_now() -> datetime:
    return datetime.now(UTC)


def _default_jitter() -> float:
    # Backoff jitter, not cryptography.
    return random.random()  # noqa: S311


class _Retry(Exception):
    def __init__(self, retry_after: float | None) -> None:
        self.retry_after = retry_after


def _parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Seconds to wait per a ``Retry-After`` header (delta-seconds or HTTP-date), or None."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, (when - now).total_seconds())


def _post(
    *,
    payload_kind: Literal["alert", "heartbeat"],
    body: dict[str, object],
    client: httpx.Client,
    url: SecretStr | None,
    timeout_seconds: float,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
    jitter: Callable[[], float],
    backoff_base_seconds: float,
    max_retry_after_seconds: float,
) -> DeliveryResult:
    if url is None:
        return DeliveryResult(
            payload_kind=payload_kind, outcome=DeliveryOutcome.NOT_CONFIGURED, attempts=0
        )
    status_code: int | None = None
    error: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        retry_after: float | None = None
        try:
            response = client.post(url.get_secret_value(), json=body, timeout=timeout_seconds)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            status_code, error = None, type(exc).__name__
            retryable = not isinstance(exc, httpx.InvalidURL | httpx.UnsupportedProtocol)
        else:
            status_code = response.status_code
            if response.is_success:
                return DeliveryResult(
                    payload_kind=payload_kind,
                    outcome=DeliveryOutcome.DELIVERED,
                    attempts=attempt,
                    status_code=status_code,
                )
            error = f"http_{status_code}"
            retryable = status_code >= 500 or status_code == _RETRYABLE_STATUS
            retry_after = _parse_retry_after(response.headers.get("Retry-After"), now())
        _log.warning(
            "notification_delivery_attempt_failed",
            extra={
                "payload_kind": payload_kind,
                "attempt": attempt,
                "error": error,
                "status_code": status_code,
            },
        )
        if not retryable or attempt == MAX_ATTEMPTS:
            break
        if retry_after is not None:
            if retry_after > max_retry_after_seconds:
                error = f"{error}_retry_after_exceeds_cap"
                break
            delay = retry_after
        else:
            delay = jitter() * backoff_base_seconds * (2 ** (attempt - 1))
        sleep(delay)
    return DeliveryResult(
        payload_kind=payload_kind,
        outcome=DeliveryOutcome.FAILED,
        attempts=attempt,
        status_code=status_code,
        error=error,
    )


def deliver_alert(
    payload: AlertPayload,
    *,
    client: httpx.Client,
    url: SecretStr | None,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _default_now,
    jitter: Callable[[], float] = _default_jitter,
    backoff_base_seconds: float = 1.0,
    max_retry_after_seconds: float = 30.0,
) -> DeliveryResult:
    """POST an alert to ``ALERT_WEBHOOK_URL``. Never raises for delivery failures."""
    return _post(
        payload_kind="alert",
        body=payload.model_dump(mode="json"),
        client=client,
        url=url,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        now=now,
        jitter=jitter,
        backoff_base_seconds=backoff_base_seconds,
        max_retry_after_seconds=max_retry_after_seconds,
    )


def deliver_heartbeat(
    payload: HeartbeatPayload,
    *,
    client: httpx.Client,
    url: SecretStr | None,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _default_now,
    jitter: Callable[[], float] = _default_jitter,
    backoff_base_seconds: float = 1.0,
    max_retry_after_seconds: float = 30.0,
) -> DeliveryResult:
    """POST a heartbeat to ``HEARTBEAT_URL``. Never raises for delivery failures."""
    return _post(
        payload_kind="heartbeat",
        body=payload.model_dump(mode="json"),
        client=client,
        url=url,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        now=now,
        jitter=jitter,
        backoff_base_seconds=backoff_base_seconds,
        max_retry_after_seconds=max_retry_after_seconds,
    )
