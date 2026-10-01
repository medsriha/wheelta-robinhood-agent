"""Run-summary email delivery over Resend (ADR-0029), after the wheelta-api email layer.

`EmailMessage` is provider-neutral; `send_resend` translates it at the provider boundary.
`send_run_summary` composes and delivers one run's summary and never raises: a failed prose
call falls back to the facts-only body, and a failed send returns a typed result the caller
records. Emails are informational and never affect run status or the exit code.

ADR-0057: one email per tick covers both agent runs of the slot. Resend requests carry an
Idempotency-Key (`run-summary/<env>/<slot>`), so retrying a
transport error, 5xx or 429 cannot send a second copy. The API key, the recipient and the
response body are never logged or returned: provider validation errors can echo recipients.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import httpx
from pydantic import SecretStr

from wheelta_robinhood_agent.integrations.notifications.delivery import (
    MAX_ATTEMPTS,
    _default_jitter,
    _default_now,
    _parse_retry_after,
)
from wheelta_robinhood_agent.integrations.notifications.summarizer import write_run_summary
from wheelta_robinhood_agent.observability.redaction import Redactor
from wheelta_robinhood_agent.observability.run_summary import (
    SlotSummaryInput,
    build_slot_subject,
    render_bodies,
    render_slot_facts_text,
    slot_summary_facts,
)

_log = logging.getLogger(__name__)

RESEND_EMAILS_URL = "https://api.resend.com/emails"
MESSAGE_TYPE = "run_summary"


@dataclass(frozen=True, slots=True)
class EmailMessage:
    """A complete message independent of the delivery provider."""

    message_type: str
    from_address: str
    to: tuple[str, ...]
    subject: str
    text: str
    html: str | None = None

    def __post_init__(self) -> None:
        if not self.message_type.strip():
            raise ValueError("email message_type must not be empty")
        if not self.from_address.strip():
            raise ValueError("email from_address must not be empty")
        if not self.to:
            raise ValueError("email must have at least one recipient")
        if not self.subject.strip():
            raise ValueError("email subject must not be empty")
        if not self.text.strip():
            raise ValueError("email must include text content")


class EmailDeliveryStatus(StrEnum):
    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EmailDeliveryResult:
    """Outcome of one summary email. Carries no recipient, key or response body."""

    status: EmailDeliveryStatus
    subject: str | None = None
    provider_message_id: str | None = None
    prose_written: bool = False
    attempts: int = 0
    status_code: int | None = None
    error: str | None = None


class ResendError(Exception):
    """Resend did not accept the message. The message never includes provider text."""

    def __init__(self, error: str, *, attempts: int, status_code: int | None) -> None:
        super().__init__(error)
        self.error = error
        self.attempts = attempts
        self.status_code = status_code


def _resend_payload(message: EmailMessage) -> dict[str, object]:
    payload: dict[str, object] = {
        "from": message.from_address,
        "to": list(message.to),
        "subject": message.subject,
        "text": message.text,
    }
    if message.html is not None:
        payload["html"] = message.html
    return payload


def send_resend(
    message: EmailMessage,
    *,
    client: httpx.Client,
    api_key: SecretStr,
    idempotency_key: str,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _default_now,
    jitter: Callable[[], float] = _default_jitter,
    backoff_base_seconds: float = 1.0,
    max_retry_after_seconds: float = 30.0,
) -> tuple[str, int]:
    """POST to Resend's send API; return (message id, attempts). Raises ResendError.

    Retries transport errors, 5xx and 429 (bounded, full jitter, `Retry-After` honoured);
    the Idempotency-Key makes a retry of an accepted request a no-op.
    """
    headers = {
        "Authorization": f"Bearer {api_key.get_secret_value()}",
        "Idempotency-Key": idempotency_key,
    }
    body = _resend_payload(message)
    status_code: int | None = None
    error = "not_attempted"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        retry_after: float | None = None
        try:
            response = client.post(
                RESEND_EMAILS_URL, json=body, headers=headers, timeout=timeout_seconds
            )
        except httpx.HTTPError as exc:
            status_code, error = None, type(exc).__name__
            retryable = True
        else:
            status_code = response.status_code
            if response.is_success:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                message_id = payload.get("id") if isinstance(payload, dict) else None
                if not isinstance(message_id, str) or not message_id:
                    raise ResendError("invalid_response", attempts=attempt, status_code=status_code)
                return message_id, attempt
            error = f"http_{status_code}"
            retryable = status_code >= 500 or status_code == 429
            retry_after = _parse_retry_after(response.headers.get("Retry-After"), now())
        _log.warning(
            "run_summary_email_attempt_failed",
            extra={"attempt": attempt, "error": error, "status_code": status_code},
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
    raise ResendError(error, attempts=attempt, status_code=status_code)


@dataclass(frozen=True)
class RunSummaryEmailConfig:
    """What `send_run_summary` needs from Settings (ADR-0029)."""

    enabled: bool
    resend_api_key: SecretStr | None
    from_address: str
    to_address: SecretStr | None
    anthropic_api_key: SecretStr
    model: str
    prose_timeout_seconds: float = 60.0
    send_timeout_seconds: float = 10.0


def idempotency_key_for(summary: SlotSummaryInput) -> str:
    return f"run-summary/{summary.environment.value}/{summary.slot.isoformat()}"


def send_run_summary(
    summary: SlotSummaryInput,
    *,
    config: RunSummaryEmailConfig,
    client: httpx.Client,
    redactor: Redactor,
    sleep: Callable[[float], None] = time.sleep,
) -> EmailDeliveryResult:
    """Write, compose and send one tick's summary email. Never raises."""
    if not config.enabled or config.resend_api_key is None or config.to_address is None:
        return EmailDeliveryResult(status=EmailDeliveryStatus.SKIPPED)
    try:
        subject = redactor.redact_text(build_slot_subject(summary))
        facts_text = render_slot_facts_text(summary, redactor)
        prose = write_run_summary(
            slot_summary_facts(summary, redactor),
            client=client,
            api_key=config.anthropic_api_key,
            model=config.model,
            timeout_seconds=config.prose_timeout_seconds,
        )
        text, html = render_bodies(prose, facts_text, redactor)
        message = EmailMessage(
            message_type=MESSAGE_TYPE,
            from_address=config.from_address,
            to=(config.to_address.get_secret_value(),),
            subject=subject,
            text=text,
            html=html,
        )
    except Exception as exc:  # noqa: BLE001 - informational email: never fail the run
        _log.warning("run_summary_email_compose_failed", extra={"error": type(exc).__name__})
        return EmailDeliveryResult(status=EmailDeliveryStatus.FAILED, error=type(exc).__name__)
    try:
        message_id, attempts = send_resend(
            message,
            client=client,
            api_key=config.resend_api_key,
            idempotency_key=idempotency_key_for(summary),
            timeout_seconds=config.send_timeout_seconds,
            sleep=sleep,
        )
    except ResendError as exc:
        return EmailDeliveryResult(
            status=EmailDeliveryStatus.FAILED,
            subject=subject,
            prose_written=prose is not None,
            attempts=exc.attempts,
            status_code=exc.status_code,
            error=exc.error,
        )
    except Exception as exc:  # noqa: BLE001 - informational email: never fail the run
        _log.warning("run_summary_email_send_failed", extra={"error": type(exc).__name__})
        return EmailDeliveryResult(
            status=EmailDeliveryStatus.FAILED,
            subject=subject,
            prose_written=prose is not None,
            error=type(exc).__name__,
        )
    _log.info(
        "run_summary_email_sent",
        extra={"provider_message_id": message_id, "prose_written": prose is not None},
    )
    return EmailDeliveryResult(
        status=EmailDeliveryStatus.SENT,
        subject=subject,
        provider_message_id=message_id,
        prose_written=prose is not None,
        attempts=attempts,
    )
