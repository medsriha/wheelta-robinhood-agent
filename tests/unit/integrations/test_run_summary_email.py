"""Run-summary email over Resend and the Anthropic Messages API (ADR-0029).

httpx.MockTransport routes by host: no sockets.
"""

import io
import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from wheelta_robinhood_agent.domain.enums import AppEnv, ExecutionMode, RunStatus
from wheelta_robinhood_agent.integrations.notifications.email import (
    EmailDeliveryStatus,
    EmailMessage,
    ResendError,
    RunSummaryEmailConfig,
    send_resend,
    send_run_summary,
)
from wheelta_robinhood_agent.integrations.notifications.summarizer import (
    ANTHROPIC_VERSION,
    write_run_summary,
)
from wheelta_robinhood_agent.observability.logging import configure_logging
from wheelta_robinhood_agent.observability.redaction import Redactor
from wheelta_robinhood_agent.observability.run_summary import (
    PROSE_UNAVAILABLE,
    ConsideredOption,
    RunSummaryInput,
)

T0 = datetime(2026, 9, 28, 14, 5, tzinfo=UTC)
RESEND_KEY = "re_TeStKeY_1234567890"  # noqa: S105 - test fixture
ANTHROPIC_KEY = "sk-ant-TeStKeY-1234567890"  # noqa: S105 - test fixture
RECIPIENT = "owner@example.com"
SUMMARY = RunSummaryInput(
    run_id="run-1",
    environment=AppEnv.LOCAL,
    slot=T0,
    status=RunStatus.COMPLETED,
    reason="completed",
    requested_execution_mode=ExecutionMode.OFF,
    effective_execution_mode=ExecutionMode.OFF,
    record=None,
    audit_status="completed",
)
REDACTOR = Redactor(secrets=(SecretStr(RESEND_KEY), SecretStr(ANTHROPIC_KEY)))


def _prose(text: str = "Nothing traded; the agent held.") -> httpx.Response:
    return httpx.Response(
        200,
        json={"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]},
    )


class Router:
    """Answers Anthropic and Resend requests from their own queues and records both."""

    def __init__(
        self,
        anthropic: list[httpx.Response | Exception] | None = None,
        resend: list[httpx.Response | Exception] | None = None,
    ) -> None:
        self.queues = {
            "api.anthropic.com": anthropic or [_prose()],
            "api.resend.com": resend or [httpx.Response(200, json={"id": "em_1"})],
        }
        self.requests: dict[str, list[httpx.Request]] = {h: [] for h in self.queues}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen = self.requests[host]
        seen.append(request)
        queue = self.queues[host]
        item = queue[min(len(seen), len(queue)) - 1]
        if isinstance(item, Exception):
            raise item
        return item


def _config(**kw: Any) -> RunSummaryEmailConfig:
    base: dict[str, Any] = {
        "enabled": True,
        "resend_api_key": SecretStr(RESEND_KEY),
        "from_address": "Wheelta Agent <agent@wheelta.com>",
        "to_address": SecretStr(RECIPIENT),
        "anthropic_api_key": SecretStr(ANTHROPIC_KEY),
        "model": "claude-test-model",
    }
    base.update(kw)
    return RunSummaryEmailConfig(**base)


def _send(router: Router, *, summary: RunSummaryInput = SUMMARY, **kw: Any) -> Any:
    client = httpx.Client(transport=httpx.MockTransport(router))
    sleeps: list[float] = []
    result = send_run_summary(
        summary, config=_config(**kw), client=client, redactor=REDACTOR, sleep=sleeps.append
    )
    return result, sleeps


def test_sends_prose_and_facts_with_an_idempotency_key() -> None:
    router = Router()
    result, _ = _send(router)
    assert result.status is EmailDeliveryStatus.SENT
    assert result.provider_message_id == "em_1" and result.prose_written
    (llm,) = router.requests["api.anthropic.com"]
    assert llm.headers["x-api-key"] == ANTHROPIC_KEY
    assert llm.headers["anthropic-version"] == ANTHROPIC_VERSION
    body = json.loads(llm.content)
    assert body["model"] == "claude-test-model"
    assert json.loads(body["messages"][0]["content"])["run"]["run_id"] == "run-1"
    assert "tools" not in body
    (mail,) = router.requests["api.resend.com"]
    assert mail.headers["authorization"] == f"Bearer {RESEND_KEY}"
    assert mail.headers["idempotency-key"] == "run-summary/local/run-1"
    sent = json.loads(mail.content)
    assert sent["to"] == [RECIPIENT] and sent["from"] == "Wheelta Agent <agent@wheelta.com>"
    assert sent["subject"].startswith("[Wheelta agent] completed · dry run")
    assert sent["text"].startswith("Nothing traded; the agent held.")
    assert "Recorded facts (authoritative)" in sent["text"] and "<pre" in sent["html"]


@pytest.mark.parametrize(
    "anthropic",
    [
        [httpx.Response(529)],
        [httpx.ConnectError("down")],
        [httpx.Response(200, content=b"not json")],
        [httpx.Response(200, json={"stop_reason": "max_tokens", "content": []})],
        [httpx.Response(200, json={"stop_reason": "end_turn", "content": [{"type": "x"}]})],
    ],
)
def test_prose_failure_still_sends_the_facts(anthropic: list[Any]) -> None:
    router = Router(anthropic=anthropic)
    result, _ = _send(router)
    assert result.status is EmailDeliveryStatus.SENT and not result.prose_written
    assert len(router.requests["api.anthropic.com"]) == 1  # never retried
    sent = json.loads(router.requests["api.resend.com"][0].content)
    assert sent["text"].startswith(PROSE_UNAVAILABLE)


def test_failure_context_reaches_writer_and_delivered_fallback_without_secrets() -> None:
    summary = SUMMARY.model_copy(
        update={
            "status": RunStatus.FAILED,
            "reason": "session_failed",
            "diagnostic_details": (f"PermissionError: invalid credential {ANTHROPIC_KEY}",),
            "candidates": (
                ConsideredOption(
                    candidate_ref="candidate:aapl",
                    underlying="AAPL",
                    occ_symbol="AAPL  261016P00190000",
                ),
            ),
        }
    )
    router = Router(anthropic=[httpx.Response(529)])
    result, _ = _send(router, summary=summary)
    assert result.status is EmailDeliveryStatus.SENT and not result.prose_written
    request_body = json.loads(router.requests["api.anthropic.com"][0].content)
    facts = json.loads(request_body["messages"][0]["content"])
    assert "PermissionError" in facts["diagnostic_details"][0]
    assert facts["candidates"][0]["selection"] == "unknown"
    sent = json.loads(router.requests["api.resend.com"][0].content)
    assert "PermissionError" in sent["text"] and "PermissionError" in sent["html"]
    assert "AAPL  261016P00190000" in sent["text"]
    assert ANTHROPIC_KEY not in repr(facts) and ANTHROPIC_KEY not in repr(sent)


def test_disabled_or_unconfigured_skips_without_any_request() -> None:
    for kw in ({"enabled": False}, {"resend_api_key": None}, {"to_address": None}):
        router = Router()
        result, _ = _send(router, **kw)
        assert result.status is EmailDeliveryStatus.SKIPPED
        assert router.requests == {"api.anthropic.com": [], "api.resend.com": []}


def test_resend_5xx_and_429_retry_with_the_same_key() -> None:
    router = Router(
        resend=[
            httpx.Response(503),
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json={"id": "em_2"}),
        ]
    )
    result, sleeps = _send(router)
    assert result.status is EmailDeliveryStatus.SENT and result.attempts == 3
    assert sleeps[1] == 2.0
    keys = {r.headers["idempotency-key"] for r in router.requests["api.resend.com"]}
    assert keys == {"run-summary/local/run-1"}


def test_resend_4xx_is_not_retried_and_fails_without_raising() -> None:
    router = Router(resend=[httpx.Response(422, json={"message": f"bad {RECIPIENT}"})])
    result, _ = _send(router)
    assert result.status is EmailDeliveryStatus.FAILED
    assert result.error == "http_422" and result.attempts == 1
    assert RECIPIENT not in repr(result)


def test_resend_gives_up_after_three_attempts() -> None:
    router = Router(resend=[httpx.ConnectError("down")])
    result, _ = _send(router)
    assert result.status is EmailDeliveryStatus.FAILED
    assert result.attempts == 3 and result.error == "ConnectError"


def test_resend_success_without_an_id_is_a_failure() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}))
    )
    message = EmailMessage(
        message_type="run_summary", from_address="a@b.co", to=("c@d.co",), subject="s", text="t"
    )
    with pytest.raises(ResendError, match="invalid_response"):
        send_resend(
            message,
            client=client,
            api_key=SecretStr(RESEND_KEY),
            idempotency_key="k",
            timeout_seconds=1.0,
        )


def test_keys_and_recipient_never_reach_the_logs() -> None:
    stream = io.StringIO()
    root = logging.getLogger()
    previous_level = root.level
    handler = configure_logging(
        "DEBUG",
        None,
        secrets=(SecretStr(RESEND_KEY), SecretStr(ANTHROPIC_KEY), SecretStr(RECIPIENT)),
        stream=stream,
    )
    router = Router(
        anthropic=[httpx.Response(401, json={"error": ANTHROPIC_KEY})],
        resend=[httpx.Response(500, json={"message": RECIPIENT})],
    )
    try:
        _send(router)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
    logged = stream.getvalue()
    assert "run_summary_prose_failed" in logged
    assert "run_summary_email_attempt_failed" in logged
    for value in (RESEND_KEY, ANTHROPIC_KEY, RECIPIENT):
        assert value not in logged


def test_write_run_summary_joins_text_blocks() -> None:
    response = httpx.Response(
        200,
        json={
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "One."}, {"type": "text", "text": " Two. "}],
        },
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda r: response))
    text = write_run_summary(
        {"run": {}},
        client=client,
        api_key=SecretStr(ANTHROPIC_KEY),
        model="m",
        timeout_seconds=1.0,
    )
    assert text == "One.\n\nTwo."


def test_email_message_requires_content() -> None:
    with pytest.raises(ValueError, match="recipient"):
        EmailMessage(message_type="t", from_address="a@b.co", to=(), subject="s", text="t")
    with pytest.raises(ValueError, match="text"):
        EmailMessage(message_type="t", from_address="a@b.co", to=("c@d.co",), subject="s", text=" ")
