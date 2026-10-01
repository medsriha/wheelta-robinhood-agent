"""Run-summary email over Resend and the Anthropic Messages API (ADR-0029; one per tick,
ADR-0057).

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

from wheelta_robinhood_agent.domain.enums import AgentRole, AppEnv, ExecutionMode, RunStatus
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
    SlotSummaryInput,
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
    agent=AgentRole.CLOSE,
    requested_execution_mode=ExecutionMode.OFF,
    effective_execution_mode=ExecutionMode.OFF,
    record=None,
    audit_status="completed",
)
KEY = f"run-summary/local/{T0.isoformat()}"
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


def _tick(*agents: RunSummaryInput) -> SlotSummaryInput:
    return SlotSummaryInput(environment=AppEnv.LOCAL, slot=T0, agents=agents)


def _send(router: Router, *, summary: RunSummaryInput = SUMMARY, **kw: Any) -> Any:
    return _send_tick(router, _tick(summary), **kw)


def _send_tick(router: Router, tick: SlotSummaryInput, **kw: Any) -> Any:
    client = httpx.Client(transport=httpx.MockTransport(router))
    sleeps: list[float] = []
    result = send_run_summary(
        tick, config=_config(**kw), client=client, redactor=REDACTOR, sleep=sleeps.append
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
    facts = json.loads(body["messages"][0]["content"])
    assert facts["agents"][0]["run"]["run_id"] == "run-1"
    assert facts["agents"][0]["agent"] == "Buy-to-Close agent"
    assert "tools" not in body
    (mail,) = router.requests["api.resend.com"]
    assert mail.headers["authorization"] == f"Bearer {RESEND_KEY}"
    assert mail.headers["idempotency-key"] == KEY
    sent = json.loads(mail.content)
    assert sent["to"] == [RECIPIENT] and sent["from"] == "Wheelta Agent <agent@wheelta.com>"
    assert sent["subject"].startswith("[Wheelta agent] dry run · close: completed, no trades")
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
    assert "PermissionError" in facts["agents"][0]["diagnostic_details"][0]
    assert facts["agents"][0]["candidates"][0]["selection"] == "unknown"
    sent = json.loads(router.requests["api.resend.com"][0].content)
    assert "PermissionError" in sent["text"] and "PermissionError" in sent["html"]
    assert "AAPL  261016P00190000" in sent["text"]
    assert ANTHROPIC_KEY not in repr(facts) and ANTHROPIC_KEY not in repr(sent)


def test_one_email_covers_both_agents_in_order() -> None:
    """ADR-0057: a skipped close agent gets one line; the sell agent its full section."""
    close = SUMMARY.model_copy(
        update={
            "run_id": "close-1",
            "status": RunStatus.SKIPPED_NO_OPEN_SHORTS,
            "reason": "no open short option positions",
            "session_started": False,
            "audit_status": None,
        }
    )
    sell = SUMMARY.model_copy(update={"run_id": "sell-1", "agent": AgentRole.SELL})
    tick = _tick(close, sell).model_copy(
        update={"next_run_at": T0.replace(hour=15), "next_run_source": "agent"}
    )
    router = Router()
    result, _ = _send_tick(router, tick)
    assert result.status is EmailDeliveryStatus.SENT
    facts = json.loads(
        json.loads(router.requests["api.anthropic.com"][0].content)["messages"][0]["content"]
    )
    assert [a["agent"] for a in facts["agents"]] == ["Buy-to-Close agent", "Sell Options agent"]
    assert facts["agents"][0]["run"]["session_started"] is False
    assert facts["agents"][1]["run"]["session_started"] is True
    assert facts["tick"]["next_run_source"] == "agent"
    sent = json.loads(router.requests["api.resend.com"][0].content)
    assert sent["subject"].startswith(
        "[Wheelta agent] dry run · close: skipped_no_open_shorts · sell: completed, no trades"
    )
    text = sent["text"]
    assert text.index("== Buy-to-Close agent ==") < text.index("== Sell Options agent ==")
    assert "Status: skipped_no_open_shorts (no open short option positions)" in text
    assert "No session started." in text
    assert "Next run of both agents not before: 2026-09-28T15:05:00+00:00 (agent)" in text


def test_a_tick_summary_needs_one_slot() -> None:
    other = SUMMARY.model_copy(update={"slot": T0.replace(minute=10)})
    with pytest.raises(ValueError, match="slot"):
        _tick(SUMMARY, other)


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
    assert keys == {KEY}


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
