"""Structured logging and redaction (CLAUDE.md §7, §16)."""

import io
import json
import logging
import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, SecretStr

from wheelta_robinhood_agent.observability.logging import (
    JsonFormatter,
    bind,
    configure_logging,
    settings_secrets,
)
from wheelta_robinhood_agent.observability.redaction import (
    REDACTED,
    Redactor,
    is_account_key,
    is_secret_key,
    mask_account,
)

ACCOUNT = "5QR12345678"
TOKEN = "wheelta-secret-abcdef123456"  # noqa: S105 - test fixture

# The "Zq" marker keeps a random secret from coinciding with a key name or "[REDACTED]".
_secret_text = st.text(
    alphabet=string.ascii_letters + string.digits + "-_.", min_size=6, max_size=40
).map(lambda s: f"Zq{s}")
_digits = st.text(alphabet=string.digits, min_size=8, max_size=14)


def _logger(name: str) -> tuple[logging.Logger, io.StringIO, logging.Handler]:
    stream = io.StringIO()
    logger = logging.getLogger(f"test.obs.{name}")
    logger.propagate = False
    handler = configure_logging(
        "DEBUG", SecretStr(ACCOUNT), secrets=[SecretStr(TOKEN)], stream=stream, logger=logger
    )
    return logger, stream, handler


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_one_json_object_per_line_with_fixed_fields() -> None:
    logger, stream, _ = _logger("fields")
    log = bind(logger, run_id="run-1", stage="preflight")
    log.info("source_status", extra={"server": "wheelta"})
    log.warning("multi\nline")
    raw = stream.getvalue().splitlines()
    assert len(raw) == 2
    first = json.loads(raw[0])
    assert first["event"] == "source_status"
    assert first["level"] == "INFO"
    assert first["run_id"] == "run-1"
    assert first["stage"] == "preflight"
    assert first["server"] == "wheelta"
    assert str(first["timestamp"]).endswith("+00:00")


def test_bind_merges_call_site_extras_and_rebinds() -> None:
    logger, stream, _ = _logger("bind")
    log = bind(logger, run_id="r", stage="a", tool="x").bind(stage="b")
    log.info("e", extra={"attempt": 2})
    (line,) = _lines(stream)
    assert (line["stage"], line["tool"], line["attempt"]) == ("b", "x", 2)


def test_fixed_field_collision_is_renamed() -> None:
    logger, stream, _ = _logger("collide")
    logger.info("e", extra={"level_x": 1, "logger_extra": 2})
    logger.info("e", extra={"timestamp": "fake"})
    assert _lines(stream)[1]["extra_timestamp"] == "fake"


def test_configure_logging_is_idempotent() -> None:
    logger, stream, _ = _logger("idem")
    configure_logging("INFO", None, stream=stream, logger=logger)
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO


def test_known_secrets_and_account_scrubbed_from_message_extras_and_exception() -> None:
    logger, stream, _ = _logger("known")
    logger.info("calling with %s for %s", TOKEN, ACCOUNT, extra={"note": f"acct={ACCOUNT}"})
    try:
        raise RuntimeError(f"boom {TOKEN}")
    except RuntimeError:
        logger.exception("failed")
    out = stream.getvalue()
    assert TOKEN not in out
    assert ACCOUNT not in out
    assert "****5678" in out


def test_secret_keys_bearer_jwt_and_userinfo() -> None:
    logger, stream, _ = _logger("patterns")
    logger.info(
        "e",
        extra={
            "headers": {"Authorization": "Bearer abcdefghijklmnop", "X-Api-Key": "k" * 10},
            "database_url": "postgres://u:p@h/db",
            "ALERT_WEBHOOK_URL": "https://hooks.example/x",
            "text": "see Bearer zzzzzzzzzzzz and eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl",
            "url": "https://user:hunter22@example.com/path",
            "input_tokens": 1200,
        },
    )
    (line,) = _lines(stream)
    assert line["headers"] == {"Authorization": REDACTED, "X-Api-Key": REDACTED}
    assert line["database_url"] == REDACTED
    assert line["ALERT_WEBHOOK_URL"] == REDACTED
    assert "zzzzzzzzzzzz" not in str(line["text"])
    assert "c2lnbmF0dXJl" not in str(line["text"])
    assert "hunter22" not in str(line["url"])
    assert line["input_tokens"] == 1200  # "tokens" count is not a secret


def test_secretstr_never_serializes() -> None:
    class Model(BaseModel):
        key: SecretStr
        plain: str

    logger, stream, _ = _logger("secretstr")
    value = "plain-secret-value-xyz"
    logger.info(
        "e", extra={"harmless": SecretStr(value), "model": Model(key=SecretStr(value), plain="p")}
    )
    out = stream.getvalue()
    assert value not in out
    line = _lines(stream)[0]
    assert line["harmless"] == REDACTED
    assert line["model"] == {"key": REDACTED, "plain": "p"}


def test_account_fields_masked_even_when_not_configured() -> None:
    redactor = Redactor()
    out = redactor.redact(
        {"account_number": "998877665544", "accountId": "ABC99887766", "account": {"id": 1234567}}
    )
    assert out == {
        "account_number": "****5544",
        "accountId": "****7766",
        "account": {"id": 1234567},
    }


def test_key_classification() -> None:
    for key in (
        "access_token",
        "Authorization",
        "api_key",
        "apiKey",
        "password",
        "Set-Cookie",
        "HEARTBEAT_URL",
        "wheelta_mcp_token",
        "client_secret",
    ):
        assert is_secret_key(key), key
    for key in ("input_tokens", "status", "auth_status", "server"):
        assert not is_secret_key(key), key
    assert is_account_key("account_number") and is_account_key("acctNo")
    assert not is_account_key("accounts_count")
    assert mask_account("12") == "****"


def test_formatter_handles_bad_format_args() -> None:
    formatter = JsonFormatter(Redactor())
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "value %d", ("nope",), None)
    assert json.loads(formatter.format(record))["event"] == "value %d"


def test_settings_secrets_excludes_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from wheelta_robinhood_agent.config.settings import Settings

    for name, value in {
        "ANTHROPIC_API_KEY": "sk-ant-aaaaaaaa",
        "AGENT_MODEL": "m",
        "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER": ACCOUNT,
        "WHEELTA_MCP_TOKEN": TOKEN,
        "DATABASE_URL": "postgres://u:p@h/db",
        "ALERT_WEBHOOK_URL": "https://hooks.example/abc",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("HEARTBEAT_URL", raising=False)
    monkeypatch.delenv("ROBINHOOD_MCP_ACCESS_TOKEN", raising=False)
    values = {s.get_secret_value() for s in settings_secrets(Settings())}
    assert values == {"sk-ant-aaaaaaaa", TOKEN, "postgres://u:p@h/db", "https://hooks.example/abc"}


@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    secret=_secret_text,
    account=_digits,
    prefix=st.text(alphabet=string.ascii_letters + " ", max_size=20),
)
def test_property_no_known_secret_or_account_ever_logged(
    secret: str, account: str, prefix: str
) -> None:
    stream = io.StringIO()
    logger = logging.getLogger("test.obs.property")
    logger.propagate = False
    configure_logging(
        "DEBUG", SecretStr(account), secrets=[SecretStr(secret)], stream=stream, logger=logger
    )
    log = bind(logger, run_id="r", stage="s")
    log.info(f"{prefix}{secret}{prefix}", extra={"detail": [f"x{secret}y", {"n": account}]})
    log.info("acct %s", account, extra={"account_number": account, "free": f"id {account}!"})
    log.info("e", extra={"password": secret, "nested": {"token": secret}, "s": SecretStr(secret)})
    for line in stream.getvalue().splitlines():
        json.loads(line)
        assert secret not in line
        assert account not in line


@given(secret=_secret_text)
def test_property_secret_keys_always_redacted_without_known_secrets(secret: str) -> None:
    out = Redactor().redact(
        {"password": secret, "Authorization": f"Bearer {secret}", "x": {"api_key": secret}}
    )
    assert secret not in json.dumps(out)
