"""Structured JSON logging on the stdlib ``logging`` module (CLAUDE.md §16, §7).

One JSON object per line with ``timestamp`` (UTC ISO 8601), ``level``, ``event``, ``logger``,
``run_id``, ``stage`` and any extra fields. Every record, including third-party library records
(httpx logs request URLs), passes through a :class:`Redactor` before it is written.

Usage::

    handler = configure_logging(settings.LOG_LEVEL, settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                                secrets=settings_secrets(settings))
    log = bind(logging.getLogger(__name__), run_id=run_id, stage="preflight")
    log.info("source_status", extra={"server": "wheelta", "status": "connected"})

The message is the event name. No global state beyond the logging module's own registry.
"""

import json
import logging
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, TextIO

from pydantic import SecretStr

from wheelta_robinhood_agent.config.settings import LogLevel, Settings
from wheelta_robinhood_agent.observability.redaction import JsonValue, Redactor

# Attributes every LogRecord carries; anything else on a record came from ``extra``.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)
_FIXED_FIELDS = ("timestamp", "level", "event", "logger", "run_id", "stage")


class JsonFormatter(logging.Formatter):
    """Formats a record as one redacted JSON object on a single line."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        extras: dict[str, object] = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_")
        }
        try:
            event = record.getMessage()
        except (TypeError, ValueError):
            event = str(record.msg)
        payload: dict[str, JsonValue] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "event": self._redactor.redact_text(event),
            "logger": record.name,
            "run_id": self._redactor.redact(extras.pop("run_id", None)),
            "stage": self._redactor.redact(extras.pop("stage", None)),
        }
        for key, value in self._redactor.redact_mapping(extras).items():
            if key in _FIXED_FIELDS:
                key = f"extra_{key}"
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self._redactor.redact_text(self.formatException(record.exc_info))
        if record.stack_info:
            payload["stack"] = self._redactor.redact_text(self.formatStack(record.stack_info))
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


class _RedactingJsonHandler(logging.StreamHandler[TextIO]):
    """Marker type so ``configure_logging`` can replace its own handler idempotently."""


def settings_secrets(settings: Settings) -> tuple[SecretStr, ...]:
    """Every secret value in Settings except the account number (passed separately)."""
    candidates = (
        settings.ANTHROPIC_API_KEY,
        settings.ROBINHOOD_MCP_ACCESS_TOKEN,
        settings.WHEELTA_MCP_TOKEN,
        settings.DATABASE_URL,
        settings.HEARTBEAT_URL,
        settings.ALERT_WEBHOOK_URL,
    )
    return tuple(secret for secret in candidates if secret is not None)


def configure_logging(
    level: LogLevel | str,
    account_number: SecretStr | None,
    *,
    secrets: Iterable[SecretStr] = (),
    stream: TextIO | None = None,
    logger: logging.Logger | None = None,
) -> logging.Handler:
    """Install a redacting JSON handler on ``logger`` (default: the root logger) and return it.

    Pass ``settings_secrets(settings)`` as ``secrets`` so known secret values (including the
    heartbeat and alert webhook URLs) are scrubbed from any record. Calling it again replaces
    the handler it installed before; other handlers are left alone.
    """
    target = logger if logger is not None else logging.getLogger()
    for existing in list(target.handlers):
        if isinstance(existing, _RedactingJsonHandler):
            target.removeHandler(existing)
    handler = _RedactingJsonHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(
        JsonFormatter(Redactor(account_number=account_number, secrets=tuple(secrets)))
    )
    target.addHandler(handler)
    target.setLevel(LogLevel(level).value if isinstance(level, str) else level.value)
    return handler


class RunLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """A logger bound to ``run_id``/``stage`` (and other fields); call-site extras merge in."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        bound: Mapping[str, object] = self.extra or {}
        kwargs["extra"] = {**bound, **(kwargs.get("extra") or {})}
        return msg, kwargs

    def bind(self, **fields: object) -> "RunLoggerAdapter":
        """Return a new adapter with additional or replaced bound fields."""
        return RunLoggerAdapter(self.logger, {**(self.extra or {}), **fields})


def bind(
    logger: logging.Logger, *, run_id: str | None, stage: str | None = None, **fields: object
) -> RunLoggerAdapter:
    """Bind ``run_id``, ``stage`` and other fields to every record logged through the result."""
    return RunLoggerAdapter(logger, {"run_id": run_id, "stage": stage, **fields})
