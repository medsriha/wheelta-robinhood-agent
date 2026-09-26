"""Redaction of secrets and account numbers before anything is logged or alerted (CLAUDE.md §7).

Three layers, applied to every value that leaves the process through logs or alert payloads:

1. Key-based: a mapping value whose key names a secret (token, password, cookie, database URL,
   webhook/heartbeat URL, the Settings secret fields, ...) is replaced by ``[REDACTED]``. A value
   under an account-number key is reduced to its last four characters.
2. Type-based: pydantic ``SecretStr``/``SecretBytes`` never serialize their value, wherever
   they appear.
3. Value-based: every string is scanned for bearer/basic credentials, JWT-shaped tokens and URL
   userinfo, and for the known secret values passed in explicitly (the configured account
   number is reduced to its last four; other known secrets are replaced).

Pure: no I/O, no global state. The known secrets are passed to the constructor.
"""

import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import TypeAlias

from pydantic import BaseModel, SecretBytes, SecretStr

REDACTED = "[REDACTED]"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

# Known secrets shorter than this are not substituted inside free text: substituting a
# two-character "secret" would mangle every log line. Key- and type-based redaction still apply.
MIN_KNOWN_SECRET_LENGTH = 6

# Nesting deeper than this is replaced wholesale rather than walked.
_MAX_DEPTH = 12

# A key is secret if any of its word parts is one of these ...
_SECRET_KEY_PARTS = frozenset(
    {
        "token",
        "authorization",
        "password",
        "passwd",
        "passphrase",
        "secret",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "bearer",
        "webhook",
        "dsn",
        "apikey",
        "sessionid",
    }
)
# ... or if its normalized form contains one of these.
_SECRET_KEY_SUBSTRINGS = (
    "api_key",
    "access_key",
    "private_key",
    "database_url",
    "heartbeat_url",
    "webhook_url",
    "set_cookie",
    "client_secret",
    "refresh_token",
    "access_token",
)
# Settings fields holding secrets (config/settings.py), lowercased.
SETTINGS_SECRET_FIELDS = frozenset(
    {
        "anthropic_api_key",
        "robinhood_mcp_access_token",
        "robinhood_agentic_account_number",
        "wheelta_mcp_token",
        "database_url",
        "heartbeat_url",
        "alert_webhook_url",
    }
)
_ACCOUNT_ID_PARTS = frozenset({"number", "num", "no", "id"})
_ACCOUNT_KEYS = frozenset({"account", "acct", "accountnumber", "account_number"})

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SPLIT = re.compile(r"[^a-z0-9]+")

_AUTH_SCHEME = re.compile(r"(?i)\b(bearer|basic|token)\s+[A-Za-z0-9._~+/=-]{8,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]*:[^/\s@]*@")
_ACCOUNT_SHAPED = re.compile(r"^[A-Za-z0-9-]{5,}$")


def _normalize_key(key: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", key).lower()


def _key_parts(normalized: str) -> list[str]:
    return [part for part in _KEY_SPLIT.split(normalized) if part]


def is_secret_key(key: str) -> bool:
    """True if a mapping key names a secret (its value must never be logged)."""
    normalized = _normalize_key(key)
    joined = "_".join(_key_parts(normalized))
    if joined in SETTINGS_SECRET_FIELDS and joined != "robinhood_agentic_account_number":
        return True
    if any(part in _SECRET_KEY_PARTS for part in _key_parts(normalized)):
        return True
    return any(sub in joined for sub in _SECRET_KEY_SUBSTRINGS)


def is_account_key(key: str) -> bool:
    """True if a mapping key names an account number (logged as its last four only)."""
    parts = _key_parts(_normalize_key(key))
    joined = "_".join(parts)
    if joined in _ACCOUNT_KEYS or joined == "robinhood_agentic_account_number":
        return True
    return ("account" in parts or "acct" in parts) and any(p in _ACCOUNT_ID_PARTS for p in parts)


def mask_account(value: str) -> str:
    """Reduce an account identifier to its last four characters (CLAUDE.md §7)."""
    return f"****{value[-4:]}" if len(value) > 4 else "****"


class Redactor:
    """Redacts secrets and account numbers from arbitrary values. Stateless after construction.

    ``account_number`` is the configured Agentic account number; it is reduced to its last four
    wherever it appears. ``secrets`` are other known secret values (tokens, URLs with embedded
    credentials, the database URL); each is replaced wherever it appears.
    """

    def __init__(
        self,
        *,
        account_number: SecretStr | None = None,
        secrets: Iterable[SecretStr] = (),
    ) -> None:
        substitutions: dict[str, str] = {}
        for secret in secrets:
            raw = secret.get_secret_value()
            if len(raw) >= MIN_KNOWN_SECRET_LENGTH:
                substitutions[raw] = REDACTED
        if account_number is not None:
            raw = account_number.get_secret_value()
            if len(raw) >= MIN_KNOWN_SECRET_LENGTH:
                substitutions[raw] = mask_account(raw)
        # Longest first, so a secret containing another secret is replaced whole.
        self._substitutions: tuple[tuple[str, str], ...] = tuple(
            sorted(substitutions.items(), key=lambda item: len(item[0]), reverse=True)
        )

    def redact_text(self, text: str) -> str:
        """Redact credentials and known secret values inside a free-text string."""
        for raw, replacement in self._substitutions:
            if raw in text:
                text = text.replace(raw, replacement)
        text = _AUTH_SCHEME.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
        text = _JWT.sub(REDACTED, text)
        return _URL_USERINFO.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)

    def redact(self, value: object) -> JsonValue:
        """Return a JSON-safe, redacted copy of ``value``."""
        return self._redact(value, 0)

    def redact_mapping(self, mapping: Mapping[str, object]) -> dict[str, JsonValue]:
        """Return a JSON-safe, redacted copy of a string-keyed mapping."""
        return {
            self.redact_text(str(k)): self._redact_item(str(k), v, 0) for k, v in mapping.items()
        }

    def _redact_item(self, key: str, value: object, depth: int) -> JsonValue:
        if value is None or isinstance(value, bool):
            return value
        if is_secret_key(key):
            return REDACTED
        if is_account_key(key) and not isinstance(value, Mapping | list | tuple | BaseModel):
            if isinstance(value, SecretStr | SecretBytes):
                return REDACTED
            text = str(value.value if isinstance(value, Enum) else value)
            if _ACCOUNT_SHAPED.match(text):
                return mask_account(text)
        return self._redact(value, depth + 1)

    def _redact(self, value: object, depth: int) -> JsonValue:
        if depth > _MAX_DEPTH:
            return REDACTED
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, SecretStr | SecretBytes):
            return REDACTED
        if isinstance(value, Enum):
            return self._redact(value.value, depth)
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, bytes):
            return REDACTED
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime | date):
            return value.isoformat()
        if isinstance(value, BaseModel):
            # model_dump keeps SecretStr objects as-is; they are redacted above.
            return self._redact(value.model_dump(mode="python"), depth)
        if isinstance(value, Mapping):
            return {
                self.redact_text(str(k)): self._redact_item(str(k), v, depth)
                for k, v in value.items()
            }
        if isinstance(value, list | tuple | set | frozenset):
            items = sorted(value, key=repr) if isinstance(value, set | frozenset) else value
            return [self._redact(item, depth + 1) for item in items]
        return self.redact_text(repr(value))
