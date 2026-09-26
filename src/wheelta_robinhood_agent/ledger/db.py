"""Ledger database connections (ADR-0002). The URL is a secret: never logged or echoed."""

import psycopg
from pydantic import SecretStr

from wheelta_robinhood_agent.ledger.errors import LedgerUnavailable

DEFAULT_STATEMENT_TIMEOUT_MS = 15_000
DEFAULT_CONNECT_TIMEOUT_S = 10
APPLICATION_NAME = "wheelta-robinhood-agent"


def connect(
    database_url: SecretStr,
    *,
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    connect_timeout_s: int = DEFAULT_CONNECT_TIMEOUT_S,
    application_name: str = APPLICATION_NAME,
) -> psycopg.Connection[tuple[object, ...]]:
    """Open an autocommit connection with an explicit server-side statement timeout.

    Autocommit is deliberate: repository functions open explicit `conn.transaction()` blocks,
    and session-level advisory locks (lock.py) must not be tied to an implicit transaction.
    Raises LedgerUnavailable without the driver message, which may contain connection details.
    """
    if statement_timeout_ms <= 0 or connect_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    try:
        return psycopg.connect(
            database_url.get_secret_value(),
            autocommit=True,
            connect_timeout=connect_timeout_s,
            application_name=application_name,
            options=f"-c statement_timeout={statement_timeout_ms} -c TimeZone=UTC",
        )
    except psycopg.Error as exc:
        raise LedgerUnavailable(
            f"could not connect to the ledger database ({type(exc).__name__})"
        ) from None
