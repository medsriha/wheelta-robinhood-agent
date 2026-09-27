"""OAuth credential repository (ADR-0021, migrations/0001_initial.sql `oauth_credentials`).

Rows are append-only. Each holds one encrypted token pair as opaque bytes: encryption and
decryption live in `integrations/robinhood/token_vault.py`; the ledger never sees plaintext.
The current credential is the latest row for (environment, provider). A refresh appends a
row that supersedes the one it consumed; the unique constraint on `supersedes_credential_id`
makes a second refresh of the same row fail instead of forking the token family.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

import psycopg
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.ledger.errors import CredentialConflict
from wheelta_robinhood_agent.ledger.ids import new_id

Conn = psycopg.Connection[tuple[object, ...]]


class OAuthProvider(StrEnum):
    ROBINHOOD = "robinhood"


class CredentialSource(StrEnum):
    SEED = "seed"  # an operator loaded a pair obtained by a browser login
    REFRESH = "refresh"  # the orchestrator obtained a new pair with the previous refresh token


class StoredCredential(BaseModel):
    """One `oauth_credentials` row. `ciphertext` is Fernet output, kept out of repr."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credential_id: uuid.UUID
    environment: AppEnv
    provider: OAuthProvider
    client_id: str
    ciphertext: bytes = Field(repr=False)
    access_expires_at: AwareDatetime
    obtained_at: AwareDatetime
    source: CredentialSource
    supersedes_credential_id: uuid.UUID | None
    recorded_at: AwareDatetime

    @field_validator("access_expires_at", "obtained_at", "recorded_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


_COLUMNS = (
    "credential_id, environment, provider, client_id, ciphertext, access_expires_at, "
    "obtained_at, source, supersedes_credential_id, recorded_at"
)


def insert_credential(
    conn: Conn,
    *,
    environment: AppEnv,
    provider: OAuthProvider,
    client_id: str,
    ciphertext: bytes,
    access_expires_at: datetime,
    obtained_at: datetime,
    source: CredentialSource,
    supersedes_credential_id: uuid.UUID | None = None,
) -> StoredCredential:
    """Append one credential row and return it.

    Raises CredentialConflict if another row already supersedes `supersedes_credential_id`.
    Driver errors are re-raised as-is (they carry no ciphertext: parameters are not echoed).
    """
    try:
        with conn.transaction():
            row = conn.execute(
                f"INSERT INTO oauth_credentials ({_COLUMNS}) "  # noqa: S608 - constant list
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, DEFAULT) "
                f"RETURNING {_COLUMNS}",
                (new_id(), environment.value, provider.value, client_id, ciphertext,
                 access_expires_at, obtained_at, source.value, supersedes_credential_id),
            ).fetchone()  # fmt: skip
    except psycopg.errors.UniqueViolation:
        raise CredentialConflict(
            "the superseded credential was already superseded by another row"
        ) from None
    assert row is not None  # noqa: S101 - INSERT ... RETURNING always yields the row
    return _credential(row)


def latest_credential(
    conn: Conn, environment: AppEnv, provider: OAuthProvider
) -> StoredCredential | None:
    """The current credential for (environment, provider), or None if none was ever stored."""
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM oauth_credentials "  # noqa: S608 - constant column list
        "WHERE environment = %s AND provider = %s "
        "ORDER BY recorded_at DESC, credential_id DESC LIMIT 1",
        (environment.value, provider.value),
    ).fetchone()
    return None if row is None else _credential(row)


def _credential(row: tuple[object, ...]) -> StoredCredential:
    ciphertext = row[4]
    if isinstance(ciphertext, memoryview):
        ciphertext = ciphertext.tobytes()
    return StoredCredential(
        credential_id=row[0],
        environment=row[1],
        provider=row[2],
        client_id=row[3],
        ciphertext=ciphertext,
        access_expires_at=row[5],
        obtained_at=row[6],
        source=row[7],
        supersedes_credential_id=row[8],
        recorded_at=row[9],
    )
