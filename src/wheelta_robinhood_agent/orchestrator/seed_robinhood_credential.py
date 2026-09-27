"""Operator command: seed the ledger with a Robinhood OAuth credential (ADR-0021).

    uv run python -m wheelta_robinhood_agent.orchestrator.seed_robinhood_credential [TOKEN_FILE]

Reads the token file written by `scripts/robinhood_oauth_probe.py` (default
`~/.config/wheelta-robinhood-agent/robinhood_oauth.json`), encrypts the token pair with
ROBINHOOD_TOKEN_ENCRYPTION_KEY, and appends a `seed` row to `oauth_credentials` in
DATABASE_URL for APP_ENV. The new row becomes the current credential for that environment.

Reads only APP_ENV, DATABASE_URL, and ROBINHOOD_TOKEN_ENCRYPTION_KEY (through config).
Refuses a token file readable or writable by anyone but its owner (mode broader than 0600).
Prints only non-secret facts: environment, credential id, expiry, token fingerprints.

Seed each environment from its own browser login: refresh tokens rotate, so two environments
seeded from one file would share a token family and the first refresh would kill the other.
Never trades and never calls a Robinhood tool.
"""

import argparse
import stat
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr, ValidationError

from wheelta_robinhood_agent.config.settings import SettingsError, load_credential_seed_settings
from wheelta_robinhood_agent.integrations.robinhood.oauth import fingerprint
from wheelta_robinhood_agent.integrations.robinhood.token_vault import TokenVault, TokenVaultError
from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.errors import LedgerError
from wheelta_robinhood_agent.ledger.oauth_credentials import (
    CredentialSource,
    OAuthProvider,
    insert_credential,
)

DEFAULT_TOKEN_FILE = Path.home() / ".config" / "wheelta-robinhood-agent" / "robinhood_oauth.json"
_OWNER_ONLY = stat.S_IRUSR | stat.S_IWUSR


class SeedError(Exception):
    """The token file or configuration is unusable. Messages never contain a token."""


class TokenFile(BaseModel):
    """The probe's token file. Unknown fields (token_type, scope) are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    client_id: str = Field(min_length=1)
    obtained_at: AwareDatetime
    access_token: SecretStr
    refresh_token: SecretStr
    expires_in: int = Field(gt=0)

    @property
    def access_expires_at(self) -> datetime:
        return self.obtained_at.astimezone(UTC) + timedelta(seconds=self.expires_in)


def read_token_file(path: Path) -> TokenFile:
    """Parse the token file after checking that only its owner can read or write it."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise SeedError(f"cannot read {path} ({type(exc).__name__})") from None
    if mode & ~_OWNER_ONLY:
        raise SeedError(f"{path} has mode {mode:04o}; run `chmod 600 {path}` first")
    try:
        return TokenFile.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) or "<file>" for e in exc.errors()})
        raise SeedError(f"{path} is not a valid token file (fields: {', '.join(fields)})") from None


def seed(token_file: Path, *, now: datetime) -> list[str]:
    """Insert a `seed` credential and return the non-secret lines to print."""
    try:
        settings = load_credential_seed_settings()
    except SettingsError as exc:
        raise SeedError(str(exc)) from None
    tokens = read_token_file(token_file)
    try:
        vault = TokenVault(settings.ROBINHOOD_TOKEN_ENCRYPTION_KEY)
    except TokenVaultError as exc:
        raise SeedError(str(exc)) from None
    ciphertext = vault.encrypt(tokens.access_token, tokens.refresh_token)
    try:
        with connect(settings.DATABASE_URL) as conn:
            stored = insert_credential(
                conn,
                environment=settings.APP_ENV,
                provider=OAuthProvider.ROBINHOOD,
                client_id=tokens.client_id,
                ciphertext=ciphertext,
                access_expires_at=tokens.access_expires_at,
                obtained_at=tokens.obtained_at,
                source=CredentialSource.SEED,
            )
    except LedgerError as exc:
        raise SeedError(str(exc)) from None
    except psycopg.Error as exc:  # the driver's detail may echo row values; report the type
        raise SeedError(f"ledger insert failed ({type(exc).__name__})") from None
    lines = [
        f"Seeded Robinhood credential {stored.credential_id} for environment "
        f"{stored.environment.value}.",
        f"Access token expires at {stored.access_expires_at.isoformat()} "
        f"(fingerprint {fingerprint(tokens.access_token)}); "
        f"refresh token fingerprint {fingerprint(tokens.refresh_token)}.",
    ]
    if stored.access_expires_at <= now:
        lines.append("The access token has already expired; the next run will refresh it.")
    lines.append(
        "Do not seed another environment from this file: refresh tokens rotate, and a shared "
        "token family breaks on the first refresh. Delete the file once every seed is done."
    )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("token_file", nargs="?", type=Path, default=DEFAULT_TOKEN_FILE)
    args = parser.parse_args(argv)
    try:
        lines = seed(args.token_file, now=datetime.now(UTC))
    except SeedError as exc:
        sys.stderr.write(f"seed failed: {exc}\n")
        return 1
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
