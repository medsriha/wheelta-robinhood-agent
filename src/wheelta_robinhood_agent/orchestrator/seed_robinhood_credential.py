"""Operator command: seed the ledger with a Robinhood OAuth credential (ADR-0021).

    uv run python -m wheelta_robinhood_agent.orchestrator.seed_robinhood_credential --env-file .env

Reads, through config (`CredentialSeedSettings`), APP_ENV, DATABASE_URL,
ROBINHOOD_TOKEN_ENCRYPTION_KEY, and the seed-only values `scripts/robinhood_oauth_probe.py`
writes to the gitignored `.env`: ROBINHOOD_OAUTH_CLIENT_ID, ROBINHOOD_OAUTH_ACCESS_TOKEN,
ROBINHOOD_OAUTH_REFRESH_TOKEN, ROBINHOOD_OAUTH_OBTAINED_AT, ROBINHOOD_OAUTH_EXPIRES_IN.
Process environment variables override `--env-file` values (so APP_ENV/DATABASE_URL/key can
target another environment from the shell). It encrypts the pair and appends a `seed` row to
`oauth_credentials`; the new row becomes the current credential for that environment.

Prints only non-secret facts: environment, credential id, expiry, token fingerprints.

Refresh tokens rotate: once any run refreshes, the `.env` values are stale and the ledger row
is the source of truth. Re-seeding needs a fresh browser login. Seed each environment from its
own login: two environments seeded from one pair share a token family, and the first refresh
kills the other. Never trades and never calls a Robinhood tool.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import psycopg

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


class SeedError(Exception):
    """Configuration or the ledger insert failed. Messages never contain a token."""


def seed(*, now: datetime, env_file: Path | None = None) -> list[str]:
    """Insert a `seed` credential and return the non-secret lines to print."""
    try:
        settings = load_credential_seed_settings(env_file)
    except SettingsError as exc:
        raise SeedError(str(exc)) from None
    access = settings.ROBINHOOD_OAUTH_ACCESS_TOKEN
    refresh = settings.ROBINHOOD_OAUTH_REFRESH_TOKEN
    try:
        vault = TokenVault(settings.ROBINHOOD_TOKEN_ENCRYPTION_KEY)
    except TokenVaultError as exc:
        raise SeedError(str(exc)) from None
    try:
        with connect(settings.DATABASE_URL) as conn:
            stored = insert_credential(
                conn,
                environment=settings.APP_ENV,
                provider=OAuthProvider.ROBINHOOD,
                client_id=settings.ROBINHOOD_OAUTH_CLIENT_ID,
                ciphertext=vault.encrypt(access, refresh),
                access_expires_at=settings.access_expires_at,
                obtained_at=settings.ROBINHOOD_OAUTH_OBTAINED_AT,
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
        f"(fingerprint {fingerprint(access)}); refresh token fingerprint {fingerprint(refresh)}.",
    ]
    if stored.access_expires_at <= now:
        lines.append("The access token has already expired; the next run will refresh it.")
    lines.append(
        "The ROBINHOOD_OAUTH_* values are now seed-only history: after the first refresh they "
        "are stale and the ledger row is the source of truth. Do not seed another environment "
        "from them; run a fresh login (scripts/robinhood_oauth_probe.py) for each environment."
    )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="dotenv file to read (e.g. .env); process environment variables take precedence",
    )
    args = parser.parse_args(argv)
    try:
        lines = seed(now=datetime.now(UTC), env_file=args.env_file)
    except SeedError as exc:
        sys.stderr.write(f"seed failed: {exc}\n")
        return 1
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
