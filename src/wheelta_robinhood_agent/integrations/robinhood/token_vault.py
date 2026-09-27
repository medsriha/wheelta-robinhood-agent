"""Encrypt and decrypt Robinhood OAuth token pairs for the ledger (ADR-0021).

The plaintext is the JSON object `{"access_token": ..., "refresh_token": ...}`; the ledger
stores only the Fernet ciphertext (AES-128-CBC + HMAC-SHA256, authenticated). The key is
`ROBINHOOD_TOKEN_ENCRYPTION_KEY`, a Railway secret. Metadata that is not secret (client_id,
expiry, obtained_at) lives in plain ledger columns so it can be inspected without the key.

Pure: no I/O, no clock.
"""

import json

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError


class TokenVaultError(Exception):
    """The key is malformed, or ciphertext could not be decrypted or parsed.

    Messages never contain the key, the ciphertext, or a token.
    """


class VaultTokens(BaseModel):
    """The decrypted secret half of a credential."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    access_token: SecretStr
    refresh_token: SecretStr


class TokenVault:
    """Fernet encryption with the configured key. Holds the key; never exposes it."""

    def __init__(self, key: SecretStr) -> None:
        try:
            self._fernet = Fernet(key.get_secret_value().encode())
        except (ValueError, TypeError):
            raise TokenVaultError("encryption key is not a valid Fernet key") from None

    def __repr__(self) -> str:
        return "TokenVault(key=**********)"

    def encrypt(self, access_token: SecretStr, refresh_token: SecretStr) -> bytes:
        plaintext = json.dumps(
            {
                "access_token": access_token.get_secret_value(),
                "refresh_token": refresh_token.get_secret_value(),
            }
        ).encode()
        return self._fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> VaultTokens:
        try:
            plaintext = self._fernet.decrypt(ciphertext)
        except (InvalidToken, TypeError):
            raise TokenVaultError(
                "credential could not be decrypted (wrong key or corrupted ciphertext)"
            ) from None
        try:
            return VaultTokens.model_validate_json(plaintext)
        except ValidationError:
            raise TokenVaultError("decrypted credential has an unexpected shape") from None
