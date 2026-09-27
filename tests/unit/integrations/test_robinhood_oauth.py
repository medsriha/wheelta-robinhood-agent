"""Robinhood OAuth refresh and token vault (ADR-0021). No network: httpx.MockTransport."""

import json
import secrets
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from cryptography.fernet import Fernet
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import SecretStr

from wheelta_robinhood_agent.integrations.robinhood.oauth import (
    _OAUTH_ERROR_CODES,
    DISCOVERY_URL,
    TOKEN_ENDPOINT,
    OAuthDiscoveryFailed,
    OAuthError,
    OAuthRefreshFailed,
    OAuthRefreshOutcomeUnknown,
    TokenPair,
    discover_token_endpoint,
    fingerprint,
    refresh_access_token,
)
from wheelta_robinhood_agent.integrations.robinhood.token_vault import TokenVault, TokenVaultError

NOW = datetime(2026, 9, 26, 15, tzinfo=UTC)
CLIENT_ID = "client-abc"


def _key() -> SecretStr:
    return SecretStr(Fernet.generate_key().decode())


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


class Recorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def _refresh(recorder: Recorder, old: str = "old-refresh") -> TokenPair:
    with _client(httpx.MockTransport(recorder)) as client:
        return refresh_access_token(
            client, TOKEN_ENDPOINT, CLIENT_ID, SecretStr(old), obtained_at=NOW
        )


def _ok(
    access: str = "new-access", refresh: str = "new-refresh", **extra: object
) -> httpx.Response:
    body = {"access_token": access, "refresh_token": refresh, "expires_in": 496235,
            "token_type": "Bearer", "scope": "internal", **extra}  # fmt: skip
    return httpx.Response(200, json=body)


# -- vault ----------------------------------------------------------------------------------------


def test_vault_round_trip() -> None:
    vault = TokenVault(_key())
    blob = vault.encrypt(SecretStr("acc-1"), SecretStr("ref-1"))
    assert b"acc-1" not in blob and b"ref-1" not in blob
    tokens = vault.decrypt(blob)
    assert tokens.access_token.get_secret_value() == "acc-1"
    assert tokens.refresh_token.get_secret_value() == "ref-1"
    assert "acc-1" not in repr(tokens) and "ref-1" not in repr(tokens)


def test_vault_wrong_key_is_a_typed_error() -> None:
    blob = TokenVault(_key()).encrypt(SecretStr("acc-1"), SecretStr("ref-1"))
    with pytest.raises(TokenVaultError, match="wrong key or corrupted") as info:
        TokenVault(_key()).decrypt(blob)
    assert "acc-1" not in str(info.value)


def test_vault_corrupted_ciphertext_and_bad_key() -> None:
    vault = TokenVault(_key())
    with pytest.raises(TokenVaultError):
        vault.decrypt(b"not-a-fernet-token")
    with pytest.raises(TokenVaultError, match="not a valid Fernet key"):
        TokenVault(SecretStr("short"))
    key = _key()
    assert key.get_secret_value() not in repr(TokenVault(key))


def test_vault_rejects_unexpected_plaintext_shape() -> None:
    key = _key()
    blob = Fernet(key.get_secret_value().encode()).encrypt(json.dumps({"x": 1}).encode())
    with pytest.raises(TokenVaultError, match="unexpected shape"):
        TokenVault(key).decrypt(blob)


# -- refresh --------------------------------------------------------------------------------------


def test_refresh_success_posts_the_refresh_grant_and_rotates() -> None:
    rec = Recorder([_ok()])
    pair = _refresh(rec)
    assert pair.access_token.get_secret_value() == "new-access"
    assert pair.refresh_token.get_secret_value() == "new-refresh"
    assert pair.expires_in == 496235
    assert pair.access_expires_at == NOW + timedelta(seconds=496235)
    assert "new-access" not in repr(pair) and "new-refresh" not in repr(pair)
    [request] = rec.requests
    assert request.method == "POST" and str(request.url) == TOKEN_ENDPOINT
    form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
    assert form == {"grant_type": "refresh_token", "refresh_token": "old-refresh",
                    "client_id": CLIENT_ID}  # fmt: skip


def test_refresh_error_surfaces_only_status_and_code_and_is_not_retried() -> None:
    body = {"error": "invalid_grant", "error_description": "token old-refresh was revoked"}
    rec = Recorder([httpx.Response(400, json=body), _ok()])
    with pytest.raises(OAuthRefreshFailed) as info:
        _refresh(rec)
    assert info.value.status_code == 400 and info.value.error_code == "invalid_grant"
    assert str(info.value) == "refresh rejected: HTTP 400, error=invalid_grant"
    assert "old-refresh" not in str(info.value) and "revoked" not in str(info.value)
    assert len(rec.requests) == 1  # never retried


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(401, text="<html>old-refresh</html>"), "unparseable"),
        (httpx.Response(400, json={"error": "Weird Code old-refresh"}), "unrecognized"),
        (httpx.Response(500, json=["x"]), "unrecognized"),
    ],
)
def test_refresh_error_code_is_sanitized(response: httpx.Response, code: str) -> None:
    with pytest.raises(OAuthRefreshFailed) as info:
        _refresh(Recorder([response]))
    assert info.value.error_code == code
    assert "old-refresh" not in str(info.value)


def test_refresh_transport_error_is_unknown_outcome_and_not_retried() -> None:
    calls = []

    def boom(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    with _client(httpx.MockTransport(boom)) as client, pytest.raises(OAuthRefreshOutcomeUnknown):
        refresh_access_token(client, TOKEN_ENDPOINT, CLIENT_ID, SecretStr("r"), obtained_at=NOW)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json new-access"),
        httpx.Response(200, json={"access_token": "new-access", "expires_in": 5}),  # no refresh
        _ok(expires_in=0),
        _ok(token_type="mac"),  # noqa: S106 - a token type, not a secret
        _ok(refresh="old-refresh"),  # not rotated
        _ok(access="has space"),
        httpx.Response(204),
    ],
)
def test_unusable_success_is_unknown_outcome(response: httpx.Response) -> None:
    with pytest.raises(OAuthRefreshOutcomeUnknown) as info:
        _refresh(Recorder([response]))
    assert "new-access" not in str(info.value) and "old-refresh" not in str(info.value)


# -- discovery ------------------------------------------------------------------------------------


def _meta(**over: object) -> dict[str, object]:
    return {"token_endpoint": TOKEN_ENDPOINT,
            "grant_types_supported": ["authorization_code", "refresh_token"], **over}  # fmt: skip


def test_discovery_matches_the_pinned_endpoint() -> None:
    rec = Recorder([httpx.Response(200, json=_meta())])
    with _client(httpx.MockTransport(rec)) as client:
        assert discover_token_endpoint(client) == TOKEN_ENDPOINT
    assert str(rec.requests[0].url) == DISCOVERY_URL


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=_meta(token_endpoint="https://evil.example/token")),  # noqa: S106
        httpx.Response(200, json=_meta(grant_types_supported=["authorization_code"])),
        httpx.Response(503),
        httpx.Response(200, text="nope"),
        httpx.Response(200, json=[1]),
    ],
)
def test_discovery_fails_closed(response: httpx.Response) -> None:
    with _client(httpx.MockTransport(Recorder([response]))) as client:
        with pytest.raises(OAuthDiscoveryFailed):
            discover_token_endpoint(client)


def test_discovery_transport_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with _client(httpx.MockTransport(boom)) as client, pytest.raises(OAuthDiscoveryFailed):
        discover_token_endpoint(client)


def test_fingerprint_is_short_and_not_the_token() -> None:
    fp = fingerprint(SecretStr("tok-123"))
    assert len(fp) == 8 and "tok" not in fp
    assert fp != fingerprint(SecretStr("tok-124"))


# -- no token value ever reaches an exception message ---------------------------------------------

# Random printable tokens; a token that happens to spell an OAuth error code is not a leak.
_TOKEN = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=16, max_size=64
).filter(lambda t: t not in _OAUTH_ERROR_CODES)


@settings(max_examples=60, deadline=None)
@given(old=_TOKEN, new_access=_TOKEN, new_refresh=_TOKEN, status=st.sampled_from([400, 401, 500]))
def test_no_token_in_any_error_message(
    old: str, new_access: str, new_refresh: str, status: int
) -> None:
    echo = {"error": old, "error_description": f"{old} {new_access}", "access_token": new_access}
    responses = [
        httpx.Response(status, json=echo),
        httpx.Response(status, text=f"{old}{new_refresh}"),
        httpx.Response(
            200, json={"access_token": new_access, "expires_in": -1, "refresh_token": new_refresh}
        ),  # fmt: skip
        httpx.Response(200, text=f"{{{old}"),
    ]
    for response in responses:
        with pytest.raises(OAuthError) as info:
            _refresh(Recorder([response]), old=old)
        text = f"{info.value!s} {info.value!r} {info.value.args!r}"
        for token in (old, new_access, new_refresh):
            assert token not in text


def test_random_tokens_do_not_leak_through_vault_errors() -> None:
    token = secrets.token_urlsafe(32)
    blob = TokenVault(_key()).encrypt(SecretStr(token), SecretStr(token + "r"))
    with pytest.raises(TokenVaultError) as info:
        TokenVault(_key()).decrypt(blob)
    assert token not in f"{info.value!s}{info.value!r}"
