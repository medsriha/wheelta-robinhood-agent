"""Operator script: can a Railway run authenticate to the Robinhood MCP with a refresh token?

Runs Robinhood's own OAuth (authorization code + PKCE, discovered from
https://agent.robinhood.com/.well-known/oauth-authorization-server/mcp/trading) once in the
owner's browser, then exercises one `refresh_token` grant and one MCP connection check, and
reports what it learned. Never trades and never calls a Robinhood tool.

Secrets: tokens are written only to the repo's gitignored `.env` (mode 0600), as the seed-only
keys ROBINHOOD_OAUTH_CLIENT_ID, ROBINHOOD_OAUTH_ACCESS_TOKEN, ROBINHOOD_OAUTH_REFRESH_TOKEN,
ROBINHOOD_OAUTH_OBTAINED_AT, and ROBINHOOD_OAUTH_EXPIRES_IN. Those keys are replaced in place
(or appended); every other `.env` line is kept as-is. Values are never printed or logged.
Non-standard response fields (`mfa_code`, `backup_code`, `user_uuid`) are discarded.

Then seed the ledger (ADR-0021):
`uv run python -m wheelta_robinhood_agent.orchestrator.seed_robinhood_credential --env-file .env`.
Refresh tokens rotate: once any run refreshes, these `.env` values are stale and the ledger row
is the source of truth; re-seeding needs a fresh run of this probe.

Usage (owner, on a desktop with a browser):  uv run python scripts/robinhood_oauth_probe.py
"""

import base64
import hashlib
import http.server
import os
import secrets
import stat
import sys
import threading
import time
import urllib.parse
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

METADATA_URL = "https://agent.robinhood.com/.well-known/oauth-authorization-server/mcp/trading"
RESOURCE = "https://agent.robinhood.com/mcp/trading"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
KEPT_FIELDS = ("access_token", "refresh_token", "expires_in")
# Record field → seed-only `.env` key (read by config.settings.CredentialSeedSettings).
ENV_KEYS = {
    "client_id": "ROBINHOOD_OAUTH_CLIENT_ID",
    "access_token": "ROBINHOOD_OAUTH_ACCESS_TOKEN",
    "refresh_token": "ROBINHOOD_OAUTH_REFRESH_TOKEN",
    "obtained_at": "ROBINHOOD_OAUTH_OBTAINED_AT",
    "expires_in": "ROBINHOOD_OAUTH_EXPIRES_IN",
}
LOGIN_TIMEOUT_SECONDS = 300


def out(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def fingerprint(secret: str) -> str:
    """A short, non-reversible label to tell two tokens apart without revealing either."""
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


def _env_key(line: str) -> str | None:
    """The variable a `.env` line assigns (`KEY=...` or `export KEY=...`), else None."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    return stripped.split("=", 1)[0].removeprefix("export ").strip()


def save_tokens(record: dict[str, Any]) -> None:
    """Replace (or append) the seed-only keys in `.env`, keeping every other line, mode 0600."""
    values: dict[str, str] = {}
    for field, key in ENV_KEYS.items():
        value = str(record[field])
        if not value or "'" in value or any(c.isspace() for c in value):
            raise SystemExit(f"Unexpected format for {key}; nothing was saved.")
        values[key] = f"'{value}'"  # single quotes: python-dotenv keeps the value literal
    lines = ENV_FILE.read_text().splitlines(keepends=True) if ENV_FILE.exists() else []
    written: set[str] = set()
    kept: list[str] = []
    for line in lines:
        key = _env_key(line)
        if key not in values:
            kept.append(line)
        elif key not in written:  # replace the first assignment, drop duplicates
            kept.append(f"{key}={values[key]}\n")
            written.add(key)
    missing = [key for key in values if key not in written]
    if missing:
        if kept and not kept[-1].endswith("\n"):
            kept[-1] += "\n"
        kept.append("# Robinhood OAuth seed values (scripts/robinhood_oauth_probe.py; ADR-0021)\n")
        kept.extend(f"{key}={values[key]}\n" for key in missing)
    tmp = ENV_FILE.with_name(".env.probe.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.writelines(kept)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, ENV_FILE)
    finally:
        tmp.unlink(missing_ok=True)


def kept(token_response: dict[str, Any]) -> dict[str, Any]:
    return {k: token_response[k] for k in KEPT_FIELDS if k in token_response}


class _Callback(http.server.BaseHTTPRequestHandler):
    result: dict[str, str] = {}

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _Callback.result = {k: v[0] for k, v in query.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Robinhood login received. You can close this tab.")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - silence access log
        return


def authorize(client: httpx.Client, meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Callback)
    redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"
    reg = client.post(
        meta["registration_endpoint"],
        json={
            "client_name": "wheelta-robinhood-agent (refresh probe)",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    reg.raise_for_status()
    client_id = reg.json()["client_id"]
    out(f"1. Registered a public OAuth client (client_id fingerprint {fingerprint(client_id)}).")

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    state = secrets.token_urlsafe(24)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": challenge.rstrip(b"=").decode(),
        "code_challenge_method": "S256",
        "scope": "internal",
        "state": state,
        "resource": RESOURCE,
    }
    url = meta["authorization_endpoint"] + "?" + urllib.parse.urlencode(params)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    out("2. Opening Robinhood's login in your browser. Log in and approve access.")
    out(f"   If no browser opens, visit: {url}")
    webbrowser.open(url)
    thread.join(LOGIN_TIMEOUT_SECONDS)
    server.server_close()
    result = _Callback.result
    if not result:
        raise SystemExit("No login callback within 5 minutes; nothing was saved.")
    if result.get("state") != state:
        raise SystemExit("OAuth state mismatch; aborting and saving nothing.")
    if "code" not in result:
        raise SystemExit(f"Authorization failed: {result.get('error', 'no code returned')}")

    token = client.post(
        meta["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )
    token.raise_for_status()
    return client_id, token.json()


def refresh(
    client: httpx.Client, meta: dict[str, Any], client_id: str, refresh_token: str
) -> dict[str, Any]:
    response = client.post(
        meta["token_endpoint"],
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
    )
    if response.status_code >= 400:
        # The body may echo request details; report only the status and error code.
        try:
            code = response.json().get("error", "unknown")
        except ValueError:
            code = "unparseable"
        raise SystemExit(f"Refresh failed: HTTP {response.status_code}, error={code}")
    return dict(response.json())


def mcp_connects(access_token: str) -> str:
    """Connect the MCP with the bearer token through the Agent SDK; no prompt, no tool call."""
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(
        permission_mode="dontAsk",
        tools=[],
        allowed_tools=[],
        setting_sources=[],
        strict_mcp_config=True,
        mcp_servers={
            "robinhood": {
                "type": "http",
                "url": RESOURCE,
                "headers": {"Authorization": f"Bearer {access_token}"},
            }
        },
    )

    async def check() -> str:
        async with ClaudeSDKClient(options=options) as sdk:
            deadline = time.monotonic() + 60
            while True:
                status = await sdk.get_mcp_status()
                rh = [s for s in status["mcpServers"] if s["name"] == "robinhood"]
                if (rh and rh[0]["status"] != "pending") or time.monotonic() > deadline:
                    break
                await asyncio.sleep(2)
        if not rh:
            return "not reported"
        return f"{rh[0]['status']} ({len(rh[0].get('tools', []))} tools)"

    return asyncio.run(check())


def main() -> None:
    with httpx.Client(timeout=30) as client:
        meta = client.get(METADATA_URL).json()
        client_id, first = authorize(client, meta)
        obtained = datetime.now(UTC)
        out("3. Login succeeded. Token response fields (values not shown):")
        out(f"   {sorted(first)}")
        out(f"   expires_in = {first.get('expires_in')} s")
        has_refresh = "refresh_token" in first
        out(f"   refresh_token present: {has_refresh}")
        save_tokens({"client_id": client_id, "obtained_at": obtained.isoformat(), **kept(first)})
        out(f"   Saved ROBINHOOD_OAUTH_* to {ENV_FILE} (mode 0600; other lines kept).")
        if not has_refresh:
            raise SystemExit("No refresh token issued: headless refresh is not possible.")

        out("4. Trying one refresh_token grant...")
        second = refresh(client, meta, client_id, first["refresh_token"])
        rotated = second.get("refresh_token") not in (None, first["refresh_token"])
        out(f"   Refresh succeeded. expires_in = {second.get('expires_in')} s")
        out(f"   New refresh token issued (rotation): {rotated}")
        out(
            "   access token changed: "
            f"{fingerprint(first['access_token']) != fingerprint(second['access_token'])}"
        )
        record = {"client_id": client_id, "obtained_at": datetime.now(UTC).isoformat(),
                  **kept(second)}  # fmt: skip
        if "refresh_token" not in second:
            record["refresh_token"] = first["refresh_token"]
        save_tokens(record)

        # Deliberately no reuse test of the old refresh token: many providers treat reuse
        # after rotation as theft and revoke the whole token family.

    out("5. Connecting the Robinhood MCP with the refreshed access token (no tool calls)...")
    out(f"   MCP status: {mcp_connects(record['access_token'])}")
    out("Done. Tokens were never printed.")
    out(
        "Next: uv run python -m wheelta_robinhood_agent.orchestrator.seed_robinhood_credential "
        "--env-file .env  (set APP_ENV, DATABASE_URL, ROBINHOOD_TOKEN_ENCRYPTION_KEY for the "
        "target environment). One login per environment."
    )


if __name__ == "__main__":
    main()
