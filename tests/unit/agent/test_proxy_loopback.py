"""ADR-0063: the Mignons' per-role proxy servers and the loopback ASGI app.

No socket: requests go to the ASGI app through httpx's in-process transport. The real CLI
over a real loopback listener is tests/e2e/test_e2e_orchestrator_tool_visibility_cli.py.
"""

from typing import Any

import anyio
import httpx
import pytest
from test_proxy import rig

from wheelta_robinhood_agent.agent import proxy as proxy_module
from wheelta_robinhood_agent.agent.mignons import Role, role_tools
from wheelta_robinhood_agent.agent.proxy import (
    LoopbackProxyApp,
    build_role_proxy_servers,
    role_path,
)
from wheelta_robinhood_agent.agent.tool_access import build_tool_access
from wheelta_robinhood_agent.domain.enums import ExecutionMode, OrderVenue
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY

TOKEN = "k" * 43
MIGNON_ROLES = (Role.MARKET, Role.COMPANY, Role.MACRO)
PROTOCOL = "2025-06-18"
HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": PROTOCOL,
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


def _allowed(mode: ExecutionMode = ExecutionMode.OFF, **kw: Any) -> tuple[str, ...]:
    return build_tool_access(
        effective_mode=mode, workspace_writes=True, registries=(ROBINHOOD_REGISTRY,), **kw
    ).allowed_tools


def _proxies() -> Any:
    r = rig()
    r.upstream._tools = tuple(t.name for t in ROBINHOOD_REGISTRY.tools)
    return {"robinhood": (r.proxy, ROBINHOOD_REGISTRY)}


Request = tuple[str, Any, dict[str, str]]


def _send(app: LoopbackProxyApp, *requests: Request) -> list[httpx.Response]:
    """POST each (path, JSON body, extra headers) in order, inside one `running()`."""

    async def go() -> list[httpx.Response]:
        async with app.running():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1:41234"
            ) as client:
                return [
                    await client.post(
                        path,
                        json=body,
                        headers=HEADERS | {"mcp-protocol-version": PROTOCOL} | extra,
                    )
                    for path, body, extra in requests
                ]

    return anyio.run(go)


def _listings(app: LoopbackProxyApp, roles: Any) -> dict[Role, set[str]]:
    auth = {"Authorization": app.authorization}
    requests: list[Request] = []
    for role in roles:
        requests += [
            (role_path(role, "robinhood"), INIT, auth),
            (role_path(role, "robinhood"), LIST, auth),
        ]
    responses = _send(app, *requests)
    out: dict[Role, set[str]] = {}
    for role, listed in zip(roles, responses[1::2], strict=True):
        assert listed.status_code == 200, listed.text
        out[role] = {f"mcp__robinhood__{t['name']}" for t in listed.json()["result"]["tools"]}
    return out


def test_each_role_lists_exactly_its_allowed_tools_of_the_source() -> None:
    allowed = _allowed()
    servers = build_role_proxy_servers(_proxies(), allowed, MIGNON_ROLES)
    assert set(servers) == set(MIGNON_ROLES)
    app = LoopbackProxyApp(TOKEN)
    app.mount(servers)
    assert app.paths == {role_path(r, "robinhood") for r in MIGNON_ROLES}
    listings = _listings(app, MIGNON_ROLES)
    for role in MIGNON_ROLES:
        expected = {t for t in allowed if t in role_tools(role) and "__robinhood__" in t}
        assert listings[role] == expected
        assert not expected & role_tools(Role.ORCHESTRATOR) - role_tools(role)


def test_a_role_with_no_tool_of_a_source_gets_no_server() -> None:
    allowed = [t for t in _allowed() if t in role_tools(Role.MARKET)]
    servers = build_role_proxy_servers(_proxies(), allowed, MIGNON_ROLES)
    assert set(servers) == {Role.MARKET, Role.COMPANY, Role.MACRO} & {
        r for r in MIGNON_ROLES if role_tools(r) & set(allowed)
    }
    assert build_role_proxy_servers(_proxies(), [], MIGNON_ROLES) == {}


def test_order_tools_never_reach_a_mignon_listing_even_in_live() -> None:
    allowed = _allowed(ExecutionMode.LIVE, venue=OrderVenue.BROKER)
    assert "mcp__robinhood__cancel_option_order" in allowed
    app = LoopbackProxyApp(TOKEN)
    app.mount(build_role_proxy_servers(_proxies(), allowed, MIGNON_ROLES))
    for listed in _listings(app, MIGNON_ROLES).values():
        assert not {t for t in listed if "option_order" in t}


def test_the_orchestrator_is_never_served_over_loopback() -> None:
    with pytest.raises(ValueError, match="in-process"):
        build_role_proxy_servers(_proxies(), _allowed(), [Role.ORCHESTRATOR])


def test_a_tier_x_tool_in_a_role_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    place = "mcp__robinhood__place_option_order"
    monkeypatch.setattr(proxy_module, "role_allowed", lambda role, allowed: (place,))
    with pytest.raises(ValueError, match="never be served to a Mignon"):
        build_role_proxy_servers(_proxies(), [place], [Role.MARKET])


def test_requests_without_the_bearer_token_are_refused() -> None:
    app = LoopbackProxyApp(TOKEN)
    app.mount(build_role_proxy_servers(_proxies(), _allowed(), MIGNON_ROLES))
    path = role_path(Role.MARKET, "robinhood")
    missing, wrong, unknown, ok = _send(
        app,
        (path, INIT, {}),
        (path, INIT, {"Authorization": "Bearer " + "x" * 43}),
        ("/mignon-market/wheelta/mcp", INIT, {"Authorization": app.authorization}),
        (path, INIT, {"Authorization": app.authorization}),
    )
    assert missing.status_code == wrong.status_code == 401
    assert unknown.status_code == 404
    assert ok.status_code == 200


def test_a_non_loopback_host_is_refused() -> None:
    app = LoopbackProxyApp(TOKEN)
    app.mount(build_role_proxy_servers(_proxies(), _allowed(), MIGNON_ROLES))
    (rebound,) = _send(
        app,
        (
            role_path(Role.MARKET, "robinhood"),
            INIT,
            {"Authorization": app.authorization, "Host": "attacker.example:41234"},
        ),
    )
    assert rebound.status_code >= 400


def test_tokens_are_long_random_and_mounted_once() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        LoopbackProxyApp("short")
    a, b = LoopbackProxyApp(), LoopbackProxyApp()
    assert a.authorization != b.authorization and len(a.authorization) >= len("Bearer ") + 32
    a.mount({})
    with pytest.raises(ValueError, match="mounted once"):
        a.mount({})
    assert a.paths == frozenset()
