"""The orchestrator sees only its own tools; each Mignon sees and calls its own (ADR-0063).

Against the bundled real CLI (claude-agent-sdk 0.2.160, CLI 2.1.283), through our real hooks,
options (`build_agent_options` with `mignon_mcp_servers`, so `assert_safe_options` checks the
inline servers), the validating proxy, the session's `_mount_mignon_servers`, and the reference
loopback listener (`integrations/loopback_http.py`). Pinned facts:

- the main loop's model requests and the init message list only the orchestrator's tools of
  the source: no Mignon-only tool schema reaches the orchestrator's context;
- a Mignon is offered exactly its role's tools (its inline loopback server's listing plus the
  parent's tools its `AgentDefinition.tools` names), calls one, and the call passes the parent
  hook with its `agent_type`, is correlated by the proxy and forwarded once;
- an orchestrator call to a Mignon-only tool is refused by the CLI before any hook or server,
  and the PreToolUse hook still denies it if it ever arrives (layer 3 is unchanged).

The spike that chose this mechanism (ADR-0063) also found, with the same CLI: `tools=` does not
filter MCP tools; `disallowed_tools` leaves an in-process server's tools listed to the main
loop but removes them from subagents; an SDK-type server inlined in an `AgentDefinition` is not
connected for the subagent. Only an inline HTTP server is offered to the subagent alone.

Skipped unless WRA_RUN_REQUIRES_CLI=1. Re-run whenever the SDK or CLI is bumped.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from typing import Any

import anyio
import cli_harness.session as hs
import pytest
from claude_agent_sdk.types import McpSdkServerConfig
from cli_harness.mcp_server import Json
from cli_harness.model_server import FinalText, ToolUse
from cli_harness.session import MODEL, SERVER, Case, SessionOutcome
from pydantic import SecretStr
from test_e2e_result_boundary_cli import QUOTES, diagnostics, run

from wheelta_robinhood_agent.agent.hooks import HookDeps, build_hooks
from wheelta_robinhood_agent.agent.mignons import MignonLimits, Role, role_tools
from wheelta_robinhood_agent.agent.options import build_agent_options
from wheelta_robinhood_agent.agent.proxy import (
    LoopbackProxyApp,
    ValidatingProxy,
    build_proxy_server,
    upstream_timeout_seconds,
)
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyDispatch
from wheelta_robinhood_agent.agent.session import LoopbackEndpoint, _mount_mignon_servers
from wheelta_robinhood_agent.agent.tool_access import build_tool_access
from wheelta_robinhood_agent.domain.enums import ExecutionMode, ToolCallStatus
from wheelta_robinhood_agent.integrations.loopback_http import serve_loopback
from wheelta_robinhood_agent.integrations.mcp_upstream import open_http_upstream
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.status import McpHttpServer

pytestmark = [pytest.mark.requires_cli, pytest.mark.allow_hosts(["127.0.0.1"])]

LIMITS = MignonLimits(max_per_run=8, max_concurrent=4, max_turns_per_mignon=10)
MARKET_T = f"mignon-market--{MODEL}"
CHAINS = "get_option_chains"
CHAINS_TOOL = ROBINHOOD_REGISTRY.qualified(CHAINS)
FINANCIALS_TOOL = ROBINHOOD_REGISTRY.qualified("get_financials")  # company role only
SPAWN = ToolUse(
    "Agent",
    {
        "description": "screen",
        "prompt": json.dumps({"objective": "Screen AAPL puts."}),
        "subagent_type": MARKET_T,
    },
)
ALLOWED = build_tool_access(
    effective_mode=ExecutionMode.OFF,
    workspace_writes=False,
    registries=(ROBINHOOD_REGISTRY,),
    mignons=True,
).allowed_tools
ORCHESTRATOR = frozenset(ALLOWED) & role_tools(Role.ORCHESTRATOR)
MARKET = frozenset(ALLOWED) & role_tools(Role.MARKET)
MIGNON_ONLY = frozenset(ALLOWED) - ORCHESTRATOR - {"Agent"}


@dataclasses.dataclass
class Wiring:
    """What the split session was built from, for assertions after the run."""

    hook_deps: HookDeps | None = None
    inline_paths: frozenset[str] = frozenset()


def _split_session(wiring: Wiring) -> Any:
    """`cli_harness.session._session` with the ADR-0063 wiring of `build_session_options`."""

    async def session(case: Case, outcome: SessionOutcome, model_url: str, mcp_url: str) -> None:
        redactor = hs.Redactor(account_number=SecretStr(hs.ACCOUNT_NUMBER))
        dispatch = ProxyDispatch(frozenset({SERVER}))
        deps = HookDeps(
            effective_mode=ExecutionMode.OFF,
            kill_switch=False,
            workspace_writes=False,
            workspace_prefix="WRA · ",
            account_number=SecretStr(hs.ACCOUNT_NUMBER),
            rules=hs.RULES,
            run_control=outcome.run_control,
            recorder=case.recorder,
            validator=hs.BoundaryValidator(redactor=redactor, mappers=case.mappers),
            ownership=hs.NoOwnership(),
            counter=hs.ZeroCounter(),
            redactor=redactor,
            clock=case.clock,
            registries=(ROBINHOOD_REGISTRY,),
            hook_timeout_seconds=case.hook_timeout_seconds,
            mignon_limits=case.mignons,
            mignon_models=(MODEL,),
            proxy_dispatch=dispatch,
        )
        wiring.hook_deps = deps
        upstream_server = McpHttpServer(
            name=SERVER, url=f"{mcp_url}/mcp", token=SecretStr("harness-token")
        )
        async with open_http_upstream(upstream_server, connect_timeout_seconds=15) as upstream:
            proxy = ValidatingProxy(
                server=SERVER,
                upstream=upstream,
                dispatch=dispatch,
                recorder=case.recorder,
                validator=deps.validator,
                run_control=outcome.run_control,
                clock=case.clock,
                upstream_timeout_seconds=upstream_timeout_seconds(case.mcp_tool_timeout_ms),
            )
            app = LoopbackProxyApp()
            async with serve_loopback(app) as base_url:
                configs = _mount_mignon_servers(
                    LoopbackEndpoint(app=app, base_url=base_url),
                    {SERVER: (proxy, ROBINHOOD_REGISTRY)},
                    ALLOWED,
                )
                wiring.inline_paths = app.paths
                main = McpSdkServerConfig(
                    type="sdk",
                    name=SERVER,
                    instance=build_proxy_server(proxy, ROBINHOOD_REGISTRY, sorted(ORCHESTRATOR)),
                )
                build = functools.partial(build_agent_options, mignon_mcp_servers=configs)
                async with app.running():
                    hs.build_agent_options = build  # type: ignore[assignment]
                    try:
                        await hs._converse(case, outcome, model_url, deps, [], {SERVER: main})
                    finally:
                        hs.build_agent_options = build_agent_options  # type: ignore[assignment]

    return session


def split_run(case: Case, monkeypatch: pytest.MonkeyPatch) -> tuple[SessionOutcome, Wiring]:
    wiring = Wiring()
    monkeypatch.setattr(hs, "_session", _split_session(wiring))
    return run(case, monkeypatch), wiring


def chains_case(*steps: Any, mignon_steps: tuple[Any, ...] = ()) -> Case:
    return Case(
        steps=list(steps),
        mignon_steps=list(mignon_steps),
        behaviors={CHAINS: Json({"ok": True}), QUOTES: Json({"ok": True})},
        mignons=LIMITS,
    )


def test_the_orchestrator_lists_only_its_tools_and_a_mignon_calls_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, wiring = split_run(
        chains_case(
            SPAWN,
            FinalText("done"),
            mignon_steps=(ToolUse(CHAINS_TOOL, {"underlying_symbol": "AAPL"}), FinalText("{}")),
        ),
        monkeypatch,
    )
    assert wiring.inline_paths == {
        f"/{role.value}/{SERVER}/mcp" for role in (Role.MARKET, Role.COMPANY, Role.MACRO)
    }
    main = out.model.main_loop()
    assert main, diagnostics(out)
    for request in main:
        offered = set(request.tool_names())
        assert offered == ORCHESTRATOR, (offered ^ ORCHESTRATOR, diagnostics(out))
        assert not offered & MIGNON_ONLY
    init = out.init_message()
    assert init is not None and not set(init.data.get("tools", [])) & MIGNON_ONLY

    mignon = out.model.mignon_loop()
    assert mignon, "no Mignon ever ran: " + diagnostics(out)
    assert set(mignon[0].tool_names()) == MARKET, set(mignon[0].tool_names()) ^ MARKET

    assert out.mcp.called(CHAINS) == 1, diagnostics(out)  # forwarded once by the proxy
    (call,) = [c for c in out.recorder.requested_calls() if c["tool"] == CHAINS]
    assert call["agent_type"] == MARKET_T and call["agent_id"]
    tool_call_id = out.recorder.ids[call["sdk_tool_use_id"]]
    statuses = [o["status"] for o in out.recorder.outcomes_for(tool_call_id)]
    assert ToolCallStatus.DENIED not in statuses, statuses
    delivered = [d for d in out.recorder.of("delivered") if d["tool_call_id"] == tool_call_id]
    assert len(delivered) == 1, diagnostics(out)


def test_a_mignon_cannot_call_another_roles_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _ = split_run(
        chains_case(
            SPAWN,
            FinalText("done"),
            mignon_steps=(ToolUse(FINANCIALS_TOOL, {"symbols": ["AAPL"]}), FinalText("{}")),
        ),
        monkeypatch,
    )
    assert out.model.mignon_loop(), diagnostics(out)
    assert out.mcp.called("get_financials") == 0
    assert "get_financials" not in {c["tool"] for c in out.recorder.requested_calls()}


def test_an_orchestrator_call_to_a_mignon_only_tool_is_refused_and_the_hook_still_denies_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, wiring = split_run(
        chains_case(ToolUse(CHAINS_TOOL, {"underlying_symbol": "AAPL"}), FinalText("done")),
        monkeypatch,
    )
    # The CLI refuses an unlisted tool before any hook or server.
    assert out.mcp.called(CHAINS) == 0
    assert CHAINS not in {c["tool"] for c in out.recorder.requested_calls()}
    (second, *_) = out.model.main_loop()[1:] or [None]
    assert second is not None, diagnostics(out)
    assert "No such tool available" in second.text()

    # Layer 3 is unchanged: the same hooks deny the orchestrator (no agent_id) that tool.
    assert wiring.hook_deps is not None
    pre = build_hooks(wiring.hook_deps)["PreToolUse"][0].hooks[0]
    hook_input: Any = {
        "session_id": "s",
        "transcript_path": "",
        "cwd": "/scratch",
        "hook_event_name": "PreToolUse",
        "tool_name": CHAINS_TOOL,
        "tool_input": {"underlying_symbol": "AAPL"},
        "tool_use_id": "toolu_direct",
    }
    decision = anyio.run(pre, hook_input, "toolu_direct", {"signal": None})
    spec = decision.get("hookSpecificOutput", {})
    assert spec.get("permissionDecision") == "deny", decision
    assert "not available to the orchestrator" in spec["permissionDecisionReason"]
