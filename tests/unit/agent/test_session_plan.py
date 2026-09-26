"""Session planning, withholding, local server options, and status observation (CLAUDE.md §8)."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import HookMatcher
from pydantic import SecretStr

from wheelta_robinhood_agent.agent.account_scope import account_scope_for, account_scope_id
from wheelta_robinhood_agent.agent.facts_tool import FACTS_TOOL_NAME
from wheelta_robinhood_agent.agent.local_server import (
    LOCAL_REGISTRY,
    LocalServerError,
    build_local_server,
)
from wheelta_robinhood_agent.agent.options import AgentOptionsError, build_agent_options
from wheelta_robinhood_agent.agent.session import (
    RemoteSource,
    SessionPlanError,
    assert_no_order_tools,
    available_tools_table,
    observe_statuses,
    plan_session,
)
from wheelta_robinhood_agent.agent.tool_access import ToolAccess
from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME, WEB_CACHE_TOOL_NAME
from wheelta_robinhood_agent.agent.withholding import ServerWithholding
from wheelta_robinhood_agent.domain.enums import ExecutionMode, SourceStatus
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    LIVE_ORDER_TOOLS,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.status import McpHttpServer, SourceObservation
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

NOW = datetime(2026, 9, 23, 15, 30, tzinfo=UTC)
RH = McpHttpServer(name="robinhood", url="https://rh.example/mcp", token=SecretStr("rh-token"))  # type: ignore[arg-type]
WT = McpHttpServer(name="wheelta", url="https://wt.example/mcp", token=SecretStr("wt-token"))  # type: ignore[arg-type]
RH_VERIFIED = ROBINHOOD_REGISTRY.model_copy(update={"verified": True})
ORDER_TOOLS = {ROBINHOOD_REGISTRY.qualified(n) for n in LIVE_ORDER_TOOLS}
LOCAL_TOOLS = {LOCAL_REGISTRY.qualified(t.name) for t in LOCAL_REGISTRY.tools}


def _plan(accepted: bool = True, rh: Any = RH, rh_registry: Any = RH_VERIFIED) -> Any:
    return plan_session(
        effective_mode=ExecutionMode.OFF,
        workspace_writes=True,
        sources=(
            RemoteSource(rh_registry, rh, required=True),
            RemoteSource(WHEELTA_REGISTRY, WT),
        ),
        observed_at=NOW,
        remote_boundary_accepted=accepted,
    )


def test_defaults_withhold_every_remote_source_and_block_the_session() -> None:
    plan = _plan(accepted=False)
    assert set(plan.withheld) == {"robinhood", "wheelta"}
    assert plan.required_unavailable == ("robinhood",)
    assert not plan.may_start
    assert plan.servers == ()
    assert set(plan.tool_access.allowed_tools) == {"WebSearch", "WebFetch", *LOCAL_TOOLS}
    assert {o.status for o in plan.observations} == {SourceStatus.DISABLED}


def test_unverified_registry_is_withheld_even_when_the_boundary_is_accepted() -> None:
    plan = _plan()
    assert "wheelta" in plan.withheld and "robinhood" not in plan.withheld
    assert plan.may_start and plan.servers == (RH,)
    allowed = set(plan.tool_access.allowed_tools)
    assert not any(t.startswith("mcp__wheelta__") for t in allowed)
    assert WHEELTA_REGISTRY.qualified("wheelta_board_query") in plan.tool_access.disallowed_tools


def test_missing_token_is_needs_auth_and_required() -> None:
    obs = SourceObservation(server="robinhood", status=SourceStatus.NEEDS_AUTH, observed_at=NOW)
    plan = _plan(rh=obs)
    assert plan.required_unavailable == ("robinhood",)
    assert obs in plan.observations


def test_order_tools_are_never_allowed_and_live_is_refused() -> None:
    plan = _plan()
    assert not ORDER_TOOLS & set(plan.tool_access.allowed_tools)
    assert ORDER_TOOLS <= set(plan.tool_access.disallowed_tools)
    with pytest.raises(SessionPlanError):
        plan_session(
            effective_mode=ExecutionMode.LIVE,
            workspace_writes=True,
            sources=(RemoteSource(RH_VERIFIED, RH, required=True),),
            observed_at=NOW,
            remote_boundary_accepted=True,
        )
    leaky = ToolAccess(
        effective_mode=ExecutionMode.OFF,
        allowed_tools=tuple(sorted(ORDER_TOOLS)),
        disallowed_tools=(),
    )
    with pytest.raises(SessionPlanError):
        assert_no_order_tools(leaky, (ROBINHOOD_REGISTRY,))


def test_tool_table_lists_only_allowed_verified_tools_and_withheld_sources() -> None:
    table = available_tools_table(_plan())
    assert f"`mcp__{LOCAL_SERVER_NAME}__{FACTS_TOOL_NAME}`" in table
    assert f"`mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}`" in table
    assert "`mcp__robinhood__get_option_quotes`" in table
    assert "place_option_order" not in table
    assert "mcp__wheelta__" not in table
    assert "- wheelta: tool registry unverified" in table


def test_observe_statuses_maps_pending_absent_and_tool_names() -> None:
    response = {
        "mcpServers": [
            {
                "name": "wra_local",
                "status": "connected",
                "tools": [{"name": f"mcp__wra_local__{t.name}"} for t in LOCAL_REGISTRY.tools],
            },
            {"name": "robinhood", "status": "pending"},
        ]
    }
    obs = observe_statuses(
        response, (RH_VERIFIED, LOCAL_REGISTRY), ["robinhood", "wra_local", "x"], NOW
    )
    by = {o.server: o for o in obs}
    assert by["robinhood"].status is SourceStatus.PENDING
    assert by["wra_local"].available
    assert by["x"].status is SourceStatus.FAILED
    missing = observe_statuses(
        {"mcpServers": [{"name": "wra_local", "status": "connected", "tools": []}]},
        (LOCAL_REGISTRY,),
        ["wra_local"],
        NOW,
    )[0]
    assert not missing.available and missing.discovery is not None and missing.discovery.missing


def test_withholding_is_add_only() -> None:
    latch = ServerWithholding()
    assert latch.withhold("wheelta", "failed")
    assert not latch.withhold("wheelta", "other")
    assert latch.reason("wheelta") == "failed"
    assert latch.reason("robinhood") is None
    assert dict(latch.snapshot()) == {"wheelta": "failed"}
    with pytest.raises(ValueError):
        latch.withhold("", "x")


def test_account_scope_id_is_stable_and_hides_the_number() -> None:
    scope = account_scope_id(SecretStr("5550001234"))
    assert scope == account_scope_id(SecretStr("5550001234"))
    assert scope != account_scope_id(SecretStr("5550001235"))
    assert "5550001234" not in scope and scope.startswith("agentic:")
    assert account_scope_for(LOCAL_SERVER_NAME, FACTS_TOOL_NAME).scope.value == (
        "not_account_scoped"
    )


# -- options with the local SDK server ------------------------------------------------------


async def _noop(*_: Any) -> Any:
    return {}


HOOKS: Any = {
    e: [HookMatcher(hooks=[_noop])] for e in ("PreToolUse", "PostToolUse", "PostToolUseFailure")
}


def _local() -> Any:
    from claude_agent_sdk import tool

    tools = [
        tool(name, "d", {"type": "object"})(_noop)
        for name in (WEB_CACHE_TOOL_NAME, FACTS_TOOL_NAME)
    ]
    return build_local_server(tools)


def _options(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "tool_access": _plan().tool_access,
        "mcp_servers": [RH],
        "hooks": HOOKS,
        "system_prompt": "prompt",
        "model": "m",
        "scratch_dir": Path("/private/tmp/wra-scratch"),
        "max_turns": 5,
        "max_budget_usd": None,
        "mcp_timeout_ms": 1000,
        "mcp_tool_timeout_ms": 1000,
        "sdk_servers": {LOCAL_SERVER_NAME: _local()},
    }
    kwargs.update(overrides)
    return build_agent_options(**kwargs)


def test_options_carry_the_local_sdk_server() -> None:
    options = _options()
    assert set(options.mcp_servers) == {"robinhood", LOCAL_SERVER_NAME}
    assert options.mcp_servers[LOCAL_SERVER_NAME]["type"] == "sdk"


def test_options_reject_unknown_or_clashing_sdk_servers() -> None:
    with pytest.raises(AgentOptionsError):
        _options(sdk_servers={"other": _local()})
    with pytest.raises(AgentOptionsError):
        _options(sdk_servers={"robinhood": _local()})


def test_local_server_must_match_its_registry() -> None:
    from claude_agent_sdk import tool

    with pytest.raises(LocalServerError):
        build_local_server([tool(WEB_CACHE_TOOL_NAME, "d", {"type": "object"})(_noop)])
