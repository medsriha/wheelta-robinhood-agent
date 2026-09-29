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
from wheelta_robinhood_agent.agent.mignons import MignonLimits, Role
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
from wheelta_robinhood_agent.domain.enums import (
    ExecutionMode,
    MignonType,
    OrderVenue,
    SourceStatus,
    ToolTier,
)
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


def _plan(
    accepted: bool = True,
    rh: Any = RH,
    rh_registry: Any = RH_VERIFIED,
    proxy_accepted: bool = False,
) -> Any:
    return plan_session(
        effective_mode=ExecutionMode.OFF,
        workspace_writes=True,
        sources=(
            RemoteSource(rh_registry, rh, required=True),
            RemoteSource(WHEELTA_REGISTRY, WT),
        ),
        observed_at=NOW,
        remote_boundary_accepted=accepted,
        proxy_accepted=proxy_accepted,
    )


def test_nothing_accepted_withholds_every_remote_source_and_blocks_the_session() -> None:
    plan = _plan(accepted=False)
    assert set(plan.withheld) == {"robinhood", "wheelta"}
    assert plan.required_unavailable == ("robinhood",)
    assert not plan.may_start
    assert plan.servers == ()
    assert set(plan.tool_access.allowed_tools) == {"Agent", "WebSearch", "WebFetch", *LOCAL_TOOLS}
    assert {o.status for o in plan.observations} == {SourceStatus.DISABLED}


def test_production_defaults_proxy_token_servers_and_never_deliver_directly() -> None:
    """ADR-0023: a verified server with a token is proxied; nothing is direct."""
    plan = plan_session(
        effective_mode=ExecutionMode.OFF,
        workspace_writes=True,
        sources=(
            RemoteSource(RH_VERIFIED, RH, required=True),
            RemoteSource(WHEELTA_REGISTRY, WT),
        ),
        observed_at=NOW,
    )
    assert plan.proxied == (RH,) and plan.servers == ()
    assert set(plan.withheld) == {"wheelta"}  # unverified registry, proxy or not
    assert plan.may_start
    assert "mcp__robinhood__get_option_quotes" in plan.tool_access.allowed_tools


def test_proxied_dry_run_gets_the_live_order_tools_on_the_simulated_venue() -> None:
    """ADR-0038: with Robinhood proxied, a dry run sees exactly the live option-order tools
    (answered by the simulated broker); every other Tier X tool stays denied."""
    plan = plan_session(
        effective_mode=ExecutionMode.OFF,
        workspace_writes=True,
        sources=(RemoteSource(RH_VERIFIED, RH, required=True),),
        observed_at=NOW,
    )
    assert plan.order_venue is OrderVenue.SIMULATED
    allowed = set(plan.tool_access.allowed_tools)
    assert ORDER_TOOLS <= allowed
    denied_x = {
        RH_VERIFIED.qualified(t.name)
        for t in RH_VERIFIED.by_tier(ToolTier.X)
        if not t.live_order_tool
    }
    assert denied_x and not denied_x & allowed
    assert denied_x <= set(plan.tool_access.disallowed_tools)


def test_direct_robinhood_dry_run_has_no_order_venue() -> None:
    """ADR-0038: a direct server cannot be intercepted, so its dry run is proposal-only."""
    plan = _plan()
    assert plan.servers == (RH,) and plan.order_venue is OrderVenue.NONE
    assert not ORDER_TOOLS & set(plan.tool_access.allowed_tools)


def test_stored_cli_login_cannot_be_proxied_and_needs_direct_acceptance() -> None:
    login = McpHttpServer(
        name="robinhood",
        url="https://rh.example/mcp",  # type: ignore[arg-type]
        uses_stored_cli_login=True,
    )
    refused = _plan(accepted=False, rh=login, proxy_accepted=True)
    assert "no bearer token for the validating proxy" in refused.withheld["robinhood"]
    assert not refused.may_start
    local = _plan(accepted=True, rh=login, proxy_accepted=True)  # ADR-0019 local opt-in
    assert local.servers == (login,) and local.proxied == ()


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


def test_order_tools_only_in_live_and_never_in_off() -> None:
    plan = _plan()
    assert plan.order_venue is OrderVenue.NONE
    assert not ORDER_TOOLS & set(plan.tool_access.allowed_tools)
    assert ORDER_TOOLS <= set(plan.tool_access.disallowed_tools)
    live = plan_session(
        effective_mode=ExecutionMode.LIVE,
        workspace_writes=True,
        sources=(RemoteSource(RH_VERIFIED, RH, required=True),),
        observed_at=NOW,
        remote_boundary_accepted=True,
    )
    assert ORDER_TOOLS <= set(live.tool_access.allowed_tools)
    denied_x = {
        RH_VERIFIED.qualified(t.name)
        for t in RH_VERIFIED.by_tier(ToolTier.X)
        if not t.live_order_tool
    }
    assert not denied_x & set(live.tool_access.allowed_tools)
    assert live.order_venue is OrderVenue.BROKER
    leaky_live = ToolAccess(
        effective_mode=ExecutionMode.LIVE,
        order_venue=OrderVenue.BROKER,
        allowed_tools=tuple(sorted(denied_x)),
        disallowed_tools=(),
    )
    with pytest.raises(SessionPlanError):
        assert_no_order_tools(leaky_live, (ROBINHOOD_REGISTRY,))
    leaky = ToolAccess(
        effective_mode=ExecutionMode.OFF,
        order_venue=OrderVenue.NONE,
        allowed_tools=tuple(sorted(ORDER_TOOLS)),
        disallowed_tools=(),
    )
    with pytest.raises(SessionPlanError):
        assert_no_order_tools(leaky, (ROBINHOOD_REGISTRY,))


def test_tool_table_lists_only_allowed_verified_tools_and_withheld_sources() -> None:
    table = available_tools_table(_plan())
    assert f"`mcp__{LOCAL_SERVER_NAME}__{FACTS_TOOL_NAME}`" in table
    # web_cache_lookup is a Mignon tool: the orchestrator's table omits it.
    assert f"`mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}`" not in table
    assert f"`mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}`" in available_tools_table(
        _plan(), Role.COMPANY
    )
    assert "`mcp__robinhood__get_option_quotes`" in table
    assert "robinhood tool" not in table  # no placeholder purposes
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
        "mignon_prompts": {m: f"{m.value} prompt" for m in MignonType},
        "mignon_limits": MignonLimits(8, 4, 40),
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


def test_orchestrator_table_lists_its_tools_then_mignon_types_and_models_once() -> None:
    plan = _plan()
    models = ("claude-haiku-4-5", "claude-opus-4-8")
    table = available_tools_table(plan, mignon_models=models)
    head, roster = table.split("Mignon types")
    assert "| `Agent` | D |" in head
    assert "`mcp__robinhood__get_option_positions`" in head
    for mignon in MignonType:
        assert roster.count(f"- `{mignon.value}`:") == 1
    for model in models:
        assert roster.count(f"- `{model}`:") == 1
    # A Mignon's tools are its own prompt's business, never the orchestrator's table.
    assert "WebSearch" not in table and "get_option_chains" not in table
    assert "$1/$5" in roster


def test_login_scoped_workspace_reads_are_offered() -> None:
    """ADR-0026: workspace reads stay listed and callable (login-scoped)."""
    plan = _plan()
    table = available_tools_table(plan)
    for tool in ("get_scans", "get_watchlists", "get_alerts", "get_option_watchlist"):
        assert f"mcp__robinhood__{tool}" in plan.tool_access.allowed_tools
        assert f"`mcp__robinhood__{tool}`" in table
    assert "mcp__robinhood__run_scan" in available_tools_table(plan, Role.MARKET)


def test_mignon_table_lists_only_its_role() -> None:
    plan = _plan()
    table = available_tools_table(plan, Role.COMPANY)
    assert "`WebFetch`" in table and "get_option_positions" not in table
    assert "Agent" not in table and "Mignon `" not in table


def test_disabled_mignons_leave_the_table_without_a_roster() -> None:
    plan = plan_session(
        effective_mode=ExecutionMode.OFF,
        workspace_writes=False,
        sources=(RemoteSource(ROBINHOOD_REGISTRY, RH, required=True),),
        observed_at=NOW,
        mignons=False,
    )
    table = available_tools_table(plan)
    assert "Agent" not in table and "Mignon `" not in table
