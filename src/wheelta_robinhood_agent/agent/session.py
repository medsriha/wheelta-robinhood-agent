"""Run the single Claude Agent SDK session for a run (ARCHITECTURE.md "Run lifecycle" 4-6).

Two parts:

- `plan_session` (pure): decide, before anything connects, how each remote server is
  delivered. A server with a bearer token is **proxied** (ADR-0023): the session serves its
  tools in-process through the validating proxy (`agent/proxy.py`), accepted by the real-CLI
  acceptance tests (`PROXY_RESULT_BOUNDARY_ACCEPTED`). A server the CLI would reach itself
  (**direct**, e.g. the stored Claude Code login, ADR-0018) is exposed only when direct
  delivery is accepted (`REMOTE_RESULT_BOUNDARY_ACCEPTED`, False; ADR-0019 local dry runs
  opt in). A server is withheld when it has no credentials (Robinhood `needs-auth`), its
  registry is unverified (CLAUDE.md §9), or no accepted delivery exists. Withheld tools move
  from `allowed_tools` to `disallowed_tools`, so the model never sees them. Robinhood withheld
  or unavailable means **no session**: without it the agent cannot trade or manage positions.
- `run_agent_session` (async): open each proxied server's upstream connection (initialize +
  `tools/list`) within the connect budget and turn it into a `SourceObservation` with a
  discovery diff (401/403 → `needs-auth`); verify in trusted code, through the Robinhood
  upstream's `get_accounts`, that the configured account is Agentic-eligible (a failed or
  negative check withholds Robinhood; a pass lets `get_portfolio` snapshots be
  `agentic_verified`); build hooks (tool access layer 3, recording,
  result boundary, web cache), the proxies, and options; connect; poll `get_mcp_status` for
  direct servers only (`pending` is intermediate; CLAUDE.md §8), withhold anything
  unavailable, and only then send the start message. In-process servers (`wra_local` and the
  proxies) are absent from `get_mcp_status` until the first query (real CLI 2.1.283), so they
  are verified by the init `SystemMessage` check instead, which stops the run if Robinhood or
  `wra_local` is not connected. A RunControl stop (signal, deadline, infrastructure failure)
  interrupts the SDK. The `ResultMessage` usage/cost feeds `RunMetrics`; its final text is
  parsed strictly into AgentDecisionOutput v5, and raw (redacted) plus parsed output are
  persisted.

`assert_no_order_tools` re-checks the plan before any session is built: without an order
venue no Tier X tool is allowed; with one only the three option-order tools are. The venue
(ADR-0038) is `broker` in armed live (ADR-0034); in a dry run it is `simulated` when
Robinhood is proxied (`simulated_broker.SimulatedBroker` answers the order tools in-process)
and `none` otherwise. `broker_ledger.BrokerLedger` records orders for both venues: broker
orders under the account scope, simulated ones under the run's own simulated scope.
"""

import contextlib
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import anyio
import psycopg
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    Transport,
)
from claude_agent_sdk.types import McpSdkServerConfig

from wheelta_robinhood_agent.agent.account_scope import (
    ROBINHOOD_ACCOUNT_SCOPE,
    AccountScopeSpec,
)
from wheelta_robinhood_agent.agent.broker_ledger import BrokerLedger
from wheelta_robinhood_agent.agent.facts_tool import (
    FACTS_TOOL_NAME,
    DecisionFactsService,
    build_facts_tool,
    load_run_evidence,
)
from wheelta_robinhood_agent.agent.hooks import HookDeps, build_hooks
from wheelta_robinhood_agent.agent.ledger_adapters import (
    LedgerWorkspaceCounter,
    LedgerWorkspaceOwnership,
    ledger_result_writer,
)
from wheelta_robinhood_agent.agent.local_server import LOCAL_REGISTRY, build_local_server
from wheelta_robinhood_agent.agent.mignons import (
    DELEGATION_TOOL,
    MIGNON_DESCRIPTIONS,
    MODEL_GUIDANCE,
    ROLE_TOOLS,
    Role,
    mignon_limits,
)
from wheelta_robinhood_agent.agent.options import build_agent_options
from wheelta_robinhood_agent.agent.proxy import (
    ValidatingProxy,
    build_proxy_server,
    upstream_timeout_seconds,
)
from wheelta_robinhood_agent.agent.proxy_dispatch import ProxyDispatch
from wheelta_robinhood_agent.agent.recorder import LedgerToolEventRecorder
from wheelta_robinhood_agent.agent.result_boundary import (
    VERIFIED_MAPPERS,
    BoundaryValidator,
    EvidenceMapper,
    PayloadError,
    PayloadKind,
    extract_mcp_payload,
)
from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason
from wheelta_robinhood_agent.agent.simulated_broker import SimulatedBroker, simulated_scope_id
from wheelta_robinhood_agent.agent.tool_access import (
    ALLOWED_BUILTINS,
    ToolAccess,
    build_tool_access,
)
from wheelta_robinhood_agent.agent.web_cache import (
    LOCAL_SERVER_NAME,
    WEB_CACHE_TOOL_NAME,
    LedgerWebCacheStore,
    build_web_cache_tool,
    cached_search_denial,
    capture_web_result,
)
from wheelta_robinhood_agent.agent.withholding import ServerWithholding
from wheelta_robinhood_agent.config.facts_rules import facts_rules_from
from wheelta_robinhood_agent.config.rules import LoadedRules
from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.account import AgenticEligibility
from wheelta_robinhood_agent.domain.decision_output import (
    DecisionOutputParseResult,
    parse_agent_decision_output,
)
from wheelta_robinhood_agent.domain.enums import (
    ExecutionMode,
    MignonType,
    OrderVenue,
    SourceStatus,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.gating import executes_orders, order_venue
from wheelta_robinhood_agent.integrations.mcp_upstream import (
    McpUpstream,
    UpstreamAuthError,
    UpstreamError,
    open_http_upstream,
)
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, diff_discovered
from wheelta_robinhood_agent.integrations.robinhood.accounts import check_eligibility
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    SERVER_NAME as ROBINHOOD,
)
from wheelta_robinhood_agent.integrations.status import (
    McpHttpServer,
    SourceObservation,
    observe_server,
)
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger import tool_calls as ledger_tool_calls
from wheelta_robinhood_agent.observability.metrics import RunMetrics
from wheelta_robinhood_agent.observability.redaction import Redactor

Conn = psycopg.Connection[tuple[object, ...]]
TransportFactory = Callable[[ClaudeAgentOptions], Transport]
UpstreamFactory = Callable[[McpHttpServer, float], AbstractAsyncContextManager[McpUpstream]]

REMOTE_RESULT_BOUNDARY_ACCEPTED: Final = False
"""Whether remote MCP results may reach the model directly through the SDK hook boundary.

DATA_QUALITY.md: direct delivery (the CLI connects to the remote server) fails five of the
real-CLI result-boundary acceptance tests (isError, transport failure, oversized output, hook
exception, hook timeout), so it is not accepted. Only an ADR-0019 local dry run opts in.
"""

PROXY_RESULT_BOUNDARY_ACCEPTED: Final = True
"""Whether remote MCP results may reach the model through the validating proxy (ADR-0023).

Accepted on 2026-09-27: every real-CLI acceptance test in
`tests/e2e/test_e2e_result_boundary_cli.py` passes through the proxy against the pinned
claude-agent-sdk 0.2.160 / bundled CLI 2.1.283. A code constant, not an environment variable:
re-run those tests (WRA_RUN_REQUIRES_CLI=1) whenever the SDK or CLI is bumped.
"""

# Bounds for the session's own waits; not trading values.
STATUS_POLL_INTERVAL_SECONDS: Final = 0.5
INTERRUPT_GRACE_SECONDS: Final = 15.0
DISCONNECT_TIMEOUT_SECONDS: Final = 20.0
DEFAULT_MAX_TURNS: Final = 200
START_MESSAGE: Final = (
    "Begin this run now. Follow the procedure in your instructions and finish with the "
    "AgentDecisionOutput JSON object only."
)


class SessionPlanError(RuntimeError):
    """The planned tool exposure would violate a safety invariant. The run must not start."""


# --------------------------------------------------------------------------------------------
# Planning (pure)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteSource:
    """One remote MCP server: its registry and either a config or a pre-connect observation
    (e.g. Robinhood without a token is `needs-auth`)."""

    registry: ToolRegistry
    server: McpHttpServer | SourceObservation
    required: bool = False


@dataclass(frozen=True)
class SessionPlan:
    effective_mode: ExecutionMode
    tool_access: ToolAccess
    servers: tuple[McpHttpServer, ...]  # direct: the CLI connects to them itself
    registries: tuple[ToolRegistry, ...]
    observations: tuple[SourceObservation, ...]
    withheld: Mapping[str, str]
    required_unavailable: tuple[str, ...]
    # Served in-process through the validating proxy (ADR-0023).
    proxied: tuple[McpHttpServer, ...] = ()

    @property
    def may_start(self) -> bool:
        return not self.required_unavailable

    @property
    def order_venue(self) -> OrderVenue:
        return self.tool_access.order_venue


def assert_no_order_tools(access: ToolAccess, registries: Sequence[ToolRegistry]) -> None:
    """No Tier X tool without an order venue; with one (broker in armed live, ADR-0034, or
    the simulated broker in a proxied dry run, ADR-0038) only the live option-order tools.
    Raises SessionPlanError."""
    allowed = set(access.allowed_tools)
    orders = executes_orders(access.order_venue)
    for registry in registries:
        for spec in registry.by_tier(ToolTier.X):
            if registry.qualified(spec.name) in allowed and not (orders and spec.live_order_tool):
                raise SessionPlanError(f"order tool {spec.name} would be exposed")


def plan_session(
    *,
    effective_mode: ExecutionMode,
    workspace_writes: bool,
    sources: Sequence[RemoteSource],
    observed_at: datetime,
    remote_boundary_accepted: bool = REMOTE_RESULT_BOUNDARY_ACCEPTED,
    proxy_accepted: bool = PROXY_RESULT_BOUNDARY_ACCEPTED,
    local_registry: ToolRegistry = LOCAL_REGISTRY,
    mignons: bool = True,
) -> SessionPlan:
    """Decide the exposed servers and tools before connecting (fail closed).

    A server with a token is proxied when the proxy is accepted; otherwise, or without a
    token our code can present, it is direct only if direct delivery is accepted.
    The order venue (ADR-0038) follows: broker in live; in off, simulated when Robinhood is
    proxied (the proxy answers order calls in-process), else none (no order tools).
    """
    registries = (*(s.registry for s in sources), local_registry)
    withheld: dict[str, str] = {}
    observations: list[SourceObservation] = []
    servers: list[McpHttpServer] = []
    proxied: list[McpHttpServer] = []
    for source in sources:
        name = source.registry.server
        if isinstance(source.server, SourceObservation):
            observations.append(source.server)
            withheld[name] = f"unavailable before connect ({source.server.status.value})"
            continue
        reason = None
        proxy = proxy_accepted and source.server.token is not None
        if not source.registry.verified:
            reason = "tool registry unverified (no captured tools/list)"
        elif not proxy and not remote_boundary_accepted:
            reason = (
                "result-boundary acceptance tests have not passed"
                if source.server.token is not None
                else "no bearer token for the validating proxy; direct delivery not accepted"
            )
        if reason is not None:
            withheld[name] = reason
            observations.append(
                SourceObservation(
                    server=name, status=SourceStatus.DISABLED, observed_at=observed_at
                )
            )
            continue
        (proxied if proxy else servers).append(source.server)
    venue = order_venue(effective_mode, robinhood_proxied=any(p.name == ROBINHOOD for p in proxied))
    base = build_tool_access(
        effective_mode=effective_mode,
        workspace_writes=workspace_writes,
        registries=registries,
        mignons=mignons,
        venue=venue,
    )
    allowed = set(base.allowed_tools)
    disallowed = set(base.disallowed_tools)
    for registry in registries:
        if registry.server in withheld:
            names = {registry.qualified(t.name) for t in registry.tools}
            disallowed |= names
            allowed -= names

    access = ToolAccess(
        effective_mode=effective_mode,
        order_venue=venue,
        allowed_tools=tuple(sorted(allowed)),
        disallowed_tools=tuple(sorted(disallowed)),
    )
    assert_no_order_tools(access, registries)
    required = tuple(
        s.registry.server for s in sources if s.required and s.registry.server in withheld
    )
    return SessionPlan(
        effective_mode=effective_mode,
        tool_access=access,
        servers=tuple(servers),
        registries=registries,
        observations=tuple(observations),
        withheld=withheld,
        required_unavailable=required,
        proxied=tuple(proxied),
    )


_TOOL_PURPOSES: Final[dict[str, str]] = {
    DELEGATION_TOOL: "Spawn one research Mignon (subagent_type, description, prompt)",
    "WebSearch": "Public web context the structured tools lack (source tiers apply)",
    "WebFetch": "Read one page from a trusted source (source tiers apply)",
    f"mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}": (
        "Fresh recorded WebSearch results for a ticker; read before searching again"
    ),
    f"mcp__{LOCAL_SERVER_NAME}__{FACTS_TOOL_NAME}": (
        "Code-computed decision facts, sizing, and facts_ref for a candidate/position ref"
    ),
}


def _role_rows(plan: SessionPlan, role: Role) -> list[str]:
    """Table rows for the role's tools allowed this run (built-ins first, then registries)."""
    allowed = set(plan.tool_access.allowed_tools) & ROLE_TOOLS[role]
    rows = []
    for name in (DELEGATION_TOOL, *ALLOWED_BUILTINS):
        if name in allowed:
            tier = "D" if name == DELEGATION_TOOL else "R"
            rows.append(f"| `{name}` | {tier} | {_TOOL_PURPOSES[name]} |")
    for registry in plan.registries:
        if not registry.verified or registry.server in plan.withheld:
            continue
        for spec in registry.tools:
            qualified = registry.qualified(spec.name)
            if qualified in allowed:
                purpose = _TOOL_PURPOSES.get(qualified, "")
                rows.append(f"| `{qualified}` | {spec.tier.value} | {purpose} |")
    return rows


def _withheld_lines(plan: SessionPlan) -> list[str]:
    if not plan.withheld:
        return []
    lines = ["", "Sources withheld this run (their tools are unavailable):"]
    lines.extend(f"- {server}: {reason}" for server, reason in sorted(plan.withheld.items()))
    return lines


def available_tools_table(
    plan: SessionPlan,
    role: Role = Role.ORCHESTRATOR,
    mignon_models: Sequence[str] = (),
) -> str:
    """A prompt's `{{available_tools}}`: the role's allowed tools, fully qualified, with tier.

    Only allowed tools of verified registries (plus allowed built-ins) are listed; withheld
    sources are named separately so the model does not look for them. The orchestrator's
    table then lists the Mignon types it can spawn this run and the models in
    `mignon_models` with their guidance, each once. A Mignon's own tools are not listed
    there: the orchestrator cannot call them, and each Mignon's prompt carries its table.
    """
    lines = ["| Tool | Tier | Purpose |", "|---|---|---|", *_role_rows(plan, role)]
    if role is Role.ORCHESTRATOR and DELEGATION_TOOL in plan.tool_access.allowed_tools:
        types = [m for m in MignonType if _role_rows(plan, Role(m.value))]
        if types and mignon_models:
            lines.extend(["", "Mignon types (spawn as `subagent_type` = `<type>--<model>`):"])
            lines.extend(f"- `{m.value}`: {MIGNON_DESCRIPTIONS[m]}" for m in types)
            lines.extend(["", "Models:"])
            lines.extend(
                f"- `{model}`: {MODEL_GUIDANCE.get(model, 'no guidance recorded')}"
                for model in mignon_models
            )
    lines.extend(_withheld_lines(plan))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------------------------


class SessionStatus(StrEnum):
    COMPLETED = "completed"  # a ResultMessage arrived without a stop request
    NOT_STARTED = "not_started"  # a required source was unavailable; no query was sent
    STOPPED = "stopped"  # the RunControl latch interrupted the session
    FAILED = "failed"  # the SDK/transport failed before a result


@dataclass(frozen=True)
class SessionDeps:
    """Everything one session needs. Safety values come from Settings, never the model."""

    conn: Conn
    run_id: uuid.UUID
    account_scope_id: str
    settings: Settings
    rules: LoadedRules
    plan: SessionPlan
    system_prompt: str
    run_control: RunControl
    clock: Callable[[], datetime]
    redactor: Redactor
    metrics: RunMetrics
    scratch_dir: Path
    connect_budget_seconds: float
    session_budget_seconds: Callable[[], float]
    deadline_check: Callable[[], None] = lambda: None
    transport_factory: TransportFactory | None = None
    # Opens one proxied server's upstream (test seam; default: streamable HTTP).
    upstream_factory: UpstreamFactory | None = None
    mappers: Mapping[tuple[str, str], EvidenceMapper] = field(
        default_factory=lambda: VERIFIED_MAPPERS
    )
    account_scope_table: Mapping[str, AccountScopeSpec] = field(
        default_factory=lambda: ROBINHOOD_ACCOUNT_SCOPE
    )
    max_turns: int = DEFAULT_MAX_TURNS
    max_budget_usd: Decimal | None = None
    # One rendered prompt per Mignon type (ADR-0025); empty when Mignons are disabled.
    mignon_prompts: Mapping[MignonType, str] = field(default_factory=dict)
    status_poll_interval: float = STATUS_POLL_INTERVAL_SECONDS
    interrupt_grace_seconds: float = INTERRUPT_GRACE_SECONDS


@dataclass
class SessionResult:
    status: SessionStatus
    observations: list[SourceObservation] = field(default_factory=list)
    withheld: dict[str, str] = field(default_factory=dict)
    raw_output: str | None = None
    parsed: DecisionOutputParseResult | None = None
    output_id: uuid.UUID | None = None
    interrupted: bool = False
    error: str | None = None
    model_id: str | None = None
    tool_drift: list[str] = field(default_factory=list)
    # The trusted Agentic-eligibility check (proxied Robinhood only); None if it did not run.
    eligibility: AgenticEligibility | None = None


def _web_cache_parts(
    deps: SessionDeps,
) -> tuple[Callable[[str, dict[str, Any]], str | None] | None, Any, Any]:
    """web_precheck / web_capture callables and the lookup tool, bound to the ledger store."""
    store = LedgerWebCacheStore(deps.conn)
    max_age = deps.rules.rules.data_quality.freshness.news_max_age_seconds
    if not isinstance(max_age, int):
        # TBD/none freshness cannot establish a fresh entry: no dedupe and no capture.
        return None, None, build_web_cache_tool(store, deps.clock, 0, limit=0)

    def precheck(tool_name: str, tool_input: dict[str, Any]) -> str | None:
        return cached_search_denial(store, tool_name, tool_input, deps.clock(), max_age)

    def capture(
        tool_call_id: uuid.UUID, tool_name: str, tool_input: dict[str, Any], result: Any
    ) -> None:
        capture_web_result(
            store,
            run_id=deps.run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            validated_result=result,
            retrieved_at=deps.clock(),
        )

    return precheck, capture, build_web_cache_tool(store, deps.clock, max_age)


def build_session_options(
    deps: SessionDeps,
    withholding: ServerWithholding,
    upstreams: Mapping[str, McpUpstream] | None = None,
    account_eligible: bool = False,
) -> ClaudeAgentOptions:
    """Hooks, local server, validating proxies (one per open upstream), and options (no I/O).

    `account_eligible` is the result of the session's trusted `get_accounts` check."""
    precheck, capture, lookup_tool = _web_cache_parts(deps)
    limits = mignon_limits(deps.rules.rules)
    recorder = LedgerToolEventRecorder(
        deps.conn, run_id=deps.run_id, result_writer=ledger_result_writer
    )
    validator = BoundaryValidator(
        redactor=deps.redactor,
        mappers=deps.mappers,
        account_eligible=account_eligible,
        board_screens=lambda: load_run_evidence(deps.conn, deps.run_id).board_screens(),
    )
    upstreams = dict(upstreams or {})
    dispatch = ProxyDispatch(frozenset(upstreams))
    settings = deps.settings
    prefix = settings.ROBINHOOD_WORKSPACE_PREFIX
    hook_deps = HookDeps(
        effective_mode=deps.plan.effective_mode,
        order_venue=deps.plan.order_venue,
        kill_switch=settings.KILL_SWITCH,
        workspace_writes=settings.workspace_writes_enabled,
        workspace_prefix=prefix,
        account_number=settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
        rules=deps.rules.rules,
        run_control=deps.run_control,
        recorder=recorder,
        validator=validator,
        ownership=LedgerWorkspaceOwnership(deps.conn, deps.account_scope_id, prefix),
        counter=LedgerWorkspaceCounter(deps.conn, deps.account_scope_id, prefix, deps.run_id),
        redactor=deps.redactor,
        clock=deps.clock,
        registries=deps.plan.registries,
        account_scope_table=deps.account_scope_table,
        web_precheck=precheck,
        web_capture=capture,
        withheld=withholding,
        proxy_dispatch=dispatch,
        mignon_limits=limits,
        mignon_models=settings.mignon_models,
    )
    facts_service = DecisionFactsService(
        conn=deps.conn,
        run_id=deps.run_id,
        account_scope_id=deps.account_scope_id,
        rules=facts_rules_from(deps.rules),
        clock=deps.clock,
    )
    local = build_local_server([lookup_tool, build_facts_tool(facts_service, deps.run_control)])
    sdk_servers: dict[str, McpSdkServerConfig] = {LOCAL_SERVER_NAME: local}
    registry_by_name = {r.server: r for r in deps.plan.registries}
    timeout = upstream_timeout_seconds(settings.MCP_TOOL_TIMEOUT)
    venue = deps.plan.order_venue
    for name, upstream in upstreams.items():
        if venue is OrderVenue.SIMULATED and name == ROBINHOOD:
            # ADR-0038: order tools are answered in-process; none reaches Robinhood.
            upstream = SimulatedBroker(
                upstream=upstream,
                registry=registry_by_name[name],
                instruments=lambda iid: load_run_evidence(deps.conn, deps.run_id).instrument(iid),
                clock=deps.clock,
            )
        proxy = ValidatingProxy(
            server=name,
            upstream=upstream,
            dispatch=dispatch,
            recorder=recorder,
            validator=validator,
            run_control=deps.run_control,
            clock=deps.clock,
            upstream_timeout_seconds=timeout,
            order_recorder=(
                BrokerLedger(
                    deps.conn,
                    deps.run_id,
                    deps.account_scope_id
                    if venue is OrderVenue.BROKER
                    else simulated_scope_id(deps.run_id),
                )
                if executes_orders(venue) and name == ROBINHOOD
                else None
            ),
        )
        sdk_servers[name] = McpSdkServerConfig(
            type="sdk",
            name=name,
            instance=build_proxy_server(
                proxy, registry_by_name[name], deps.plan.tool_access.allowed_tools
            ),
        )
    return build_agent_options(
        tool_access=deps.plan.tool_access,
        mcp_servers=deps.plan.servers,
        sdk_servers=sdk_servers,
        hooks=build_hooks(hook_deps),
        system_prompt=deps.system_prompt,
        model=settings.AGENT_MODEL,
        scratch_dir=deps.scratch_dir,
        max_turns=deps.max_turns,
        max_budget_usd=deps.max_budget_usd,
        mcp_timeout_ms=settings.MCP_TIMEOUT,
        mcp_tool_timeout_ms=settings.MCP_TOOL_TIMEOUT,
        mignon_prompts=deps.mignon_prompts or None,
        mignon_limits=limits,
        mignon_models=settings.mignon_models,
    )


def _tool_names(server: str, entry: Mapping[str, Any]) -> list[str] | None:
    tools = entry.get("tools")
    if not isinstance(tools, list):
        return None
    prefix = f"mcp__{server}__"
    names: list[str] = []
    for item in tools:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str):
            return None
        names.append(name.removeprefix(prefix))
    return names


def observe_statuses(
    response: Mapping[str, Any],
    registries: Sequence[ToolRegistry],
    configured: Sequence[str],
    observed_at: datetime,
) -> list[SourceObservation]:
    """One observation per configured server; a server absent from the response is FAILED."""
    entries = response.get("mcpServers")
    by_name: dict[str, Mapping[str, Any]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                by_name[entry["name"]] = entry
    registry_by_name = {r.server: r for r in registries}
    observations = []
    for name in configured:
        entry = by_name.get(name, {})
        observations.append(
            observe_server(
                name,
                entry.get("status"),
                observed_at,
                registry=registry_by_name.get(name),
                discovered_tools=_tool_names(name, entry),
            )
        )
    return observations


async def _poll_status(
    client: ClaudeSDKClient, deps: SessionDeps, configured: Sequence[str]
) -> list[SourceObservation]:
    """Poll until no configured server is pending or the connect budget ends (§8)."""
    observations: list[SourceObservation] = []
    with anyio.move_on_after(deps.connect_budget_seconds):
        while True:
            response = await client.get_mcp_status()
            observations = observe_statuses(
                dict(response), deps.plan.registries, configured, deps.clock()
            )
            if all(o.status is not SourceStatus.PENDING for o in observations):
                return observations
            if deps.run_control.stop_requested:
                return observations
            await anyio.sleep(deps.status_poll_interval)
    return observations


def _withhold_unavailable(
    observations: Sequence[SourceObservation],
    withholding: ServerWithholding,
    result: SessionResult,
) -> None:
    for obs in observations:
        if obs.available:
            continue
        if obs.discovery is not None and obs.discovery.missing:
            result.tool_drift.append(obs.server)
        reason = f"status {obs.status.value}" + (
            "" if obs.discovery is None or obs.discovery.ok else "; expected tools missing"
        )
        if obs.status is SourceStatus.CONNECTED and obs.discovery is None:
            reason = "connected without a verifiable tool list"
        if withholding.withhold(obs.server, reason):
            result.withheld[obs.server] = reason


def _default_upstream(
    server: McpHttpServer, connect_timeout_seconds: float
) -> AbstractAsyncContextManager[McpUpstream]:
    return open_http_upstream(server, connect_timeout_seconds=connect_timeout_seconds)


async def _open_upstreams(
    stack: AsyncExitStack, deps: SessionDeps, result: SessionResult
) -> dict[str, McpUpstream]:
    """Connect each proxied server within the shared connect budget (CLAUDE.md §8).

    A refused credential is `needs-auth`; any other failure is `failed`; a connected server
    gets a discovery diff from its full tool list. Failures are observations, never raised.
    """
    factory = deps.upstream_factory or _default_upstream
    registry_by_name = {r.server: r for r in deps.plan.registries}
    deadline = anyio.current_time() + deps.connect_budget_seconds
    upstreams: dict[str, McpUpstream] = {}
    for server in deps.plan.proxied:
        remaining = deadline - anyio.current_time()
        status = SourceStatus.FAILED
        try:
            if remaining <= 0:
                raise UpstreamError(f"{server.name} connect: connect budget exhausted")
            upstream = await stack.enter_async_context(factory(server, remaining))
        except UpstreamAuthError:
            status = SourceStatus.NEEDS_AUTH
        except UpstreamError:
            pass
        else:
            upstreams[server.name] = upstream
            result.observations.append(
                SourceObservation(
                    server=server.name,
                    status=SourceStatus.CONNECTED,
                    observed_at=deps.clock(),
                    discovery=diff_discovered(
                        registry_by_name[server.name], (t.name for t in upstream.tools)
                    ),
                )
            )
            continue
        result.observations.append(
            SourceObservation(server=server.name, status=status, observed_at=deps.clock())
        )
    return upstreams


ACCOUNT_LISTING_TOOL: Final = "get_accounts"


async def _check_agentic_account(
    deps: SessionDeps,
    upstream: McpUpstream,
    withholding: ServerWithholding,
    result: SessionResult,
) -> None:
    """CLAUDE.md §9: verify, in trusted code, that the configured account is the Agentic
    account before the model may touch it. The listing goes through the upstream directly
    (never the model) and is reduced to the configured account's redacted eligibility; no
    other account's data is kept. A failed or negative check withholds Robinhood, so no
    session starts (fail closed)."""
    try:
        response = await upstream.call_tool(
            ACCOUNT_LISTING_TOOL,
            {},
            timeout_seconds=upstream_timeout_seconds(deps.settings.MCP_TOOL_TIMEOUT),
        )
        kind, payload = extract_mcp_payload(response.response)
        if kind is PayloadKind.TOOL_ERROR:
            raise UpstreamError(f"{ACCOUNT_LISTING_TOOL}: the tool returned an error")
        eligibility = check_eligibility(
            payload, deps.settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER, deps.clock()
        )
    except (UpstreamError, PayloadError, ValueError, TypeError) as exc:
        reason = f"Agentic eligibility check failed ({type(exc).__name__})"
        if withholding.withhold(ROBINHOOD, reason):
            result.withheld[ROBINHOOD] = reason
        return
    result.eligibility = eligibility
    if not eligibility.eligible:
        reason = "configured account is not Agentic-eligible: " + "; ".join(eligibility.reasons)
        if withholding.withhold(ROBINHOOD, reason):
            result.withheld[ROBINHOOD] = reason


def _start_message(withheld: Mapping[str, str]) -> str:
    if not withheld:
        return START_MESSAGE
    detail = "; ".join(f"{name} ({why})" for name, why in sorted(withheld.items()))
    return f"{START_MESSAGE} Sources unavailable this run (their tools are denied): {detail}."


def _init_check(
    message: SystemMessage,
    deps: SessionDeps,
    configured: Sequence[str],
    withholding: ServerWithholding,
    result: SessionResult,
) -> None:
    """Re-check the init message; a required source that is not connected stops the run."""
    servers = message.data.get("mcp_servers")
    status_by_name: dict[str, object] = {}
    if isinstance(servers, list):
        for entry in servers:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                status_by_name[entry["name"]] = entry.get("status")
    for name in configured:
        raw = status_by_name.get(name)
        if raw == SourceStatus.CONNECTED.value:
            continue
        reason = f"init status {raw if isinstance(raw, str) else 'absent'}"
        if withholding.withhold(name, reason):
            result.withheld[name] = reason
        result.observations.append(observe_server(name, raw, deps.clock()))
        if name in (ROBINHOOD, LOCAL_SERVER_NAME):
            deps.run_control.request_stop(StopReason.INFRASTRUCTURE_FAILURE, deps.clock())


def _record_usage(message: ResultMessage, metrics: RunMetrics) -> None:
    usage = message.usage or {}

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    cost = message.total_cost_usd
    metrics.llm_usage(
        input_tokens=count("input_tokens")
        + count("cache_read_input_tokens")
        + count("cache_creation_input_tokens"),
        output_tokens=count("output_tokens"),
        cost_usd=Decimal(str(cost)) if cost is not None and cost >= 0 else Decimal(0),
    )


async def _converse(
    client: ClaudeSDKClient,
    deps: SessionDeps,
    configured: Sequence[str],
    withholding: ServerWithholding,
    result: SessionResult,
) -> None:
    """Send the start message and read until the ResultMessage, interrupting on stop."""
    done = anyio.Event()

    async def watch(scope: anyio.CancelScope) -> None:
        while not done.is_set():
            deps.deadline_check()
            if deps.run_control.stop_requested and not result.interrupted:
                result.interrupted = True
                with contextlib.suppress(Exception), anyio.move_on_after(10):
                    await client.interrupt()
                with anyio.move_on_after(deps.interrupt_grace_seconds):
                    await done.wait()
                scope.cancel()
                return
            with anyio.move_on_after(deps.status_poll_interval):
                await done.wait()

    async with anyio.create_task_group() as tg:
        receive_scope = anyio.CancelScope()
        tg.start_soon(watch, receive_scope)
        with receive_scope, anyio.move_on_after(max(deps.session_budget_seconds(), 0.0)):
            await client.query(_start_message(result.withheld))
            async for message in client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    _init_check(message, deps, configured, withholding, result)
                elif isinstance(message, AssistantMessage):
                    # Mignon turns carry their spawning Agent call's id; the run records the
                    # orchestrator's model (a Mignon's model is part of its agent_type).
                    if message.parent_tool_use_id is None:
                        result.model_id = message.model or result.model_id
                elif isinstance(message, ResultMessage):
                    _record_usage(message, deps.metrics)
                    if isinstance(message.result, str) and not message.is_error:
                        result.raw_output = message.result
                    elif message.is_error:
                        result.error = f"result error ({message.subtype})"
        done.set()
    if not result.interrupted and result.raw_output is None and result.error is None:
        # The budget elapsed (or the stream ended) without a ResultMessage.
        deps.deadline_check()
        result.error = "session ended without a result message"


def persist_output(deps: SessionDeps, result: SessionResult) -> None:
    """Store the raw (redacted) final response and its strict parse (ledger/evidence.py).

    A missing response is stored as None: output coverage is then unknown, never "no trades".
    """
    raw = deps.redactor.redact_text(result.raw_output) if result.raw_output is not None else None
    result.output_id = ledger_evidence.insert_agent_output(
        deps.conn, run_id=deps.run_id, raw_redacted=raw, observed_at=deps.clock()
    )
    if raw is None:
        return
    result.parsed = parse_agent_decision_output(raw)
    ledger_evidence.insert_agent_decision(
        deps.conn, run_id=deps.run_id, output_id=result.output_id, result=result.parsed
    )


async def run_agent_session(deps: SessionDeps) -> SessionResult:
    """Run one session per the module docstring. Never raises for SDK/transport failures:
    they return `FAILED` with the error type (the orchestrator records and alerts)."""
    result = SessionResult(status=SessionStatus.NOT_STARTED, withheld=dict(deps.plan.withheld))
    result.observations.extend(deps.plan.observations)
    if not deps.plan.may_start:
        return result
    withholding = ServerWithholding()
    for server, reason in deps.plan.withheld.items():
        withholding.withhold(server, reason)
    async with AsyncExitStack() as upstream_stack:
        upstreams = await _open_upstreams(upstream_stack, deps, result)
        _withhold_unavailable(result.observations, withholding, result)
        # A proxy is served only for a verified connection; a withheld one is not configured.
        upstreams = {n: u for n, u in upstreams.items() if withholding.reason(n) is None}
        if ROBINHOOD in upstreams:
            await _check_agentic_account(deps, upstreams[ROBINHOOD], withholding, result)
        if ROBINHOOD in result.withheld or deps.run_control.stop_requested:
            return result
        await _run_client(deps, withholding, upstreams, result)
    if result.status is not SessionStatus.NOT_STARTED:
        close_unresolved_calls(deps)
        persist_output(deps, result)
    return result


async def _run_client(
    deps: SessionDeps,
    withholding: ServerWithholding,
    upstreams: Mapping[str, McpUpstream],
    result: SessionResult,
) -> None:
    direct = [s.name for s in deps.plan.servers]
    configured = [*direct, *upstreams, LOCAL_SERVER_NAME]
    eligible = result.eligibility is not None and result.eligibility.eligible
    options = build_session_options(deps, withholding, upstreams, account_eligible=eligible)
    transport = deps.transport_factory(options) if deps.transport_factory else None
    client = ClaudeSDKClient(options, transport=transport)
    try:
        with anyio.fail_after(max(deps.connect_budget_seconds, 0.001)):
            await client.connect()
        if direct:
            observations = await _poll_status(client, deps, direct)
            result.observations.extend(observations)
            _withhold_unavailable(observations, withholding, result)
        if ROBINHOOD in result.withheld or deps.run_control.stop_requested:
            result.status = SessionStatus.NOT_STARTED
            return
        await _converse(client, deps, configured, withholding, result)
        result.status = (
            SessionStatus.STOPPED
            if deps.run_control.stop_requested
            else SessionStatus.FAILED
            if result.raw_output is None and result.error is not None
            else SessionStatus.COMPLETED
        )
    except Exception as exc:  # noqa: BLE001 - SDK/transport failure: recorded, never retried
        result.status = SessionStatus.FAILED
        result.error = f"session failed ({type(exc).__name__})"
    finally:
        with contextlib.suppress(Exception), anyio.move_on_after(DISCONNECT_TIMEOUT_SECONDS):
            await client.disconnect()


SESSION_ENDED_DEDUP_KEY = "session_ended_without_outcome"


def close_unresolved_calls(deps: SessionDeps) -> int:
    """Record `unknown` for every call dispatched but never resolved when the session ended.

    The real CLI fires no PostToolUseFailure for a call in flight at interrupt (real-CLI
    acceptance test 10, DATA_QUALITY.md), so the outcome is recorded here rather than waiting
    for a later recovery. `unknown` is exact: nothing reported what happened, so a financial
    action is never inferred to have failed (INTERFACES.md). Returns the number closed.
    """
    closed = 0
    for record in ledger_tool_calls.tool_call_records(deps.conn, deps.run_id):
        if record.dispatched_at is None or record.status is not ToolCallStatus.REQUESTED:
            continue
        ledger_tool_calls.append_tool_call_outcome(
            deps.conn,
            record.identity.tool_call_id,
            ToolCallStatus.UNKNOWN,
            observed_at=deps.clock(),
            dedup_key=SESSION_ENDED_DEDUP_KEY,
            reason="session ended before an outcome was reported",
        )
        closed += 1
    return closed


def run_session_sync(deps: SessionDeps) -> SessionResult:
    """Run the session on a fresh asyncio event loop (the orchestrator is synchronous)."""
    runner: Callable[[], Awaitable[SessionResult]] = lambda: run_agent_session(deps)  # noqa: E731
    return anyio.run(runner)


__all__ = [
    "PROXY_RESULT_BOUNDARY_ACCEPTED",
    "REMOTE_RESULT_BOUNDARY_ACCEPTED",
    "RemoteSource",
    "SessionDeps",
    "SessionPlan",
    "SessionPlanError",
    "SessionResult",
    "SessionStatus",
    "assert_no_order_tools",
    "available_tools_table",
    "observe_statuses",
    "plan_session",
    "run_agent_session",
    "run_session_sync",
]
