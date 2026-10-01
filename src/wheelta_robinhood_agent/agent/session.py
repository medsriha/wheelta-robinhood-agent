"""Run the Claude Agent SDK session of one run (ARCHITECTURE.md "Run lifecycle" 4-6).

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
  `agentic_verified`); for the Buy-to-Close and Sell Options agents (ADR-0057), check the role's
  start condition the same way (`agent/start_probe.py`): not met means `SKIPPED`, unknown means
  no session (`NOT_STARTED`, fail closed); build hooks (tool access layer 3, recording, result
  boundary, web cache), the proxies, and options; connect; poll `get_mcp_status` for direct
  servers only (`pending` is intermediate; CLAUDE.md §8), withhold anything unavailable, and
  only then send the start message. In-process servers (`wra_local` and the proxies) are absent
  from `get_mcp_status` until the first query (real CLI 2.1.283), so they are verified by the
  init `SystemMessage` check instead, which stops the run if Robinhood or `wra_local` is not
  connected. A RunControl stop (signal, deadline, infrastructure failure) interrupts the SDK.
  The last `ResultMessage` usage/cost (the session's running totals) feeds `RunMetrics`. Its
  final text is parsed into AgentDecisionOutput v6; an invalid output gets up to
  `MAX_OUTPUT_REPAIRS` follow-up turns listing the issues, with every tool denied (ADR-0044).
  Each raw (redacted) response and its parse are persisted, each repair correcting the one
  before. A valid output whose references do not resolve, or that leaves an order action it
  could cite unclaimed, gets up to `MAX_REFERENCE_REPAIRS` such turns listing those issues
  (ADR-0052, `SessionDeps.reference_check`).

ADR-0063: with `SessionDeps.loopback_server` (a loopback listener from `integrations/`), the
orchestrator's in-process proxies list only the orchestrator's tools of each source (a source
with none is not configured in the session at all) and each Mignon role reaches its own tools
through an inline loopback server (`proxy.LoopbackProxyApp`), so Mignon-only tool schemas
never enter the orchestrator's context. Without it (the default, and any custom transport
that cannot reach a loopback server) every proxy lists every allowed tool, as before; the
PreToolUse hook enforces each role's tools either way.

`assert_no_order_tools` re-checks the plan before any session is built: without an order
venue no Tier X tool is allowed; with one only the three option-order tools are. The venue
(ADR-0038) is `broker` in armed live (ADR-0034); in a dry run it is `simulated` when
Robinhood is proxied (`simulated_broker.SimulatedBroker` answers the order tools in-process)
and `none` otherwise. `broker_ledger.BrokerLedger` records orders for both venues: broker
orders under the account scope, simulated ones under the tick's simulated scope (ADR-0057:
the first run's id, shared with the second run, which also shares the simulated state).
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
from claude_agent_sdk.types import McpHttpServerConfig, McpSdkServerConfig

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
from wheelta_robinhood_agent.agent.hooks import (
    HookDeps,
    OutputRepairGate,
    build_hooks_with_gate,
)
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
    Role,
    mignon_limits,
    role_tools,
)
from wheelta_robinhood_agent.agent.options import build_agent_options
from wheelta_robinhood_agent.agent.order_cleanup import (
    CLEANUP_DENIAL,
    CLEANUP_TOOLS,
    MAX_ORDER_CLEANUPS,
    ORDER_WIND_DOWN_SECONDS,
    WIND_DOWN_DENIAL,
    cleanup_message,
    needs_cleanup,
    order_scope,
    unresolved_orders,
)
from wheelta_robinhood_agent.agent.order_walk import (
    AWAIT_TOOL,
    ORDER_WORK_REGISTRY,
    ORDER_WORK_SERVER,
    WORK_TOOL,
    CallOutcome,
    OrderWorkRunner,
)
from wheelta_robinhood_agent.agent.order_work_server import (
    OrderWorkServer,
    build_order_work_server,
)
from wheelta_robinhood_agent.agent.pretrade_gate import (
    PlacementState,
    PretradeGate,
    WalkInProgress,
)
from wheelta_robinhood_agent.agent.proxy import (
    LOOPBACK_HOST,
    LoopbackProxyApp,
    ValidatingProxy,
    build_proxy_server,
    build_role_proxy_servers,
    role_path,
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
    mapped_evidence_of,
)
from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason
from wheelta_robinhood_agent.agent.simulated_broker import (
    SimulatedBroker,
    SimulatedState,
    simulated_scope_id,
)
from wheelta_robinhood_agent.agent.start_probe import probe_start_condition
from wheelta_robinhood_agent.agent.tool_access import (
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
from wheelta_robinhood_agent.config.facts_rules import facts_rules_from, pretrade_rules_from
from wheelta_robinhood_agent.config.rules import LoadedRules, RuleMarker
from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.account import AgenticEligibility
from wheelta_robinhood_agent.domain.decision_output import (
    ROLE_ACTIONS,
    DecisionOutputParsed,
    DecisionOutputParseResult,
    ParseIssue,
    parse_agent_decision_output,
)
from wheelta_robinhood_agent.domain.enums import (
    AgentRole,
    ExecutionMode,
    MignonType,
    OrderVenue,
    SourceStatus,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.gating import executes_orders, order_venue
from wheelta_robinhood_agent.domain.orders import OrderRecord
from wheelta_robinhood_agent.domain.start_conditions import (
    STARTS_SESSION,
    StartCondition,
    StartOutcome,
    unchecked_start,
)
from wheelta_robinhood_agent.integrations.mcp_upstream import (
    McpUpstream,
    UpstreamAuthError,
    UpstreamError,
    open_http_upstream,
)
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, diff_discovered
from wheelta_robinhood_agent.integrations.robinhood.accounts import check_eligibility
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    PLACE_ORDER_TOOL,
)
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    SERVER_NAME as ROBINHOOD,
)
from wheelta_robinhood_agent.integrations.status import (
    McpHttpServer,
    SourceObservation,
    observe_server,
)
from wheelta_robinhood_agent.integrations.websearch.registry import (
    EXTRACT_TOOL,
    SEARCH_TOOL,
)
from wheelta_robinhood_agent.integrations.websearch.registry import (
    SERVER_NAME as TAVILY,
)
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger import orders as ledger_orders
from wheelta_robinhood_agent.ledger import runs as ledger_runs
from wheelta_robinhood_agent.ledger import tool_calls as ledger_tool_calls
from wheelta_robinhood_agent.observability.metrics import RunMetrics
from wheelta_robinhood_agent.observability.redaction import Redactor

Conn = psycopg.Connection[tuple[object, ...]]
TransportFactory = Callable[[ClaudeAgentOptions], Transport]
# ADR-0063: serves the app on 127.0.0.1 for the session and yields its base URL
# (`http://127.0.0.1:<port>`). The listener lives in `integrations/` (CLAUDE.md §3).
LoopbackServer = Callable[[LoopbackProxyApp], AbstractAsyncContextManager[str]]


@dataclass(frozen=True)
class LoopbackEndpoint:
    """A started loopback listener (ADR-0063): the app it serves and its base URL."""

    app: LoopbackProxyApp
    base_url: str


# ADR-0052: a parsed output -> the reference issues to send back (empty: none).
ReferenceCheck = Callable[[DecisionOutputParsed], tuple[str, ...]]
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
    # ADR-0059: whose session this plan is; it decides the orchestrator's tools.
    agent: AgentRole = AgentRole.WHEEL

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
    agent: AgentRole = AgentRole.WHEEL,
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
    robinhood_proxied = any(p.name == ROBINHOOD for p in proxied)
    venue = order_venue(effective_mode, robinhood_proxied=robinhood_proxied)
    if executes_orders(venue) and robinhood_proxied:
        # ADR-0066: code works orders through the Robinhood proxy; the order-work tools exist
        # only with an order venue and that proxy.
        registries = (*registries, ORDER_WORK_REGISTRY)
    base = build_tool_access(
        effective_mode=effective_mode,
        workspace_writes=workspace_writes,
        registries=registries,
        mignons=mignons,
        venue=venue,
        agent=agent,
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
        agent=agent,
    )


_TOOL_PURPOSES: Final[dict[str, str]] = {
    DELEGATION_TOOL: "Spawn one research Mignon (subagent_type, description, prompt)",
    f"mcp__{TAVILY}__{SEARCH_TOOL}": (
        "Web search for public context the structured tools lack: leads, not sources"
    ),
    f"mcp__{TAVILY}__{EXTRACT_TOOL}": (
        "Read up to 5 pages; only pages it returned are citable (source tiers apply)"
    ),
    f"mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}": (
        "Fresh recorded tavily_search results for a ticker; read before searching again"
    ),
    f"mcp__{LOCAL_SERVER_NAME}__{FACTS_TOOL_NAME}": (
        "Code-computed decision facts, sizing, and facts_ref for a candidate/position ref"
    ),
    f"mcp__{ORDER_WORK_SERVER}__{WORK_TOOL}": (
        "Work one order in the order window: code steps start_price to worst_price (ADR-0066)"
    ),
    f"mcp__{ORDER_WORK_SERVER}__{AWAIT_TOOL}": (
        "Wait for an order-work job to end; its status, steps, and fills"
    ),
}


def _role_rows(plan: SessionPlan, role: Role) -> list[str]:
    """Table rows for the role's tools allowed this run (`Agent` first, then registries)."""
    allowed = set(plan.tool_access.allowed_tools) & role_tools(role, plan.agent)
    rows = []
    if DELEGATION_TOOL in allowed:
        rows.append(f"| `{DELEGATION_TOOL}` | D | {_TOOL_PURPOSES[DELEGATION_TOOL]} |")
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
    # ADR-0057: the agent's start condition was not met (`SessionResult.start_condition`);
    # no query was sent.
    SKIPPED = "skipped"
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
    # ADR-0063: the Mignons' loopback listener. None: every proxy lists every allowed tool.
    loopback_server: LoopbackServer | None = None
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
    # ADR-0052: assembles a parsed output against this run's recorded calls (the
    # orchestrator binds `run_loader.check_references`). None: no reference turns.
    reference_check: ReferenceCheck | None = None
    # ADR-0057: which agent this session is. CLOSE and SELL check their start condition
    # before the model connects and may decide only their own actions.
    role: AgentRole = AgentRole.WHEEL
    # Earlier runs of the same tick (the sell run: the close run). Their unresolved orders
    # count as working for this run's placements (ADR-0051).
    related_run_ids: tuple[uuid.UUID, ...] = ()
    # The run whose id names the tick's simulated order scope (default: this run).
    order_scope_run_id: uuid.UUID | None = None
    # The tick's simulated broker state (simulated venue only; None: a fresh one).
    simulated_state: SimulatedState | None = None

    def __post_init__(self) -> None:
        # ADR-0059: the plan's tools were chosen for one agent; never run another on them.
        if self.plan.agent is not self.role:
            raise SessionPlanError(
                f"session plan is for the {self.plan.agent.value} agent, not {self.role.value}"
            )


@dataclass
class SessionResult:
    status: SessionStatus
    observations: list[SourceObservation] = field(default_factory=list)
    withheld: dict[str, str] = field(default_factory=dict)
    raw_output: str | None = None
    # Every final response in order (ADR-0044): an invalid one, then its repairs. The last
    # is `raw_output`, the effective output.
    raw_outputs: list[str] = field(default_factory=list)
    parsed: DecisionOutputParseResult | None = None
    output_id: uuid.UUID | None = None
    interrupted: bool = False
    error: str | None = None
    error_details: tuple[str, ...] = ()
    model_id: str | None = None
    tool_drift: list[str] = field(default_factory=list)
    # The trusted Agentic-eligibility check (proxied Robinhood only); None if it did not run.
    eligibility: AgenticEligibility | None = None
    # ADR-0050: cleanup turns sent, and owned orders still unresolved when the session ended
    # (broker order IDs, or the place call ID when no broker order is known).
    order_cleanups: int = 0
    orders_left_unresolved: tuple[str, ...] = ()
    # ADR-0052: reference turns sent, the issues still open after the last check (None when
    # no check ran), and a check that failed (its error type; the output is then accepted).
    reference_repairs: int = 0
    reference_issues: tuple[str, ...] | None = None
    reference_check_error: str | None = None
    # ADR-0057: the trusted start-condition check (CLOSE/SELL); None if it did not run.
    start_condition: StartCondition | None = None
    # ADR-0066: turns reporting order-work jobs that ended after the agent returned.
    order_work_turns: int = 0


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
    output_gate: OutputRepairGate | None = None,
    loopback: LoopbackEndpoint | None = None,
    runner: OrderWorkRunner | None = None,
) -> ClaudeAgentOptions:
    """Hooks, local server, validating proxies (one per open upstream), and options (no I/O).

    ADR-0066: with `runner` (an order venue and a proxied Robinhood) the `wra_orders` server
    is served and the runner is bound to the hooks' executor gate and the Robinhood proxy.

    `account_eligible` is the result of the session's trusted `get_accounts` check;
    `output_gate` is closed by the session during final-output repair turns (ADR-0044).
    With `loopback` (ADR-0063, module docstring) the Mignon roles' servers are mounted on its
    app and the in-process proxies list only the orchestrator's tools."""
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
    if runner is not None and ROBINHOOD not in upstreams:
        raise SessionPlanError("the order executor needs the proxied Robinhood upstream")
    dispatch = ProxyDispatch(
        frozenset(upstreams) | ({ORDER_WORK_SERVER} if runner is not None else frozenset())
    )
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
        agent=deps.plan.agent,
        account_scope_table=deps.account_scope_table,
        web_precheck=precheck,
        web_capture=capture,
        withheld=withholding,
        proxy_dispatch=dispatch,
        mignon_limits=limits,
        mignon_models=settings.mignon_models,
        output_gate=output_gate,
        pretrade_gate=PretradeGate(
            evidence=lambda: load_run_evidence(deps.conn, deps.run_id),
            rules=pretrade_rules_from(deps.rules),
            clock=deps.clock,
            placements=lambda: placement_state(deps, runner),
            role=deps.role,
            run_orders=lambda: ledger_orders.run_order_records(deps.conn, deps.run_id),
        ),
        order_work=runner,
        session_remaining=deps.session_budget_seconds if runner is not None else None,
        wind_down_seconds=ORDER_WIND_DOWN_SECONDS,
    )
    hooks, executor_gate = build_hooks_with_gate(hook_deps)
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
    allowed = deps.plan.tool_access.allowed_tools
    split = loopback is not None and DELEGATION_TOOL in allowed and limits is not None
    main_allowed = (
        tuple(t for t in allowed if t in role_tools(Role.ORCHESTRATOR, deps.plan.agent))
        if split
        else allowed
    )
    proxies: dict[str, tuple[ValidatingProxy, ToolRegistry]] = {}
    for name, upstream in upstreams.items():
        if venue is OrderVenue.SIMULATED and name == ROBINHOOD:
            # ADR-0038: order tools are answered in-process; none reaches Robinhood.
            upstream = SimulatedBroker(
                upstream=upstream,
                registry=registry_by_name[name],
                instruments=lambda iid: load_run_evidence(deps.conn, deps.run_id).instrument(iid),
                clock=deps.clock,
                state=deps.simulated_state or SimulatedState(),
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
                    else simulated_scope_id(deps.order_scope_run_id or deps.run_id),
                )
                if executes_orders(venue) and name == ROBINHOOD
                else None
            ),
        )
        proxies[name] = (proxy, registry_by_name[name])
        main_tools = [t for t in main_allowed if t.startswith(f"mcp__{name}__")]
        if split and not main_tools:
            continue  # ADR-0063: a Mignon-only source is not in the orchestrator's session
        sdk_servers[name] = McpSdkServerConfig(
            type="sdk",
            name=name,
            instance=build_proxy_server(proxy, registry_by_name[name], main_allowed),
        )
    if runner is not None:
        robinhood_proxy = proxies[ROBINHOOD][0]

        async def transport(use_id: str, *, timeout_seconds: float | None = None) -> CallOutcome:
            done = await robinhood_proxy.execute(use_id, timeout_seconds=timeout_seconds)
            evidence = (
                mapped_evidence_of(done.payload)
                if done.status is ToolCallStatus.SUCCEEDED
                else None
            )
            return CallOutcome(status=done.status, evidence=evidence)

        runner.bind(executor_gate, transport)
        sdk_servers[ORDER_WORK_SERVER] = McpSdkServerConfig(
            type="sdk",
            name=ORDER_WORK_SERVER,
            instance=build_order_work_server(
                OrderWorkServer(
                    runner=runner,
                    dispatch=dispatch,
                    recorder=recorder,
                    run_control=deps.run_control,
                    clock=deps.clock,
                )
            ),
        )
    mignon_servers = (
        _mount_mignon_servers(loopback, proxies, allowed) if split and loopback else None
    )
    return build_agent_options(
        tool_access=deps.plan.tool_access,
        mcp_servers=deps.plan.servers,
        sdk_servers=sdk_servers,
        hooks=hooks,
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
        mignon_mcp_servers=mignon_servers,
    )


def _mount_mignon_servers(
    loopback: LoopbackEndpoint,
    proxies: Mapping[str, tuple[ValidatingProxy, ToolRegistry]],
    allowed: Sequence[str],
) -> dict[Role, dict[str, McpHttpServerConfig]]:
    """Mount each Mignon role's proxy servers on the loopback app; their inline configs."""
    if not loopback.base_url.startswith(f"http://{LOOPBACK_HOST}:"):
        raise SessionPlanError(f"the loopback listener is not on {LOOPBACK_HOST}")
    roles = [r for r in Role if r is not Role.ORCHESTRATOR]
    servers = build_role_proxy_servers(proxies, allowed, roles)
    loopback.app.mount(servers)
    header = {"Authorization": loopback.app.authorization}
    return {
        role: {
            name: McpHttpServerConfig(
                type="http", url=f"{loopback.base_url}{role_path(role, name)}", headers=header
            )
            for name in by_name
        }
        for role, by_name in servers.items()
    }


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


async def _check_start_condition(deps: SessionDeps, upstream: McpUpstream | None) -> StartCondition:
    """ADR-0057: the role's start condition from trusted reads (`agent/start_probe.py`).

    Without a proxied Robinhood upstream there is no trusted channel: a proposal-only dry run
    (Robinhood served directly, order venue `none`, ADR-0019) starts unchecked; any other
    venue fails closed (UNAVAILABLE)."""
    if upstream is None:
        if deps.plan.order_venue is OrderVenue.NONE:
            return unchecked_start(deps.role)
        return StartCondition(
            role=deps.role,
            outcome=StartOutcome.UNAVAILABLE,
            reason="no trusted Robinhood connection to read the start condition through",
        )
    return await probe_start_condition(
        deps.role,
        upstream,
        account_number=deps.settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
        min_settled_cash_usd=deps.rules.rules.sessions.sell_min_settled_cash_usd,
        clock=deps.clock,
        timeout_seconds=upstream_timeout_seconds(deps.settings.MCP_TOOL_TIMEOUT),
        simulated=deps.simulated_state if deps.plan.order_venue is OrderVenue.SIMULATED else None,
        mappers=deps.mappers,
    )


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
        if name in (ROBINHOOD, LOCAL_SERVER_NAME, ORDER_WORK_SERVER):
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


# ADR-0044: follow-up turns that return an invalid final output's issues to the agent.
MAX_OUTPUT_REPAIRS: Final = 2
# Issues listed in one repair message; the rest are counted.
MAX_REPAIR_ISSUES: Final = 20
OUTPUT_REPAIR_DENIAL: Final = (
    "tools are disabled while the final output is corrected; reply with the corrected output only"
)


# ADR-0052: follow-up turns returning a valid output's reference issues to the agent.
MAX_REFERENCE_REPAIRS: Final = 2

# ADR-0066: follow-up turns reporting order-work jobs that ended after the agent returned.
MAX_ORDER_WORK_TURNS: Final = 3


def order_work_message(jobs: Sequence[Mapping[str, object]], attempt: int) -> str:
    """The follow-up turn sent once the order-work jobs the agent left running have ended."""
    lines = [
        f"- {j['work_ref']}: {j['status']}, filled {j['filled_quantity']} of {j['quantity']}"
        + (f" ({j['reason']})" if j.get("reason") else "")
        for j in jobs
    ]
    return (
        f"Order work you left running has ended (report {attempt} of {MAX_ORDER_WORK_TURNS}):\n"
        + "\n".join(lines)
        + "\n\nReturn your complete AgentDecisionOutput JSON object again with these outcomes: "
        "cite each job's work_ref in the decision's execution_refs. You may start other trades "
        "the rules still allow."
    )


def reference_message(issues: Sequence[str], attempt: int) -> str:
    """The follow-up turn sent when a valid output's references do not resolve (ADR-0052)."""
    shown = [f"- {i}" for i in issues[:MAX_REPAIR_ISSUES]]
    if len(issues) > MAX_REPAIR_ISSUES:
        shown.append(f"- and {len(issues) - MAX_REPAIR_ISSUES} more")
    return (
        f"Your final output is valid JSON, but code could not resolve or associate some of "
        f"its references (reference check {attempt} of {MAX_REFERENCE_REPAIRS}). Issues:\n"
        + "\n".join(shown)
        + "\n\nTools are now disabled; any call is denied. Reply with the complete corrected "
        "AgentDecisionOutput JSON object. Use only references exactly as code supplied them "
        "(an order call's `order_call_ref`, never its bare tool_call_id). Keep the same "
        "decisions, rationale, and next run; change only the references listed. Do not "
        "describe or repeat actions."
    )


def repair_message(issues: Sequence[ParseIssue], attempt: int) -> str:
    """The follow-up turn sent when the final output fails to parse (ADR-0044)."""
    shown = [
        f"- {i.loc or '(top level)'}: {i.message} [{i.kind}]" for i in issues[:MAX_REPAIR_ISSUES]
    ]
    if len(issues) > MAX_REPAIR_ISSUES:
        shown.append(f"- and {len(issues) - MAX_REPAIR_ISSUES} more")
    return (
        f"Your final output did not validate as AgentDecisionOutput (repair {attempt} of "
        f"{MAX_OUTPUT_REPAIRS}). Issues:\n" + "\n".join(shown) + "\n\n"
        "Tools are now disabled; any call is denied. Reply with the corrected "
        "AgentDecisionOutput JSON object. Keep the same decisions, references, and next run; "
        "fix only the issues listed. Do not describe or repeat actions."
    )


async def _converse(
    client: ClaudeSDKClient,
    deps: SessionDeps,
    configured: Sequence[str],
    withholding: ServerWithholding,
    result: SessionResult,
    output_gate: OutputRepairGate | None = None,
    runner: OrderWorkRunner | None = None,
) -> None:
    """Send the start message and read until the ResultMessage, interrupting on stop.

    ADR-0066: an output returned while order-work jobs are still running is not final: code
    waits for the jobs (the watcher still enforces stop and deadline), then sends up to
    `MAX_ORDER_WORK_TURNS` turns listing their outcomes, before any cleanup or repair.

    ADR-0044: a final output that fails to parse gets up to `MAX_OUTPUT_REPAIRS` follow-up
    turns in the same session, each listing the issues, with every tool denied (the gate).
    ADR-0050 (order venue only): in the last `ORDER_WIND_DOWN_SECONDS` of the budget the gate
    lets only order reads and cancels through, and a final output returned while owned orders
    are unresolved gets up to `MAX_ORDER_CLEANUPS` turns to cancel them, before any repair.
    ADR-0052: a valid output is then checked with `deps.reference_check`; its issues get up
    to `MAX_REFERENCE_REPAIRS` turns, with every tool denied as in a repair. A failing check
    is recorded and the output accepted: the post-run assembly records the same findings.
    """
    done = anyio.Event()
    last_result: ResultMessage | None = None
    scope_id = order_scope(
        deps.plan.order_venue, deps.account_scope_id, deps.order_scope_run_id or deps.run_id
    )

    async def watch(scope: anyio.CancelScope) -> None:
        while not done.is_set():
            deps.deadline_check()
            if (
                scope_id is not None
                and output_gate is not None
                and deps.session_budget_seconds() <= ORDER_WIND_DOWN_SECONDS
            ):
                output_gate.restrict(WIND_DOWN_DENIAL, CLEANUP_TOOLS)
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

    async def turn(prompt: str) -> str | None:
        nonlocal last_result
        await client.query(prompt)
        text: str | None = None
        async for message in client.receive_response():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                _init_check(message, deps, configured, withholding, result)
            elif isinstance(message, AssistantMessage):
                # Mignon turns carry their spawning Agent call's id; the run records the
                # orchestrator's model (a Mignon's model is part of its agent_type).
                if message.parent_tool_use_id is None:
                    result.model_id = message.model or result.model_id
            elif isinstance(message, ResultMessage):
                last_result = message
                if isinstance(message.result, str) and not message.is_error:
                    text = message.result
                elif message.is_error:
                    result.error = f"result error ({message.subtype})"
                    result.error_details = tuple(
                        deps.redactor.redact_text(detail)
                        for detail in (message.errors or [])
                        + ([message.result] if message.result else [])
                    )
        return text

    async with anyio.create_task_group() as tg:
        receive_scope = anyio.CancelScope()
        tg.start_soon(watch, receive_scope)
        with receive_scope, anyio.move_on_after(max(deps.session_budget_seconds(), 0.0)):
            text = await turn(_start_message(result.withheld))
            repairs = 0
            # ADR-0052: the last valid output and its reference issues. A reference turn may
            # only improve on it; otherwise it is restored as the effective output.
            best: tuple[str, DecisionOutputParsed, tuple[str, ...]] | None = None
            while text is not None:
                result.raw_outputs.append(text)
                result.raw_output = text
                result.reference_issues = None
                if deps.run_control.stop_requested:
                    break
                if runner is not None and (runner.active() or runner.pending()):
                    # Never past a running job: the cleanup and reference checks below read
                    # the shared connection and the order state the jobs are changing.
                    left = runner.active()
                    await runner.wait_all()
                    if deps.run_control.stop_requested:
                        break
                    if left and result.order_work_turns < MAX_ORDER_WORK_TURNS:
                        result.order_work_turns += 1
                        text = await turn(
                            order_work_message([j.view() for j in left], result.order_work_turns)
                        )
                        continue
                    if output_gate is not None:
                        # Out of report turns: no new job may start behind the checks.
                        output_gate.restrict(CLEANUP_DENIAL, CLEANUP_TOOLS)
                if (
                    repairs == 0
                    and result.reference_repairs == 0
                    and result.order_cleanups < MAX_ORDER_CLEANUPS
                ):
                    unresolved = cleanup_candidates(deps)
                    if unresolved:
                        result.order_cleanups += 1
                        if output_gate is not None:
                            output_gate.restrict(CLEANUP_DENIAL, CLEANUP_TOOLS)
                        text = await turn(cleanup_message(unresolved, result.order_cleanups))
                        continue
                parsed = parse_agent_decision_output(
                    deps.redactor.redact_text(text), ROLE_ACTIONS[deps.role]
                )
                if isinstance(parsed, DecisionOutputParsed):
                    # Off the event loop, so the watcher still enforces stop and deadline.
                    issues = await anyio.to_thread.run_sync(_reference_issues, deps, parsed, result)
                    if best is not None and not _improves(best, parsed, issues, result):
                        _restore(result, best)
                        break
                    best = (text, parsed, issues)
                    if not issues or result.reference_repairs >= MAX_REFERENCE_REPAIRS:
                        break
                    result.reference_repairs += 1
                    if output_gate is not None:
                        output_gate.close(OUTPUT_REPAIR_DENIAL)
                    text = await turn(reference_message(issues, result.reference_repairs))
                    continue
                if best is not None:
                    # The reply to a reference turn did not parse: keep the valid output.
                    _restore(result, best)
                    break
                if repairs >= MAX_OUTPUT_REPAIRS:
                    break
                repairs += 1
                if output_gate is not None:
                    output_gate.close(OUTPUT_REPAIR_DENIAL)
                text = await turn(repair_message(parsed.issues, repairs))
        done.set()
    if last_result is not None:
        # ResultMessage totals are the session's running totals, so only the last counts.
        _record_usage(last_result, deps.metrics)
    if not result.interrupted and result.raw_output is None and result.error is None:
        # The budget elapsed (or the stream ended) without a ResultMessage.
        deps.deadline_check()
        result.error = "session ended without a result message"


def _improves(
    best: tuple[str, DecisionOutputParsed, tuple[str, ...]],
    parsed: DecisionOutputParsed,
    issues: tuple[str, ...],
    result: SessionResult,
) -> bool:
    """Whether a reply to a reference turn may replace the last valid output (ADR-0052): the
    check ran, the reply has fewer issues, and it keeps the same decision actions in order."""
    _, before, before_issues = best
    same = [d.action for d in parsed.output.decisions] == [
        d.action for d in before.output.decisions
    ]
    return result.reference_check_error is None and same and len(issues) < len(before_issues)


def _restore(
    result: SessionResult, best: tuple[str, DecisionOutputParsed, tuple[str, ...]]
) -> None:
    """Make the last valid output effective again: it is stored once more as the final
    response, correcting the rejected reply (persist_output chains corrections)."""
    text, _, issues = best
    result.raw_outputs.append(text)
    result.raw_output = text
    result.reference_issues = issues


def _reference_issues(
    deps: SessionDeps, parsed: DecisionOutputParsed, result: SessionResult
) -> tuple[str, ...]:
    """Run the reference check (ADR-0052); record its issues, or its failure as none."""
    if deps.reference_check is None or result.reference_check_error is not None:
        return ()
    try:
        issues = deps.reference_check(parsed)
    except Exception as exc:  # noqa: BLE001 - feedback only; the assembly records the same
        result.reference_check_error = type(exc).__name__
        return ()
    result.reference_issues = issues
    return issues


def persist_output(deps: SessionDeps, result: SessionResult) -> None:
    """Store each raw (redacted) final response and its parse (ledger/evidence.py).

    ADR-0044: a repaired output corrects the one before it (`corrects_output_id`), so the
    last response is the effective output and every earlier one stays on record. A missing
    response is stored as None: output coverage is then unknown, never "no trades".
    """
    texts = result.raw_outputs or ([result.raw_output] if result.raw_output is not None else [])
    if not texts:
        result.output_id = ledger_evidence.insert_agent_output(
            deps.conn, run_id=deps.run_id, raw_redacted=None, observed_at=deps.clock()
        )
        return
    previous: uuid.UUID | None = None
    for text in texts:
        raw = deps.redactor.redact_text(text)
        output_id = ledger_evidence.insert_agent_output(
            deps.conn,
            run_id=deps.run_id,
            raw_redacted=raw,
            observed_at=deps.clock(),
            corrects_output_id=previous,
        )
        parsed = parse_agent_decision_output(raw, ROLE_ACTIONS[deps.role])
        ledger_evidence.insert_agent_decision(
            deps.conn, run_id=deps.run_id, output_id=output_id, result=parsed
        )
        previous, result.output_id, result.parsed = output_id, output_id, parsed


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
        if deps.role is not AgentRole.WHEEL:
            result.start_condition = await _check_start_condition(deps, upstreams.get(ROBINHOOD))
            if result.start_condition.outcome not in STARTS_SESSION:
                if result.start_condition.outcome is StartOutcome.NOT_MET:
                    result.status = SessionStatus.SKIPPED
                return result
        try:
            await _run_client(deps, withholding, upstreams, result)
        finally:
            # Even if something escaped the client: unknown calls closed, output and the
            # leftover orders recorded (CLAUDE.md §17).
            if result.status is not SessionStatus.NOT_STARTED:
                close_unresolved_calls(deps)
                persist_output(deps, result)
                result.orders_left_unresolved = orders_left_unresolved(deps)
    return result


def placement_state(deps: SessionDeps, runner: OrderWorkRunner | None = None) -> PlacementState:
    """This run's unresolved owned orders and its place calls without an outcome (ADR-0051).

    Only this tick's placements count (this run's and, ADR-0057, those of its earlier runs):
    an older order the ledger cannot resolve (paged order history, ADR-0034) must not block
    trading; the prompt's step 1 and ADR-0050 handle those.
    """
    scope = order_scope(
        deps.plan.order_venue, deps.account_scope_id, deps.order_scope_run_id or deps.run_id
    )
    unresolved = tuple(
        r
        for r in unresolved_orders(deps.conn, scope)
        if r.intent is not None and r.intent.run_id in (deps.run_id, *deps.related_run_ids)
    )
    # ADR-0066: a running job is counted as working through `active_jobs`, so its own
    # placement in flight is not also "another placement" (executor placements are
    # serialized by the runner).
    running = {j.job_id for j in runner.active()} if runner is not None else set()
    records = ledger_tool_calls.tool_call_records(deps.conn, deps.run_id)
    in_flight = sum(
        1
        for r in records
        if r.identity.tool == PLACE_ORDER_TOOL
        and r.status is ToolCallStatus.REQUESTED
        and r.identity.parent_tool_call_id not in running
    )
    # A running job's working order is counted once, as the job (at its worst price).
    jobs_places = {
        r.identity.tool_call_id for r in records if r.identity.parent_tool_call_id in running
    }
    unresolved = tuple(
        r for r in unresolved if r.intent is None or r.intent.place_tool_call_id not in jobs_places
    )
    jobs = (
        tuple(
            WalkInProgress(
                job_id=j.job_id,
                option_id=j.option_id,
                closing=j.closing,
                remaining_quantity=j.remaining_quantity,
                worst_price=j.worst_price,
            )
            for j in runner.active_jobs()
        )
        if runner is not None
        else ()
    )
    return PlacementState(unresolved=unresolved, placements_in_flight=in_flight, active_jobs=jobs)


def cleanup_candidates(deps: SessionDeps) -> tuple[OrderRecord, ...]:
    """Unresolved owned orders that can still be working (ADR-0050, `needs_cleanup`)."""
    scope = order_scope(
        deps.plan.order_venue, deps.account_scope_id, deps.order_scope_run_id or deps.run_id
    )
    unresolved = unresolved_orders(deps.conn, scope)
    if not unresolved:
        return ()
    call_ids = {
        r.identity.tool_call_id for r in ledger_tool_calls.tool_call_records(deps.conn, deps.run_id)
    }
    as_of = deps.clock()
    return tuple(
        r
        for r in unresolved
        if needs_cleanup(r, run_id=deps.run_id, run_tool_call_ids=call_ids, as_of=as_of)
    )


LEFT_LOOKUP_FAILED: Final = "unknown: the order lookup failed"


def orders_left_unresolved(deps: SessionDeps) -> tuple[str, ...]:
    """Owned orders still unresolved once the session ended (ADR-0050): broker order IDs, or
    `place:<tool call ID>` for an intent with no known broker order. A failed lookup is
    reported as `LEFT_LOOKUP_FAILED`, so the alert fires rather than claiming none are left."""
    try:
        records = cleanup_candidates(deps)
    except Exception:  # noqa: BLE001 - unknown is reported, never read as "none left"
        return (LEFT_LOOKUP_FAILED,)
    left: list[str] = []
    for record in records:
        if record.broker_order is not None:
            left.append(record.broker_order.broker_order_id)
        elif record.intent is not None:
            left.append(f"place:{record.intent.place_tool_call_id}")
    return tuple(left)


def _order_work_runner(
    deps: SessionDeps, upstreams: Mapping[str, McpUpstream]
) -> OrderWorkRunner | None:
    """ADR-0066: the session's order-walk runner when it has an order venue and Robinhood is
    proxied (the executor works every order through the proxy); else None."""
    if not executes_orders(deps.plan.order_venue) or ROBINHOOD not in upstreams:
        return None
    walk = deps.rules.rules.orders.walk

    def record(key: str, payload: Mapping[str, object], calls: tuple[uuid.UUID, ...]) -> None:
        ledger_runs.append_run_event(
            deps.conn,
            deps.run_id,
            RunEventType.METADATA,
            observed_at=deps.clock(),
            dedup_key=key,
            payload=payload,
            source_tool_call_ids=calls,
        )

    max_age = deps.rules.rules.data_quality.freshness.option_quote_max_age_seconds
    # await_order_work must answer before the CLI's per-call timeout (proxy margin kept).
    await_cap = upstream_timeout_seconds(deps.settings.MCP_TOOL_TIMEOUT) - 5.0
    return OrderWorkRunner(
        max_await_seconds=await_cap,
        # A `none` rule has no age limit; TBD (or any marker but none) fails every quote.
        quote_max_age_seconds=max_age
        if isinstance(max_age, int)
        else None
        if max_age is RuleMarker.NONE
        else 0,
        timing=walk.timing(),
        partial_fill=walk.partial_fill,
        instruments=lambda iid: load_run_evidence(deps.conn, deps.run_id).instrument(iid),
        run_control=deps.run_control,
        clock=deps.clock,
        events=record,
    )


async def _run_client(
    deps: SessionDeps,
    withholding: ServerWithholding,
    upstreams: Mapping[str, McpUpstream],
    result: SessionResult,
) -> None:
    async with AsyncExitStack() as stack:
        try:
            loopback = await _start_loopback(stack, deps, upstreams)
        except Exception as exc:  # noqa: BLE001 - no listener, no session; recorded, not retried
            result.status = SessionStatus.FAILED
            result.error = f"Mignon loopback server failed ({type(exc).__name__})"
            result.error_details = (deps.redactor.redact_text(f"{type(exc).__name__}: {exc}"),)
            return
        await _run_client_with(stack, deps, withholding, upstreams, result, loopback)


async def _start_loopback(
    stack: AsyncExitStack, deps: SessionDeps, upstreams: Mapping[str, McpUpstream]
) -> LoopbackEndpoint | None:
    """ADR-0063: start the Mignons' loopback listener when there are Mignons and proxies."""
    if deps.loopback_server is None or not upstreams or not deps.mignon_prompts:
        return None
    if DELEGATION_TOOL not in deps.plan.tool_access.allowed_tools:
        return None
    app = LoopbackProxyApp()
    base_url = await stack.enter_async_context(deps.loopback_server(app))
    return LoopbackEndpoint(app=app, base_url=base_url)


async def _run_client_with(
    stack: AsyncExitStack,
    deps: SessionDeps,
    withholding: ServerWithholding,
    upstreams: Mapping[str, McpUpstream],
    result: SessionResult,
    loopback: LoopbackEndpoint | None,
) -> None:
    direct = [s.name for s in deps.plan.servers]
    eligible = result.eligibility is not None and result.eligibility.eligible
    gate = OutputRepairGate()
    runner = _order_work_runner(deps, upstreams)
    options = build_session_options(
        deps,
        withholding,
        upstreams,
        account_eligible=eligible,
        output_gate=gate,
        loopback=loopback,
        runner=runner,
    )
    if runner is not None:
        # ADR-0066: jobs outlive no session; leaving waits for each (latch: one cancel).
        await stack.enter_async_context(runner)
    if loopback is not None:
        await stack.enter_async_context(loopback.app.running())
    # ADR-0063: a Mignon-only source is not in the session, so the init check skips it.
    in_session = options.mcp_servers if isinstance(options.mcp_servers, dict) else {}
    configured = [*direct, *(n for n in upstreams if n in in_session), LOCAL_SERVER_NAME]
    if runner is not None:
        configured.append(ORDER_WORK_SERVER)
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
        await _converse(client, deps, configured, withholding, result, gate, runner)
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
        result.error_details = (deps.redactor.redact_text(f"{type(exc).__name__}: {exc}"),)
    finally:
        if runner is not None:
            # ADR-0066: no job outlives its agent; after a normal end none is running.
            runner.halt()
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
