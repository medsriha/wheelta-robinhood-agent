"""Run the single Claude Agent SDK session for a run (ARCHITECTURE.md "Run lifecycle" 4-6).

Two parts:

- `plan_session` (pure): decide, before anything connects, which remote servers are exposed.
  A server is withheld when it has no credentials (Robinhood `needs-auth`, ADR-0004), its
  registry is unverified (CLAUDE.md §9), or its result boundary has not passed acceptance
  (`REMOTE_RESULT_BOUNDARY_ACCEPTED`, DATA_QUALITY.md). Withheld tools move from
  `allowed_tools` to `disallowed_tools`, so the model never sees them. Robinhood withheld or
  unavailable means **no session**: without it the agent cannot trade or manage positions.
- `run_agent_session` (async): build hooks (tool access layer 3, recording, result boundary,
  web cache) and options, connect, poll `get_mcp_status` within the connect budget
  (`pending` is intermediate; CLAUDE.md §8), turn statuses and tool lists into
  `SourceObservation`s with discovery diffs, withhold anything unavailable, and only then send
  the start message. During the session the init `SystemMessage` is re-checked; a
  RunControl stop (signal, deadline, infrastructure failure) interrupts the SDK. The
  `ResultMessage` usage/cost feeds `RunMetrics`; its final text is parsed strictly into
  AgentDecisionOutput v5, and raw (redacted) plus parsed output are persisted.

Effective mode is capped at off in phase 1 (ADR-0013). `assert_no_order_tools` re-checks
that no Tier X tool is allowed before any session is built, whatever the settings say.
"""

import contextlib
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
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

from wheelta_robinhood_agent.agent.account_scope import (
    ROBINHOOD_ACCOUNT_SCOPE,
    AccountScopeSpec,
)
from wheelta_robinhood_agent.agent.facts_tool import (
    FACTS_TOOL_NAME,
    DecisionFactsService,
    build_facts_tool,
)
from wheelta_robinhood_agent.agent.hooks import HookDeps, build_hooks
from wheelta_robinhood_agent.agent.ledger_adapters import (
    LedgerWorkspaceCounter,
    LedgerWorkspaceOwnership,
    ledger_result_writer,
)
from wheelta_robinhood_agent.agent.local_server import LOCAL_REGISTRY, build_local_server
from wheelta_robinhood_agent.agent.options import build_agent_options
from wheelta_robinhood_agent.agent.recorder import LedgerToolEventRecorder
from wheelta_robinhood_agent.agent.result_boundary import (
    VERIFIED_MAPPERS,
    BoundaryValidator,
    EvidenceMapper,
)
from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason
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
from wheelta_robinhood_agent.domain.decision_output import (
    DecisionOutputParseResult,
    parse_agent_decision_output,
)
from wheelta_robinhood_agent.domain.enums import ExecutionMode, SourceStatus, ToolTier
from wheelta_robinhood_agent.integrations.registry import ToolRegistry
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    SERVER_NAME as ROBINHOOD,
)
from wheelta_robinhood_agent.integrations.status import (
    McpHttpServer,
    SourceObservation,
    observe_server,
)
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.observability.metrics import RunMetrics
from wheelta_robinhood_agent.observability.redaction import Redactor

Conn = psycopg.Connection[tuple[object, ...]]
TransportFactory = Callable[[ClaudeAgentOptions], Transport]

REMOTE_RESULT_BOUNDARY_ACCEPTED: Final = False
"""Whether remote MCP results may reach the model through the SDK hook boundary.

DATA_QUALITY.md: direct SDK tool delivery may be used only after the pinned SDK passes the
result-boundary acceptance tests (invalid payloads, isError, transport failures, oversized
output, hook exceptions/timeouts, ledger failures; raw data must never reach a later model
request). Those tests need the real CLI (`tests/e2e/test_e2e_result_boundary_cli.py`, marked
`requires_cli`) and have not passed, so every remote server is withheld. This is a code
constant, not an environment variable: flipping it is a reviewed change with test evidence.
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
    servers: tuple[McpHttpServer, ...]
    registries: tuple[ToolRegistry, ...]
    observations: tuple[SourceObservation, ...]
    withheld: Mapping[str, str]
    required_unavailable: tuple[str, ...]

    @property
    def may_start(self) -> bool:
        return not self.required_unavailable


def assert_no_order_tools(access: ToolAccess, registries: Sequence[ToolRegistry]) -> None:
    """Phase 1 (ADR-0013): no Tier X tool may ever be allowed. Raises SessionPlanError."""
    allowed = set(access.allowed_tools)
    for registry in registries:
        for spec in registry.by_tier(ToolTier.X):
            if registry.qualified(spec.name) in allowed:
                raise SessionPlanError(f"order tool {spec.name} would be exposed")
    if access.effective_mode is not ExecutionMode.OFF:
        raise SessionPlanError("effective mode above the phase-1 ceiling (off)")


def plan_session(
    *,
    effective_mode: ExecutionMode,
    workspace_writes: bool,
    sources: Sequence[RemoteSource],
    observed_at: datetime,
    remote_boundary_accepted: bool = REMOTE_RESULT_BOUNDARY_ACCEPTED,
    local_registry: ToolRegistry = LOCAL_REGISTRY,
) -> SessionPlan:
    """Decide the exposed servers and tools before connecting (fail closed)."""
    registries = (*(s.registry for s in sources), local_registry)
    base = build_tool_access(
        effective_mode=effective_mode, workspace_writes=workspace_writes, registries=registries
    )
    withheld: dict[str, str] = {}
    observations: list[SourceObservation] = []
    servers: list[McpHttpServer] = []
    for source in sources:
        name = source.registry.server
        if isinstance(source.server, SourceObservation):
            observations.append(source.server)
            withheld[name] = f"unavailable before connect ({source.server.status.value})"
            continue
        reason = None
        if not source.registry.verified:
            reason = "tool registry unverified (no captured tools/list)"
        elif not remote_boundary_accepted:
            reason = "result-boundary acceptance tests have not passed"
        if reason is not None:
            withheld[name] = reason
            observations.append(
                SourceObservation(
                    server=name, status=SourceStatus.DISABLED, observed_at=observed_at
                )
            )
            continue
        servers.append(source.server)
    allowed = set(base.allowed_tools)
    disallowed = set(base.disallowed_tools)
    for registry in registries:
        if registry.server in withheld:
            names = {registry.qualified(t.name) for t in registry.tools}
            disallowed |= names
            allowed -= names
    access = ToolAccess(
        effective_mode=effective_mode,
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
    )


_TOOL_PURPOSES: Final[dict[str, str]] = {
    "WebSearch": "Public web context the structured tools lack (source tiers apply)",
    "WebFetch": "Read one page from a trusted source (source tiers apply)",
    f"mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}": (
        "Fresh recorded WebSearch results for a ticker; read before searching again"
    ),
    f"mcp__{LOCAL_SERVER_NAME}__{FACTS_TOOL_NAME}": (
        "Code-computed decision facts, sizing, and facts_ref for a candidate/position ref"
    ),
}


def available_tools_table(plan: SessionPlan) -> str:
    """The prompt's `{{available_tools}}`: every allowed tool, fully qualified, with its tier.

    Only allowed tools of verified registries (plus the two built-ins) are listed; withheld
    sources are named separately so the model does not look for them.
    """
    lines = ["| Tool | Tier | Purpose |", "|---|---|---|"]
    allowed = set(plan.tool_access.allowed_tools)
    for name in ALLOWED_BUILTINS:
        if name in allowed:
            lines.append(f"| `{name}` | R | {_TOOL_PURPOSES[name]} |")
    for registry in plan.registries:
        if not registry.verified or registry.server in plan.withheld:
            continue
        for spec in registry.tools:
            qualified = registry.qualified(spec.name)
            if qualified in allowed:
                purpose = _TOOL_PURPOSES.get(qualified, f"{registry.server} tool")
                lines.append(f"| `{qualified}` | {spec.tier.value} | {purpose} |")
    if plan.withheld:
        lines.append("")
        lines.append("Sources withheld this run (their tools are unavailable):")
        for server, reason in sorted(plan.withheld.items()):
            lines.append(f"- {server}: {reason}")
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
    mappers: Mapping[tuple[str, str], EvidenceMapper] = field(
        default_factory=lambda: VERIFIED_MAPPERS
    )
    account_scope_table: Mapping[str, AccountScopeSpec] = field(
        default_factory=lambda: ROBINHOOD_ACCOUNT_SCOPE
    )
    max_turns: int = DEFAULT_MAX_TURNS
    max_budget_usd: Decimal | None = None
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


def build_session_options(deps: SessionDeps, withholding: ServerWithholding) -> ClaudeAgentOptions:
    """Hooks, local server, and options for this session (no I/O)."""
    precheck, capture, lookup_tool = _web_cache_parts(deps)
    recorder = LedgerToolEventRecorder(
        deps.conn, run_id=deps.run_id, result_writer=ledger_result_writer
    )
    settings = deps.settings
    prefix = settings.ROBINHOOD_WORKSPACE_PREFIX
    hook_deps = HookDeps(
        effective_mode=deps.plan.effective_mode,
        kill_switch=settings.KILL_SWITCH,
        workspace_writes=settings.ROBINHOOD_WORKSPACE_WRITES,
        workspace_prefix=prefix,
        account_number=settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
        rules=deps.rules.rules,
        run_control=deps.run_control,
        recorder=recorder,
        validator=BoundaryValidator(redactor=deps.redactor, mappers=deps.mappers),
        ownership=LedgerWorkspaceOwnership(deps.conn, deps.account_scope_id, prefix),
        counter=LedgerWorkspaceCounter(deps.conn, deps.account_scope_id, prefix, deps.run_id),
        redactor=deps.redactor,
        clock=deps.clock,
        registries=deps.plan.registries,
        account_scope_table=deps.account_scope_table,
        web_precheck=precheck,
        web_capture=capture,
        withheld=withholding,
    )
    facts_service = DecisionFactsService(
        conn=deps.conn,
        run_id=deps.run_id,
        account_scope_id=deps.account_scope_id,
        rules=facts_rules_from(deps.rules),
        clock=deps.clock,
    )
    local = build_local_server([lookup_tool, build_facts_tool(facts_service, deps.run_control)])
    return build_agent_options(
        tool_access=deps.plan.tool_access,
        mcp_servers=deps.plan.servers,
        sdk_servers={LOCAL_SERVER_NAME: local},
        hooks=build_hooks(hook_deps),
        system_prompt=deps.system_prompt,
        model=settings.AGENT_MODEL,
        scratch_dir=deps.scratch_dir,
        max_turns=deps.max_turns,
        max_budget_usd=deps.max_budget_usd,
        mcp_timeout_ms=settings.MCP_TIMEOUT,
        mcp_tool_timeout_ms=settings.MCP_TOOL_TIMEOUT,
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
    configured = [*(s.name for s in deps.plan.servers), LOCAL_SERVER_NAME]
    options = build_session_options(deps, withholding)
    transport = deps.transport_factory(options) if deps.transport_factory else None
    client = ClaudeSDKClient(options, transport=transport)
    try:
        with anyio.fail_after(max(deps.connect_budget_seconds, 0.001)):
            await client.connect()
        observations = await _poll_status(client, deps, configured)
        result.observations.extend(observations)
        _withhold_unavailable(observations, withholding, result)
        required = [s for s in (ROBINHOOD, LOCAL_SERVER_NAME) if s in result.withheld]
        if required or deps.run_control.stop_requested:
            result.status = SessionStatus.NOT_STARTED
            return result
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
    if result.status is not SessionStatus.NOT_STARTED:
        persist_output(deps, result)
    return result


def run_session_sync(deps: SessionDeps) -> SessionResult:
    """Run the session on a fresh asyncio event loop (the orchestrator is synchronous)."""
    runner: Callable[[], Awaitable[SessionResult]] = lambda: run_agent_session(deps)  # noqa: E731
    return anyio.run(runner)


__all__ = [
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
