"""Run entrypoint: the full lifecycle of one cron fire (ARCHITECTURE.md "Run lifecycle").

ADR-0057: a tick runs two agents in order, each its own run of the slot: the Buy-to-Close
agent (`AgentRole.CLOSE`: close, roll, or hold existing shorts), then the Sell Options agent
(`AgentRole.SELL`: new CSPs and CCs). The slot gate below (kill switch, NYSE session, next-run
time) runs once, on the close run; a gate skip ends the tick. The sell run starts once the
close run is final, whatever its status (completed, skipped, failed, timed out, or stopped),
after re-checking the kill switch and the session. Each agent's session starts only when its
start condition holds (`domain/start_conditions.py`, read in trusted code). One heartbeat
and one run-summary email cover the tick; the exit code is the more severe of the two runs'.
The close run's budget is `CLOSE_AGENT_TIMEOUT_SECONDS`; the sell run gets the rest of
`RUN_TIMEOUT_SECONDS`. A re-fired slot only recovers interrupted runs: it never starts a
session (CLAUDE.md §15).

Per run: boot (settings, rules, prompts; fail fast) → effective mode off without `--run-now`: exit
`skipped_dry_run_not_requested` before the ledger (ADR-0038) → logging → slot/run_id → ledger
connection → single-flight lock (`skipped_concurrent`) → run slot (completed → no-op;
interrupted → reconcile and finalize without a new session) → preflight (kill switch; a dry
run then proceeds at any time; live checks the NYSE session and next-run time: ADR-0028,
`skipped_not_due`, and a due live tick records the fallback next run first) → Robinhood
credential (refresh_token mode only: load, refresh near expiry, persist before use; ADR-0021)
→ session plan (order venue, ADR-0038: armed live → the broker; a proxied dry run → the
simulated broker, with the same three option-order tools and a live-rendered prompt; a
direct-Robinhood dry run → no order tool) → prompt → agent session → the agent's `next_run`,
if valid, replaces the fallback (live only; a dry run records it unapplied) →
`assemble_run_record` → position notes (ADR-0018) → `run_audit` → persist → alerts/heartbeat
→ run-summary email (ADR-0029: once per tick, when a session started) → exit code.

Contains no trading logic. Everything the run decides is recorded as run events.
`python -m wheelta_robinhood_agent.orchestrator [--run-now]` calls `main()`. `--run-now`
starts a local dry run on demand (ADR-0038, ADR-0039) and is refused outside APP_ENV=local;
live runs only in production, on its schedule.

`OrchestratorDeps` carries the injectable boundaries (clock, database connect, calendar,
notifier, SDK transport). Its remaining fields (`remote_boundary_accepted`,
`upstream_factory`, `registries`, `mappers`, `account_scope_table`) are test seams for fake
servers; `main()` always uses the production values, which withhold every unverified remote
tool and deliver remote results only through the validating proxy (ADR-0023).
"""

import contextlib
import json
import logging
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import httpx
import psycopg
from pydantic import SecretStr

from wheelta_robinhood_agent.agent.account_scope import (
    ROBINHOOD_ACCOUNT_SCOPE,
    AccountScopeSpec,
    account_scope_id,
)
from wheelta_robinhood_agent.agent.audit.runner import AuditResult, run_audit
from wheelta_robinhood_agent.agent.board_probe import (
    BoardStatusContext,
    read_board_status,
    skip_reason,
    unavailable,
)
from wheelta_robinhood_agent.agent.mignons import DELEGATION_TOOL, Role, mignon_limits
from wheelta_robinhood_agent.agent.order_cleanup import ORDER_WIND_DOWN_SECONDS
from wheelta_robinhood_agent.agent.proxy import upstream_timeout_seconds
from wheelta_robinhood_agent.agent.result_boundary import VERIFIED_MAPPERS, EvidenceMapper
from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason
from wheelta_robinhood_agent.agent.run_loader import (
    RunMeta,
    check_references,
    load_assembly_context,
    load_audit_context,
    load_decisions,
)
from wheelta_robinhood_agent.agent.session import (
    INTERRUPT_GRACE_SECONDS,
    REMOTE_RESULT_BOUNDARY_ACCEPTED,
    STATUS_POLL_INTERVAL_SECONDS,
    LoopbackServer,
    RemoteSource,
    SessionDeps,
    SessionPlan,
    SessionPlanError,
    SessionResult,
    SessionStatus,
    TransportFactory,
    UpstreamFactory,
    available_tools_table,
    plan_session,
    run_session_sync,
)
from wheelta_robinhood_agent.agent.simulated_broker import SimulatedState
from wheelta_robinhood_agent.agent.summary_loader import load_summary_research
from wheelta_robinhood_agent.agent.trace_loader import load_decision_trace
from wheelta_robinhood_agent.config.prompts import (
    PromptError,
    PromptTemplate,
    RenderedPrompt,
    load_agent_prompts,
    load_mignon_prompts,
    render_prompt,
)
from wheelta_robinhood_agent.config.rules import LoadedRules, RulesError, load_rules
from wheelta_robinhood_agent.config.settings import (
    RobinhoodMcpAuth,
    Settings,
    SettingsError,
    load_settings,
)
from wheelta_robinhood_agent.domain.assembly import DecisionsInput, assemble_run_record
from wheelta_robinhood_agent.domain.decision_output import (
    DecisionOutputParsed,
    DecisionOutputParseFailure,
)
from wheelta_robinhood_agent.domain.enums import (
    AgentRole,
    AppEnv,
    AuditOutcome,
    ExecutionMode,
    MignonType,
    OrderVenue,
    RunStatus,
    SourceStatus,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.gating import order_venue, prompt_execution_mode
from wheelta_robinhood_agent.domain.position_notes import notes_from_run_record, opened_order_ids
from wheelta_robinhood_agent.domain.positions import PositionBook
from wheelta_robinhood_agent.domain.run import AuditStatus
from wheelta_robinhood_agent.domain.run_identity import run_id_for, slot_for
from wheelta_robinhood_agent.domain.run_record import RunRecord
from wheelta_robinhood_agent.domain.start_conditions import StartCondition, StartOutcome
from wheelta_robinhood_agent.integrations.loopback_http import serve_loopback
from wheelta_robinhood_agent.integrations.notifications.delivery import (
    DeliveryOutcome,
    DeliveryResult,
    deliver_alert,
    deliver_heartbeat,
)
from wheelta_robinhood_agent.integrations.notifications.email import (
    EmailDeliveryResult,
    EmailDeliveryStatus,
    RunSummaryEmailConfig,
    send_run_summary,
)
from wheelta_robinhood_agent.integrations.registry import ToolRegistry
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME as ROBINHOOD
from wheelta_robinhood_agent.integrations.robinhood.server import build_robinhood_server
from wheelta_robinhood_agent.integrations.robinhood.token_vault import TokenVault
from wheelta_robinhood_agent.integrations.status import SourceObservation
from wheelta_robinhood_agent.integrations.websearch.registry import TAVILY_REGISTRY
from wheelta_robinhood_agent.integrations.websearch.server import build_tavily_server
from wheelta_robinhood_agent.integrations.wheelta.registry import SERVER_NAME as WHEELTA
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.server import build_wheelta_server
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger import oauth_credentials as ledger_credentials
from wheelta_robinhood_agent.ledger import orders as ledger_orders
from wheelta_robinhood_agent.ledger import positions as ledger_positions
from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.errors import LedgerError
from wheelta_robinhood_agent.ledger.lock import single_flight
from wheelta_robinhood_agent.ledger.runs import (
    SlotState,
    append_run_event,
    latest_next_run_not_before,
    open_run_slot,
    run_event_payloads,
)
from wheelta_robinhood_agent.ledger.tool_calls import (
    append_tool_call_outcome,
    tool_call_records,
)
from wheelta_robinhood_agent.observability.alerts import (
    AlertKind,
    AlertPayload,
    HeartbeatPayload,
    build_alert,
    build_heartbeat,
    heartbeat_status_for,
)
from wheelta_robinhood_agent.observability.decision_trace import decision_log_events
from wheelta_robinhood_agent.observability.logging import (
    RunLoggerAdapter,
    bind,
    configure_logging,
    settings_secrets,
)
from wheelta_robinhood_agent.observability.metrics import RunMetrics
from wheelta_robinhood_agent.observability.redaction import Redactor
from wheelta_robinhood_agent.observability.run_summary import RunSummaryInput, SlotSummaryInput
from wheelta_robinhood_agent.orchestrator.exit_codes import (
    EXIT_FAILED,
    EXIT_OK,
    combined_exit_code,
    exit_code_for,
)
from wheelta_robinhood_agent.orchestrator.market_session import (
    TradingCalendar,
    build_nyse_calendar,
    evaluate_market_session,
)
from wheelta_robinhood_agent.orchestrator.preflight import (
    PreflightProceed,
    PreflightReason,
    PreflightSkip,
    decide_preflight,
)
from wheelta_robinhood_agent.orchestrator.robinhood_credential import (
    CredentialInserter,
    CredentialResolution,
    CredentialStatus,
    OAuthRefresher,
    refresh_via_http,
    resolve_robinhood_credential,
)
from wheelta_robinhood_agent.orchestrator.schedule import (
    NextRun,
    ScheduleSource,
    calendar_window,
    fallback_requested_at,
    is_due,
    latest_next_run,
    next_run,
)
from wheelta_robinhood_agent.orchestrator.signals import (
    RunDeadline,
    install_stop_signal_handlers,
    trip_if_deadline_passed,
)

Conn = psycopg.Connection[tuple[object, ...]]
_LOG = logging.getLogger("wheelta_robinhood_agent.run")
NOTIFY_TIMEOUT_SECONDS = 10.0
# ADR-0029: summary email deliveries share the alerts_sent ledger table under this kind.
RUN_SUMMARY_EMAIL_KIND = "run_summary_email"
# Time kept back from the run budget for assembly, audit, and finalization.
FINALIZE_RESERVE_SECONDS = 60.0


class Notifier(Protocol):
    """Alert and heartbeat delivery. Never raises; returns the typed delivery result."""

    def alert(self, payload: AlertPayload) -> DeliveryResult: ...

    def heartbeat(self, payload: HeartbeatPayload) -> DeliveryResult: ...


@dataclass
class HttpNotifier:
    """`Notifier` over integrations/notifications (webhook URLs are secrets from Settings)."""

    alert_url: SecretStr | None
    heartbeat_url: SecretStr | None
    _client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=NOTIFY_TIMEOUT_SECONDS)
        return self._client

    def alert(self, payload: AlertPayload) -> DeliveryResult:
        return deliver_alert(
            payload, client=self._http(), url=self.alert_url, timeout_seconds=NOTIFY_TIMEOUT_SECONDS
        )

    def heartbeat(self, payload: HeartbeatPayload) -> DeliveryResult:
        return deliver_heartbeat(
            payload,
            client=self._http(),
            url=self.heartbeat_url,
            timeout_seconds=NOTIFY_TIMEOUT_SECONDS,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


class SummaryMailer(Protocol):
    """Run-summary email delivery (ADR-0029). Never raises; returns the typed result."""

    def send(self, summary: SlotSummaryInput, redactor: Redactor) -> EmailDeliveryResult: ...


@dataclass
class HttpSummaryMailer:
    """`SummaryMailer` over integrations/notifications (Resend + Anthropic Messages)."""

    config: RunSummaryEmailConfig
    _client: httpx.Client | None = None

    def send(self, summary: SlotSummaryInput, redactor: Redactor) -> EmailDeliveryResult:
        if self._client is None:
            self._client = httpx.Client()
        return send_run_summary(summary, config=self.config, client=self._client, redactor=redactor)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def summary_mailer_for(settings: Settings) -> HttpSummaryMailer | None:
    """The mailer when RUN_SUMMARY_EMAIL_ENABLED (Settings then guarantees key and recipient)."""
    if not settings.RUN_SUMMARY_EMAIL_ENABLED:
        return None
    return HttpSummaryMailer(
        RunSummaryEmailConfig(
            enabled=True,
            resend_api_key=settings.RESEND_API_KEY,
            from_address=settings.RUN_SUMMARY_EMAIL_FROM,
            to_address=settings.RUN_SUMMARY_EMAIL_TO,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            model=settings.run_summary_model,
        )
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _calendar(start: date, end: date) -> TradingCalendar:
    return build_nyse_calendar(start, end)


@dataclass(frozen=True)
class OrchestratorDeps:
    notifier: Notifier
    clock: Callable[[], datetime] = _utc_now
    # ADR-0029: None means no run-summary email.
    summary_mailer: SummaryMailer | None = None
    connect_db: Callable[[SecretStr], Conn] = connect
    calendar_factory: Callable[[date, date], TradingCalendar] = _calendar
    transport_factory: TransportFactory | None = None
    install_signals: bool = True
    scratch_root: Path | None = None
    # Test seams (module docstring): production always uses the defaults.
    remote_boundary_accepted: bool = REMOTE_RESULT_BOUNDARY_ACCEPTED
    upstream_factory: UpstreamFactory | None = None
    # ADR-0063: serves each Mignon role's tools on loopback so the orchestrator never lists
    # them. Used only with the real CLI: a fake transport cannot reach inline servers.
    loopback_server: LoopbackServer | None = serve_loopback
    registries: tuple[ToolRegistry, ToolRegistry] = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY)
    # ADR-0058: Tavily web search (optional source; without a key it is disabled).
    tavily_registry: ToolRegistry = TAVILY_REGISTRY
    mappers: Mapping[tuple[str, str], EvidenceMapper] = field(
        default_factory=lambda: VERIFIED_MAPPERS
    )
    account_scope_table: Mapping[str, AccountScopeSpec] = field(
        default_factory=lambda: ROBINHOOD_ACCOUNT_SCOPE
    )
    # ADR-0021 (ROBINHOOD_MCP_AUTH=refresh_token): the OAuth refresh and the credential insert.
    oauth_refresher: OAuthRefresher = refresh_via_http
    insert_credential: CredentialInserter = ledger_credentials.insert_credential
    connect_budget_seconds: float | None = None
    interrupt_grace_seconds: float | None = None
    status_poll_interval: float | None = None


def _heartbeat(
    settings: Settings,
    deps: OrchestratorDeps,
    status: RunStatus,
    run_id: uuid.UUID | None,
    slot: datetime | None,
    reason: str | None,
) -> None:
    payload = build_heartbeat(
        status,
        run_id=str(run_id) if run_id else None,
        environment=settings.APP_ENV,
        slot=slot,
        occurred_at=deps.clock(),
        reason=reason,
    )
    deps.notifier.heartbeat(payload)


def run_once(
    settings: Settings,
    rules: LoadedRules,
    templates: Mapping[AgentRole, PromptTemplate],
    deps: OrchestratorDeps,
    mignon_templates: Mapping[MignonType, PromptTemplate] | None = None,
    *,
    run_now: bool = False,
) -> int:
    """Run one cron fire and return the process exit code (exit_codes.py).

    `templates` holds the close and sell agents' prompts (`load_agent_prompts`).
    `mignon_templates` defaults to the packaged Mignon prompts (`main` loads them at startup
    so a missing file fails before any network call). Dry runs are local and on demand
    (ADR-0038, ADR-0039): live runs only in production, on its calendar and schedule; with the
    effective mode off (always, outside production), outside APP_ENV=local nothing runs
    (`skipped_dry_run_not_local`), and locally only a `run_now` invocation runs, at any time
    (the kill switch still applies); otherwise `skipped_dry_run_not_requested`. Both skips
    happen before the ledger. `run_now` outside APP_ENV=local raises ValueError.
    """
    if run_now and settings.APP_ENV is not AppEnv.LOCAL:
        raise ValueError("run_now starts a local dry run; it requires APP_ENV=local")
    if mignon_templates is None:
        mignon_templates = load_mignon_prompts()
    started = deps.clock()
    slot = slot_for(started)
    run_id = run_id_for(settings.APP_ENV, slot, AgentRole.CLOSE)
    log = bind(_LOG, run_id=str(run_id), stage="boot", slot=slot.isoformat())
    if settings.effective_execution_mode is ExecutionMode.OFF:
        # ADR-0039: dry runs are local and on demand; outside local an off mode runs nothing.
        if settings.APP_ENV is not AppEnv.LOCAL:
            status, reason = RunStatus.SKIPPED_DRY_RUN_NOT_LOCAL, "dry_run_not_local"
        elif not run_now:
            status, reason = RunStatus.SKIPPED_DRY_RUN_NOT_REQUESTED, "dry_run_not_requested"
        else:
            status = None
        if status is not None:
            log.info("no dry run here", extra={"status": status.value, "reason": reason})
            _heartbeat(settings, deps, status, run_id, slot, reason)
            return exit_code_for(status)
    try:
        conn = deps.connect_db(settings.DATABASE_URL)
    except LedgerError as exc:
        log.error("ledger unavailable", extra={"error_type": type(exc).__name__})
        _heartbeat(settings, deps, RunStatus.FAILED, run_id, slot, "ledger_unavailable")
        return EXIT_FAILED
    with conn, single_flight(conn, settings.APP_ENV) as acquired:
        if not acquired:
            log.info("another run holds the lock", extra={"status": "skipped_concurrent"})
            _heartbeat(
                settings, deps, RunStatus.SKIPPED_CONCURRENT, run_id, slot, "lock_contention"
            )
            return exit_code_for(RunStatus.SKIPPED_CONCURRENT)
        tick = _Tick(settings, rules, templates, deps, conn, slot, started, log)
        tick.mignon_templates = dict(mignon_templates)
        tick.run_now = run_now
        return tick.run()


# ADR-0057: a close run finalized by the slot gate ends the tick: no sell run.
SLOT_GATE_STATUSES = frozenset(
    {
        RunStatus.SKIPPED_KILLED,
        RunStatus.SKIPPED_MARKET_CLOSED,
        RunStatus.SKIPPED_NOT_DUE,
        RunStatus.SKIPPED_DRY_RUN_NOT_REQUESTED,
    }
)


class _Tick:
    """One cron fire: the close run, then the sell run (module docstring, ADR-0057)."""

    def __init__(
        self,
        settings: Settings,
        rules: LoadedRules,
        templates: Mapping[AgentRole, PromptTemplate],
        deps: OrchestratorDeps,
        conn: Conn,
        slot: datetime,
        started: datetime,
        log: RunLoggerAdapter,
    ) -> None:
        self.settings = settings
        self.rules = rules
        self.templates = templates
        self.deps = deps
        self.conn = conn
        self.slot = slot
        self.started = started
        self.log = log
        self.mignon_templates: dict[MignonType, PromptTemplate] = {}
        self.run_now = False
        # ADR-0057: a dry-run tick's simulated broker state, shared by both runs.
        self.simulated = SimulatedState()
        # ADR-0021: resolved (and refreshed, at most once) by the tick's first run.
        self.credential: list[CredentialResolution] = []
        self.tick_alerts: set[AlertKind] = set()
        self.runs: list[_Run] = []

    def _new_run(self, role: AgentRole, run_id: uuid.UUID) -> "_Run":
        budget = (
            self.settings.CLOSE_AGENT_TIMEOUT_SECONDS
            if role is AgentRole.CLOSE
            else self.settings.RUN_TIMEOUT_SECONDS
        )
        run = _Run(
            self.settings,
            self.rules,
            self.templates[role],
            self.deps,
            self.conn,
            run_id,
            self.slot,
            self.started,
            bind(_LOG, run_id=str(run_id), stage="boot", slot=self.slot.isoformat()),
            role=role,
            budget_seconds=budget,
        )
        run.mignon_templates = dict(self.mignon_templates)
        run.run_now = self.run_now
        run.simulated_state = self.simulated
        run.tick_credential = self.credential
        run.tick_alerts = self.tick_alerts
        run.order_scope_run_id = run_id_for(self.settings.APP_ENV, self.slot, AgentRole.CLOSE)
        self.runs.append(run)
        return run

    def run(self) -> int:
        env = self.settings.APP_ENV
        close_slot = open_run_slot(self.conn, env, self.slot, AgentRole.CLOSE)
        if close_slot.state is not SlotState.NEW:
            return self._refire(close_slot.state)
        close = self._new_run(AgentRole.CLOSE, close_slot.run_id)
        close.execute()
        if close.final_status in SLOT_GATE_STATUSES or not close.gate_passed:
            return self._finish_tick()
        # Built before the sell row exists: nothing may fail between creating it and starting.
        outcome = close.outcome_json()
        sell_slot = open_run_slot(self.conn, env, self.slot, AgentRole.SELL)
        if sell_slot.state is not SlotState.NEW:
            # Only a racing process could have opened it; the lock forbids that.
            raise LedgerError("the sell run of a new tick already exists")
        sell = self._new_run(AgentRole.SELL, sell_slot.run_id)
        sell.gate_at = close.gate_at
        sell.related_run_ids = (close.run_id,)
        sell.close_agent_outcome = outcome
        sell.execute()
        return self._finish_tick()

    def _refire(self, close_state: SlotState) -> int:
        """A slot that already has a close run: recover interrupted runs, start no session."""
        env = self.settings.APP_ENV
        if close_state is SlotState.INTERRUPTED:
            self._new_run(AgentRole.CLOSE, run_id_for(env, self.slot, AgentRole.CLOSE)).recover()
        sell_id = run_id_for(env, self.slot, AgentRole.SELL)
        exists = self.conn.execute("SELECT 1 FROM runs WHERE run_id = %s", (sell_id,)).fetchone()
        if exists is not None:
            sell_slot = open_run_slot(self.conn, env, self.slot, AgentRole.SELL)
            if sell_slot.state is SlotState.INTERRUPTED:
                self._new_run(AgentRole.SELL, sell_slot.run_id).recover()
        if not self.runs:
            self.log.info("slot already finalized; nothing to do", extra={"status": "noop"})
            return EXIT_OK
        return self._finish_tick()

    def _finish_tick(self) -> int:
        """One heartbeat and one summary email for the tick (ADR-0057); the exit code is the
        most severe of its runs'."""
        finished = [r for r in self.runs if r.final_status is not None]
        if not finished:
            return EXIT_OK
        failing = [
            r
            for r in finished
            if r.final_status is not None
            and heartbeat_status_for(r.final_status)
            is not heartbeat_status_for(RunStatus.COMPLETED)
        ]
        lead = failing[0] if failing else finished[-1]
        reason = "; ".join(
            f"{r.role.value}={r.final_status.value}"
            + (f" ({r.final_reason})" if r.final_reason else "")
            for r in finished
            if r.final_status is not None
        )
        if lead.final_status is not None:
            _heartbeat(self.settings, self.deps, lead.final_status, lead.run_id, self.slot, reason)
        if any(r.session_started for r in finished):
            self._send_summary(finished)
        return combined_exit_code(
            tuple(exit_code_for(r.final_status) for r in finished if r.final_status)
        )

    def _next_run(self, runs: Sequence["_Run"]) -> NextRun | None:
        """The tick's effective next run: the earliest agent request, else the fallback."""
        chosen = [
            r.next_run for r in runs if r.next_run and r.next_run.source is ScheduleSource.AGENT
        ]
        if chosen:
            return min(chosen, key=lambda n: n.not_before)
        fallback = [r.next_run for r in runs if r.next_run is not None]
        return fallback[0] if fallback else None

    def _send_summary(self, runs: Sequence["_Run"]) -> None:
        """ADR-0029, ADR-0057: email the tick summary. Informational: never changes a status
        or the exit code."""
        mailer = self.deps.summary_mailer
        if mailer is None:
            return
        lead = runs[0]
        try:
            next_run = self._next_run(runs)
            summary = SlotSummaryInput(
                environment=self.settings.APP_ENV,
                slot=self.slot,
                agents=tuple(r.summary_input() for r in runs),
                next_run_at=next_run.not_before if next_run else None,
                next_run_source=next_run.source.value if next_run else None,
            )
            result = mailer.send(summary, lead.redactor)
        except Exception as exc:  # noqa: BLE001 - informational email: never fail the run
            self.log.warning("run summary email failed", extra={"error_type": type(exc).__name__})
            return
        self.log.bind(stage="summary_email").info(
            "run summary email",
            extra={"delivery_status": result.status.value, "error": result.error},
        )
        if result.status is EmailDeliveryStatus.SKIPPED:
            return
        for run in runs:
            if run.session_started:
                run.record_summary_email(result, self.slot)


class _Run:
    """One run's mutable bookkeeping. Every step appends to the ledger before moving on."""

    def __init__(
        self,
        settings: Settings,
        rules: LoadedRules,
        template: PromptTemplate,
        deps: OrchestratorDeps,
        conn: Conn,
        run_id: uuid.UUID,
        slot: datetime,
        started: datetime,
        log: RunLoggerAdapter,
        *,
        role: AgentRole,
        budget_seconds: int,
    ) -> None:
        self.settings = settings
        self.rules = rules
        self.template = template
        # ADR-0057: which agent this run is.
        self.role = role
        # ADR-0025: set by run_once; rendered per run in _render when Mignons are allowed.
        self.mignon_templates: dict[MignonType, PromptTemplate] = {}
        self.mignon_prompts: dict[MignonType, RenderedPrompt] = {}
        self.deps = deps
        self.conn = conn
        self.run_id = run_id
        self.slot = slot
        self.started = started
        self.log = log
        self.metrics = RunMetrics(str(run_id))
        self.control = RunControl()
        self.redactor = Redactor(
            account_number=settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            secrets=settings_secrets(settings),
        )
        self.scope_id = account_scope_id(settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER)
        # ADR-0038: where order tools go; the plan decides it (`_plan`), recovery reads it back.
        self.order_venue = order_venue(settings.effective_execution_mode, robinhood_proxied=False)
        # ADR-0057: measured from the tick's start, so the sell run gets what is left.
        self.deadline = RunDeadline(started, budget_seconds)
        self._event_counter = 0
        self.alerts_sent: list[AlertKind] = []
        # ADR-0028: a hand-started local run (`--run-now`); set by run_once.
        self.run_now = False
        # ADR-0028: when this run passed the schedule gate; the fallback and the maximum gap
        # are measured from it. Set in _execute on a due tick.
        self.gate_at: datetime | None = None
        # ADR-0029: what the run-summary email reports. The email is sent only when a session
        # started; the rest is filled in by _finish, _audit and record_next_run.
        self.session_started = False
        self.summary_record: RunRecord | None = None
        self.summary_diagnostics: list[str] = []
        self.audit_result: AuditResult | None = None
        self.audit_ran = False
        self.next_run: NextRun | None = None
        self.next_run_rationale: str | None = None
        # ADR-0057: set by finalize; read by the tick for the heartbeat, email, and exit code.
        self.final_status: RunStatus | None = None
        self.final_reason: str | None = None
        # The slot gate passed (close) or was re-checked and passed (sell).
        self.gate_passed = False
        self.start_condition: StartCondition | None = None
        # Set by the tick: the close run of the same tick (sell only), the tick's order scope
        # run, the shared simulated state, and the close run's outcome for the sell prompt.
        self.related_run_ids: tuple[uuid.UUID, ...] = ()
        self.order_scope_run_id: uuid.UUID | None = None
        self.simulated_state: SimulatedState | None = None
        self.close_agent_outcome: str | None = None
        # ADR-0057, ADR-0021: the tick's one credential resolution, shared by both runs; a
        # refresh is never attempted twice in a tick (a second try would be a retry).
        self.tick_credential: list[CredentialResolution] = []
        self.credential_reused = False
        # Kinds already alerted by a run of this tick (`tick_alert`); shared by the tick.
        self.tick_alerts: set[AlertKind] = set()

    # -- ledger helpers -------------------------------------------------------------------

    def event(
        self,
        event_type: RunEventType,
        payload: Mapping[str, object] | None = None,
        status: RunStatus | None = None,
        key: str | None = None,
    ) -> None:
        self._event_counter += 1
        dedup = key or f"{event_type.value}:{self._event_counter}:{uuid.uuid4().hex}"
        append_run_event(
            self.conn,
            self.run_id,
            event_type,
            observed_at=self.deps.clock(),
            dedup_key=dedup,
            payload=payload,
            status=status,
        )

    def alert(
        self, kind: AlertKind, message: str, details: Mapping[str, object] | None = None
    ) -> None:
        payload = build_alert(
            kind,
            run_id=str(self.run_id),
            environment=self.settings.APP_ENV,
            occurred_at=self.deps.clock(),
            message=message,
            details=details,
            redactor=self.redactor,
        )
        result = self.deps.notifier.alert(payload)
        self.alerts_sent.append(kind)
        with contextlib.suppress(Exception):
            ledger_evidence.record_alert_sent(
                self.conn,
                run_id=self.run_id,
                alert_kind=kind.value,
                dedup_key=f"{kind.value}:{len(self.alerts_sent)}",
                payload=payload.model_dump(mode="json"),
                delivery_status=ledger_evidence.DeliveryStatus.SENT
                if result.outcome is DeliveryOutcome.DELIVERED
                else ledger_evidence.DeliveryStatus.FAILED,
                attempted_at=self.deps.clock(),
            )

    def tick_alert(
        self, kind: AlertKind, message: str, details: Mapping[str, object] | None = None
    ) -> None:
        """An alert about a cause both runs of a tick share (the Robinhood credential): sent
        by the first run that meets it only (ADR-0057)."""
        if kind in self.tick_alerts:
            return
        self.tick_alerts.add(kind)
        self.alert(kind, message, details)

    def observe_sources(self, observations: Sequence[SourceObservation]) -> None:
        for obs in observations:
            self.event(RunEventType.SOURCE_STATUS, obs.model_dump(mode="json"))

    def finalize(self, status: RunStatus, reason: str | None) -> int:
        payload: dict[str, object] = {"reason": reason} if reason else {}
        if self.summary_diagnostics:
            payload["diagnostic_details"] = [
                self.redactor.redact_text(detail) for detail in self.summary_diagnostics
            ]
        self.event(RunEventType.STATUS, payload or None, status=status)
        # ADR-0057: the tick sends one heartbeat and one email for both runs.
        self.final_status, self.final_reason = status, reason
        snapshot = self.metrics.snapshot().model_dump(mode="json")
        self.log.bind(stage="finalize").info(
            "run finished",
            extra={
                "agent": self.role.value,
                "status": status.value,
                "reason": reason,
                "metrics": snapshot,
            },
        )
        return exit_code_for(status)

    def summary_input(self) -> RunSummaryInput:
        """This run's section of the tick's summary email (ADR-0029, ADR-0057)."""
        if self.final_status is None:
            raise SessionPlanError("a run summary is built only after the run is final")
        if not self.session_started:
            return RunSummaryInput(
                run_id=str(self.run_id),
                environment=self.settings.APP_ENV,
                slot=self.slot,
                status=self.final_status,
                reason=self.final_reason,
                agent=self.role,
                session_started=False,
                requested_execution_mode=self.settings.requested_execution_mode,
                effective_execution_mode=self.settings.effective_execution_mode,
                order_venue=self.order_venue,
                record=None,
                diagnostic_details=tuple(self.summary_diagnostics),
                alerts=tuple(k.value for k in self.alerts_sent),
            )
        audit = self.audit_result
        research_unavailable = None
        try:
            candidates, reports = load_summary_research(self.conn, self.run_id)
        except Exception as exc:  # noqa: BLE001 - preserve the email if context loading fails
            candidates, reports = (), ()
            research_unavailable = type(exc).__name__
        audit_details = (
            tuple(
                f"{f.check_id.value} {f.outcome.value}: {f.detail}"
                for f in audit.findings
                if f.outcome is not AuditOutcome.PASS
            )
            + tuple(f"{e.check_id.value}: {e.error_type}: {e.message}" for e in audit.errors)
            if audit is not None
            else ()
        )
        return RunSummaryInput(
            run_id=str(self.run_id),
            environment=self.settings.APP_ENV,
            slot=self.slot,
            status=self.final_status,
            reason=self.final_reason,
            agent=self.role,
            requested_execution_mode=self.settings.requested_execution_mode,
            effective_execution_mode=self.settings.effective_execution_mode,
            order_venue=self.order_venue,
            record=self.summary_record,
            candidates=candidates,
            research_reports=reports,
            research_unavailable=research_unavailable,
            diagnostic_details=tuple(self.summary_diagnostics),
            audit_details=audit_details,
            audit_status=(
                (audit.status.value if audit is not None else AuditStatus.FAILED.value)
                if self.audit_ran
                else None
            ),
            audit_violations=len(audit.violations) if audit is not None else 0,
            audit_unverifiable=len(audit.unverifiable_checks) if audit is not None else 0,
            alerts=tuple(k.value for k in self.alerts_sent),
            next_run_at=self.next_run.not_before if self.next_run else None,
            next_run_source=self.next_run.source.value if self.next_run else None,
            next_run_rationale=self.next_run_rationale,
        )

    def record_summary_email(self, result: EmailDeliveryResult, slot: datetime) -> None:
        """Record the tick's email delivery on this run (alerts_sent, ADR-0029)."""
        with contextlib.suppress(Exception):
            ledger_evidence.record_alert_sent(
                self.conn,
                run_id=self.run_id,
                alert_kind=RUN_SUMMARY_EMAIL_KIND,
                dedup_key=f"run-summary/{slot.isoformat()}",
                payload={
                    "subject": result.subject,
                    "provider_message_id": result.provider_message_id,
                    "prose_written": result.prose_written,
                    "attempts": result.attempts,
                    "status_code": result.status_code,
                    "error": result.error,
                },
                delivery_status=ledger_evidence.DeliveryStatus.SENT
                if result.status is EmailDeliveryStatus.SENT
                else ledger_evidence.DeliveryStatus.FAILED,
                attempted_at=self.deps.clock(),
            )

    def outcome_json(self) -> str:
        """ADR-0057: this (close) run's outcome for the sell prompt, from the ledger: status,
        reason, start condition, and the orders it placed with their recorded state. Context
        only: if the orders cannot be read, they are reported as unavailable (never as none),
        so the sell run still starts."""
        orders: list[dict[str, object]] | str = []
        try:
            orders = self._outcome_orders()
        except Exception as exc:  # noqa: BLE001 - context for the next agent; never blocks it
            self.log.warning(
                "close-run orders unavailable for the sell prompt",
                extra={"error_type": type(exc).__name__},
            )
            orders = f"unavailable ({type(exc).__name__}); read orders and positions yourself"
        outcome = {
            "status": self.final_status.value if self.final_status else None,
            "reason": self.final_reason,
            "start_condition": self.start_condition.model_dump(mode="json")
            if self.start_condition
            else None,
            "orders": orders,
        }
        return json.dumps(outcome, sort_keys=True)

    def _outcome_orders(self) -> list[dict[str, object]]:
        orders: list[dict[str, object]] = []
        for record in ledger_orders.run_order_records(self.conn, self.run_id):
            intent = record.intent
            orders.append(
                {
                    "occ_symbol": str(intent.occ_symbol)
                    if intent is not None and intent.occ_symbol is not None
                    else None,
                    "side": intent.side_raw if intent is not None else None,
                    "quantity": intent.quantity if intent is not None else None,
                    "limit_price": str(intent.limit_price)
                    if intent is not None and intent.limit_price is not None
                    else None,
                    "status": record.status.value,
                    "filled_quantity": record.filled_quantity,
                }
            )
        return orders

    def meta(self, *, prompt: RenderedPrompt | None, model_id: str | None) -> RunMeta:
        return RunMeta(
            run_id=self.run_id,
            environment=self.settings.APP_ENV,
            slot=self.slot,
            requested_mode=self.settings.requested_execution_mode,
            effective_mode=self.settings.effective_execution_mode,
            order_venue=self.order_venue,
            account_scope_id=self.scope_id,
            rules=self.rules,
            prompt_id=self.template.prompt_id if prompt else None,
            prompt_hash=self.template.sha256 if prompt else None,
            model_id=model_id,
            role=self.role,
        )

    # -- lifecycle ------------------------------------------------------------------------

    def execute(self) -> int:
        settings = self.settings
        self.event(
            RunEventType.STARTED,
            {
                "agent": self.role.value,
                "config_snapshot": settings.config_snapshot(),
                "rules_version": self.rules.version,
                "rules_hash": self.rules.sha256,
                "prompt_id": self.template.prompt_id,
                "prompt_version": self.template.version,
                "prompt_template_hash": self.template.sha256,
                "execution_ceiling": settings.execution_ceiling.value,
            },
            key="started",
        )
        self.event(RunEventType.STATUS, status=RunStatus.RUNNING, key="status:running")
        try:
            return self._execute()
        except Exception as exc:  # noqa: BLE001 - fail closed: record, alert via heartbeat
            self.log.exception("run failed", extra={"error_type": type(exc).__name__})
            self.metrics.error(type(exc).__name__)
            self.summary_diagnostics.append(
                self.redactor.redact_text(f"Orchestrator: {type(exc).__name__}: {exc}")
            )
            return self.finalize(RunStatus.FAILED, f"error:{type(exc).__name__}")

    def _execute(self) -> int:
        settings = self.settings
        now = self.deps.clock()
        calendar = self.deps.calendar_factory(
            now.date() - timedelta(days=7), now.date() + timedelta(days=7)
        )
        market = evaluate_market_session(now, calendar)
        self.event(
            RunEventType.MARKET_SESSION,
            {
                "as_of": market.as_of.isoformat(),
                "session": market.session.value,
                "may_proceed": market.may_proceed,
                "session_date": market.session_date.isoformat(),
                "calendar": {
                    "library": market.provenance.library,
                    "library_version": market.provenance.library_version,
                    "calendar_code": market.provenance.calendar_code,
                },
            },
        )
        # ADR-0038: a dry run is on demand only and never reads or writes the live schedule.
        dry_run = settings.effective_execution_mode is ExecutionMode.OFF
        if self.role is AgentRole.SELL:
            # ADR-0057: the close run passed the schedule gate for the tick; its own next run,
            # recorded since, must not make the sell run "not due".
            not_before, due = None, True
            schedule_check: dict[str, object] = {
                "due": True,
                "carried_from": [str(r) for r in self.related_run_ids],
                "run_now": self.run_now,
            }
        else:
            not_before = latest_next_run_not_before(self.conn, settings.APP_ENV)
            due = is_due(now, not_before)
            schedule_check = {
                "not_before": not_before.isoformat() if not_before else None,
                "due": due,
                "run_now": self.run_now,
            }
        self.event(
            RunEventType.METADATA,
            {"schedule_check": schedule_check},
            key="metadata:schedule_check",
        )
        decision = decide_preflight(
            kill_switch=settings.KILL_SWITCH,
            market=market,
            due=due,
            requested_mode=settings.requested_execution_mode,
            armed=settings.EXECUTION_ARMED,
            ceiling=settings.execution_ceiling,
            on_demand=self.run_now,
        )
        if (
            self.role is not AgentRole.SELL
            and not dry_run
            and due
            and not (
                isinstance(decision, PreflightSkip)
                and decision.reason is PreflightReason.OUTSIDE_REGULAR_SESSION
            )
        ):
            # ADR-0028: a due tick is this cadence's run, even when killed. The fallback keeps
            # a crashed or output-less run (and the kill alert) on the hourly cadence.
            self.gate_at = now
            minutes = self.rules.rules.scheduling.fallback_next_run_minutes
            self.record_next_run(fallback_requested_at(now, minutes), ScheduleSource.FALLBACK)
        if isinstance(decision, PreflightSkip):
            if (
                decision.status is RunStatus.SKIPPED_KILLED
                and (due or dry_run)
                and self.role is not AgentRole.SELL
            ):
                self.alert(AlertKind.KILL_SWITCH_ENGAGED, "KILL_SWITCH=true; the run did not start")
            return self.finalize(decision.status, decision.reason.value)
        self.gate_passed = isinstance(decision, PreflightProceed)
        if self._session_budget() <= ORDER_WIND_DOWN_SECONDS:
            # ADR-0057: what is left of the tick budget (the sell run after a long close run)
            # cannot hold a session outside the wind-down, so none is started.
            self.alert(
                AlertKind.RUN_TIMEOUT,
                f"the {self.role.value} agent's run budget was used up before its session",
            )
            return self.finalize(RunStatus.TIMED_OUT, "no_budget_left")
        restore = (
            install_stop_signal_handlers(self.control, self.deps.clock)
            if (self.deps.install_signals)
            else None
        )
        try:
            return self._agent_run(decision.effective_mode)
        finally:
            if restore is not None:
                restore()

    def record_next_run(self, requested_at: datetime, source: ScheduleSource) -> NextRun:
        """Record when the next session may start (ADR-0028): `requested_at` capped at the
        maximum gap after the gate, then moved into the NYSE regular session. Raises
        CalendarOutOfRange/ValueError if it cannot be placed, and SessionPlanError before the
        gate has passed."""
        if self.gate_at is None:
            raise SessionPlanError("a next run is recorded only after the schedule gate")
        latest = latest_next_run(self.gate_at, self.rules.rules.scheduling.max_next_run_gap_hours)
        start, end = calendar_window(min(requested_at, latest))
        scheduled = next_run(
            requested_at, source, self.deps.calendar_factory(start, end), latest_at=latest
        )
        self.event(RunEventType.SCHEDULE, scheduled.event_payload(), key=f"schedule:{source.value}")
        self.next_run = scheduled
        self.log.bind(stage="schedule").info("next run scheduled", extra=scheduled.event_payload())
        return scheduled

    def _apply_agent_next_run(self, decisions: DecisionsInput) -> None:
        """A valid `next_run` replaces the fallback. A time that cannot be placed (no session
        within the search window, or out of the date range) is recorded as rejected, and the
        fallback stands. Never raises: the run's status does not depend on this."""
        if not isinstance(decisions, DecisionOutputParsed) or decisions.output.next_run is None:
            return
        requested = decisions.output.next_run.at
        if self.settings.effective_execution_mode is ExecutionMode.OFF:
            # ADR-0038: recorded for comparison with live, never scheduled (dry runs are on
            # demand and must not move the live schedule).
            self.event(
                RunEventType.METADATA,
                {
                    "next_run_not_applied": {
                        "requested_at": requested.isoformat(),
                        "rationale": decisions.output.next_run.rationale,
                        "reason": "dry_run_on_demand",
                    }
                },
                key="metadata:next_run_not_applied",
            )
            return
        try:
            self.record_next_run(requested, ScheduleSource.AGENT)
            self.next_run_rationale = decisions.output.next_run.rationale
        except Exception as exc:  # noqa: BLE001 - model-chosen input: any failure keeps the fallback
            rejected = {"requested_at": requested.isoformat(), "error_type": type(exc).__name__}
            self.log.warning("agent next run not placeable; fallback stands", extra=rejected)
            self.event(
                RunEventType.METADATA,
                {"next_run_rejected": rejected},
                key="metadata:next_run_rejected",
            )

    def _robinhood_credential(self) -> CredentialResolution | None:
        """ADR-0021: in refresh_token mode, the access token for this run (refreshed and
        persisted first if near expiry). None in the other modes. Recorded without secrets;
        every token seen is added to the run's redactor. ADR-0057: resolved once per tick; the
        second run reuses the first's resolution, success or failure, so a refresh is never
        retried."""
        settings = self.settings
        if settings.ROBINHOOD_MCP_AUTH is not RobinhoodMcpAuth.REFRESH_TOKEN:
            return None
        key = settings.ROBINHOOD_TOKEN_ENCRYPTION_KEY
        if key is None:  # Settings rejects this; kept so the type is narrowed without assert
            raise SessionPlanError("refresh_token mode without an encryption key")
        reused = bool(self.tick_credential)
        self.credential_reused = reused
        if reused:
            resolution = self.tick_credential[0]
        else:
            resolution = resolve_robinhood_credential(
                self.conn,
                environment=settings.APP_ENV,
                vault=TokenVault(key),
                clock=self.deps.clock,
                refresher=self.deps.oauth_refresher,
                insert=self.deps.insert_credential,
            )
            self.tick_credential.append(resolution)
        self.redactor = Redactor(
            account_number=settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            secrets=(*settings_secrets(settings), *resolution.secrets),
        )
        self.event(
            RunEventType.METADATA,
            {"robinhood_credential": {**resolution.event_payload(), "reused_in_tick": reused}},
            key="metadata:robinhood_credential",
        )
        log = self.log.bind(stage="robinhood_auth")
        if resolution.usable:
            log.info("robinhood credential resolved", extra=resolution.event_payload())
        else:
            log.error("robinhood credential unavailable", extra=resolution.event_payload())
        return resolution

    def _plan(
        self, effective_mode: ExecutionMode, robinhood_token: SecretStr | None = None
    ) -> SessionPlan:
        rh_registry, wt_registry = self.deps.registries
        now = self.deps.clock()
        plan = plan_session(
            effective_mode=effective_mode,
            workspace_writes=self.settings.workspace_writes_enabled,
            sources=(
                RemoteSource(
                    rh_registry,
                    build_robinhood_server(self.settings, now, robinhood_token),
                    required=True,
                ),
                RemoteSource(wt_registry, build_wheelta_server(self.settings)),
                RemoteSource(self.deps.tavily_registry, build_tavily_server(self.settings, now)),
            ),
            observed_at=now,
            remote_boundary_accepted=(
                self.deps.remote_boundary_accepted or self.settings.remote_result_risk_accepted
            ),
            mignons=mignon_limits(self.rules.rules) is not None,
            agent=self.role,
        )
        self.order_venue = plan.order_venue
        self.event(
            RunEventType.METADATA,
            {
                "tool_access": {
                    "effective_mode": plan.tool_access.effective_mode.value,
                    "order_venue": plan.order_venue.value,
                    "allowed_tools": list(plan.tool_access.allowed_tools),
                    "disallowed_tools": list(plan.tool_access.disallowed_tools),
                },
                "withheld": dict(plan.withheld),
            },
        )
        self.observe_sources(plan.observations)
        return plan

    def _render(self, plan: SessionPlan, book: PositionBook) -> RenderedPrompt:
        now = self.deps.clock()
        owned = ledger_orders.owned_unresolved_orders(self.conn, self.scope_id)
        # ADR-0053: when the wind-down starts (ADR-0050), so the agent can judge whether
        # another discovery round fits.
        work = max(self._session_budget() - ORDER_WIND_DOWN_SECONDS, 0.0)
        values = {
            "as_of": now.isoformat(),
            "work_deadline": (now + timedelta(seconds=work)).isoformat(),
            # ADR-0038: a simulated-venue dry run is told it is live, so it follows the live
            # procedure exactly; the recorded effective mode and venue say what it was.
            "execution_mode": prompt_execution_mode(plan.order_venue).value,
            "account_ref": self.settings.account_last4,
            "workspace_prefix": self.settings.ROBINHOOD_WORKSPACE_PREFIX,
            "policy_version": str(self.rules.version),
            "policy": self.rules.rendered,
            "available_tools": available_tools_table(
                plan, mignon_models=self.settings.mignon_models
            ),
            "position_book": book.model_dump_json(),
            "owned_orders": json.dumps([r.model_dump(mode="json") for r in owned], sort_keys=True),
            "recent_decisions": "[]",
        }
        if self.role is AgentRole.SELL:
            # ADR-0057: what the Buy-to-Close agent did this tick (context, from the ledger).
            values["close_agent_outcome"] = self.close_agent_outcome or "null"
        if "board_status" in self.template.placeholders:
            # ADR-0062: the board status, read by trusted code (the orchestrator may not).
            values["board_status"] = self._board_status(plan).prompt_value()
        rendered = render_prompt(self.template, values)
        self.mignon_prompts = self._render_mignons(plan, now)
        self.event(
            RunEventType.METADATA,
            {
                "agent": self.role.value,
                "prompt_id": rendered.prompt_id,
                "prompt_version": rendered.version,
                "prompt_template_hash": rendered.template_sha256,
                "rendered_prompt_hash": rendered.sha256,
                "prompt_execution_mode": values["execution_mode"],
                "model_id": self.settings.AGENT_MODEL,
                "mignon_models": list(self.settings.mignon_models) if self.mignon_prompts else [],
                "position_book": book.model_dump(mode="json"),
                "mignon_prompts": {
                    mignon.value: {
                        "prompt_id": r.prompt_id,
                        "prompt_version": r.version,
                        "prompt_template_hash": r.template_sha256,
                        "rendered_prompt_hash": r.sha256,
                    }
                    for mignon, r in self.mignon_prompts.items()
                },
            },
            key="metadata:prompt",
        )
        return rendered

    def _board_status(self, plan: SessionPlan) -> BoardStatusContext:
        """ADR-0062: the Wheelta board status through the run's proxied Wheelta server and
        the session's upstream factory, bounded by what is left of the session budget, and
        recorded as run metadata. Never raises: a failure renders as unavailable."""
        wheelta = next((s for s in plan.proxied if s.name == WHEELTA), None)
        registry = next((r for r in plan.registries if r.server == WHEELTA), None)
        reason = skip_reason(
            allowed_tools=frozenset(plan.tool_access.allowed_tools),
            wheelta_proxied=wheelta is not None and registry is not None,
            wheelta_withheld=plan.withheld.get(WHEELTA),
        )
        if reason is None and self.control.stop_requested:
            reason = "the run is stopping"
        if reason is not None or wheelta is None or registry is None:
            context = unavailable(reason or "Wheelta is not proxied this run", self.deps.clock())
        else:
            budget = self._session_budget() - ORDER_WIND_DOWN_SECONDS
            connect = self.deps.connect_budget_seconds or self.settings.MCP_TIMEOUT / 1000
            try:
                tool_timeout = upstream_timeout_seconds(self.settings.MCP_TOOL_TIMEOUT)
            except ValueError:
                tool_timeout = 0.0
            context = read_board_status(
                wheelta,
                registry,
                opener=self.deps.upstream_factory,
                connect_timeout_seconds=min(connect, budget),
                tool_timeout_seconds=min(tool_timeout, budget),
                clock=self.deps.clock,
            )
        self.event(
            RunEventType.METADATA,
            {"board_status": context.event_payload()},
            key="metadata:board_status",
        )
        self.log.bind(stage="board_status").info(
            "board status read", extra={"board_status": context.status.value}
        )
        return context

    def _render_mignons(self, plan: SessionPlan, now: datetime) -> dict[MignonType, RenderedPrompt]:
        """Each Mignon type's prompt with its own tool table (ADR-0025); none when `Agent` is
        not allowed this run (rules.mignons unset)."""
        if DELEGATION_TOOL not in plan.tool_access.allowed_tools:
            return {}
        return {
            mignon: render_prompt(
                self.mignon_templates[mignon],
                {
                    "as_of": now.isoformat(),
                    "policy_version": str(self.rules.version),
                    "policy": self.rules.rendered,
                    "available_tools": available_tools_table(plan, Role(mignon.value)),
                },
            )
            for mignon in MignonType
        }

    def _deadline_check(self) -> None:
        trip_if_deadline_passed(self.control, self.deadline, self.deps.clock)

    def _session_budget(self) -> float:
        remaining = self.deadline.remaining_seconds(self.deps.clock())
        return max(remaining - FINALIZE_RESERVE_SECONDS, 0.0)

    def _agent_run(self, effective_mode: ExecutionMode) -> int:
        credential = self._robinhood_credential()
        plan = self._plan(effective_mode, credential.access_token if credential else None)
        unavailable_reason: str | None = None
        if credential is not None and credential.status is CredentialStatus.PERSIST_FAILED:
            unavailable_reason = "robinhood_credential_unsaved"
            self.tick_alert(
                AlertKind.ROBINHOOD_CREDENTIAL_UNSAVED,
                credential.operator_message or "rotated Robinhood credential not saved",
                {"credential": credential.event_payload()},
            )
        elif ROBINHOOD in plan.withheld:
            obs = next((o for o in plan.observations if o.server == ROBINHOOD), None)
            if obs is not None and obs.status is SourceStatus.NEEDS_AUTH:
                message = (
                    credential.operator_message
                    if credential is not None and credential.operator_message
                    else "Robinhood needs authentication; no trading session was started"
                )
                self.tick_alert(
                    AlertKind.ROBINHOOD_NEEDS_AUTH,
                    message,
                    {"credential": credential.event_payload()} if credential else None,
                )
        book = ledger_positions.position_book(self.conn, self.scope_id, as_of=self.deps.clock())
        session: SessionResult | None = None
        prompt: RenderedPrompt | None = None
        if plan.may_start:
            prompt = self._render(plan, book)
            session = self._session(plan, prompt, book)
        else:
            self.log.warning(
                "required source unavailable; no session",
                extra={"withheld": dict(plan.withheld)},
            )
        return self._finish(plan, book, prompt, session, unavailable_reason)

    def _session(
        self, plan: SessionPlan, prompt: RenderedPrompt, book: PositionBook
    ) -> SessionResult:
        scratch = Path(tempfile.mkdtemp(prefix="wra-agent-", dir=self.deps.scratch_root))
        meta = self.meta(prompt=prompt, model_id=None)

        def reference_check(parsed: DecisionOutputParsed) -> tuple[str, ...]:
            # ADR-0052: the same load and assembly as the run record, before it is final.
            return check_references(
                self.conn, meta, as_of=self.deps.clock(), book=book, decisions=parsed
            )

        deps = SessionDeps(
            conn=self.conn,
            run_id=self.run_id,
            account_scope_id=self.scope_id,
            settings=self.settings,
            rules=self.rules,
            plan=plan,
            system_prompt=prompt.text,
            mignon_prompts={m: r.text for m, r in self.mignon_prompts.items()},
            run_control=self.control,
            clock=self.deps.clock,
            redactor=self.redactor,
            metrics=self.metrics,
            scratch_dir=scratch.resolve(),
            connect_budget_seconds=self.deps.connect_budget_seconds
            or self.settings.MCP_TIMEOUT / 1000,
            session_budget_seconds=self._session_budget,
            deadline_check=self._deadline_check,
            transport_factory=self.deps.transport_factory,
            upstream_factory=self.deps.upstream_factory,
            loopback_server=(
                self.deps.loopback_server if self.deps.transport_factory is None else None
            ),
            mappers=self.deps.mappers,
            account_scope_table=self.deps.account_scope_table,
            interrupt_grace_seconds=self.deps.interrupt_grace_seconds or INTERRUPT_GRACE_SECONDS,
            status_poll_interval=self.deps.status_poll_interval or STATUS_POLL_INTERVAL_SECONDS,
            reference_check=reference_check,
            role=self.role,
            related_run_ids=self.related_run_ids,
            order_scope_run_id=self.order_scope_run_id,
            simulated_state=self.simulated_state,
        )
        self.metrics.stage_started("agent", self.deps.clock())
        try:
            return run_session_sync(deps)
        finally:
            self.metrics.stage_finished("agent", self.deps.clock())
            shutil.rmtree(scratch, ignore_errors=True)

    def _finish(
        self,
        plan: SessionPlan,
        book: PositionBook,
        prompt: RenderedPrompt | None,
        session: SessionResult | None,
        unavailable_reason: str | None = None,
    ) -> int:
        self.session_started = session is not None and session.status not in (
            SessionStatus.NOT_STARTED,
            SessionStatus.SKIPPED,
        )
        if session is not None and session.start_condition is not None:
            self.start_condition = session.start_condition
            self.event(
                RunEventType.METADATA,
                {"start_condition": session.start_condition.model_dump(mode="json")},
                key="metadata:start_condition",
            )
            self.log.bind(stage="start_condition").info(
                "start condition", extra=session.start_condition.model_dump(mode="json")
            )
            if session.status is SessionStatus.SKIPPED:
                # ADR-0057: nothing for this agent to do; no session, assembly, or audit.
                self.observe_sources(session.observations[len(plan.observations) :])
                if session.eligibility is not None:
                    self.event(
                        RunEventType.METADATA,
                        {"agentic_eligibility": session.eligibility.model_dump(mode="json")},
                    )
                return self.finalize(
                    self._skip_status(), self._skip_reason(session.start_condition)
                )
            if session.start_condition.outcome is StartOutcome.UNAVAILABLE:
                unavailable_reason = unavailable_reason or "start_condition_unavailable"
                self.summary_diagnostics.append(
                    f"Start condition unavailable: {session.start_condition.reason}"
                )
                self.alert(
                    AlertKind.START_CONDITION_UNAVAILABLE,
                    f"the {self.role.value} agent's start condition could not be read; "
                    "no session was started",
                    {"reason": session.start_condition.reason},
                )
        if session is not None:
            self.summary_diagnostics.extend(session.error_details)
            self.summary_diagnostics.extend(
                f"Source unavailable: {source}: {reason}"
                for source, reason in session.withheld.items()
            )
            self.observe_sources(session.observations[len(plan.observations) :])
            if session.eligibility is not None:
                # Only the configured account's redacted eligibility is persisted (§9, §24).
                self.event(
                    RunEventType.METADATA,
                    {"agentic_eligibility": session.eligibility.model_dump(mode="json")},
                )
                if not session.eligibility.eligible:
                    unavailable_reason = unavailable_reason or "robinhood_account_not_agentic"
                    self.log.error(
                        "configured Robinhood account is not Agentic-eligible; no session",
                        extra={"reasons": list(session.eligibility.reasons)},
                    )
            if session.tool_drift:
                self.alert(
                    AlertKind.TOOL_DRIFT,
                    "expected tools missing on a connected MCP server",
                    {"servers": list(session.tool_drift)},
                )
            if session.order_cleanups or session.orders_left_unresolved:
                self.event(
                    RunEventType.METADATA,
                    {
                        "order_cleanup": {
                            "cleanup_turns": session.order_cleanups,
                            "left_unresolved": list(session.orders_left_unresolved),
                        }
                    },
                )
            if session.reference_repairs or session.reference_check_error:
                self.event(
                    RunEventType.METADATA,
                    {
                        "reference_check": {
                            "reference_turns": session.reference_repairs,
                            "issues_left": list(session.reference_issues or ()),
                            "error": session.reference_check_error,
                        }
                    },
                )
            if session.orders_left_unresolved:
                # ADR-0050: the next run's step 1 cancels them; an operator may cancel sooner.
                self.alert(
                    AlertKind.ORDERS_LEFT_WORKING,
                    "owned orders were still unresolved when the session ended",
                    {"orders": list(session.orders_left_unresolved)},
                )
            if session.observations and any(
                o.server == ROBINHOOD and o.status is SourceStatus.NEEDS_AUTH
                for o in session.observations[len(plan.observations) :]
            ):
                self.tick_alert(AlertKind.ROBINHOOD_NEEDS_AUTH, "Robinhood reported needs-auth")
        stop = self.control.stop_record
        if stop is not None:
            self.event(
                RunEventType.CONTROL,
                {"stop_reason": stop.reason.value, "requested_at": stop.requested_at.isoformat()},
                key="control:stop",
            )
        meta = self.meta(prompt=prompt, model_id=(session.model_id if session else None))
        decisions, output_id = load_decisions(self.conn, self.run_id)
        if isinstance(decisions, DecisionOutputParseFailure):
            self.summary_diagnostics.extend(
                f"Invalid agent output at {i.loc or '(root)'}: {i.message} ({i.kind})"
                for i in decisions.issues
            )
        elif decisions is None:
            self.summary_diagnostics.append("No final decision output was recorded.")
        record = self._assemble(meta, book, decisions, output_id)
        self.summary_record = record
        audit_ok = self._audit(meta, book, decisions, record)
        self._log_decisions()
        # After assembly and audit, so nothing in the agent's schedule can keep them from running.
        self._apply_agent_next_run(decisions)
        status, reason = self._status(plan, session, decisions)
        if unavailable_reason is not None and reason == "required_source_unavailable":
            reason = unavailable_reason
        if not audit_ok and status is RunStatus.COMPLETED:
            status, reason = RunStatus.FAILED, "audit_failed"
        return self.finalize(status, reason)

    def _skip_status(self) -> RunStatus:
        return (
            RunStatus.SKIPPED_NO_OPEN_SHORTS
            if self.role is AgentRole.CLOSE
            else RunStatus.SKIPPED_INSUFFICIENT_BALANCE
        )

    @staticmethod
    def _skip_reason(condition: StartCondition) -> str:
        return condition.reason

    def _log_decisions(self) -> None:
        """One structured log line per decision from the recorded trace (observability only:
        never changes the run's status). The full trace: `scripts/trace_run.py`."""
        log = self.log.bind(stage="decisions")
        try:
            trace = load_decision_trace(self.conn, self.run_id)
            for payload in decision_log_events(trace):
                log.info("decision", extra=payload)
            log.info(
                "decision trace recorded",
                extra={
                    "decisions": len(trace.decisions),
                    "order_venue": trace.order_venue.value if trace.order_venue else None,
                    "audit": trace.audit_counts,
                    "tool_calls": trace.tool_call_counts,
                },
            )
        except Exception as exc:  # noqa: BLE001 - informational: never fail the run
            log.warning("decision trace unavailable", extra={"error_type": type(exc).__name__})

    def _status(
        self, plan: SessionPlan, session: SessionResult | None, decisions: DecisionsInput
    ) -> tuple[RunStatus, str]:
        stop = self.control.stop_record
        if stop is not None:
            if stop.reason is StopReason.DEADLINE:
                self.alert(AlertKind.RUN_TIMEOUT, "run budget exhausted; the session was stopped")
                return RunStatus.TIMED_OUT, "deadline"
            if stop.reason in (StopReason.SIGTERM, StopReason.SIGINT):
                return RunStatus.STOPPED, stop.reason.value
            return RunStatus.FAILED, stop.reason.value
        if session is None or session.status is SessionStatus.NOT_STARTED:
            return RunStatus.FAILED, "required_source_unavailable"
        if session.status is SessionStatus.FAILED:
            return RunStatus.FAILED, session.error or "session_failed"
        if not isinstance(decisions, DecisionOutputParsed):
            self.alert(
                AlertKind.INVALID_AGENT_OUTPUT,
                "agent output missing or invalid; the run was assembled from events",
            )
            return RunStatus.FAILED, "invalid_agent_output"
        for decision in decisions.output.decisions:
            self.metrics.decision(decision.action)
        return RunStatus.COMPLETED, "completed"

    def _assemble(
        self,
        meta: RunMeta,
        book: PositionBook | None,
        decisions: DecisionsInput,
        output_id: uuid.UUID | None,
    ) -> RunRecord | None:
        try:
            terminated = self.deps.clock()
            context = load_assembly_context(
                self.conn, meta, terminated_at=terminated, book=book, output_id=output_id
            )
            record = assemble_run_record(context, decisions)
            ledger_evidence.insert_run_record(self.conn, record=record, assembled_at=terminated)
        except Exception as exc:  # noqa: BLE001 - assembly failure is recorded; audit still runs
            self.log.exception("assembly failed", extra={"error_type": type(exc).__name__})
            self.metrics.error(f"assembly:{type(exc).__name__}")
            self.summary_diagnostics.append(
                self.redactor.redact_text(f"Record assembly: {type(exc).__name__}: {exc}")
            )
            return None
        for decision_record in record.decisions:
            for leg in decision_record.legs:
                for attempt in leg.attempts:
                    self.metrics.order(attempt.status)
        for calls in tool_call_records(self.conn, self.run_id):
            self.metrics.tool_call(calls.identity.server)
        # ADR-0039: a dry run's judgments are not position memory for production's positions.
        if book is not None and self.settings.effective_execution_mode is ExecutionMode.LIVE:
            self._record_notes(record, book)
        return record

    def _record_notes(self, record: RunRecord, book: PositionBook) -> None:
        """Carry this run's judgments on active lineages into later runs (ADR-0018).

        An OPEN decision's note goes to the lineages its filled orders created (ADR-0055).
        Notes are context only, so a failure is logged and counted but does not fail the run.
        """
        try:
            lineages = ledger_positions.entry_lineages(
                self.conn, self.scope_id, opened_order_ids(record)
            )
            for item in notes_from_run_record(record, book, lineages):
                ledger_positions.record_note(
                    self.conn, item.position_id, dedup_key=item.dedup_key, note=item.note
                )
        except Exception as exc:  # noqa: BLE001 - notes are context; the record is stored
            self.log.exception("position notes failed", extra={"error_type": type(exc).__name__})
            self.metrics.error(f"position_notes:{type(exc).__name__}")

    def _audit(
        self,
        meta: RunMeta,
        book: PositionBook | None,
        decisions: DecisionsInput,
        record: RunRecord | None,
    ) -> bool:
        output = decisions.output if isinstance(decisions, DecisionOutputParsed) else None
        try:
            context = load_audit_context(
                self.conn, meta, book=book, decision_output=output, run_record=record
            )
            result: AuditResult | None = run_audit(context)
        except Exception as exc:  # noqa: BLE001 - an audit that cannot run is a failed audit
            self.log.exception("audit failed", extra={"error_type": type(exc).__name__})
            self.summary_diagnostics.append(
                self.redactor.redact_text(f"Audit: {type(exc).__name__}: {exc}")
            )
            result = None
        self.audit_ran = True
        self.audit_result = result
        if result is not None:
            for finding in result.findings:
                ledger_evidence.insert_audit_finding(self.conn, finding)
        status = result.status if result is not None else AuditStatus.FAILED
        self.event(
            RunEventType.AUDIT_STATUS,
            {
                "status": status.value,
                "audit_version": result.audit_version if result else None,
                "context_hash": result.context_hash if result else None,
                "violations": len(result.violations) if result else None,
                "errors": [e.model_dump(mode="json") for e in result.errors] if result else [],
            },
            key="audit_status",
        )
        if result is None or result.status is AuditStatus.FAILED:
            self.alert(AlertKind.AUDIT_FAILURE, "the post-run audit could not complete")
            return False
        if any(f.outcome is AuditOutcome.VIOLATION for f in result.findings):
            self.alert(
                AlertKind.AUDIT_VIOLATION,
                "the post-run audit found violations",
                {"checks": sorted({f.check_id.value for f in result.violations})},
            )
        return True

    def recover(self) -> int:
        """Finalize an interrupted slot from its recorded events; never start a session.

        Calls with no recorded outcome get one: reads `failed`, workspace/order actions and
        untiered calls `unknown` (never inferred to have failed or succeeded; CLAUDE.md §14).
        """
        self.event(RunEventType.RECOVERY_STARTED, {"observed_by": "orchestrator"})
        now = self.deps.clock()
        for call in tool_call_records(self.conn, self.run_id):
            if call.status is not ToolCallStatus.REQUESTED:
                continue
            tier = call.identity.tier
            # Reads and Mignon spawns are not actions with an outside effect: failed.
            status = (
                ToolCallStatus.FAILED
                if tier in (ToolTier.R, ToolTier.D)
                else ToolCallStatus.UNKNOWN
            )
            append_tool_call_outcome(
                self.conn,
                call.identity.tool_call_id,
                status,
                observed_at=now,
                dedup_key="recovery",
                reason="the run was interrupted before an outcome was recorded",
            )
        book_payloads = [
            p.get("position_book")
            for p in run_event_payloads(self.conn, self.run_id, RunEventType.METADATA)
        ]
        stored_books = [b for b in book_payloads if isinstance(b, dict)]
        book = PositionBook.model_validate(stored_books[-1]) if stored_books else None
        venues = [
            access.get("order_venue")
            for p in run_event_payloads(self.conn, self.run_id, RunEventType.METADATA)
            if isinstance(access := p.get("tool_access"), dict)
        ]
        if venues and isinstance(venues[-1], str):
            self.order_venue = OrderVenue(venues[-1])
        meta = self.meta(prompt=None, model_id=None)
        decisions, output_id = load_decisions(self.conn, self.run_id)
        record = self._assemble(meta, book, decisions, output_id)
        self._audit(meta, book, decisions, record)
        self._log_decisions()
        return self.finalize(RunStatus.FAILED, "interrupted_run_recovered")


RUN_NOW_FLAG = "--run-now"


def main(argv: Sequence[str] | None = None) -> int:
    """Production entrypoint: load and validate everything before any network call.

    `argv` excludes the program name; None means no arguments. The only argument is
    `--run-now`, which starts an on-demand dry run and is refused when the effective mode is
    live (ADR-0038). Anything else fails fast.
    """
    args = list(argv or ())
    unknown = [a for a in args if a != RUN_NOW_FLAG]
    if unknown:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
        _LOG.error("unknown arguments: %s (only %s is accepted)", " ".join(unknown), RUN_NOW_FLAG)
        return EXIT_FAILED
    run_now = RUN_NOW_FLAG in args
    try:
        settings = load_settings()
        rules = load_rules()
        templates = load_agent_prompts()
        mignon_templates = load_mignon_prompts()
    except (SettingsError, RulesError, PromptError) as exc:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
        _LOG.error("startup configuration invalid: %s", exc)
        return EXIT_FAILED
    if run_now and settings.APP_ENV is not AppEnv.LOCAL:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
        _LOG.error("%s starts a local dry run; it requires APP_ENV=local", RUN_NOW_FLAG)
        return EXIT_FAILED
    configure_logging(
        settings.LOG_LEVEL,
        settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
        secrets=settings_secrets(settings),
    )
    notifier = HttpNotifier(settings.ALERT_WEBHOOK_URL, settings.HEARTBEAT_URL)
    mailer = summary_mailer_for(settings)
    try:
        return run_once(
            settings,
            rules,
            templates,
            OrchestratorDeps(notifier=notifier, summary_mailer=mailer),
            mignon_templates,
            run_now=run_now,
        )
    except psycopg.errors.UndefinedTable:
        # The ledger schema is missing: migrations haven't run on this database.
        _LOG.error(
            "ledger schema missing; run `python -m wheelta_robinhood_agent.ledger.migrate` "
            "(the container entrypoint and Railway preDeployCommand both do)"
        )
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - last resort: fail closed, never crash-dump
        # One redacted line instead of a raw traceback (CLAUDE.md §7, §14). The run is
        # recorded as far as it got; the exit code tells Railway and alerting it failed.
        _LOG.error("run failed with an unhandled %s", type(exc).__name__, exc_info=True)
        return EXIT_FAILED
    finally:
        notifier.close()
        if mailer is not None:
            mailer.close()
