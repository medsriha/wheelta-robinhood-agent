"""Run entrypoint: the full lifecycle of one cron fire (ARCHITECTURE.md "Run lifecycle").

boot (settings, rules, prompt; fail fast) → logging → slot/run_id → ledger connection →
single-flight lock (`skipped_concurrent`) → run slot (completed → no-op; interrupted →
reconcile and finalize without a new session) → preflight (kill switch, NYSE session, next-run
time: ADR-0028, `skipped_not_due`; a due tick records the fallback next run first) →
Robinhood credential (refresh_token mode only: load, refresh near expiry, persist before use;
ADR-0021) → session plan (effective mode capped at off; no order tool can be exposed) → dry
runs are local only (ADR-0024: outside APP_ENV=local an off-mode run ends
`skipped_dry_run_not_local` here, after the credential and its alerts) → prompt v6 →
agent session → the agent's `next_run`, if valid, replaces the fallback → `assemble_run_record` →
position notes (ADR-0018) → `run_audit` → persist →
alerts/heartbeat → run-summary email (ADR-0029: only when a session started) → exit code.

Contains no trading logic. Everything the run decides is recorded as run events.
`python -m wheelta_robinhood_agent.orchestrator [--run-now]` calls `main()`. `--run-now`
(APP_ENV=local only) makes a hand-started local run due whatever the recorded next run
(ADR-0028).

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
from wheelta_robinhood_agent.agent.mignons import DELEGATION_TOOL, Role, mignon_limits
from wheelta_robinhood_agent.agent.result_boundary import VERIFIED_MAPPERS, EvidenceMapper
from wheelta_robinhood_agent.agent.run_control import RunControl, StopReason
from wheelta_robinhood_agent.agent.run_loader import (
    RunMeta,
    load_assembly_context,
    load_audit_context,
    load_decisions,
)
from wheelta_robinhood_agent.agent.session import (
    INTERRUPT_GRACE_SECONDS,
    REMOTE_RESULT_BOUNDARY_ACCEPTED,
    STATUS_POLL_INTERVAL_SECONDS,
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
from wheelta_robinhood_agent.config.prompts import (
    PromptError,
    PromptTemplate,
    RenderedPrompt,
    load_mignon_prompts,
    load_prompt,
    render_prompt,
)
from wheelta_robinhood_agent.config.rules import LoadedRules, RulesError, load_rules
from wheelta_robinhood_agent.config.settings import (
    PHASE_EXECUTION_CEILING,
    RobinhoodMcpAuth,
    Settings,
    SettingsError,
    load_settings,
)
from wheelta_robinhood_agent.domain.assembly import DecisionsInput, assemble_run_record
from wheelta_robinhood_agent.domain.decision_output import DecisionOutputParsed
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AuditOutcome,
    ExecutionMode,
    MignonType,
    RunStatus,
    SourceStatus,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.position_notes import notes_from_run_record
from wheelta_robinhood_agent.domain.positions import PositionBook
from wheelta_robinhood_agent.domain.run import AuditStatus
from wheelta_robinhood_agent.domain.run_identity import run_id_for, slot_for
from wheelta_robinhood_agent.domain.run_record import RunRecord
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
)
from wheelta_robinhood_agent.observability.logging import (
    RunLoggerAdapter,
    bind,
    configure_logging,
    settings_secrets,
)
from wheelta_robinhood_agent.observability.metrics import RunMetrics
from wheelta_robinhood_agent.observability.redaction import Redactor
from wheelta_robinhood_agent.observability.run_summary import RunSummaryInput
from wheelta_robinhood_agent.orchestrator.exit_codes import EXIT_FAILED, EXIT_OK, exit_code_for
from wheelta_robinhood_agent.orchestrator.market_session import (
    TradingCalendar,
    build_nyse_calendar,
    evaluate_market_session,
)
from wheelta_robinhood_agent.orchestrator.preflight import (
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

    def send(self, summary: RunSummaryInput, redactor: Redactor) -> EmailDeliveryResult: ...


@dataclass
class HttpSummaryMailer:
    """`SummaryMailer` over integrations/notifications (Resend + Anthropic Messages)."""

    config: RunSummaryEmailConfig
    _client: httpx.Client | None = None

    def send(self, summary: RunSummaryInput, redactor: Redactor) -> EmailDeliveryResult:
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
    registries: tuple[ToolRegistry, ToolRegistry] = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY)
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
    template: PromptTemplate,
    deps: OrchestratorDeps,
    mignon_templates: Mapping[MignonType, PromptTemplate] | None = None,
    *,
    run_now: bool = False,
) -> int:
    """Run one cron fire and return the process exit code (exit_codes.py).

    `mignon_templates` defaults to the packaged Mignon prompts (`main` loads them at startup
    so a missing file fails before any network call). `run_now` (local only, ADR-0028) makes
    the tick due whatever the recorded next run; the kill switch and market session still
    apply. Raises ValueError outside APP_ENV=local."""
    if run_now and settings.APP_ENV is not AppEnv.LOCAL:
        raise ValueError("run_now is allowed only with APP_ENV=local")
    if mignon_templates is None:
        mignon_templates = load_mignon_prompts()
    started = deps.clock()
    slot = slot_for(started)
    run_id = run_id_for(settings.APP_ENV, slot)
    log = bind(_LOG, run_id=str(run_id), stage="boot", slot=slot.isoformat())
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
        run_slot = open_run_slot(conn, settings.APP_ENV, slot)
        if run_slot.state is SlotState.COMPLETED:
            log.info("slot already finalized; nothing to do", extra={"status": "noop"})
            return EXIT_OK
        run = _Run(settings, rules, template, deps, conn, run_id, slot, started, log)
        run.mignon_templates = dict(mignon_templates)
        run.run_now = run_now
        if run_slot.state is SlotState.INTERRUPTED:
            return run.recover()
        return run.execute()


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
    ) -> None:
        self.settings = settings
        self.rules = rules
        self.template = template
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
        self.deadline = RunDeadline(started, settings.RUN_TIMEOUT_SECONDS)
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
        self.audit_result: AuditResult | None = None
        self.audit_ran = False
        self.next_run: NextRun | None = None
        self.next_run_rationale: str | None = None

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

    def observe_sources(self, observations: Sequence[SourceObservation]) -> None:
        for obs in observations:
            self.event(RunEventType.SOURCE_STATUS, obs.model_dump(mode="json"))

    def finalize(self, status: RunStatus, reason: str | None) -> int:
        self.event(RunEventType.STATUS, {"reason": reason} if reason else None, status=status)
        _heartbeat(self.settings, self.deps, status, self.run_id, self.slot, reason)
        if self.session_started:
            self.send_summary(status, reason)
        snapshot = self.metrics.snapshot().model_dump(mode="json")
        self.log.bind(stage="finalize").info(
            "run finished", extra={"status": status.value, "reason": reason, "metrics": snapshot}
        )
        return exit_code_for(status)

    def send_summary(self, status: RunStatus, reason: str | None) -> None:
        """ADR-0029: email the run summary. Informational: never changes status or exit code."""
        mailer = self.deps.summary_mailer
        if mailer is None:
            return
        try:
            audit = self.audit_result
            summary = RunSummaryInput(
                run_id=str(self.run_id),
                environment=self.settings.APP_ENV,
                slot=self.slot,
                status=status,
                reason=reason,
                requested_execution_mode=self.settings.requested_execution_mode,
                effective_execution_mode=self.settings.effective_execution_mode,
                record=self.summary_record,
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
            result = mailer.send(summary, self.redactor)
        except Exception as exc:  # noqa: BLE001 - informational email: never fail the run
            self.log.warning("run summary email failed", extra={"error_type": type(exc).__name__})
            return
        log = self.log.bind(stage="summary_email")
        log.info(
            "run summary email",
            extra={"delivery_status": result.status.value, "error": result.error},
        )
        if result.status is EmailDeliveryStatus.SKIPPED:
            return
        with contextlib.suppress(Exception):
            ledger_evidence.record_alert_sent(
                self.conn,
                run_id=self.run_id,
                alert_kind=RUN_SUMMARY_EMAIL_KIND,
                dedup_key=f"run-summary/{self.run_id}",
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

    def meta(self, *, prompt: RenderedPrompt | None, model_id: str | None) -> RunMeta:
        return RunMeta(
            run_id=self.run_id,
            environment=self.settings.APP_ENV,
            slot=self.slot,
            requested_mode=self.settings.requested_execution_mode,
            effective_mode=self.settings.effective_execution_mode,
            account_scope_id=self.scope_id,
            rules=self.rules,
            prompt_id=self.template.prompt_id if prompt else None,
            prompt_hash=self.template.sha256 if prompt else None,
            model_id=model_id,
        )

    # -- lifecycle ------------------------------------------------------------------------

    def execute(self) -> int:
        settings = self.settings
        self.event(
            RunEventType.STARTED,
            {
                "config_snapshot": settings.config_snapshot(),
                "rules_version": self.rules.version,
                "rules_hash": self.rules.sha256,
                "prompt_id": self.template.prompt_id,
                "prompt_version": self.template.version,
                "prompt_template_hash": self.template.sha256,
                "execution_ceiling": PHASE_EXECUTION_CEILING.value,
            },
            key="started",
        )
        self.event(RunEventType.STATUS, status=RunStatus.RUNNING, key="status:running")
        try:
            return self._execute()
        except Exception as exc:  # noqa: BLE001 - fail closed: record, alert via heartbeat
            self.log.exception("run failed", extra={"error_type": type(exc).__name__})
            self.metrics.error(type(exc).__name__)
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
        not_before = latest_next_run_not_before(self.conn, settings.APP_ENV)
        due = self.run_now or is_due(now, not_before)
        self.event(
            RunEventType.METADATA,
            {
                "schedule_check": {
                    "not_before": not_before.isoformat() if not_before else None,
                    "due": due,
                    "run_now": self.run_now,
                }
            },
            key="metadata:schedule_check",
        )
        decision = decide_preflight(
            kill_switch=settings.KILL_SWITCH,
            market=market,
            due=due,
            requested_mode=settings.requested_execution_mode,
            armed=settings.EXECUTION_ARMED,
            ceiling=PHASE_EXECUTION_CEILING,
        )
        if due and not (
            isinstance(decision, PreflightSkip)
            and decision.reason is PreflightReason.OUTSIDE_REGULAR_SESSION
        ):
            # ADR-0028: a due tick is this cadence's run, even when killed. The fallback keeps
            # a crashed or output-less run (and the kill alert) on the hourly cadence.
            self.gate_at = now
            minutes = self.rules.rules.scheduling.fallback_next_run_minutes
            self.record_next_run(fallback_requested_at(now, minutes), ScheduleSource.FALLBACK)
        if isinstance(decision, PreflightSkip):
            if decision.status is RunStatus.SKIPPED_KILLED and due:
                self.alert(AlertKind.KILL_SWITCH_ENGAGED, "KILL_SWITCH=true; the run did not start")
            return self.finalize(decision.status, decision.reason.value)
        if decision.effective_mode is not ExecutionMode.OFF:
            raise SessionPlanError("effective mode above the phase-1 ceiling")

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
        every token seen is added to the run's redactor."""
        settings = self.settings
        if settings.ROBINHOOD_MCP_AUTH is not RobinhoodMcpAuth.REFRESH_TOKEN:
            return None
        key = settings.ROBINHOOD_TOKEN_ENCRYPTION_KEY
        if key is None:  # Settings rejects this; kept so the type is narrowed without assert
            raise SessionPlanError("refresh_token mode without an encryption key")
        resolution = resolve_robinhood_credential(
            self.conn,
            environment=settings.APP_ENV,
            vault=TokenVault(key),
            clock=self.deps.clock,
            refresher=self.deps.oauth_refresher,
            insert=self.deps.insert_credential,
        )
        self.redactor = Redactor(
            account_number=settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            secrets=(*settings_secrets(settings), *resolution.secrets),
        )
        self.event(
            RunEventType.METADATA,
            {"robinhood_credential": resolution.event_payload()},
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
            workspace_writes=self.settings.ROBINHOOD_WORKSPACE_WRITES,
            sources=(
                RemoteSource(
                    rh_registry,
                    build_robinhood_server(self.settings, now, robinhood_token),
                    required=True,
                ),
                RemoteSource(wt_registry, build_wheelta_server(self.settings)),
            ),
            observed_at=now,
            remote_boundary_accepted=(
                self.deps.remote_boundary_accepted or self.settings.remote_result_risk_accepted
            ),
            mignons=mignon_limits(self.rules.rules) is not None,
        )
        self.event(
            RunEventType.METADATA,
            {
                "tool_access": {
                    "effective_mode": plan.tool_access.effective_mode.value,
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
        values = {
            "as_of": now.isoformat(),
            "execution_mode": plan.effective_mode.value,
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
        rendered = render_prompt(self.template, values)
        self.mignon_prompts = self._render_mignons(plan, now)
        self.event(
            RunEventType.METADATA,
            {
                "prompt_id": rendered.prompt_id,
                "prompt_version": rendered.version,
                "prompt_template_hash": rendered.template_sha256,
                "rendered_prompt_hash": rendered.sha256,
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

    def dry_run_outside_local(self, effective_mode: ExecutionMode) -> bool:
        """ADR-0024: a dry run (effective mode off) never starts a session outside local."""
        return effective_mode is ExecutionMode.OFF and self.settings.APP_ENV is not AppEnv.LOCAL

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
            self.alert(
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
                self.alert(
                    AlertKind.ROBINHOOD_NEEDS_AUTH,
                    message,
                    {"credential": credential.event_payload()} if credential else None,
                )
        if plan.may_start and self.dry_run_outside_local(effective_mode):
            self.log.info(
                "dry runs run locally only; no session outside APP_ENV=local",
                extra={"app_env": self.settings.APP_ENV.value},
            )
            return self.finalize(RunStatus.SKIPPED_DRY_RUN_NOT_LOCAL, "dry_run_local_only")
        book = ledger_positions.position_book(self.conn, self.scope_id, as_of=self.deps.clock())
        session: SessionResult | None = None
        prompt: RenderedPrompt | None = None
        if plan.may_start:
            prompt = self._render(plan, book)
            session = self._session(plan, prompt)
        else:
            self.log.warning(
                "required source unavailable; no session",
                extra={"withheld": dict(plan.withheld)},
            )
        return self._finish(plan, book, prompt, session, unavailable_reason)

    def _session(self, plan: SessionPlan, prompt: RenderedPrompt) -> SessionResult:
        scratch = Path(tempfile.mkdtemp(prefix="wra-agent-", dir=self.deps.scratch_root))
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
            mappers=self.deps.mappers,
            account_scope_table=self.deps.account_scope_table,
            interrupt_grace_seconds=self.deps.interrupt_grace_seconds or INTERRUPT_GRACE_SECONDS,
            status_poll_interval=self.deps.status_poll_interval or STATUS_POLL_INTERVAL_SECONDS,
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
        self.session_started = session is not None and session.status is not (
            SessionStatus.NOT_STARTED
        )
        if session is not None:
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
            if session.observations and any(
                o.server == ROBINHOOD and o.status is SourceStatus.NEEDS_AUTH
                for o in session.observations[len(plan.observations) :]
            ):
                self.alert(AlertKind.ROBINHOOD_NEEDS_AUTH, "Robinhood reported needs-auth")
        stop = self.control.stop_record
        if stop is not None:
            self.event(
                RunEventType.CONTROL,
                {"stop_reason": stop.reason.value, "requested_at": stop.requested_at.isoformat()},
                key="control:stop",
            )
        meta = self.meta(prompt=prompt, model_id=(session.model_id if session else None))
        decisions, output_id = load_decisions(self.conn, self.run_id)
        record = self._assemble(meta, book, decisions, output_id)
        self.summary_record = record
        audit_ok = self._audit(meta, book, decisions, record)
        # After assembly and audit, so nothing in the agent's schedule can keep them from running.
        self._apply_agent_next_run(decisions)
        status, reason = self._status(plan, session, decisions)
        if unavailable_reason is not None and reason == "required_source_unavailable":
            reason = unavailable_reason
        if not audit_ok and status is RunStatus.COMPLETED:
            status, reason = RunStatus.FAILED, "audit_failed"
        return self.finalize(status, reason)

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
            return None
        for decision_record in record.decisions:
            for leg in decision_record.legs:
                for attempt in leg.attempts:
                    self.metrics.order(attempt.status)
        for calls in tool_call_records(self.conn, self.run_id):
            self.metrics.tool_call(calls.identity.server)
        if book is not None:
            self._record_notes(record, book)
        return record

    def _record_notes(self, record: RunRecord, book: PositionBook) -> None:
        """Carry this run's judgments on active lineages into later runs (ADR-0018).

        Notes are context only, so a failure is logged and counted but does not fail the run.
        """
        try:
            for item in notes_from_run_record(record, book):
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
        meta = self.meta(prompt=None, model_id=None)
        decisions, output_id = load_decisions(self.conn, self.run_id)
        record = self._assemble(meta, book, decisions, output_id)
        self._audit(meta, book, decisions, record)
        return self.finalize(RunStatus.FAILED, "interrupted_run_recovered")


RUN_NOW_FLAG = "--run-now"


def main(argv: Sequence[str] | None = None) -> int:
    """Production entrypoint: load and validate everything before any network call.

    `argv` excludes the program name; None means no arguments. The only argument is
    `--run-now`, accepted only with APP_ENV=local (ADR-0028). Anything else fails fast.
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
        template = load_prompt()
        mignon_templates = load_mignon_prompts()
    except (SettingsError, RulesError, PromptError) as exc:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
        _LOG.error("startup configuration invalid: %s", exc)
        return EXIT_FAILED
    if run_now and settings.APP_ENV is not AppEnv.LOCAL:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
        _LOG.error("%s is allowed only with APP_ENV=local", RUN_NOW_FLAG)
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
            template,
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
