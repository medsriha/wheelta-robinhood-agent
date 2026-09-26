"""Run entrypoint: the full lifecycle of one cron fire (ARCHITECTURE.md "Run lifecycle").

boot (settings, rules, prompt; fail fast) → logging → slot/run_id → ledger connection →
single-flight lock (`skipped_concurrent`) → run slot (completed → no-op; interrupted →
reconcile and finalize without a new session) → preflight (kill switch, NYSE session) →
session plan (effective mode capped at off; no order tool can be exposed) → prompt v5 →
agent session → `assemble_run_record` → `run_audit` → persist → alerts/heartbeat → exit code.

Contains no trading logic. Everything the run decides is recorded as run events.
`python -m wheelta_robinhood_agent.orchestrator` calls `main()`.

`OrchestratorDeps` carries the injectable boundaries (clock, database connect, calendar,
notifier, SDK transport). Its remaining fields (`remote_boundary_accepted`, `registries`,
`mappers`, `account_scope_table`) are test seams for fake servers; `main()` always uses the
production values, which withhold every unverified remote tool.
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
    available_tools_table,
    plan_session,
    run_session_sync,
)
from wheelta_robinhood_agent.config.prompts import (
    PromptError,
    PromptTemplate,
    RenderedPrompt,
    load_prompt,
    render_prompt,
)
from wheelta_robinhood_agent.config.rules import LoadedRules, RulesError, load_rules
from wheelta_robinhood_agent.config.settings import (
    PHASE_EXECUTION_CEILING,
    Settings,
    SettingsError,
    load_settings,
)
from wheelta_robinhood_agent.domain.assembly import DecisionsInput, assemble_run_record
from wheelta_robinhood_agent.domain.decision_output import DecisionOutputParsed
from wheelta_robinhood_agent.domain.enums import (
    AuditOutcome,
    ExecutionMode,
    RunStatus,
    SourceStatus,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.events import RunEventType
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
from wheelta_robinhood_agent.integrations.registry import ToolRegistry
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME as ROBINHOOD
from wheelta_robinhood_agent.integrations.robinhood.server import build_robinhood_server
from wheelta_robinhood_agent.integrations.status import SourceObservation
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY
from wheelta_robinhood_agent.integrations.wheelta.server import build_wheelta_server
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger import orders as ledger_orders
from wheelta_robinhood_agent.ledger import positions as ledger_positions
from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.errors import LedgerError
from wheelta_robinhood_agent.ledger.lock import single_flight
from wheelta_robinhood_agent.ledger.runs import (
    SlotState,
    append_run_event,
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
from wheelta_robinhood_agent.orchestrator.exit_codes import EXIT_FAILED, EXIT_OK, exit_code_for
from wheelta_robinhood_agent.orchestrator.market_session import (
    TradingCalendar,
    build_nyse_calendar,
    evaluate_market_session,
)
from wheelta_robinhood_agent.orchestrator.preflight import PreflightSkip, decide_preflight
from wheelta_robinhood_agent.orchestrator.signals import (
    RunDeadline,
    install_stop_signal_handlers,
    trip_if_deadline_passed,
)

Conn = psycopg.Connection[tuple[object, ...]]
_LOG = logging.getLogger("wheelta_robinhood_agent.run")
NOTIFY_TIMEOUT_SECONDS = 10.0
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _calendar(start: date, end: date) -> TradingCalendar:
    return build_nyse_calendar(start, end)


@dataclass(frozen=True)
class OrchestratorDeps:
    notifier: Notifier
    clock: Callable[[], datetime] = _utc_now
    connect_db: Callable[[SecretStr], Conn] = connect
    calendar_factory: Callable[[date, date], TradingCalendar] = _calendar
    transport_factory: TransportFactory | None = None
    install_signals: bool = True
    scratch_root: Path | None = None
    # Test seams (module docstring): production always uses the defaults.
    remote_boundary_accepted: bool = REMOTE_RESULT_BOUNDARY_ACCEPTED
    registries: tuple[ToolRegistry, ToolRegistry] = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY)
    mappers: Mapping[tuple[str, str], EvidenceMapper] = field(
        default_factory=lambda: VERIFIED_MAPPERS
    )
    account_scope_table: Mapping[str, AccountScopeSpec] = field(
        default_factory=lambda: ROBINHOOD_ACCOUNT_SCOPE
    )
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
) -> int:
    """Run one cron fire and return the process exit code (exit_codes.py)."""
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
        snapshot = self.metrics.snapshot().model_dump(mode="json")
        self.log.bind(stage="finalize").info(
            "run finished", extra={"status": status.value, "reason": reason, "metrics": snapshot}
        )
        return exit_code_for(status)

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
        decision = decide_preflight(
            kill_switch=settings.KILL_SWITCH,
            market=market,
            requested_mode=settings.requested_execution_mode,
            armed=settings.EXECUTION_ARMED,
            ceiling=PHASE_EXECUTION_CEILING,
        )
        if isinstance(decision, PreflightSkip):
            if decision.status is RunStatus.SKIPPED_KILLED:
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

    def _plan(self, effective_mode: ExecutionMode) -> SessionPlan:
        rh_registry, wt_registry = self.deps.registries
        now = self.deps.clock()
        plan = plan_session(
            effective_mode=effective_mode,
            workspace_writes=self.settings.ROBINHOOD_WORKSPACE_WRITES,
            sources=(
                RemoteSource(
                    rh_registry, build_robinhood_server(self.settings, now), required=True
                ),
                RemoteSource(wt_registry, build_wheelta_server(self.settings)),
            ),
            observed_at=now,
            remote_boundary_accepted=self.deps.remote_boundary_accepted,
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
            "available_tools": available_tools_table(plan),
            "position_book": book.model_dump_json(),
            "owned_orders": json.dumps([r.model_dump(mode="json") for r in owned], sort_keys=True),
            "recent_decisions": "[]",
        }
        rendered = render_prompt(self.template, values)
        self.event(
            RunEventType.METADATA,
            {
                "prompt_id": rendered.prompt_id,
                "prompt_version": rendered.version,
                "prompt_template_hash": rendered.template_sha256,
                "rendered_prompt_hash": rendered.sha256,
                "model_id": self.settings.AGENT_MODEL,
                "position_book": book.model_dump(mode="json"),
            },
            key="metadata:prompt",
        )
        return rendered

    def _deadline_check(self) -> None:
        trip_if_deadline_passed(self.control, self.deadline, self.deps.clock)

    def _session_budget(self) -> float:
        remaining = self.deadline.remaining_seconds(self.deps.clock())
        return max(remaining - FINALIZE_RESERVE_SECONDS, 0.0)

    def _agent_run(self, effective_mode: ExecutionMode) -> int:
        plan = self._plan(effective_mode)
        if ROBINHOOD in plan.withheld:
            obs = next((o for o in plan.observations if o.server == ROBINHOOD), None)
            if obs is not None and obs.status is SourceStatus.NEEDS_AUTH:
                self.alert(
                    AlertKind.ROBINHOOD_NEEDS_AUTH,
                    "Robinhood needs authentication; no trading session was started (ADR-0004)",
                )
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
        return self._finish(plan, book, prompt, session)

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
    ) -> int:
        if session is not None:
            self.observe_sources(session.observations[len(plan.observations) :])
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
        audit_ok = self._audit(meta, book, decisions, record)
        status, reason = self._status(plan, session, decisions)
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
        return record

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
            status = ToolCallStatus.FAILED if tier is ToolTier.R else ToolCallStatus.UNKNOWN
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


def main(argv: Sequence[str] | None = None) -> int:
    """Production entrypoint: load and validate everything before any network call."""
    del argv
    try:
        settings = load_settings()
        rules = load_rules()
        template = load_prompt()
    except (SettingsError, RulesError, PromptError) as exc:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
        _LOG.error("startup configuration invalid: %s", exc)
        return EXIT_FAILED
    configure_logging(
        settings.LOG_LEVEL,
        settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
        secrets=settings_secrets(settings),
    )
    notifier = HttpNotifier(settings.ALERT_WEBHOOK_URL, settings.HEARTBEAT_URL)
    try:
        return run_once(settings, rules, template, OrchestratorDeps(notifier=notifier))
    finally:
        notifier.close()
