"""Full runs of `orchestrator.run_once` against fake MCP servers and a scripted fake CLI.

Real pieces: Settings, rules, prompt v6, the orchestrator, `ClaudeSDKClient`/`Query` (hook and
in-process MCP dispatch), our hooks, result boundary, facts tool, web cache, assembler, audit,
and a throwaway Postgres ledger. Fake pieces: the CLI transport (e2e_fake_cli.py), the remote
servers and their fixture result mappers (e2e_fakes.py), the clock, and the notifier.
"""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
from e2e_fake_cli import FakeCli, FakeModel, FakeToolFailure, factory, world_upstreams
from e2e_fakes import (
    ACCOUNT_NUMBER,
    COMPANY,
    E2E_MODEL,
    FIXTURE_MAPPERS,
    FIXTURE_SCOPE_TABLE,
    MACRO,
    MARKET,
    OTHER_ACCOUNT_NICKNAME,
    OTHER_ACCOUNT_NUMBER,
    RAW_MARKER,
    SIMULATED_MAPPERS,
    build_world,
    decision_json,
    dry_run_script,
    market_mignon,
    mignon_report,
    research,
    simulated_order_script,
    simulated_price_walk_script,
    simulated_world,
)
from e2e_support import (
    SESSION_TIME,
    WEEKEND_TIME,
    FakeClock,
    RecordingMailer,
    RecordingNotifier,
)

from wheelta_robinhood_agent.agent.account_scope import AGENTIC_ACCOUNT_PLACEHOLDER
from wheelta_robinhood_agent.agent.session import plan_session
from wheelta_robinhood_agent.config.prompts import load_prompt
from wheelta_robinhood_agent.config.rules import load_rules
from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    ExecutionMode,
    OrderVenue,
    RunStatus,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.run_identity import run_id_for, slot_for
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    LIVE_ORDER_TOOLS,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.lock import RUN_LOCK_OBJID, try_advisory_lock
from wheelta_robinhood_agent.ledger.runs import open_run_slot, run_projection
from wheelta_robinhood_agent.ledger.tool_calls import tool_call_records
from wheelta_robinhood_agent.orchestrator.main import OrchestratorDeps, run_once
from wheelta_robinhood_agent.orchestrator.market_session import (
    CalendarOutOfRange,
    build_nyse_calendar,
)

RULES = load_rules()
TEMPLATE = load_prompt()
ORDER_TOOLS = {ROBINHOOD_REGISTRY.qualified(n) for n in LIVE_ORDER_TOOLS}
VERIFIED_RH = ROBINHOOD_REGISTRY.model_copy(update={"verified": True})
Script = Callable[[FakeModel], Any]
# Schedule and calendar gates apply to armed live only (ADR-0038), which runs only in
# production (ADR-0039).
LIVE: dict[str, Any] = {"APP_ENV": "production", "EXECUTION_MODE": "live", "EXECUTION_ARMED": True}


class Harness:
    def __init__(self, settings: Settings, notifier: RecordingNotifier, clock: FakeClock) -> None:
        self.settings = settings
        self.notifier = notifier
        self.clock = clock
        self.clis: list[FakeCli] = []
        self.mailer = RecordingMailer()
        self.world = build_world(clock.now)
        self.run_id = run_id_for(settings.APP_ENV, slot_for(clock.now))

    def deps(self, script: Script, **overrides: Any) -> OrchestratorDeps:
        values: dict[str, Any] = {
            "notifier": self.notifier,
            "summary_mailer": self.mailer,
            "clock": self.clock,
            "transport_factory": factory(self.world, script, self.clis),
            "upstream_factory": world_upstreams(self.world),
            "install_signals": False,
            "remote_boundary_accepted": True,
            "registries": (VERIFIED_RH, WHEELTA_REGISTRY),
            "mappers": FIXTURE_MAPPERS,
            "account_scope_table": FIXTURE_SCOPE_TABLE,
            "connect_budget_seconds": 5.0,
            "interrupt_grace_seconds": 5.0,
            "status_poll_interval": 0.05,
        }
        values.update(overrides)
        return OrchestratorDeps(**values)

    def run(
        self, script: Script = dry_run_script, run_now: bool | None = None, **overrides: Any
    ) -> int:
        """`run_now` defaults to on demand in effective off (ADR-0038: dry runs start only
        on demand) and to a scheduled tick in live."""
        if run_now is None:
            run_now = (
                self.settings.effective_execution_mode is ExecutionMode.OFF
                and self.settings.APP_ENV is AppEnv.LOCAL
            )
        return run_once(
            self.settings, RULES, TEMPLATE, self.deps(script, **overrides), run_now=run_now
        )

    def conn(self) -> Any:
        return connect(self.settings.DATABASE_URL)

    def status(self) -> RunStatus | None:
        with self.conn() as c:
            return run_projection(c, self.run_id).status

    def events(self, event_type: RunEventType) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT payload FROM run_events WHERE entity_id = %s AND event_type = %s "
                "ORDER BY sequence",
                (self.run_id, event_type.value),
            ).fetchall()
        return [r[0] for r in rows]


@pytest.fixture
def harness(make_settings: Any, notifier: RecordingNotifier) -> Callable[..., Harness]:
    def make(at: datetime = SESSION_TIME, **settings: Any) -> Harness:
        return Harness(make_settings(**settings), notifier, FakeClock(at))

    return make


# -- preflight skips ----------------------------------------------------------------------------


def test_kill_switch_skips_with_exit_0(harness: Callable[..., Harness]) -> None:
    h = harness(KILL_SWITCH=True)
    assert h.run() == 0
    assert h.status() is RunStatus.SKIPPED_KILLED
    assert h.clis == []
    assert "kill_switch_engaged" in h.notifier.alert_kinds()
    assert h.notifier.heartbeats[-1].run_status is RunStatus.SKIPPED_KILLED


def test_market_closed_skips_with_exit_0(harness: Callable[..., Harness]) -> None:
    h = harness(at=WEEKEND_TIME, **LIVE)
    assert h.run() == 0
    assert h.status() is RunStatus.SKIPPED_MARKET_CLOSED
    assert h.clis == []
    assert h.events(RunEventType.MARKET_SESSION)[0]["session"] == "closed"


def test_lock_contention_skips_without_touching_the_slot(harness: Callable[..., Harness]) -> None:
    h = harness()
    with h.conn() as holder:
        assert try_advisory_lock(holder, RUN_LOCK_OBJID[AppEnv.LOCAL])
        assert h.run() == 0
        rows = holder.execute("SELECT count(*) FROM runs").fetchone()
    assert rows is not None and rows[0] == 0
    assert h.clis == []
    assert h.notifier.heartbeats[-1].run_status is RunStatus.SKIPPED_CONCURRENT


def test_completed_slot_is_a_no_op(harness: Callable[..., Harness]) -> None:
    h = harness(at=WEEKEND_TIME)
    assert h.run() == 0
    with h.conn() as c:
        before = c.execute("SELECT count(*) FROM run_events").fetchone()
    assert h.run() == 0
    with h.conn() as c:
        after = c.execute("SELECT count(*) FROM run_events").fetchone()
    assert before == after
    assert len(h.notifier.heartbeats) == 1


# -- agent-chosen next run (ADR-0028) ------------------------------------------------------------


def _tick(h: Harness, at: datetime) -> None:
    """Move the harness to a later cron tick: its slot, run_id, and a fresh fake world."""
    h.clock.now = at
    h.run_id = run_id_for(h.settings.APP_ENV, slot_for(at))
    h.world = build_world(at)


def _next_run_script(at: str) -> Script:
    async def script(model: FakeModel) -> str:
        candidate_ref, facts_ref = await research(model)
        return decision_json(
            candidate_ref, facts_ref, next_run={"at": at, "rationale": "Scripted schedule."}
        )

    return script


def test_initial_run_records_the_hourly_fallback(harness: Callable[..., Harness]) -> None:
    h = harness(**LIVE)
    assert h.run() == 0, h.notifier.alert_kinds()
    assert h.events(RunEventType.SCHEDULE) == [
        {
            "not_before": (SESSION_TIME + timedelta(hours=1)).isoformat(),
            "source": "fallback",
            "requested_at": (SESSION_TIME + timedelta(hours=1)).isoformat(),
            "latest_at": (SESSION_TIME + timedelta(hours=48)).isoformat(),
            "capped": False,
            "moved_to_session_open": False,
        }
    ]
    (check,) = [
        e["schedule_check"] for e in h.events(RunEventType.METADATA) if "schedule_check" in e
    ]
    assert check == {"not_before": None, "due": True, "run_now": False}


def test_agent_next_run_gates_later_ticks(harness: Callable[..., Harness]) -> None:
    h = harness(**LIVE)
    chosen = SESSION_TIME + timedelta(minutes=42)  # 16:12 UTC, inside the session
    assert h.run(_next_run_script(chosen.isoformat().replace("+00:00", "Z"))) == 0
    assert [e["source"] for e in h.events(RunEventType.SCHEDULE)] == ["fallback", "agent"]
    assert h.events(RunEventType.SCHEDULE)[-1]["not_before"] == chosen.isoformat()

    _tick(h, SESSION_TIME + timedelta(minutes=40))  # before the chosen time
    assert h.run() == 0
    assert h.status() is RunStatus.SKIPPED_NOT_DUE
    assert len(h.clis) == 1 and h.events(RunEventType.SCHEDULE) == []
    assert h.notifier.heartbeats[-1].run_status is RunStatus.SKIPPED_NOT_DUE
    assert h.notifier.heartbeats[-1].status.value == "success"

    _tick(h, SESSION_TIME + timedelta(minutes=45))  # the first tick at or after it
    assert h.run() == 0, h.notifier.alert_kinds()
    assert h.status() is RunStatus.COMPLETED
    assert len(h.clis) == 2


def test_dry_run_is_on_demand_only_and_never_touches_the_schedule(
    harness: Callable[..., Harness],
) -> None:
    """ADR-0038: a scheduled tick in off mode does nothing, before the ledger; an on-demand
    dry run records its next run unapplied and no schedule event."""
    h = harness()
    assert h.run(run_now=False) == 0
    assert h.clis == []
    assert h.notifier.heartbeats[-1].run_status is RunStatus.SKIPPED_DRY_RUN_NOT_REQUESTED
    assert h.notifier.heartbeats[-1].status.value == "success"
    with h.conn() as c:
        rows = c.execute("SELECT count(*) FROM runs").fetchone()
    assert rows is not None and rows[0] == 0  # the slot stays free for an on-demand run
    assert h.run(_next_run_script("2026-09-24T15:00:00Z"), run_now=True) == 0
    assert h.status() is RunStatus.COMPLETED and len(h.clis) == 1
    assert h.events(RunEventType.SCHEDULE) == []
    (unapplied,) = [e for e in h.events(RunEventType.METADATA) if "next_run_not_applied" in e]
    assert unapplied["next_run_not_applied"]["requested_at"] == "2026-09-24T15:00:00+00:00"
    from wheelta_robinhood_agent.agent.trace_loader import load_decision_trace

    with h.conn() as c:
        trace = load_decision_trace(c, h.run_id)
    assert trace.next_run is not None and trace.next_run["applied"] is False
    assert trace.next_run["reason"] == "dry_run_on_demand"


def test_on_demand_dry_run_ignores_the_calendar(harness: Callable[..., Harness]) -> None:
    h = harness(at=WEEKEND_TIME)
    assert h.run(run_now=True) == 0, h.notifier.alert_kinds()
    assert h.status() is RunStatus.COMPLETED and len(h.clis) == 1
    assert h.events(RunEventType.MARKET_SESSION)[0]["session"] == "closed"


def test_run_now_still_respects_the_kill_switch_and_the_session(
    harness: Callable[..., Harness],
) -> None:
    killed = harness(KILL_SWITCH=True)
    assert killed.run(run_now=True) == 0
    assert killed.status() is RunStatus.SKIPPED_KILLED and killed.clis == []


@pytest.mark.parametrize("settings", [LIVE, {"APP_ENV": "production"}])
def test_run_now_is_refused_outside_local(
    harness: Callable[..., Harness], settings: dict[str, Any]
) -> None:
    """ADR-0039: production runs live on its calendar and schedule only; a dry run is local."""
    h = harness(**settings)
    with pytest.raises(ValueError, match="APP_ENV=local"):
        h.run(run_now=True)
    assert h.clis == []


def test_off_mode_in_production_runs_nothing(harness: Callable[..., Harness]) -> None:
    """ADR-0039: dry runs never run on Railway; an off production tick touches no ledger."""
    h = harness(APP_ENV="production", EXECUTION_MODE="off")
    assert h.run() == 0
    assert h.clis == []
    assert h.notifier.heartbeats[-1].run_status is RunStatus.SKIPPED_DRY_RUN_NOT_LOCAL
    with h.conn() as c:
        rows = c.execute("SELECT count(*) FROM runs").fetchone()
    assert rows is not None and rows[0] == 0


def test_a_local_dry_run_writes_no_workspace_objects_or_position_notes(
    harness: Callable[..., Harness],
) -> None:
    """ADR-0039: the dry run shares the Robinhood account with production, so its Tier S
    writes are withheld, even with ROBINHOOD_WORKSPACE_WRITES=true."""
    h = harness(ROBINHOOD_WORKSPACE_WRITES=True)
    assert h.run() == 0, h.notifier.alert_kinds()
    (cli,) = h.clis
    tier_s = {VERIFIED_RH.qualified(t.name) for t in VERIFIED_RH.tools if t.tier.value == "S"}
    assert tier_s and not tier_s & set(cli.options.allowed_tools)
    assert tier_s <= set(cli.options.disallowed_tools)


def test_agent_next_run_outside_the_session_moves_to_the_next_open(
    harness: Callable[..., Harness],
) -> None:
    h = harness(**LIVE)
    assert h.run(_next_run_script("2026-09-23T18:00:00-04:00")) == 0  # after Wednesday's close
    agent = h.events(RunEventType.SCHEDULE)[-1]
    assert agent == {
        "not_before": "2026-09-24T13:30:00+00:00",
        "source": "agent",
        "requested_at": "2026-09-23T22:00:00+00:00",
        "latest_at": "2026-09-25T15:30:00+00:00",
        "capped": False,
        "moved_to_session_open": True,
    }


# Friday 2026-09-25 11:30 America/New_York: inside the session, before a weekend.
FRIDAY_TIME = SESSION_TIME + timedelta(days=2)


def test_agent_next_run_on_a_weekend_moves_to_monday_open(
    harness: Callable[..., Harness],
) -> None:
    h = harness(at=FRIDAY_TIME, **LIVE)
    assert h.run(_next_run_script("2026-09-27T12:00:00Z")) == 0  # Sunday, within 48 h
    agent = h.events(RunEventType.SCHEDULE)[-1]
    assert agent["not_before"] == "2026-09-28T13:30:00+00:00"
    assert agent["capped"] is False and agent["moved_to_session_open"] is True


@pytest.mark.parametrize(
    ("at", "requested", "not_before"),
    [
        # Wednesday: capped at Friday 15:30 UTC, inside the session.
        (SESSION_TIME, "2026-10-07T15:00:00Z", "2026-09-25T15:30:00+00:00"),
        (SESSION_TIME, "9999-12-31T00:00:00Z", "2026-09-25T15:30:00+00:00"),
        # Friday: capped at Sunday 15:30 UTC, then moved to Monday's open.
        (FRIDAY_TIME, "2026-10-07T15:00:00Z", "2026-09-28T13:30:00+00:00"),
    ],
)
def test_agent_next_run_beyond_the_max_gap_is_capped(
    harness: Callable[..., Harness], at: datetime, requested: str, not_before: str
) -> None:
    h = harness(at=at, **LIVE)
    assert h.run(_next_run_script(requested)) == 0, h.notifier.alert_kinds()
    agent = h.events(RunEventType.SCHEDULE)[-1]
    assert agent["source"] == "agent" and agent["capped"] is True
    assert agent["latest_at"] == (at + timedelta(hours=48)).isoformat()
    assert agent["not_before"] == not_before


def test_unplaceable_agent_next_run_keeps_the_fallback_and_the_audit(
    harness: Callable[..., Harness],
) -> None:
    # The third calendar built in a run places the agent's time (after the market-session
    # and fallback calendars); make it fail as a broken calendar would.
    built: list[object] = []

    def calendar_factory(start: Any, end: Any) -> Any:
        built.append((start, end))
        if len(built) == 3:
            raise CalendarOutOfRange("simulated calendar failure")
        return build_nyse_calendar(start, end)

    h = harness(**LIVE)
    script = _next_run_script("2026-09-24T15:00:00Z")
    assert h.run(script, calendar_factory=calendar_factory) == 0, h.notifier.alert_kinds()
    assert len(built) == 3
    assert h.status() is RunStatus.COMPLETED
    assert [e["source"] for e in h.events(RunEventType.SCHEDULE)] == ["fallback"]
    rejected = [e for e in h.events(RunEventType.METADATA) if "next_run_rejected" in e]
    assert rejected == [
        {
            "next_run_rejected": {
                "requested_at": "2026-09-24T15:00:00+00:00",
                "error_type": "CalendarOutOfRange",
            }
        }
    ]
    assert h.events(RunEventType.AUDIT_STATUS)[0]["status"] == "completed"
    with h.conn() as c:
        assert ledger_evidence.run_records_for_run(c, h.run_id)


def test_invalid_output_keeps_the_fallback(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str:
        await research(model)
        return '{"next_run": {"at": "soon", "rationale": "x"}}'

    h = harness(**LIVE)
    assert h.run(script) == 1
    assert [e["source"] for e in h.events(RunEventType.SCHEDULE)] == ["fallback"]


def test_kill_switch_alerts_only_on_due_ticks(harness: Callable[..., Harness]) -> None:
    h = harness(KILL_SWITCH=True, **LIVE)
    assert h.run() == 0
    assert [e["source"] for e in h.events(RunEventType.SCHEDULE)] == ["fallback"]
    _tick(h, SESSION_TIME + timedelta(minutes=5))
    assert h.run() == 0
    assert h.status() is RunStatus.SKIPPED_KILLED
    assert h.notifier.alert_kinds().count("kill_switch_engaged") == 1
    assert h.clis == []


def test_market_closed_tick_records_no_schedule(harness: Callable[..., Harness]) -> None:
    h = harness(at=WEEKEND_TIME, **LIVE)
    assert h.run() == 0
    assert h.events(RunEventType.SCHEDULE) == []


# -- Robinhood availability -----------------------------------------------------------------------


def test_robinhood_without_token_is_needs_auth_and_no_session(
    harness: Callable[..., Harness],
) -> None:
    h = harness(ROBINHOOD_MCP_ACCESS_TOKEN=None)
    assert h.run() == 1
    assert h.clis == []
    assert h.status() is RunStatus.FAILED
    assert "robinhood_needs_auth" in h.notifier.alert_kinds()
    statuses = {e["server"]: e["status"] for e in h.events(RunEventType.SOURCE_STATUS)}
    assert statuses["robinhood"] == "needs-auth"
    with h.conn() as c:
        assert ledger_evidence.run_records_for_run(c, h.run_id)  # still assembled


def test_local_off_mode_still_runs_the_dry_run_session(harness: Callable[..., Harness]) -> None:
    """A local dry run starts the session (as production does since ADR-0033)."""
    h = harness()
    assert h.run() == 0
    assert len(h.clis) == 1 and h.status() is RunStatus.COMPLETED


def test_robinhood_needs_auth_at_connect_sends_no_query(harness: Callable[..., Harness]) -> None:
    h = harness()
    h.world.statuses["robinhood"] = "needs-auth"
    assert h.run() == 1
    # The proxy's upstream connect is refused before any CLI session is created (ADR-0023).
    assert h.clis == []
    statuses = {e["server"]: e["status"] for e in h.events(RunEventType.SOURCE_STATUS)}
    assert statuses["robinhood"] == "needs-auth"
    assert "robinhood_needs_auth" in h.notifier.alert_kinds()
    assert h.status() is RunStatus.FAILED


def test_production_defaults_proxy_robinhood_and_withhold_unverified_wheelta(
    harness: Callable[..., Harness],
) -> None:
    """Direct delivery is not accepted, so a remote source reaches the model only through the
    validating proxy (ADR-0023), and only with a verified registry."""
    h = harness()
    h.run(remote_boundary_accepted=False, registries=(ROBINHOOD_REGISTRY, WHEELTA_REGISTRY))
    assert len(h.clis) == 1
    (cli,) = h.clis
    servers = cli.options.mcp_servers
    assert isinstance(servers, dict)
    assert {n: c["type"] for n, c in servers.items()} == {"robinhood": "sdk", "wra_local": "sdk"}
    statuses = {e["server"]: e["status"] for e in h.events(RunEventType.SOURCE_STATUS)}
    assert statuses["robinhood"] == "connected"
    assert statuses["wheelta"] == "disabled"


# -- Agentic-account eligibility (trusted get_accounts check, CLAUDE.md §9) ----------------------


def _metadata(h: Harness, key: str) -> list[Any]:
    return [e[key] for e in h.events(RunEventType.METADATA) if key in e]


def test_eligible_account_is_recorded_redacted_and_other_accounts_never_persist(
    harness: Callable[..., Harness],
) -> None:
    h = harness()
    assert h.run() == 0, h.notifier.alert_kinds()
    (eligibility,) = _metadata(h, "agentic_eligibility")
    assert eligibility["eligible"] is True and eligibility["reasons"] == []
    assert eligibility["account_ref"] == f"****{ACCOUNT_NUMBER[-4:]}"
    # The listing went through the upstream in trusted code, never through the model.
    (cli,) = h.clis
    assert all(t.name != "mcp__robinhood__get_accounts" for t in cli.turns)
    with h.conn() as c:
        dump = json.dumps(
            [
                c.execute("SELECT payload::text FROM run_events").fetchall(),
                c.execute("SELECT payload::text FROM results").fetchall(),
            ]
        )
    assert OTHER_ACCOUNT_NUMBER not in dump and OTHER_ACCOUNT_NICKNAME not in dump
    assert ACCOUNT_NUMBER not in dump


def test_ineligible_account_starts_no_session(harness: Callable[..., Harness]) -> None:
    h = harness()
    listing = h.world.handlers["robinhood"]["get_accounts"]

    def not_agentic(args: dict[str, Any]) -> dict[str, Any]:
        response = listing(args)
        payload = json.loads(response["content"][0]["text"])
        payload["data"]["accounts"][0]["agentic_allowed"] = False
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}

    h.world.handlers["robinhood"]["get_accounts"] = not_agentic
    assert h.run() == 1
    assert h.clis == []
    (eligibility,) = _metadata(h, "agentic_eligibility")
    assert eligibility["eligible"] is False
    assert eligibility["reasons"] == ["agentic_allowed is false"]
    statuses = [e for e in h.events(RunEventType.STATUS) if e and "reason" in e]
    assert statuses[-1]["reason"] == "robinhood_account_not_agentic"


def test_failed_eligibility_check_fails_closed(harness: Callable[..., Harness]) -> None:
    h = harness()

    def broken(args: dict[str, Any]) -> dict[str, Any]:
        raise FakeToolFailure("transport down")

    h.world.handlers["robinhood"]["get_accounts"] = broken
    assert h.run() == 1
    assert h.clis == []
    assert _metadata(h, "agentic_eligibility") == []
    assert h.status() is RunStatus.FAILED


# -- the dry run ----------------------------------------------------------------------------------


def test_dry_run_records_an_unsubmitted_proposal_and_audits_it(
    harness: Callable[..., Harness],
) -> None:
    h = harness()
    assert h.run() == 0, h.notifier.alert_kinds()
    assert h.status() is RunStatus.COMPLETED
    with h.conn() as c:
        (stored,) = ledger_evidence.run_records_for_run(c, h.run_id)
        findings = ledger_evidence.audit_findings_for_run(c, h.run_id)
        facts = ledger_evidence.decision_facts_for_run(c, h.run_id)
    record = stored.record
    assert record.effective_execution_mode is ExecutionMode.OFF
    (decision,) = record.decisions
    (leg,) = decision.legs
    # ADR-0038: a simulated-venue dry run is assembled like live: a proposal the agent did
    # not submit is a leg with its computed target and no attempt (no DRY_RUN stand-in).
    assert record.order_venue is OrderVenue.SIMULATED
    # min(cash 30000, 20% of 150000 = 30000, cap 10 contracts) / (150 x 100) = 2
    assert leg.target_quantity == 2 and leg.attempts == ()
    assert facts and facts[0].facts.initial_quantity == 2
    outcomes = {(f.check_id.value, f.outcome.value) for f in findings}
    assert not any(o == "violation" for _, o in outcomes), outcomes
    assert {c for c, _ in outcomes} == {"V1", "V2", "V3", "V4", "V5", "V6", "V7"}
    assert record.findings == ()
    assert findings, "the audit recorded findings"
    assert h.events(RunEventType.AUDIT_STATUS)[0]["status"] == "completed"
    # Wheelta is unverified in the registry: withheld, and the model was told so.
    assert "wheelta" in h.clis[0].user_messages[0]


def test_invalid_agent_output_still_assembles_from_events(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str:
        await research(model)
        return "I would sell the AAPL put."  # not JSON

    h = harness()
    assert h.run(script) == 1
    assert h.status() is RunStatus.FAILED
    assert "invalid_agent_output" in h.notifier.alert_kinds()
    with h.conn() as c:
        (stored,) = ledger_evidence.run_records_for_run(c, h.run_id)
        (output,) = ledger_evidence.agent_outputs_for_run(c, h.run_id)
        calls = tool_call_records(c, h.run_id)
    assert output.raw_redacted == "I would sell the AAPL put."
    assert stored.record.decisions == ()
    assert stored.record.decision_output_status.value != "valid"
    # Agent + the Mignon's chain and quote + the orchestrator's re-quote, 3 account reads, facts.
    assert len(calls) == 8


def test_position_notes_carry_forward_until_close(harness: Callable[..., Harness]) -> None:
    """ADR-0018: earlier notes reach the prompt; this run's HOLD and question become notes."""
    import json
    from datetime import timedelta

    from wheelta_robinhood_agent.agent.account_scope import account_scope_id
    from wheelta_robinhood_agent.domain.enums import DecisionAction, StrategyKind
    from wheelta_robinhood_agent.domain.options import OccSymbol
    from wheelta_robinhood_agent.domain.positions import (
        PositionInstrument,
        PositionNote,
        PositionNoteKind,
    )
    from wheelta_robinhood_agent.ledger.positions import open_position, position_book, record_note

    h = harness(**LIVE)  # position notes are production memory (ADR-0039)
    scope = account_scope_id(h.settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER)
    earlier = h.clock.now - timedelta(hours=1)
    with h.conn() as c:
        prior = open_run_slot(c, AppEnv.PRODUCTION, slot_for(earlier)).run_id
        position_id = open_position(
            c,
            run_id=prior,
            account_scope_id=scope,
            underlying="AAPL",
            strategy=StrategyKind.CASH_SECURED_PUT,
            instruments=[
                PositionInstrument(
                    occ_symbol=OccSymbol.parse("AAPL  261016P00150000"),
                    broker_instrument_id="inst-aapl-150p",
                    short_quantity=1,
                )
            ],
            observed_at=earlier,
            imported=True,
        )
        record_note(
            c,
            position_id,
            dedup_key=f"note:{prior}:decision:0",
            note=PositionNote(
                run_id=prior,
                noted_at=earlier,
                kind=PositionNoteKind.DECISION,
                action=DecisionAction.HOLD,
                decision_ref="decision:0",
                text="Watching the supplier report before deciding.",
            ),
        )
    position_ref = f"position:{position_id}"

    async def script(model: FakeModel) -> str:
        # No positions read: a live complete read without this contract would close it.
        hold = {
            "action": "HOLD",
            "target_ref": position_ref,
            "replacement_ref": None,
            "funding_close_refs": [],
            "proposed_legs": [],
            "execution_refs": [],
            "rationale": "Supplier report was neutral; keep holding.",
            "thesis": None,
            "invalidation_conditions": [],
            "evidence_refs": [],
        }
        question = {
            "target_ref": position_ref,
            "question": "When is guidance?",
            "evidence_refs": [],
        }
        return json.dumps(
            {
                "decisions": [hold],
                "cancellation_rationales": [],
                "unresolved_questions": [question],
                "next_run": None,
            }
        )

    assert h.run(script) == 0, h.notifier.alert_kinds()
    assert "Watching the supplier report before deciding." in str(h.clis[0].options.system_prompt)
    with h.conn() as c:
        (entry,) = position_book(c, scope, as_of=h.clock.now).entries
    assert [(n.kind, n.action, n.text) for n in entry.notes] == [
        (
            PositionNoteKind.DECISION,
            DecisionAction.HOLD,
            "Watching the supplier report before deciding.",
        ),
        (
            PositionNoteKind.DECISION,
            DecisionAction.HOLD,
            "Supplier report was neutral; keep holding.",
        ),
        (PositionNoteKind.QUESTION, None, "When is guidance?"),
    ]
    assert entry.notes[-1].run_id == h.run_id


# -- tool exposure ------------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "armed"), [("off", False), ("live", False), ("bogus", True)])
def test_dry_run_order_tools_go_to_the_simulated_broker(
    harness: Callable[..., Harness], mode: str, armed: bool
) -> None:
    """ADR-0038: every effective-off run gets the live option-order tools and the live prompt,
    answered by the simulated broker; no order tool ever reaches Robinhood, and the simulated
    order is recorded under the run's own scope, never the account's."""
    from wheelta_robinhood_agent.agent.account_scope import account_scope_id
    from wheelta_robinhood_agent.agent.simulated_broker import simulated_scope_id
    from wheelta_robinhood_agent.domain.enums import OrderVenue
    from wheelta_robinhood_agent.ledger.orders import owned_unresolved_orders, run_order_records

    h = harness(EXECUTION_MODE=mode, EXECUTION_ARMED=armed)
    h.world = simulated_world(h.clock.now)
    script = simulated_order_script(lambda: h.clock.advance(1))
    assert h.run(script, mappers=SIMULATED_MAPPERS) == 0, h.notifier.alert_kinds()
    assert h.status() is RunStatus.COMPLETED
    (cli,) = h.clis
    assert ORDER_TOOLS <= cli.visible_tools()
    assert ORDER_TOOLS <= set(cli.options.allowed_tools)
    assert "mcp__robinhood__place_equity_order" in cli.options.disallowed_tools
    assert "Effective execution mode: live" in (cli.options.system_prompt or "")
    upstream = [tool for _, tool, _ in h.world.upstream_calls]
    assert not {t for t in upstream if t.endswith("_option_order")}, upstream
    with h.conn() as c:
        calls = tool_call_records(c, h.run_id)
        orders = run_order_records(c, h.run_id)
        scope = account_scope_id(h.settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER)
        owned = owned_unresolved_orders(c, scope)
        (stored,) = ledger_evidence.run_records_for_run(c, h.run_id)
    by_tool = {r.identity.tool: r.status for r in calls if r.identity.tool.endswith("_order")}
    assert by_tool == {
        "review_option_order": ToolCallStatus.SUCCEEDED,
        "place_option_order": ToolCallStatus.SUCCEEDED,
        "cancel_option_order": ToolCallStatus.SUCCEEDED,
    }
    (record,) = [r for r in orders if r.intent is not None]
    assert record.intent.account_scope_id == simulated_scope_id(h.run_id)
    assert record.status is AttemptStatus.CANCELLED
    assert owned == ()  # nothing simulated leaks into the account's owned orders
    assert stored.record.order_venue is OrderVenue.SIMULATED
    assert stored.record.effective_execution_mode is ExecutionMode.OFF
    metadata = h.events(RunEventType.METADATA)
    (tool_access,) = [e["tool_access"] for e in metadata if "tool_access" in e]
    assert tool_access["order_venue"] == "simulated"
    _assert_simulated_trace(h)


def _assert_simulated_trace(h: Harness) -> None:
    """The operator trace resolves the run end to end from the ledger alone."""
    from wheelta_robinhood_agent.agent.trace_loader import list_runs, load_decision_trace
    from wheelta_robinhood_agent.domain.enums import OrderVenue
    from wheelta_robinhood_agent.observability.decision_trace import render_trace_markdown

    with h.conn() as c:
        trace = load_decision_trace(c, h.run_id)
        (listing,) = list_runs(c, AppEnv.LOCAL)
    assert trace.order_venue is OrderVenue.SIMULATED
    assert trace.effective_execution_mode is ExecutionMode.OFF
    assert trace.prompt_execution_mode == "live"
    (decision,) = trace.decisions
    assert decision.action == "OPEN_CSP" and decision.rationale == "Scripted e2e choice."
    assert all(e.resolved for e in decision.evidence)
    (leg,) = decision.legs
    assert leg.facts is not None and leg.facts.initial_quantity == 2
    # The SPY place and its cancel, which the output did not select, with their calls.
    unlinked = {u.kind: [c.tool for c in u.calls] for u in trace.unassociated}
    assert unlinked == {
        "place": ["review_option_order", "place_option_order", "cancel_option_order"],
        "cancel": ["cancel_option_order"],
    }
    # The audit sees the simulated order (ADR-0038): the script placed a 740 put on 30,000 of
    # cash without quoting it. Its time in force matches the rule ("gfd", ADR-0039). Every
    # such finding sits under the unlinked place, not the run level.
    place = next(u for u in trace.unassociated if u.kind == "place")
    checks = {v.split(":")[0] for v in place.findings.violations}
    assert checks == {"V1.3", "V2.1", "V7.3", "V7.4"}
    assert place.broker_order_id is not None and place.status == "cancelled"
    assert not trace.run_findings.violations
    callers = {c.caller for c in trace.timeline}
    assert callers == {"orchestrator", MARKET}
    assert trace.mignon_spawns == 1
    assert trace.next_run is None  # the scripted output requests no next run
    text = render_trace_markdown(trace)
    assert "order venue **simulated**" in text and "place_option_order" in text
    assert listing.run_id == h.run_id and listing.placed == 1


def test_armed_live_exposes_exactly_the_option_order_tools(
    harness: Callable[..., Harness],
) -> None:
    """ADR-0034: armed live allowlists review/place/cancel option orders and nothing else X."""
    h = harness(EXECUTION_MODE="live", EXECUTION_ARMED=True)
    assert h.run(dry_run_script) == 0
    (cli,) = h.clis
    assert set(cli.options.allowed_tools) & ORDER_TOOLS == ORDER_TOOLS
    assert not ORDER_TOOLS & set(cli.options.disallowed_tools)
    assert "mcp__robinhood__place_equity_order" in cli.options.disallowed_tools


def test_plan_accepts_an_effective_live_mode() -> None:
    plan = plan_session(
        effective_mode=ExecutionMode.LIVE,
        workspace_writes=False,
        sources=(),
        observed_at=SESSION_TIME,
    )
    assert plan.effective_mode is ExecutionMode.LIVE


# -- result boundary (fake transport observation) ------------------------------------------------


def test_unmapped_remote_result_never_reaches_the_model(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str | None:
        turn = await model.call("mcp__robinhood__get_equity_quotes", {"symbols": ["AAPL"]})
        assert turn.output["kind"] == "missing"
        withheld = await model.call("mcp__wheelta__wheelta_board_status", {})
        assert withheld.denied and "withheld" in (withheld.reason or "")
        return await dry_run_script(model)

    h = harness()
    assert h.run(script) == 0
    (cli,) = h.clis
    assert RAW_MARKER not in repr(cli.model_inputs)
    with h.conn() as c:
        raw = [
            r
            for r in ledger_evidence.results_for_run(c, h.run_id)
            if r.kind is ledger_evidence.ResultKind.RAW_INVALID
        ]
    assert raw and RAW_MARKER in repr(raw[0].payload)  # kept as restricted evidence only


# -- web cache ---------------------------------------------------------------------------------


def test_web_cache_hit_denies_an_identical_search(harness: Callable[..., Harness]) -> None:
    async def company(model: FakeModel) -> str:
        first = await model.call("WebSearch", {"query": "AAPL earnings date"})
        assert not first.denied
        second = await model.call("WebSearch", {"query": "  aapl   EARNINGS date "})
        assert second.denied and "web_cache_lookup" in (second.reason or "")
        cached = await model.call("mcp__wra_local__web_cache_lookup", {"ticker": "AAPL"})
        assert cached.data["entries"][0]["query"] == "AAPL earnings date"
        return mignon_report("Find the AAPL earnings date.")

    async def script(model: FakeModel) -> str | None:
        turn = await model.spawn(COMPANY, "Find the AAPL earnings date.", company)
        assert turn.output["kind"] == "validated", turn.output
        return await dry_run_script(model)

    h = harness()
    assert h.run(script) == 0
    assert [n for n, _ in h.world.calls].count("WebSearch") == 1


# -- runtime stop ---------------------------------------------------------------------------------


def test_sigterm_latches_stop_and_interrupts_the_session(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str | None:
        await model.call("mcp__robinhood__get_option_quotes", {"instrument_ids": ["x"]})
        signal.raise_signal(signal.SIGTERM)
        for _ in range(200):  # the orchestrator interrupts; the next call is refused
            if model.interrupted:
                break
            await asyncio.sleep(0.01)
        await model.call("mcp__robinhood__get_option_quotes", {"instrument_ids": ["x"]})
        return "{}"

    previous = signal.getsignal(signal.SIGTERM)
    h = harness()
    assert h.run(script, install_signals=True) == 3
    assert signal.getsignal(signal.SIGTERM) == previous
    assert h.status() is RunStatus.STOPPED
    (cli,) = h.clis
    assert cli.interrupted.is_set()
    assert h.events(RunEventType.CONTROL)[0]["stop_reason"] == "sigterm"
    with h.conn() as c:
        tools = [r.identity.tool for r in tool_call_records(c, h.run_id)]
        assert ledger_evidence.run_records_for_run(c, h.run_id)
    assert tools == ["get_option_quotes"]


def test_deadline_times_out_the_session(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str | None:
        await model.call("mcp__robinhood__get_option_quotes", {"instrument_ids": ["x"]})
        h.clock.advance(2000)  # past RUN_TIMEOUT_SECONDS
        for _ in range(200):
            if model.interrupted:
                break
            await asyncio.sleep(0.01)
        return None

    h = harness()
    assert h.run(script) == 2
    assert h.status() is RunStatus.TIMED_OUT
    assert "run_timeout" in h.notifier.alert_kinds()


# -- interrupted slot recovery ------------------------------------------------------------------


def test_interrupted_slot_is_finalized_without_a_new_session(
    harness: Callable[..., Harness],
) -> None:
    from wheelta_robinhood_agent.domain.enums import ToolTier
    from wheelta_robinhood_agent.ledger.runs import append_run_event
    from wheelta_robinhood_agent.ledger.tool_calls import (
        record_tool_call_dispatched,
        record_tool_call_requested,
    )

    h = harness()
    with h.conn() as c:
        slot = open_run_slot(c, AppEnv.LOCAL, slot_for(h.clock.now))
        append_run_event(
            c,
            slot.run_id,
            RunEventType.STATUS,
            observed_at=h.clock.now,
            dedup_key="status:running",
            status=RunStatus.RUNNING,
        )
        ref = record_tool_call_requested(
            c,
            run_id=slot.run_id,
            sdk_tool_use_id="toolu_crashed",
            stage="agent",
            server="robinhood",
            tool="get_option_chains",
            tier=ToolTier.R,
            arguments_redacted={"symbol": "AAPL"},
            requested_at=h.clock.now,
        )
        record_tool_call_dispatched(
            c,
            ref.tool_call_id,
            dispatched_at=h.clock.now,
            effective_arguments_redacted={"symbol": "AAPL"},
        )
    assert h.run() == 1
    assert h.clis == []
    assert h.status() is RunStatus.FAILED
    with h.conn() as c:
        (call,) = tool_call_records(c, h.run_id)
        assert ledger_evidence.run_records_for_run(c, h.run_id)
    assert call.status is ToolCallStatus.FAILED
    assert h.events(RunEventType.RECOVERY_STARTED)
    assert h.mailer.summaries == []  # ADR-0029: recovery starts no session


# -- orchestrator and Mignons (ADR-0025) ---------------------------------------------------------


def test_dry_run_delegates_research_and_the_ledger_attributes_every_call(
    harness: Callable[..., Harness],
) -> None:
    h = harness()
    assert h.run() == 0, h.notifier.alert_kinds()
    (cli,) = h.clis
    assert set(cli.options.agents or {}) == {MARKET, COMPANY, MACRO}
    assert cli.options.env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert "- `mignon-market`:" in str(cli.options.system_prompt)
    with h.conn() as c:
        calls = tool_call_records(c, h.run_id)
        prompt_meta = next(e for e in h.events(RunEventType.METADATA) if "mignon_prompts" in e)[
            "mignon_prompts"
        ]
    spawn, chain, mignon_quote, requote, *rest = calls
    assert (spawn.identity.tool, spawn.identity.tier.value if spawn.identity.tier else None) == (
        "Agent",
        "D",
    )
    assert spawn.status is ToolCallStatus.SUCCEEDED and spawn.identity.agent_id is None
    for call in (chain, mignon_quote):
        assert call.identity.agent_type == MARKET and call.identity.agent_id
    assert requote.identity.tool == "get_option_quotes" and requote.identity.agent_id is None
    assert all(c.identity.agent_id is None for c in rest)
    assert set(prompt_meta) == {"mignon-market", "mignon-company", "mignon-macro"}


def test_follow_up_mignon_may_cite_refs_handed_over_by_the_orchestrator(
    harness: Callable[..., Harness],
) -> None:
    async def script(model: FakeModel) -> str | None:
        first = await model.spawn(MARKET, "Screen AAPL puts.", market_mignon)
        ref = first.data["report"]["findings"][0]["refs"][0]

        async def follow_up(m: FakeModel) -> str:
            return mignon_report("Check the spread.", ("The screened contract.", [ref]))

        second = await model.spawn(MARKET, f"Follow up on {ref}: spread?", follow_up)
        assert second.output["kind"] == "validated", second.output
        return await dry_run_script(model)

    h = harness()
    assert h.run(script) == 0, h.notifier.alert_kinds()


def test_invalid_mignon_report_is_missing_research_not_a_failed_run(
    harness: Callable[..., Harness],
) -> None:
    async def liar(model: FakeModel) -> str:
        return mignon_report("Screen.", ("The bid is 9.99.", ["evidence:invented"]))

    async def script(model: FakeModel) -> str | None:
        turn = await model.spawn(MARKET, "Screen AAPL puts.", liar)
        assert turn.output["kind"] == "missing"
        assert any("not delivered" in g for g in turn.output["gaps"])
        return await dry_run_script(model)

    h = harness()
    assert h.run(script) == 0, h.notifier.alert_kinds()
    (cli,) = h.clis
    assert "9.99" not in repr(cli.model_inputs[0])  # the unsupported claim never reached it
    with h.conn() as c:
        spawn = tool_call_records(c, h.run_id)[0]
    assert spawn.identity.tool == "Agent" and spawn.status is ToolCallStatus.FAILED


def test_roles_are_enforced_both_ways(harness: Callable[..., Harness]) -> None:
    async def nosy(model: FakeModel) -> str:
        positions = await model.call(
            "mcp__robinhood__get_option_positions", {"account_number": ACCOUNT_NUMBER}
        )
        assert positions.denied
        nested = await model.spawn(MACRO, "Recurse.", nosy)
        assert nested.denied
        return mignon_report("Nothing.")

    async def script(model: FakeModel) -> str | None:
        web = await model.call("WebSearch", {"query": "AAPL"})
        assert web.denied and "not available to the orchestrator" in (web.reason or "")
        general = await model.spawn("general-purpose", "Do anything.", nosy)
        assert general.denied and "Mignon type" in (general.reason or "")
        background = await model.spawn(MARKET, "Screen.", market_mignon, run_in_background=True)
        assert background.denied
        await model.spawn(COMPANY, "Poke around.", nosy)
        return await dry_run_script(model)

    h = harness()
    assert h.run(script) == 0, h.notifier.alert_kinds()
    assert (
        "mcp__robinhood__get_option_positions",
        {"account_number": AGENTIC_ACCOUNT_PLACEHOLDER},
    ) in h.world.calls  # only the orchestrator's own read reached the broker
    positions_calls = [n for n, _ in h.world.calls if n.endswith("get_option_positions")]
    assert len(positions_calls) == 1


def test_spawns_stop_at_max_per_run(harness: Callable[..., Harness]) -> None:
    async def empty(model: FakeModel) -> str:
        return mignon_report("Nothing to report.")

    async def script(model: FakeModel) -> str | None:
        turns = [await model.spawn(MACRO, f"Task {i}.", empty) for i in range(8)]
        assert not any(t.denied for t in turns)
        ninth = await model.spawn(MACRO, "Task 9.", empty)
        assert ninth.denied and "max_per_run=8" in (ninth.reason or "")
        return "{}"

    h = harness()
    h.run(script)
    assert [n for n, _ in h.world.calls].count("Agent") == 8


def test_the_orchestrator_assigns_each_mignon_a_model_from_the_allowlist(
    harness: Callable[..., Harness],
) -> None:
    async def empty(model: FakeModel) -> str:
        return mignon_report("Nothing to report.")

    async def script(model: FakeModel) -> str | None:
        cheap = await model.spawn("mignon-macro--claude-haiku-4-5", "Calendar.", empty)
        assert cheap.output["kind"] == "validated", cheap.output
        outside = await model.spawn("mignon-macro--claude-sonnet-5", "Calendar.", empty)
        assert outside.denied and "allowed model" in (outside.reason or "")
        return await dry_run_script(model)

    h = harness(MIGNON_AGENT_MODELS=f"claude-haiku-4-5,{E2E_MODEL}")
    assert h.run(script) == 0, h.notifier.alert_kinds()
    (cli,) = h.clis
    agents = cli.options.agents or {}
    assert agents["mignon-macro--claude-haiku-4-5"].model == "claude-haiku-4-5"
    assert agents[MARKET].model == E2E_MODEL and len(agents) == 6
    prompt = str(cli.options.system_prompt)
    assert "- `claude-haiku-4-5`: $1/$5 per 1M tokens" in prompt
    with h.conn() as c:
        types = [r.identity.agent_type for r in tool_call_records(c, h.run_id)]
    assert "mignon-macro--claude-haiku-4-5" not in types  # the empty Mignon made no call
    assert MARKET in types  # the dry-run research ran on the session model


# -- run-summary email (ADR-0029) ----------------------------------------------------------------


def _summary_rows(h: Harness) -> list[ledger_evidence.StoredAlert]:
    with h.conn() as c:
        rows = ledger_evidence.alerts_for_run(c, h.run_id)
    return [r for r in rows if r.alert_kind == "run_summary_email"]


def test_completed_session_sends_one_summary_email(harness: Callable[..., Harness]) -> None:
    h = harness()
    assert h.run() == 0, h.notifier.alert_kinds()
    (summary,) = h.mailer.summaries
    assert summary.status is RunStatus.COMPLETED and summary.run_id == str(h.run_id)
    assert summary.effective_execution_mode is ExecutionMode.OFF
    assert summary.order_venue is OrderVenue.SIMULATED
    assert summary.record is not None and len(summary.record.decisions) == 1
    assert summary.audit_status == "completed" and summary.audit_violations == 0
    # ADR-0038: an on-demand dry run schedules nothing.
    assert summary.next_run_at is None and summary.next_run_source is None
    (row,) = _summary_rows(h)
    assert row.delivery_status is ledger_evidence.DeliveryStatus.SENT
    assert row.payload["provider_message_id"] == "em_1"
    assert row.dedup_key == f"run-summary/{h.run_id}"


def test_summary_email_reports_the_agents_next_run(harness: Callable[..., Harness]) -> None:
    h = harness(**LIVE)
    chosen = SESSION_TIME + timedelta(minutes=42)
    assert h.run(_next_run_script(chosen.isoformat().replace("+00:00", "Z"))) == 0
    (summary,) = h.mailer.summaries
    assert summary.next_run_at == chosen and summary.next_run_source == "agent"
    assert summary.next_run_rationale == "Scripted schedule."


def test_failed_session_still_sends_a_summary_email(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str:
        await research(model)
        return "I would sell the AAPL put."  # not JSON

    h = harness()
    assert h.run(script) == 1
    (summary,) = h.mailer.summaries
    assert summary.status is RunStatus.FAILED and summary.reason == "invalid_agent_output"
    assert "invalid_agent_output" in summary.alerts


def test_timed_out_session_sends_a_summary_email(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str | None:
        await model.call("mcp__robinhood__get_option_quotes", {"instrument_ids": ["x"]})
        h.clock.advance(2000)
        for _ in range(200):
            if model.interrupted:
                break
            await asyncio.sleep(0.01)
        return None

    h = harness()
    assert h.run(script) == 2
    (summary,) = h.mailer.summaries
    assert summary.status is RunStatus.TIMED_OUT


@pytest.mark.parametrize(
    "setup",
    ["kill_switch", "market_closed", "needs_auth", "not_agentic"],
)
def test_runs_without_a_session_send_no_summary_email(
    harness: Callable[..., Harness], setup: str
) -> None:
    if setup == "kill_switch":
        h = harness(KILL_SWITCH=True)
    elif setup == "market_closed":
        h = harness(at=WEEKEND_TIME, **LIVE)
    elif setup == "needs_auth":
        h = harness(ROBINHOOD_MCP_ACCESS_TOKEN=None)
    else:
        h = harness()
        listing = h.world.handlers["robinhood"]["get_accounts"]

        def not_agentic(args: dict[str, Any]) -> dict[str, Any]:
            response = listing(args)
            payload = json.loads(response["content"][0]["text"])
            payload["data"]["accounts"][0]["agentic_allowed"] = False
            return {"content": [{"type": "text", "text": json.dumps(payload)}]}

        h.world.handlers["robinhood"]["get_accounts"] = not_agentic
    h.run()
    assert h.clis == []
    assert h.mailer.summaries == []
    assert _summary_rows(h) == []


def test_lock_contention_sends_no_summary_email(harness: Callable[..., Harness]) -> None:
    h = harness()
    with h.conn() as holder:
        assert try_advisory_lock(holder, RUN_LOCK_OBJID[AppEnv.LOCAL])
        assert h.run() == 0
    assert h.mailer.summaries == []


def test_not_due_tick_sends_no_summary_email(harness: Callable[..., Harness]) -> None:
    h = harness(**LIVE)
    chosen = SESSION_TIME + timedelta(minutes=42)
    assert h.run(_next_run_script(chosen.isoformat().replace("+00:00", "Z"))) == 0
    _tick(h, SESSION_TIME + timedelta(minutes=40))
    assert h.run() == 0
    assert h.status() is RunStatus.SKIPPED_NOT_DUE
    assert len(h.mailer.summaries) == 1  # only the first, due run


def test_a_failing_mailer_never_changes_the_run_outcome(harness: Callable[..., Harness]) -> None:
    h = harness()
    h.mailer.error = RuntimeError("mail down")
    assert h.run() == 0
    assert h.status() is RunStatus.COMPLETED
    assert len(h.mailer.summaries) == 1
    assert _summary_rows(h) == []  # nothing came back to record


def test_a_failed_delivery_is_recorded_as_failed(harness: Callable[..., Harness]) -> None:
    from wheelta_robinhood_agent.integrations.notifications.email import (
        EmailDeliveryResult,
        EmailDeliveryStatus,
    )

    h = harness()
    h.mailer.result = EmailDeliveryResult(
        status=EmailDeliveryStatus.FAILED, subject="s", attempts=3, error="http_503"
    )
    assert h.run() == 0
    (row,) = _summary_rows(h)
    assert row.delivery_status is ledger_evidence.DeliveryStatus.FAILED
    assert row.payload["error"] == "http_503"


# -- account placeholder (ADR-0030) ---------------------------------------------------------------


def test_account_placeholder_reaches_the_broker_as_the_configured_number(
    harness: Callable[..., Harness],
) -> None:
    h = harness()
    assert h.run() == 0, h.notifier.alert_kinds()
    account_reads = [
        (tool, args)
        for server, tool, args in h.world.upstream_calls
        if server == "robinhood" and "account_number" in args
    ]
    assert {t for t, _ in account_reads} == {
        "get_portfolio",
        "get_option_positions",
        "get_option_orders",
    }
    assert all(a["account_number"] == ACCOUNT_NUMBER for _, a in account_reads)
    # The model sent only the placeholder and never received the number back.
    (cli,) = h.clis
    model_sent = [args for name, args in h.world.calls if name == "mcp__robinhood__get_portfolio"]
    assert model_sent == [{"account_number": AGENTIC_ACCOUNT_PLACEHOLDER}]
    assert all(ACCOUNT_NUMBER not in json.dumps(t.output, default=str) for t in cli.turns)
    with h.conn() as c:
        dump = json.dumps(
            [
                c.execute("SELECT arguments_redacted::text FROM tool_calls").fetchall(),
                c.execute(
                    "SELECT effective_arguments_redacted::text, payload::text FROM tool_call_events"
                ).fetchall(),
            ]
        )
        (stored,) = ledger_evidence.run_records_for_run(c, h.run_id)
    assert ACCOUNT_NUMBER not in dump
    assert stored.record.decisions  # account state was available, so the run decided


def test_another_account_number_is_still_denied(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str | None:
        other = await model.call(
            "mcp__robinhood__get_portfolio", {"account_number": OTHER_ACCOUNT_NUMBER}
        )
        assert other.denied and "does not match" in (other.reason or "")
        last_four = await model.call(
            "mcp__robinhood__get_portfolio", {"account_number": ACCOUNT_NUMBER[-4:]}
        )
        assert last_four.denied
        return await dry_run_script(model)

    h = harness()
    assert h.run(script) == 0, h.notifier.alert_kinds()
    sent = [a for _, t, a in h.world.upstream_calls if t == "get_portfolio"]
    assert sent == [{"account_number": ACCOUNT_NUMBER}]


def test_trace_script_prints_the_run_and_the_listing(
    harness: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """scripts/trace_run.py: read-only trace and listing from the ledger (OPERATIONS.md)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "trace_run.py"
    spec = importlib.util.spec_from_file_location("trace_run", path)
    assert spec is not None and spec.loader is not None
    trace_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace_run)

    h = harness()
    assert h.run() == 0, h.notifier.alert_kinds()
    monkeypatch.setattr(trace_run, "load_settings", lambda: h.settings)
    assert trace_run.main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert str(h.run_id) in listing and "simulated" in listing
    assert trace_run.main([str(h.run_id)]) == 0
    assert "### OPEN_CSP AAPL" in capsys.readouterr().out
    assert trace_run.main(["--slot", slot_for(h.clock.now).isoformat(), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == str(h.run_id)
    assert trace_run.main(["--slot", "2020-01-01T00:00:00+00:00"]) == 1
    assert "no run at" in capsys.readouterr().err


def test_a_price_walk_audits_each_step_against_state_that_includes_the_last(
    harness: Callable[..., Harness],
) -> None:
    """The second step's pre-order state reflects the first step's place and confirmed cancel,
    so V1/V6/V7 judge it instead of reporting the state incomplete (run_loader broker_states)."""
    h = harness()
    h.world = simulated_world(h.clock.now)
    script = simulated_price_walk_script(lambda: h.clock.advance(1))
    assert h.run(script, mappers=SIMULATED_MAPPERS) == 0, h.notifier.alert_kinds()
    with h.conn() as c:
        findings = ledger_evidence.audit_findings_for_run(c, h.run_id)
        calls = tool_call_records(c, h.run_id)
    places = [r for r in calls if r.identity.tool == "place_option_order"]
    assert len(places) == 2
    second = str(places[1].identity.tool_call_id)
    mine = [f for f in findings if places[1].identity.tool_call_id in f.tool_call_ids]
    assert mine, second
    assert not [f for f in findings if "earlier mutation" in f.detail]
    v62 = [f for f in mine if f.check_id.value == "V6" and f.sub_item == "2"]
    assert [f.outcome.value for f in v62] == ["pass"]
    v13 = [f for f in mine if f.check_id.value == "V1" and f.sub_item == "3"]
    assert [f.outcome.value for f in v13] == ["violation"]  # 74,000 collateral on 30,000 cash


def test_a_dry_run_adds_no_position_notes(harness: Callable[..., Harness]) -> None:
    """ADR-0039: a dry run's HOLD on a real position is not recorded as its memory."""
    import json

    from wheelta_robinhood_agent.agent.account_scope import account_scope_id
    from wheelta_robinhood_agent.domain.enums import StrategyKind
    from wheelta_robinhood_agent.domain.options import OccSymbol
    from wheelta_robinhood_agent.domain.positions import PositionInstrument
    from wheelta_robinhood_agent.ledger.positions import open_position, position_book

    h = harness()
    scope = account_scope_id(h.settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER)
    earlier = h.clock.now - timedelta(hours=1)
    with h.conn() as c:
        prior = open_run_slot(c, AppEnv.LOCAL, slot_for(earlier)).run_id
        position_id = open_position(
            c,
            run_id=prior,
            account_scope_id=scope,
            underlying="AAPL",
            strategy=StrategyKind.CASH_SECURED_PUT,
            instruments=[
                PositionInstrument(
                    occ_symbol=OccSymbol.parse("AAPL  261016P00150000"),
                    broker_instrument_id="inst-aapl-150p",
                    short_quantity=1,
                )
            ],
            observed_at=earlier,
            imported=True,
        )

    async def script(model: FakeModel) -> str:
        hold = {
            "action": "HOLD",
            "target_ref": f"position:{position_id}",
            "replacement_ref": None,
            "funding_close_refs": [],
            "proposed_legs": [],
            "execution_refs": [],
            "rationale": "Dry-run judgment.",
            "thesis": None,
            "invalidation_conditions": [],
            "evidence_refs": [],
        }
        return json.dumps(
            {
                "decisions": [hold],
                "cancellation_rationales": [],
                "unresolved_questions": [],
                "next_run": None,
            }
        )

    assert h.run(script) == 0, h.notifier.alert_kinds()
    with h.conn() as c:
        (entry,) = position_book(c, scope, as_of=h.clock.now).entries
        (stored,) = ledger_evidence.run_records_for_run(c, h.run_id)
    assert [d.rationale for d in stored.record.decisions] == ["Dry-run judgment."]
    assert entry.notes == ()
