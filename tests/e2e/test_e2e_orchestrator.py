"""Full runs of `orchestrator.run_once` against fake MCP servers and a scripted fake CLI.

Real pieces: Settings, rules, prompt v5, the orchestrator, `ClaudeSDKClient`/`Query` (hook and
in-process MCP dispatch), our hooks, result boundary, facts tool, web cache, assembler, audit,
and a throwaway Postgres ledger. Fake pieces: the CLI transport (e2e_fake_cli.py), the remote
servers and their fixture result mappers (e2e_fakes.py), the clock, and the notifier.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from e2e_fake_cli import FakeCli, FakeModel, factory
from e2e_fakes import (
    FIXTURE_MAPPERS,
    FIXTURE_SCOPE_TABLE,
    RAW_MARKER,
    build_world,
    dry_run_script,
    research,
)
from e2e_support import SESSION_TIME, WEEKEND_TIME, FakeClock, RecordingNotifier

from wheelta_robinhood_agent.agent.session import plan_session
from wheelta_robinhood_agent.config.prompts import load_prompt
from wheelta_robinhood_agent.config.rules import load_rules
from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    ExecutionMode,
    RunStatus,
    ToolCallStatus,
)
from wheelta_robinhood_agent.domain.events import RunEventType
from wheelta_robinhood_agent.domain.orders import ReasonCode
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

RULES = load_rules()
TEMPLATE = load_prompt()
ORDER_TOOLS = {ROBINHOOD_REGISTRY.qualified(n) for n in LIVE_ORDER_TOOLS}
VERIFIED_RH = ROBINHOOD_REGISTRY.model_copy(update={"verified": True})
Script = Callable[[FakeModel], Any]


class Harness:
    def __init__(self, settings: Settings, notifier: RecordingNotifier, clock: FakeClock) -> None:
        self.settings = settings
        self.notifier = notifier
        self.clock = clock
        self.clis: list[FakeCli] = []
        self.world = build_world(clock.now)
        self.run_id = run_id_for(AppEnv.LOCAL, slot_for(clock.now))

    def deps(self, script: Script, **overrides: Any) -> OrchestratorDeps:
        values: dict[str, Any] = {
            "notifier": self.notifier,
            "clock": self.clock,
            "transport_factory": factory(self.world, script, self.clis),
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

    def run(self, script: Script = dry_run_script, **overrides: Any) -> int:
        return run_once(self.settings, RULES, TEMPLATE, self.deps(script, **overrides))

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
    h = harness(at=WEEKEND_TIME)
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


def test_robinhood_needs_auth_at_connect_sends_no_query(harness: Callable[..., Harness]) -> None:
    h = harness()
    h.world.statuses["robinhood"] = "needs-auth"
    assert h.run() == 1
    assert len(h.clis) == 1 and h.clis[0].user_messages == []
    assert "robinhood_needs_auth" in h.notifier.alert_kinds()
    assert h.status() is RunStatus.FAILED


def test_production_defaults_withhold_every_remote_source(harness: Callable[..., Harness]) -> None:
    h = harness()
    code = h.run(
        remote_boundary_accepted=False,
        registries=(ROBINHOOD_REGISTRY, WHEELTA_REGISTRY),
    )
    assert code == 1 and h.clis == []
    statuses = {e["server"]: e["status"] for e in h.events(RunEventType.SOURCE_STATUS)}
    assert statuses == {"robinhood": "disabled", "wheelta": "disabled"}


# -- the dry run ----------------------------------------------------------------------------------


def test_dry_run_produces_a_dry_run_proposal_and_audit(harness: Callable[..., Harness]) -> None:
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
    (attempt,) = leg.attempts
    assert attempt.status is AttemptStatus.NOT_PLACED
    assert ReasonCode.DRY_RUN in attempt.reason_codes
    assert attempt.requested_quantity == 2  # min(cash 30000, cap 10) / (150 x 100)
    assert attempt.place_tool_call_id is None and attempt.broker_order_id is None
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
    assert len(calls) == 6


# -- tool exposure ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "armed"), [("off", False), ("live", False), ("live", True), ("bogus", True)]
)
def test_order_tools_absent_in_every_mode(
    harness: Callable[..., Harness], mode: str, armed: bool
) -> None:
    async def script(model: FakeModel) -> str | None:
        turn = await model.call(
            "mcp__robinhood__place_option_order", {"account_number": "5550001234"}
        )
        assert turn.denied
        return await dry_run_script(model)

    h = harness(EXECUTION_MODE=mode, EXECUTION_ARMED=armed)
    assert h.run(script) == 0
    (cli,) = h.clis
    assert not ORDER_TOOLS & cli.visible_tools()
    assert not ORDER_TOOLS & set(cli.options.allowed_tools)
    assert ORDER_TOOLS <= set(cli.options.disallowed_tools)
    assert all(name != "mcp__robinhood__place_option_order" for name, _ in h.world.calls)
    with h.conn() as c:
        place = [
            r for r in tool_call_records(c, h.run_id) if r.identity.tool == "place_option_order"
        ]
    assert [r.status for r in place] == [ToolCallStatus.DENIED]


def test_plan_refuses_an_effective_live_mode() -> None:
    from wheelta_robinhood_agent.agent.session import SessionPlanError

    with pytest.raises(SessionPlanError):
        plan_session(
            effective_mode=ExecutionMode.LIVE,
            workspace_writes=False,
            sources=(),
            observed_at=SESSION_TIME,
        )


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
    async def script(model: FakeModel) -> str | None:
        first = await model.call("WebSearch", {"query": "AAPL earnings date"})
        assert not first.denied
        second = await model.call("WebSearch", {"query": "  aapl   EARNINGS date "})
        assert second.denied and "web_cache_lookup" in (second.reason or "")
        cached = await model.call("mcp__wra_local__web_cache_lookup", {"ticker": "AAPL"})
        assert cached.data["entries"][0]["query"] == "AAPL earnings date"
        return await dry_run_script(model)

    h = harness()
    assert h.run(script) == 0
    assert [n for n, _ in h.world.calls].count("WebSearch") == 1


# -- runtime stop ---------------------------------------------------------------------------------


def test_sigterm_latches_stop_and_interrupts_the_session(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str | None:
        await model.call("mcp__robinhood__get_option_chains", {"symbol": "AAPL"})
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
    assert tools == ["get_option_chains"]


def test_deadline_times_out_the_session(harness: Callable[..., Harness]) -> None:
    async def script(model: FakeModel) -> str | None:
        await model.call("mcp__robinhood__get_option_chains", {"symbol": "AAPL"})
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
