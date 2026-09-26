from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AuditCheck,
    AuditOutcome,
    ExecutionMode,
    MarketSession,
    RunStatus,
    SourceStatus,
)
from wheelta_robinhood_agent.domain.run import (
    AuditFinding,
    ConfigSnapshot,
    MarketSessionInfo,
    Run,
    RunIdentity,
    SourceStatusObservation,
)
from wheelta_robinhood_agent.domain.run_identity import run_id_for

SLOT = datetime(2026, 9, 25, 15, tzinfo=UTC)
OFF, LIVE = ExecutionMode.OFF, ExecutionMode.LIVE


def _identity() -> RunIdentity:
    return RunIdentity(
        run_id=run_id_for(AppEnv.STAGING, SLOT), environment=AppEnv.STAGING, slot=SLOT
    )


def test_identity() -> None:
    assert _identity().run_id == run_id_for(AppEnv.STAGING, SLOT)
    with pytest.raises(ValidationError, match="does not match"):
        RunIdentity(run_id=uuid4(), environment=AppEnv.STAGING, slot=SLOT)
    with pytest.raises(ValidationError, match="whole UTC hour"):
        RunIdentity(run_id=uuid4(), environment=AppEnv.STAGING, slot=SLOT + timedelta(minutes=1))


def _config(**kw: object) -> ConfigSnapshot:
    base: dict[str, object] = {
        "requested_execution_mode": OFF,
        "effective_execution_mode": OFF,
        "execution_armed": False,
        "kill_switch": False,
        "rules_version": "5",
        "rules_hash": "sha256:abc",
        "values": {"effective_execution_mode": "off", "app_env": "staging"},
    }
    base.update(kw)
    return ConfigSnapshot.model_validate(base)


def test_config_snapshot() -> None:
    _config()
    _config(requested_execution_mode=LIVE, execution_armed=True, effective_execution_mode=OFF)
    _config(
        requested_execution_mode=LIVE,
        execution_armed=True,
        effective_execution_mode=LIVE,
        values={},
    )
    with pytest.raises(ValidationError, match="armed"):
        _config(requested_execution_mode=LIVE, effective_execution_mode=LIVE, values={})
    with pytest.raises(ValidationError, match="disagrees"):
        _config(values={"kill_switch": True})


def _run(**kw: object) -> Run:
    base: dict[str, object] = {
        "identity": _identity(),
        "started_at": SLOT,
        "finished_at": SLOT + timedelta(minutes=5),
        "status": RunStatus.COMPLETED,
        "git_sha": "abc123",
        "image_digest": None,
        "config_snapshot": _config(),
        "market_session": MarketSessionInfo(
            session=MarketSession.REGULAR, calendar_provenance="exchange_calendars XNYS"
        ),
        "model_id": "claude-x",
    }
    base.update(kw)
    return Run.model_validate(base)


def test_run() -> None:
    r = _run()
    assert r.run_id == _identity().run_id
    assert r.slot == SLOT
    _run(status=RunStatus.RUNNING, finished_at=None)
    with pytest.raises(ValidationError, match="terminal"):
        _run(status=RunStatus.RUNNING)
    with pytest.raises(ValidationError, match="terminal"):
        _run(finished_at=None)
    with pytest.raises(ValidationError, match="finished before"):
        _run(finished_at=SLOT - timedelta(seconds=1))
    with pytest.raises(ValidationError, match="killed"):
        _run(status=RunStatus.SKIPPED_KILLED)
    obs = SourceStatusObservation(
        server="robinhood", status=SourceStatus.NEEDS_AUTH, observed_at=SLOT
    )
    with pytest.raises(ValidationError, match="duplicate"):
        _run(source_status=(obs, obs))


def _finding(**kw: object) -> AuditFinding:
    base: dict[str, object] = {
        "finding_id": uuid4(),
        "run_id": uuid4(),
        "check_id": AuditCheck.V2,
        "outcome": AuditOutcome.UNVERIFIABLE,
        "effective_execution_mode": OFF,
        "detail": "dry_run_no_execution",
        "audit_version": "1",
        "context_hash": "sha256:x",
    }
    base.update(kw)
    return AuditFinding.model_validate(base)


def test_audit_finding() -> None:
    f = _finding(leg_ref="leg:0:0", attempt_index=0, rule_value="0.25")
    assert f.rule_value == "0.25"
    with pytest.raises(ValidationError, match="correct itself"):
        _finding(finding_id=f.finding_id, corrects_finding_id=f.finding_id)
    with pytest.raises(ValidationError, match="leg_ref"):
        _finding(attempt_index=0)
    t = uuid4()
    with pytest.raises(ValidationError, match="duplicate"):
        _finding(tool_call_ids=(t, t))
