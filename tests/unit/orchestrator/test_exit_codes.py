import pytest

from wheelta_robinhood_agent.domain.enums import RunStatus
from wheelta_robinhood_agent.orchestrator.exit_codes import ExitCodeUndefined, exit_code_for

EXPECTED = {
    RunStatus.COMPLETED: 0,
    RunStatus.SKIPPED_CONCURRENT: 0,
    RunStatus.SKIPPED_MARKET_CLOSED: 0,
    RunStatus.SKIPPED_KILLED: 0,
    RunStatus.FAILED: 1,
    RunStatus.TIMED_OUT: 2,
    RunStatus.STOPPED: 3,
}
UNDEFINED = {RunStatus.RUNNING}


def test_every_status_is_classified() -> None:
    assert set(EXPECTED) | UNDEFINED == set(RunStatus)


@pytest.mark.parametrize(("status", "code"), EXPECTED.items())
def test_mapping(status: RunStatus, code: int) -> None:
    assert exit_code_for(status) == code


@pytest.mark.parametrize("status", sorted(UNDEFINED))
def test_undefined_statuses_raise(status: RunStatus) -> None:
    with pytest.raises(ExitCodeUndefined):
        exit_code_for(status)


def test_failure_codes_distinct_and_nonzero() -> None:
    codes = [EXPECTED[s] for s in (RunStatus.FAILED, RunStatus.TIMED_OUT, RunStatus.STOPPED)]
    assert len(set(codes)) == 3
    assert 0 not in codes
