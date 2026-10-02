"""RunEvidence joins shares-only and options-only positions reads (ADR-0031). Pure: no ledger."""

import uuid
from datetime import UTC, datetime, timedelta

from wheelta_robinhood_agent.agent.facts_tool import RunEvidence
from wheelta_robinhood_agent.agent.mapped_evidence import MappedEvidence
from wheelta_robinhood_agent.domain.enums import PositionsCoverage
from wheelta_robinhood_agent.domain.facts_compute import PositionsRead

T0 = datetime(2026, 9, 28, 18, 0, tzinfo=UTC)


def _read(n: int, covers: set[PositionsCoverage], age: int = 0) -> PositionsRead:
    return PositionsRead(
        evidence_id=uuid.UUID(int=n),
        as_of=T0 - timedelta(seconds=age),
        source_tool_call_ids=(uuid.UUID(int=100 + n),),
        covers=frozenset(covers),
    )


def _evidence(*reads: PositionsRead) -> RunEvidence:
    return RunEvidence(tuple(MappedEvidence(positions=(r,)) for r in reads))


SHARES = {PositionsCoverage.SHARES}
OPTIONS = {PositionsCoverage.OPTIONS}


def test_both_halves_combine_into_a_complete_read() -> None:
    read = _evidence(_read(1, SHARES, age=5), _read(2, OPTIONS, age=20)).positions()
    assert read is not None and read.complete
    assert read.as_of == T0 - timedelta(seconds=20)
    assert read.source_tool_call_ids == (uuid.UUID(int=101), uuid.UUID(int=102))


def test_a_lone_half_is_returned_incomplete() -> None:
    read = _evidence(_read(1, SHARES)).positions()
    assert read is not None and not read.complete


def test_the_latest_of_each_half_is_used() -> None:
    old = _read(1, SHARES, age=90)
    new = _read(3, SHARES, age=10)
    read = _evidence(old, _read(2, OPTIONS, age=15), new).positions()
    assert read is not None and uuid.UUID(int=103) in read.source_tool_call_ids
    assert uuid.UUID(int=101) not in read.source_tool_call_ids


def test_a_newer_complete_read_wins_over_older_halves() -> None:
    full = _read(9, SHARES | OPTIONS, age=1)
    read = _evidence(_read(1, SHARES, age=50), _read(2, OPTIONS, age=50), full).positions()
    assert read == full


def test_no_reads_is_none() -> None:
    assert RunEvidence(()).positions() is None


def test_first_picks_the_earliest_complete_read() -> None:
    """The cash baseline uses the run's opening positions (ADR-0072)."""
    old = _read(1, SHARES, age=90)
    reads = _evidence(old, _read(2, OPTIONS, age=80), _read(3, SHARES, age=10))
    read = reads.positions(first=True)
    assert read is not None and read.complete
    assert read.source_tool_call_ids == (uuid.UUID(int=101), uuid.UUID(int=102))


def test_no_baseline_without_snapshot_or_orders() -> None:
    assert _evidence(_read(9, SHARES | OPTIONS)).cash_baseline() is None
