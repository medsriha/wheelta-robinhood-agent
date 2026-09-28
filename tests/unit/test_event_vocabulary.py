"""The domain event enums equal the event_type CHECK sets of the migrated schema.

0001_initial.sql creates each set inline; a later migration may replace one with
`ADD CONSTRAINT <table>_event_type_check CHECK (event_type IN (...))` (0004 does for
run_events). The last definition in migration order wins.
"""

import re
from enum import StrEnum
from pathlib import Path

import pytest

from wheelta_robinhood_agent.domain.events import (
    OrderEventType,
    OrderIntentEventType,
    PositionEventType,
    RunEventType,
    ToolCallEventType,
    WorkspaceEventType,
)

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
MIGRATION = MIGRATIONS / "0001_initial.sql"

VOCABULARIES: dict[str, type[StrEnum]] = {
    "run_events": RunEventType,
    "tool_call_events": ToolCallEventType,
    "order_intent_events": OrderIntentEventType,
    "order_events": OrderEventType,
    "position_events": PositionEventType,
    "workspace_events": WorkspaceEventType,
}

_TABLE = re.compile(r"CREATE TABLE (\w+) \((.*?)\n\);", re.S)
_CHECK = re.compile(r"event_type\s+text\s+NOT NULL CHECK \(event_type IN \((.*?)\)\)", re.S)
_REPLACED = re.compile(
    r"ALTER TABLE (\w+) ADD CONSTRAINT \1_event_type_check CHECK \(event_type IN \((.*?)\)\)",
    re.S,
)


def _event_type_checks() -> dict[str, tuple[str, ...]]:
    sql = MIGRATION.read_text()
    out: dict[str, tuple[str, ...]] = {}
    for name, body in _TABLE.findall(sql):
        match = _CHECK.search(body)
        if match is not None:
            out[name] = tuple(re.findall(r"'([^']*)'", match.group(1)))
    for later in sorted(MIGRATIONS.glob("*.sql")):
        if later == MIGRATION:
            continue
        for name, values in _REPLACED.findall(later.read_text()):
            assert name in out, f"{later.name} replaces an unknown table's event types"
            out[name] = tuple(re.findall(r"'([^']*)'", values))
    return out


def test_every_event_table_has_exactly_one_domain_enum() -> None:
    assert set(_event_type_checks()) == set(VOCABULARIES)


@pytest.mark.parametrize("table", sorted(VOCABULARIES))
def test_domain_enum_equals_sql_check(table: str) -> None:
    sql_values = _event_type_checks()[table]
    assert len(sql_values) == len(set(sql_values))
    assert [m.value for m in VOCABULARIES[table]] == list(sql_values)
