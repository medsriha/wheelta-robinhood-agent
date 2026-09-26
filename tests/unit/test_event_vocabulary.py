"""The domain event enums equal the event_type CHECK sets in migrations/0001_initial.sql."""

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

MIGRATION = Path(__file__).resolve().parents[2] / "migrations" / "0001_initial.sql"

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


def _event_type_checks() -> dict[str, tuple[str, ...]]:
    sql = MIGRATION.read_text()
    out: dict[str, tuple[str, ...]] = {}
    for name, body in _TABLE.findall(sql):
        match = _CHECK.search(body)
        if match is not None:
            out[name] = tuple(re.findall(r"'([^']*)'", match.group(1)))
    return out


def test_every_event_table_has_exactly_one_domain_enum() -> None:
    assert set(_event_type_checks()) == set(VOCABULARIES)


@pytest.mark.parametrize("table", sorted(VOCABULARIES))
def test_domain_enum_equals_sql_check(table: str) -> None:
    sql_values = _event_type_checks()[table]
    assert len(sql_values) == len(set(sql_values))
    assert [m.value for m in VOCABULARIES[table]] == list(sql_values)
