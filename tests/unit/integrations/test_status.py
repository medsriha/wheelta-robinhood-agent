"""SDK status mapping (fail closed) and per-server observations (CLAUDE.md §8)."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import SourceStatus
from wheelta_robinhood_agent.integrations.registry import DiscoveryDiff
from wheelta_robinhood_agent.integrations.status import (
    SourceObservation,
    map_sdk_status,
    observe_server,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import WHEELTA_REGISTRY

NOW = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
ALL_WHEELTA = [t.name for t in WHEELTA_REGISTRY.tools]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("connected", SourceStatus.CONNECTED),
        ("pending", SourceStatus.PENDING),
        ("needs-auth", SourceStatus.NEEDS_AUTH),
        ("failed", SourceStatus.FAILED),
        ("disabled", SourceStatus.DISABLED),
    ],
)
def test_known_statuses(raw: str, expected: SourceStatus) -> None:
    assert map_sdk_status(raw) is expected


@pytest.mark.parametrize("raw", ["Connected", "needs_auth", "ok", "", None, 1, True, ["connected"]])
def test_unknown_status_fails_closed(raw: object) -> None:
    assert map_sdk_status(raw) is SourceStatus.FAILED


def test_connected_with_full_discovery_is_available() -> None:
    obs = observe_server(
        "wheelta", "connected", NOW, registry=WHEELTA_REGISTRY, discovered_tools=ALL_WHEELTA
    )
    assert obs.discovery == DiscoveryDiff(unknown=frozenset(), missing=frozenset())
    assert obs.available


def test_missing_and_unknown_tools_recorded() -> None:
    found = [*ALL_WHEELTA[1:], "wheelta_new_tool"]
    obs = observe_server(
        "wheelta", "connected", NOW, registry=WHEELTA_REGISTRY, discovered_tools=found
    )
    assert obs.discovery is not None
    assert obs.discovery.missing == {ALL_WHEELTA[0]}
    assert obs.discovery.unknown == {"wheelta_new_tool"}
    assert not obs.available


def test_connected_without_discovery_is_not_available() -> None:
    assert not observe_server("wheelta", "connected", NOW).available


def test_pending_ignores_incomplete_tool_list() -> None:
    obs = observe_server(
        "wheelta", "pending", NOW, registry=WHEELTA_REGISTRY, discovered_tools=ALL_WHEELTA
    )
    assert obs.status is SourceStatus.PENDING
    assert obs.discovery is None
    assert not obs.available


def test_raw_status_recorded_and_truncated() -> None:
    assert observe_server("x", "weird", NOW).raw_status == "weird"
    assert observe_server("x", "y" * 500, NOW).raw_status == "y" * 64
    assert observe_server("x", 42, NOW).raw_status is None


def test_observed_at_normalized_to_utc_and_naive_rejected() -> None:
    est = timezone(timedelta(hours=-4))
    obs = observe_server("x", "failed", datetime(2026, 9, 25, 10, 0, tzinfo=est))
    assert obs.observed_at == NOW and obs.observed_at.tzinfo is UTC
    with pytest.raises(ValidationError):
        observe_server("x", "failed", datetime(2026, 9, 25, 10, 0))  # noqa: DTZ001


def test_discovery_rejected_unless_connected() -> None:
    with pytest.raises(ValidationError):
        SourceObservation(
            server="x",
            status=SourceStatus.FAILED,
            observed_at=NOW,
            discovery=DiscoveryDiff(unknown=frozenset(), missing=frozenset()),
        )


def test_observation_is_frozen() -> None:
    obs = observe_server("x", "failed", NOW)
    with pytest.raises(ValidationError):
        obs.status = SourceStatus.CONNECTED  # type: ignore[misc]
