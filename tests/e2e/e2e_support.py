"""Shared e2e helpers: fixed session times, a settable clock, a recording notifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from wheelta_robinhood_agent.integrations.notifications.delivery import (
    DeliveryOutcome,
    DeliveryResult,
)
from wheelta_robinhood_agent.observability.alerts import AlertPayload, HeartbeatPayload

# Wednesday 2026-09-23 11:30 America/New_York: inside the NYSE regular session.
SESSION_TIME = datetime(2026, 9, 23, 15, 30, tzinfo=UTC)
# Saturday 2026-09-26: no session.
WEEKEND_TIME = datetime(2026, 9, 26, 15, 30, tzinfo=UTC)


@dataclass
class FakeClock:
    now: datetime

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


@dataclass
class RecordingNotifier:
    alerts: list[AlertPayload] = field(default_factory=list)
    heartbeats: list[HeartbeatPayload] = field(default_factory=list)

    def alert(self, payload: AlertPayload) -> DeliveryResult:
        self.alerts.append(payload)
        return DeliveryResult(payload_kind="alert", outcome=DeliveryOutcome.DELIVERED, attempts=1)

    def heartbeat(self, payload: HeartbeatPayload) -> DeliveryResult:
        self.heartbeats.append(payload)
        return DeliveryResult(
            payload_kind="heartbeat", outcome=DeliveryOutcome.DELIVERED, attempts=1
        )

    def alert_kinds(self) -> list[str]:
        return [a.kind.value for a in self.alerts]
