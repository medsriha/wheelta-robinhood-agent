"""Leave no owned order working when the session ends (ADR-0050).

Two session-level controls, both enforced through `hooks.OutputRepairGate.restrict`:

- **Wind-down.** Once the session budget has at most `ORDER_WIND_DOWN_SECONDS` left, only
  `CLEANUP_TOOLS` (order and position reads, cancel) pass the hook. A new placement, review,
  workspace write, Mignon, or other read is denied with `WIND_DOWN_DENIAL`, so the agent
  spends the remaining time cancelling and returning its output.
- **Cleanup turns.** When the agent returns its final output while the ledger still projects
  an owned order as unresolved (`ledger.orders.owned_unresolved_orders` in the venue's order
  scope), the session sends up to `MAX_ORDER_CLEANUPS` follow-up turns listing each order and
  what to do with it (`cleanup_action`), under the same restriction.

What the agent may do per order follows the prompt's rule 6: an order whose cancellation was
already sent (accepted or uncertain) is only read, never cancelled again. Every other order is
read first and cancelled only if that read shows it still working: the ledger projection is
only as fresh as the last read, and cancelling an order that already filled or expired is a
cancel error, which ends placement (rule 6). Code never cancels an order itself: the agent
does, through the hooks, and the broker ledger records the outcome. An order still unresolved
after the last turn stays open; the orchestrator alerts (R19).

Which orders count (`needs_cleanup`): those this run placed, those a call of this run observed
or cancelled, and older ones that can still be working at the broker (time in force other
than `gfd`, or placed on today's New York date). An older day order the ledger never saw
resolve (paged order history, ADR-0034) has expired at the broker; it is not sent back to the
agent every run.
"""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Final
from zoneinfo import ZoneInfo

import psycopg

from wheelta_robinhood_agent.agent.simulated_broker import simulated_scope_id
from wheelta_robinhood_agent.domain.enums import AttemptStatus, CancellationStatus, OrderVenue
from wheelta_robinhood_agent.domain.orders import OrderRecord
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME as ROBINHOOD
from wheelta_robinhood_agent.ledger import orders as ledger_orders

Conn = psycopg.Connection[tuple[object, ...]]

# Follow-up turns asking the agent to cancel its working orders before it finishes.
MAX_ORDER_CLEANUPS: Final = 2
# Session time kept for cancelling and returning the output. An operational margin sized for
# a few order reads, cancels, and one final response; not a trading rule.
ORDER_WIND_DOWN_SECONDS: Final = 180.0
CLEANUP_TOOLS: Final = frozenset(
    f"mcp__{ROBINHOOD}__{tool}"
    for tool in ("get_option_orders", "get_option_positions", "cancel_option_order")
)
WIND_DOWN_DENIAL: Final = (
    "the run is winding down: only get_option_orders, get_option_positions, and "
    "cancel_option_order are allowed. Cancel your working orders, confirm them, and return "
    "your final output"
)
CLEANUP_DENIAL: Final = (
    "orders are being cleaned up: only get_option_orders, get_option_positions, and "
    "cancel_option_order are allowed"
)


_MARKET_TZ: Final = ZoneInfo("America/New_York")


class CleanupAction(StrEnum):
    CANCEL = "cancel"  # last seen working, no cancel sent: read, then cancel if still working
    CONFIRM = "confirm"  # a cancel was already sent: read only, never cancel again
    READ = "read"  # state unknown: read first; cancel only if the read shows it working


def order_scope(venue: OrderVenue | None, account_scope_id: str, run_id: uuid.UUID) -> str | None:
    """The account scope the venue records owned orders under, or None without a venue."""
    if venue is OrderVenue.BROKER:
        return account_scope_id
    if venue is OrderVenue.SIMULATED:
        return simulated_scope_id(run_id)
    return None


def unresolved_orders(conn: Conn, scope: str | None) -> tuple[OrderRecord, ...]:
    """Every owned order the ledger projects as unresolved in `scope` (none without one)."""
    return () if scope is None else ledger_orders.owned_unresolved_orders(conn, scope)


def needs_cleanup(
    record: OrderRecord,
    *,
    run_id: uuid.UUID,
    run_tool_call_ids: Collection[uuid.UUID],
    as_of: datetime,
) -> bool:
    """Whether an unresolved owned order can still be working and belongs in cleanup
    (module docstring)."""
    intent = record.intent
    if intent is None or intent.run_id == run_id:
        return True
    if any(s.tool_call_id in run_tool_call_ids for s in record.status_history):
        return True
    if any(c.cancel_tool_call_id in run_tool_call_ids for c in record.cancellations):
        return True
    if (intent.time_in_force_raw or "").strip().lower() != "gfd":
        return True
    placed = intent.requested_at.astimezone(_MARKET_TZ).date()
    return placed == as_of.astimezone(_MARKET_TZ).date()


def cleanup_action(record: OrderRecord) -> CleanupAction:
    """What the agent should do with one unresolved order (module docstring)."""
    if any(
        c.status in (CancellationStatus.PENDING, CancellationStatus.UNKNOWN)
        for c in record.cancellations
    ):
        return CleanupAction.CONFIRM
    if record.broker_order is None or record.status is AttemptStatus.UNKNOWN:
        return CleanupAction.READ
    return CleanupAction.CANCEL


_ACTION_TEXT: Final = {
    CleanupAction.CANCEL: (
        "last seen working: read get_option_orders; if it is still working, cancel it and "
        "confirm its terminal state"
    ),
    CleanupAction.CONFIRM: (
        "a cancel was already sent: read get_option_orders to confirm; do not cancel it again"
    ),
    CleanupAction.READ: (
        "state unknown: read get_option_orders; cancel only if it is working and you have "
        "sent no cancel for it"
    ),
}


def _describe(record: OrderRecord) -> str:
    intent = record.intent
    parts = [
        f"broker order {record.broker_order.broker_order_id}"
        if record.broker_order is not None
        else "no broker order known"
    ]
    if intent is not None:
        parts.append(
            " ".join(
                str(v)
                for v in (
                    intent.side_raw,
                    intent.quantity,
                    intent.occ_symbol or intent.broker_instrument_id,
                    f"@ {intent.limit_price}" if intent.limit_price is not None else None,
                )
                if v is not None
            )
        )
    parts.append(f"status {record.status.value}")
    return ", ".join(parts)


def cleanup_message(records: Sequence[OrderRecord], attempt: int) -> str:
    """The follow-up turn sent when the agent returns output with orders still unresolved."""
    lines = [f"- {_describe(r)}: {_ACTION_TEXT[cleanup_action(r)]}" for r in records]
    return (
        f"You returned your final output while orders of yours are not confirmed terminal "
        f"(cleanup {attempt} of {MAX_ORDER_CLEANUPS}; orders.working):\n"
        + "\n".join(lines)
        + "\n\nOnly get_option_orders, get_option_positions, and cancel_option_order are "
        "allowed now; placing is not. Rule 6 still applies: never repeat an uncertain "
        "cancellation, and after a cancel error read only. Count fills that happened during "
        "cancellation. Then return your complete AgentDecisionOutput again, associating the "
        "cancel calls with their decisions (execution_refs) or listing them in "
        "cancellation_rationales."
    )


__all__ = [
    "CLEANUP_DENIAL",
    "CLEANUP_TOOLS",
    "MAX_ORDER_CLEANUPS",
    "ORDER_WIND_DOWN_SECONDS",
    "WIND_DOWN_DENIAL",
    "CleanupAction",
    "cleanup_action",
    "cleanup_message",
    "needs_cleanup",
    "order_scope",
    "unresolved_orders",
]
