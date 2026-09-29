"""Closed sets used across the codebase (CLAUDE.md §4: enums over strings)."""

from enum import StrEnum


class ExecutionMode(StrEnum):
    """Execution mode (ADR-0006). `off` is a dry run: no order reaches the broker; `live`
    places real orders. Where an off run's orders go is its `OrderVenue` (ADR-0038)."""

    OFF = "off"
    LIVE = "live"


class OrderVenue(StrEnum):
    """Where the agent's option-order tools go this run (ADR-0038).

    BROKER: Robinhood (armed live only). SIMULATED: a dry run whose review/place/cancel calls
    the validating proxy answers in-process, never reaching Robinhood. NONE: a dry run with no
    order tools (Robinhood served directly, ADR-0019, so no simulator can intercept).
    """

    BROKER = "broker"
    SIMULATED = "simulated"
    NONE = "none"


class ToolTier(StrEnum):
    """Robinhood tool tiers (ADR-0005, amended by ADR-0006).

    R: read/research. S: Agentic workspace writes on owned objects. X: financial actions.
    D: delegation, the built-in `Agent` tool the orchestrator uses to spawn a Mignon
    (ADR-0025). EXCLUDED: out of scope (crypto); never allowlisted.
    """

    R = "R"
    S = "S"
    X = "X"
    D = "D"
    EXCLUDED = "excluded"


class MignonType(StrEnum):
    """Research subagent types the orchestrator may spawn (ADR-0025, `agent/mignons.py`)."""

    MARKET = "mignon-market"
    COMPANY = "mignon-company"
    MACRO = "mignon-macro"


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"


class StrategyKind(StrEnum):
    """The only two strategies the agent trades (CLAUDE.md §1)."""

    CASH_SECURED_PUT = "csp"
    COVERED_CALL = "cc"


class AppEnv(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class RunStatus(StrEnum):
    """Run status (INTERFACES.md "Run and RunControl")."""

    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED_CONCURRENT = "skipped_concurrent"
    SKIPPED_KILLED = "skipped_killed"
    SKIPPED_MARKET_CLOSED = "skipped_market_closed"
    # ADR-0039 (as ADR-0024 first had it): effective off outside APP_ENV=local runs nothing;
    # dry runs are local only. Reported by heartbeat before any ledger row.
    SKIPPED_DRY_RUN_NOT_LOCAL = "skipped_dry_run_not_local"
    # ADR-0028: the cron tick is before the recorded next-run time; no session starts.
    SKIPPED_NOT_DUE = "skipped_not_due"
    # ADR-0038: a scheduled tick while the effective mode is off; dry runs start only on
    # demand (`--run-now`).
    SKIPPED_DRY_RUN_NOT_REQUESTED = "skipped_dry_run_not_requested"
    STOPPED = "stopped"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class MarketSession(StrEnum):
    REGULAR = "regular"
    PRE = "pre"
    POST = "post"
    CLOSED = "closed"


class SourceStatus(StrEnum):
    """MCP server status as observed at preflight (INTERFACES.md `source_status`)."""

    PENDING = "pending"
    CONNECTED = "connected"
    NEEDS_AUTH = "needs-auth"
    FAILED = "failed"
    DISABLED = "disabled"


class DataQuality(StrEnum):
    """Quality label on every sourced value (INTERFACES.md "Evidence and citations")."""

    OK = "ok"
    STALE = "stale"
    MISSING = "missing"
    CONTRADICTORY = "contradictory"
    LOW_CONFIDENCE = "low_confidence"


class SourceTier(StrEnum):
    """Web source tiers (`data_quality.source_tiers`); tier 3 never supports a decision."""

    TIER_1 = "1"
    TIER_2 = "2"
    TIER_3 = "3"


class ToolCallStatus(StrEnum):
    """ToolCallRecord status. An unresolved dispatched financial action is UNKNOWN."""

    REQUESTED = "requested"
    DENIED = "denied"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


class DecisionAction(StrEnum):
    """AgentDecisionOutput v5 actions (OUTPUT_ASSEMBLY.md)."""

    OPEN_CSP = "OPEN_CSP"
    OPEN_CC = "OPEN_CC"
    CLOSE = "CLOSE"
    ROLL = "ROLL"
    HOLD = "HOLD"


class PositionsCoverage(StrEnum):
    """Which holdings a positions read attests completely (ADR-0031): Robinhood reports shares
    (`get_equity_positions`) and options (`get_option_positions`) in separate tools."""

    SHARES = "shares"
    OPTIONS = "options"


class OrderSide(StrEnum):
    """The only two order sides the agent may use (CLAUDE.md §18)."""

    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"


class AttemptStatus(StrEnum):
    """Order attempt status (INTERFACES.md RunRecord)."""

    NOT_PLACED = "not_placed"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class CancellationStatus(StrEnum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    UNKNOWN = "unknown"


class CancelDispatchStatus(StrEnum):
    """Whether a cancel call reached the broker, from its ToolCallRecord (OUTPUT_ASSEMBLY.md).

    DENIED: the hook denied it (never dispatched; the reason is recorded).
    NO_DISPATCH_RECORDED: requested, with neither a dispatch nor a denial recorded.
    """

    DISPATCHED = "dispatched"
    DENIED = "denied"
    NO_DISPATCH_RECORDED = "no_dispatch_recorded"


class BoardComparison(StrEnum):
    COMPARED = "compared"
    NOT_APPLICABLE = "not_applicable"
    UNVERIFIABLE = "unverifiable"


class CandidateOrigin(StrEnum):
    BOARD = "board"
    ROBINHOOD = "robinhood"
    HOLDINGS = "holdings"
    EXISTING_POSITION = "existing_position"


class AuditOutcome(StrEnum):
    """Post-run audit outcomes (VALIDATION.md); unverifiable is never coerced to pass."""

    PASS = "pass"  # noqa: S105 - an audit outcome, not a credential
    VIOLATION = "violation"
    UNVERIFIABLE = "unverifiable"


class AuditCheck(StrEnum):
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"
    V7 = "V7"
