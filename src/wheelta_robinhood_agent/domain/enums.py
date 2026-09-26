"""Closed sets used across the codebase (CLAUDE.md §4: enums over strings)."""

from enum import StrEnum


class ExecutionMode(StrEnum):
    """Execution mode (ADR-0006). `off` is a dry run with no order tools; `live` places orders."""

    OFF = "off"
    LIVE = "live"


class ToolTier(StrEnum):
    """Robinhood tool tiers (ADR-0005, amended by ADR-0006).

    R: read/research. S: Agentic workspace writes on owned objects. X: financial actions.
    EXCLUDED: out of scope (crypto); never allowlisted.
    """

    R = "R"
    S = "S"
    X = "X"
    EXCLUDED = "excluded"


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"


class StrategyKind(StrEnum):
    """The only two strategies the agent trades (CLAUDE.md §1)."""

    CASH_SECURED_PUT = "csp"
    COVERED_CALL = "cc"
