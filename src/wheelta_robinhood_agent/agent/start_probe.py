"""Trusted reads behind each agent's start condition (ADR-0057).

Before the model connects, the session reads live Robinhood state through the upstream
directly (never the model), like the Agentic-eligibility check (CLAUDE.md §9), and pure
`domain.start_conditions` decides whether the session starts:

- Buy-to-Close agent: `get_option_positions`.
- Sell Options agent: `get_portfolio`, then (only if settled cash is below the minimum)
  `get_equity_positions` and `get_option_positions`.

Each result is parsed by the same mapper the model's calls use: the session's mapper table
(`SessionDeps.mappers`, the verified `agent/robinhood_mappers.py` in production), falling back
to the verified mapper for a tool the table does not list. The configured account number goes
only to the upstream; the mappers see the `AGENTIC_ACCOUNT` placeholder. A failed call, an error
result, a payload that fails its mapper, or an incomplete read (paged or narrowed) leaves that
fact `None`, so the condition is UNAVAILABLE rather than guessed. Only the derived counts and
amounts are kept (`StartCondition`); no payload is persisted.

In a simulated-venue dry run, the option rows are overlaid with the tick's simulated fills
(`SimulatedState.short_delta`), as the simulated broker overlays the model's positions reads,
and settled cash is reduced by the tick's simulated debits (a buy-to-close's
`processed_premium`). Simulated credits are not added: a sale's proceeds do not count as
settled cash this tick, so the dry-run check can only under-count.
"""

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
from typing import Final

from pydantic import SecretStr

from wheelta_robinhood_agent.agent.account_scope import AGENTIC_ACCOUNT_PLACEHOLDER
from wheelta_robinhood_agent.agent.mapped_evidence import MappedEvidence, MappingRequest
from wheelta_robinhood_agent.agent.result_boundary import (
    EvidenceMapper,
    PayloadError,
    PayloadKind,
    extract_mcp_payload,
)
from wheelta_robinhood_agent.agent.robinhood_mappers import (
    map_equity_positions,
    map_option_positions,
    map_portfolio,
)
from wheelta_robinhood_agent.agent.simulated_broker import SimulatedState
from wheelta_robinhood_agent.domain.enums import AgentRole, PositionsCoverage
from wheelta_robinhood_agent.domain.start_conditions import (
    Shares,
    ShortRow,
    StartCondition,
    evaluate_close_start,
    evaluate_sell_start,
)
from wheelta_robinhood_agent.integrations.mcp_upstream import McpUpstream, UpstreamError

OPTION_POSITIONS_TOOL: Final = "get_option_positions"
EQUITY_POSITIONS_TOOL: Final = "get_equity_positions"
PORTFOLIO_TOOL: Final = "get_portfolio"
_DEFAULT_MAPPERS: Final[Mapping[str, EvidenceMapper]] = {
    OPTION_POSITIONS_TOOL: map_option_positions,
    EQUITY_POSITIONS_TOOL: map_equity_positions,
    PORTFOLIO_TOOL: map_portfolio,
}


class _ProbeFailed(Exception):
    """A probe read that cannot establish its fact."""


async def _read(
    upstream: McpUpstream,
    tool: str,
    mapper: EvidenceMapper,
    *,
    account_number: SecretStr,
    clock: Callable[[], datetime],
    timeout_seconds: float,
) -> MappedEvidence:
    try:
        response = await upstream.call_tool(
            tool,
            {"account_number": account_number.get_secret_value()},
            timeout_seconds=timeout_seconds,
        )
        kind, payload = extract_mcp_payload(response.response)
        if kind is PayloadKind.TOOL_ERROR:
            raise _ProbeFailed(f"{tool}: the tool returned an error")
        return mapper(
            MappingRequest(
                tool_call_id=uuid.uuid4(),
                server=upstream.server,
                tool=tool,
                effective_input={"account_number": AGENTIC_ACCOUNT_PLACEHOLDER},
                payload=payload,
                retrieved_at=clock(),
                account_eligible=True,
            ),
            uuid.uuid4,
        )
    except (UpstreamError, PayloadError, ValueError, TypeError) as exc:
        raise _ProbeFailed(f"{tool}: {type(exc).__name__}") from None


def short_rows(evidence: MappedEvidence) -> tuple[ShortRow, ...] | None:
    """The short option rows of a complete options-positions read, or None."""
    if evidence.pending_option_positions:
        return tuple(
            ShortRow(
                underlying=row.underlying,
                short_quantity=row.short_quantity,
                multiplier=row.multiplier,
            )
            for row in evidence.pending_option_positions[-1].rows
            if row.short_quantity > 0
        )
    if any(PositionsCoverage.OPTIONS in read.covers for read in evidence.positions):
        return ()
    return None


def share_lots(evidence: MappedEvidence) -> tuple[Shares, ...] | None:
    """The share holdings of a complete shares read, or None."""
    reads = [r for r in evidence.positions if PositionsCoverage.SHARES in r.covers]
    if not reads:
        return None
    return tuple(Shares(symbol=h.symbol, quantity=h.quantity) for h in reads[-1].share_holdings)


def settled_cash(evidence: MappedEvidence) -> Decimal | None:
    snapshots = evidence.account_snapshots
    return snapshots[-1].available_settled_cash_usd if snapshots else None


def simulated_debits_usd(state: SimulatedState | None) -> Decimal:
    """The tick's filled simulated debit orders' premium (module docstring). An order whose
    premium cannot be read raises ValueError: the cash fact is then unknown."""
    if state is None:
        return Decimal(0)
    total = Decimal(0)
    for order in state.orders.values():
        if order.get("direction") != "debit" or order.get("state") != "filled":
            continue
        premium = order.get("processed_premium")
        if not isinstance(premium, str):
            raise ValueError("a simulated debit order has no processed_premium")
        total += Decimal(premium)
    return total


def overlay_simulated(
    rows: tuple[ShortRow, ...] | None,
    state: SimulatedState | None,
    option_ids: tuple[str, ...] = (),
) -> tuple[ShortRow, ...] | None:
    """Rows after the tick's simulated fills (module docstring). Unknown rows stay unknown.

    `option_ids` are the real rows' instrument ids in `rows` order (empty without a state)."""
    if rows is None or state is None or not state.short_delta:
        return rows
    held: dict[str, ShortRow] = dict(zip(option_ids, rows, strict=True))
    for option_id, delta in state.short_delta.items():
        instrument = state.filled_instruments.get(option_id)
        current = held.get(option_id)
        quantity = (current.short_quantity if current else 0) + delta
        if current is None and instrument is None:
            return None  # a simulated fill on a contract no read identifies
        if quantity <= 0:
            held.pop(option_id, None)
            continue
        if current is not None:
            held[option_id] = current.model_copy(update={"short_quantity": quantity})
        elif instrument is not None and instrument.multiplier is not None:
            held[option_id] = ShortRow(
                underlying=instrument.occ_symbol.root,
                short_quantity=quantity,
                multiplier=instrument.multiplier,
            )
        else:
            return None
    return tuple(held.values())


async def probe_start_condition(
    role: AgentRole,
    upstream: McpUpstream,
    *,
    account_number: SecretStr,
    min_settled_cash_usd: Decimal,
    clock: Callable[[], datetime],
    timeout_seconds: float,
    simulated: SimulatedState | None = None,
    mappers: Mapping[tuple[str, str], EvidenceMapper] | None = None,
) -> StartCondition:
    """Read and decide one role's start condition (module docstring). Never raises for a
    failed read: the condition is then UNAVAILABLE with the failure named."""
    table = mappers or {}
    read = _Reader(upstream, account_number, clock, timeout_seconds, table)
    failures: list[str] = []

    async def options() -> tuple[ShortRow, ...] | None:
        try:
            evidence = await read(OPTION_POSITIONS_TOOL)
        except _ProbeFailed as exc:
            failures.append(str(exc))
            return None
        rows = short_rows(evidence)
        ids = (
            tuple(
                r.broker_instrument_id
                for r in evidence.pending_option_positions[-1].rows
                if r.short_quantity > 0
            )
            if evidence.pending_option_positions
            else ()
        )
        return overlay_simulated(rows, simulated, ids)

    if role is AgentRole.CLOSE:
        result = evaluate_close_start(await options())
    elif role is AgentRole.SELL:
        cash: Decimal | None = None
        try:
            cash = settled_cash(await read(PORTFOLIO_TOOL))
        except _ProbeFailed as exc:
            failures.append(str(exc))
        if cash is not None and simulated is not None:
            try:
                cash -= simulated_debits_usd(simulated)
            except (ValueError, ArithmeticError) as exc:
                cash = None
                failures.append(f"simulated debits: {type(exc).__name__}")
        shares: tuple[Shares, ...] | None = None
        rows: tuple[ShortRow, ...] | None = None
        if cash is None or cash < min_settled_cash_usd:
            try:
                shares = share_lots(await read(EQUITY_POSITIONS_TOOL))
            except _ProbeFailed as exc:
                failures.append(str(exc))
            rows = await options()
        result = evaluate_sell_start(
            settled_cash_usd=cash,
            min_settled_cash_usd=min_settled_cash_usd,
            shares=shares,
            short_rows=rows,
        )
    else:
        raise ValueError(f"no start condition for agent role {role.value}")
    if failures:
        result = result.model_copy(
            update={"reason": f"{result.reason} (read failed: {'; '.join(failures)})"}
        )
    return result


class _Reader:
    def __init__(
        self,
        upstream: McpUpstream,
        account_number: SecretStr,
        clock: Callable[[], datetime],
        timeout_seconds: float,
        mappers: Mapping[tuple[str, str], EvidenceMapper],
    ) -> None:
        self._upstream = upstream
        self._account_number = account_number
        self._clock = clock
        self._timeout = timeout_seconds
        self._mappers = mappers

    async def __call__(self, tool: str) -> MappedEvidence:
        mapper = self._mappers.get((self._upstream.server, tool), _DEFAULT_MAPPERS[tool])
        return await _read(
            self._upstream,
            tool,
            mapper,
            account_number=self._account_number,
            clock=self._clock,
            timeout_seconds=self._timeout,
        )


def start_condition_payload(condition: StartCondition) -> Mapping[str, object]:
    """The run-metadata form of a start condition (no account data beyond amounts)."""
    return condition.model_dump(mode="json")


__all__ = [
    "EQUITY_POSITIONS_TOOL",
    "OPTION_POSITIONS_TOOL",
    "PORTFOLIO_TOOL",
    "overlay_simulated",
    "probe_start_condition",
    "settled_cash",
    "simulated_debits_usd",
    "share_lots",
    "short_rows",
    "start_condition_payload",
]
