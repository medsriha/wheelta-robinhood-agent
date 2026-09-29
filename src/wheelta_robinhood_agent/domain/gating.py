"""Execution-mode and order-venue resolution. Pure; the inputs come from Settings, the
session plan, and code constants."""

from wheelta_robinhood_agent.domain.enums import ExecutionMode, OrderVenue


def effective_execution_mode(
    requested: ExecutionMode, *, armed: bool, ceiling: ExecutionMode
) -> ExecutionMode:
    """Return the mode the run actually uses for tools, prompt, and audit.

    Live only if live was requested, the run is armed, and the phase ceiling permits live;
    otherwise off (INTERFACES.md "Run and RunControl"; ADR-0013 caps phase 1 at off).
    """
    if requested is ExecutionMode.LIVE and armed and ceiling is ExecutionMode.LIVE:
        return ExecutionMode.LIVE
    return ExecutionMode.OFF


def order_venue(effective: ExecutionMode, *, robinhood_proxied: bool) -> OrderVenue:
    """Where order tools go (ADR-0038): the broker in live; in off, the simulated broker when
    Robinhood is served through the validating proxy (the only place a call can be answered
    without reaching Robinhood), otherwise no order tools at all."""
    if effective is ExecutionMode.LIVE:
        return OrderVenue.BROKER
    return OrderVenue.SIMULATED if robinhood_proxied else OrderVenue.NONE


def executes_orders(venue: OrderVenue) -> bool:
    """True when the agent has order tools and the run records attempts from its place and
    cancel calls (broker or simulated); False for a proposal-only dry run (ADR-0038)."""
    return venue is not OrderVenue.NONE


def check_venue(effective: ExecutionMode, venue: OrderVenue) -> None:
    """Raise ValueError unless the venue is consistent with the mode: the broker only in
    live, and live only with the broker."""
    if (venue is OrderVenue.BROKER) != (effective is ExecutionMode.LIVE):
        raise ValueError(f"order venue {venue.value} is invalid in effective mode {effective}")


def with_default_venue(data: object) -> object:
    """Model input with `order_venue` defaulted from `effective_execution_mode` when absent:
    broker for live, none for off. Records written before ADR-0038 carry no venue, and an off
    run then had no order tools."""
    if not isinstance(data, dict) or data.get("order_venue") is not None:
        return data
    mode = data.get("effective_execution_mode")
    if mode is None:
        return data
    live = ExecutionMode(mode) is ExecutionMode.LIVE
    return {**data, "order_venue": OrderVenue.BROKER if live else OrderVenue.NONE}


def prompt_execution_mode(venue: OrderVenue) -> ExecutionMode:
    """The mode the agent prompt states (ADR-0038): live whenever the run executes orders, so
    a simulated-venue dry run follows the live procedure exactly; off for proposal-only."""
    return ExecutionMode.LIVE if executes_orders(venue) else ExecutionMode.OFF
