import itertools

import pytest

from wheelta_robinhood_agent.domain.enums import ExecutionMode, OrderVenue
from wheelta_robinhood_agent.domain.gating import (
    check_venue,
    effective_execution_mode,
    executes_orders,
    order_venue,
    prompt_execution_mode,
    with_default_venue,
)

CASES = list(itertools.product(ExecutionMode, [True, False], ExecutionMode))


@pytest.mark.parametrize(("requested", "armed", "ceiling"), CASES)
def test_live_only_when_requested_armed_and_permitted(
    requested: ExecutionMode, armed: bool, ceiling: ExecutionMode
) -> None:
    result = effective_execution_mode(requested, armed=armed, ceiling=ceiling)
    expected_live = requested is ExecutionMode.LIVE and armed and ceiling is ExecutionMode.LIVE
    assert result is (ExecutionMode.LIVE if expected_live else ExecutionMode.OFF)


@pytest.mark.parametrize(
    ("mode", "proxied", "venue"),
    [
        (ExecutionMode.LIVE, True, OrderVenue.BROKER),
        (ExecutionMode.LIVE, False, OrderVenue.BROKER),
        (ExecutionMode.OFF, True, OrderVenue.SIMULATED),
        (ExecutionMode.OFF, False, OrderVenue.NONE),
    ],
)
def test_order_venue(mode: ExecutionMode, proxied: bool, venue: OrderVenue) -> None:
    """ADR-0038: broker only in live; a dry run is simulated only through the proxy."""
    assert order_venue(mode, robinhood_proxied=proxied) is venue
    check_venue(mode, venue)


@pytest.mark.parametrize(
    ("mode", "venue"),
    [
        (ExecutionMode.OFF, OrderVenue.BROKER),
        (ExecutionMode.LIVE, OrderVenue.SIMULATED),
        (ExecutionMode.LIVE, OrderVenue.NONE),
    ],
)
def test_inconsistent_venue_raises(mode: ExecutionMode, venue: OrderVenue) -> None:
    with pytest.raises(ValueError, match="invalid"):
        check_venue(mode, venue)


def test_executes_orders_and_the_prompt_mode() -> None:
    assert executes_orders(OrderVenue.BROKER) and executes_orders(OrderVenue.SIMULATED)
    assert not executes_orders(OrderVenue.NONE)
    assert prompt_execution_mode(OrderVenue.SIMULATED) is ExecutionMode.LIVE
    assert prompt_execution_mode(OrderVenue.BROKER) is ExecutionMode.LIVE
    assert prompt_execution_mode(OrderVenue.NONE) is ExecutionMode.OFF


def test_default_venue_from_the_mode() -> None:
    assert with_default_venue({"effective_execution_mode": "live"}) == {
        "effective_execution_mode": "live",
        "order_venue": OrderVenue.BROKER,
    }
    assert with_default_venue({"effective_execution_mode": "off"})["order_venue"] is (  # type: ignore[index]
        OrderVenue.NONE
    )
    kept = {"effective_execution_mode": "off", "order_venue": "simulated"}
    assert with_default_venue(kept) is kept
    assert with_default_venue({"x": 1}) == {"x": 1}
    assert with_default_venue("raw") == "raw"
