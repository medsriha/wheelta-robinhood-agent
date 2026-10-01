"""ADR-0057: which order sides each agent may place (pure)."""

from wheelta_robinhood_agent.domain.enums import AgentRole, OrderSide
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.role_gate import (
    ROLE_DENIAL_PREFIX,
    RoleLeg,
    RunFill,
    check_role,
    roll_capacity,
)

OLD_PUT = OccSymbol.parse("AAPL  261016P00190000")
NEW_PUT = OccSymbol.parse("AAPL  261120P00185000")
OTHER_CALL = OccSymbol.parse("AAPL  261120C00220000")


def _close_leg() -> RoleLeg:
    return RoleLeg(option_id="old", closing=True, opening=False)


def _open_leg(occ: OccSymbol | None = NEW_PUT) -> RoleLeg:
    return RoleLeg(option_id="new", closing=False, opening=True, occ_symbol=occ)


def _btc(filled: int) -> RunFill:
    return RunFill(side=OrderSide.BUY_TO_CLOSE, occ_symbol=OLD_PUT, quantity=filled)


def test_sell_agent_never_buys_to_close() -> None:
    denial = check_role(AgentRole.SELL, (_close_leg(),), quantity=1, run=())
    assert denial is not None and denial.startswith(ROLE_DENIAL_PREFIX)
    assert "only sells to open" in denial
    assert check_role(AgentRole.SELL, (_open_leg(),), quantity=1, run=()) is None


def test_close_agent_closes_freely() -> None:
    assert check_role(AgentRole.CLOSE, (_close_leg(),), quantity=5, run=()) is None


def test_close_agent_opens_only_a_filled_roll_replacement() -> None:
    assert check_role(AgentRole.CLOSE, (_open_leg(),), quantity=2, run=(_btc(2),)) is None
    over = check_role(AgentRole.CLOSE, (_open_leg(),), quantity=3, run=(_btc(2),))
    assert over is not None and "allow 2" in over
    unfilled = check_role(AgentRole.CLOSE, (_open_leg(),), quantity=1, run=(_btc(0),))
    assert unfilled is not None and "allow 0" in unfilled
    nothing = check_role(AgentRole.CLOSE, (_open_leg(),), quantity=1, run=())
    assert nothing is not None and "New positions are the Sell Options agent's" in nothing


def test_replacement_must_match_root_and_right_and_count_earlier_replacements() -> None:
    wrong_right = check_role(AgentRole.CLOSE, (_open_leg(OTHER_CALL),), quantity=1, run=(_btc(1),))
    assert wrong_right is not None
    opened = RunFill(side=OrderSide.SELL_TO_OPEN, occ_symbol=NEW_PUT, quantity=1)
    assert roll_capacity((_btc(2), opened), "AAPL", NEW_PUT.right) == 1
    again = check_role(AgentRole.CLOSE, (_open_leg(),), quantity=2, run=(_btc(2), opened))
    assert again is not None and "allow 1" in again


def test_unknown_contract_or_quantity_is_denied() -> None:
    unknown = check_role(AgentRole.CLOSE, (_open_leg(None),), quantity=1, run=(_btc(1),))
    assert unknown is not None and "no validated instrument" in unknown
    no_qty = check_role(AgentRole.CLOSE, (_open_leg(),), quantity=None, run=(_btc(1),))
    assert no_qty is not None and "quantity" in no_qty


def test_legacy_agent_is_not_limited() -> None:
    assert (
        check_role(AgentRole.WHEEL, (_open_leg(None), _close_leg()), quantity=None, run=()) is None
    )


def test_an_earlier_replacement_with_an_unknown_contract_blocks_further_replacements() -> None:
    unknown = RunFill(side=OrderSide.SELL_TO_OPEN, occ_symbol=None, quantity=1)
    denial = check_role(AgentRole.CLOSE, (_open_leg(),), quantity=1, run=(_btc(5), unknown))
    assert denial is not None and "cannot be attributed" in denial
    # Closes are never limited by it.
    assert check_role(AgentRole.CLOSE, (_close_leg(),), quantity=1, run=(unknown,)) is None
