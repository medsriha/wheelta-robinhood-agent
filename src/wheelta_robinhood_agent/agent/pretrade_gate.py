"""Pre-trade validation of `place_option_order` (ADR-0048).

The PreToolUse hook calls `PretradeGate` for every `place_option_order` request, after the
tool, venue, kill-switch, and account-scope checks and before the order is dispatched (to the
broker in live mode, or to the simulated broker in a dry run). Each sell-to-open leg is
checked by pure `domain.pretrade.validate_opening_leg` against the latest validated
instrument, option quote, and underlying quote recorded in this run (the evidence the
decision-facts tool reads). A leg is identified only by its `option_id`; nothing is matched
by ticker, price, or time.

The gate returns a denial reason naming each failed check and its values, or None to allow
the order. The hook denies the call with that reason, so the agent receives the feedback and
can adjust the trade and try again; the reason is recorded as the call's `denied` outcome.
Buy-to-close legs pass unchecked (filters apply to sell-to-open legs only). A leg whose side,
position effect, or option_id cannot be read is denied: an unidentified contract cannot be
validated.

Before the leg checks, `placements` (when wired) supplies this run's unresolved orders and
in-flight placements for pure `domain.order_concurrency.check_concurrency` (ADR-0051): only
buy-to-close orders on different contracts may work at the same time, funded together by the
fresh account snapshot. Its denial reaches the agent the same way. ADR-0057: for the Sell
Options agent those include orders the same tick's Buy-to-Close run left unresolved.

First of all, the agent role (ADR-0057) limits the sides, by pure `domain.role_gate.check_role`:
the Sell Options agent never buys to close, and the Buy-to-Close agent sells to open only as a
roll's replacement, on the underlying and right of a buy-to-close this run filled, for no
more contracts than those fills less the replacements this run already placed (`run_orders`).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from wheelta_robinhood_agent.agent.facts_tool import RunEvidence
from wheelta_robinhood_agent.domain.enums import AgentRole, OrderSide
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.order_concurrency import (
    NewPlacement,
    PlacementLeg,
    WorkingPlacement,
    check_concurrency,
)
from wheelta_robinhood_agent.domain.orders import OrderRecord
from wheelta_robinhood_agent.domain.pretrade import (
    PRETRADE_DENIAL_PREFIX,
    OpeningLeg,
    PretradeRules,
    pretrade_feedback,
    validate_opening_leg,
)
from wheelta_robinhood_agent.domain.role_gate import RoleLeg, RunFill, check_role


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _count(value: object) -> int | None:
    text = str(value).strip() if isinstance(value, int | str) and value is not True else ""
    return int(text) if text.isdigit() and int(text) > 0 else None


def _price(value: object) -> Decimal | None:
    if not isinstance(value, str | int) or isinstance(value, bool):
        return None
    try:
        price = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return price if price.is_finite() and price > 0 else None


@dataclass(frozen=True)
class PlacementState:
    """This run's unresolved owned orders and its place calls that have no outcome yet
    (the one being checked included)."""

    unresolved: tuple[OrderRecord, ...]
    placements_in_flight: int


def _working(record: OrderRecord) -> WorkingPlacement:
    intent = record.intent
    label = (
        record.broker_order.broker_order_id
        if record.broker_order is not None
        else f"place:{intent.place_tool_call_id}"
        if intent is not None
        else "unknown order"
    )
    filled = record.filled_quantity
    remaining = intent.quantity - (filled or 0) if intent is not None and intent.quantity else None
    return WorkingPlacement(
        label=label,
        option_id=intent.broker_instrument_id if intent is not None else None,
        # side_raw is the recorded "<side>_to_<position_effect>" (broker_ledger._leg_side).
        closing=intent is not None and (intent.side_raw or "").lower() == "buy_to_close",
        remaining_quantity=remaining,
        limit_price=intent.limit_price if intent is not None else None,
    )


@dataclass(frozen=True)
class PretradeGate:
    """Validates one `place_option_order` input: concurrency (ADR-0051), then the
    sell-to-open legs (ADR-0048)."""

    evidence: Callable[[], RunEvidence]
    rules: PretradeRules
    clock: Callable[[], datetime]
    placements: Callable[[], PlacementState] | None = None
    # ADR-0057: the agent placing the order, and every order this run placed (resolved or
    # not) for the Buy-to-Close agent's roll-replacement check.
    role: AgentRole = AgentRole.WHEEL
    run_orders: Callable[[], tuple[OrderRecord, ...]] = lambda: ()

    def __call__(self, tool_input: Mapping[str, object]) -> str | None:
        """A denial reason with the failed checks and values, or None when all legs pass."""
        legs = tool_input.get("legs")
        if not isinstance(legs, list) or not legs:
            return (
                PRETRADE_DENIAL_PREFIX
                + "The order has no legs to validate; list each leg's option_id."
            )
        opening: list[str] = []
        parsed: list[PlacementLeg] = []
        for index, leg in enumerate(legs):
            if not isinstance(leg, Mapping):
                return PRETRADE_DENIAL_PREFIX + f"Leg {index} is not an object."
            side = _text(leg.get("side"))
            effect = _text(leg.get("position_effect"))
            option_id = _text(leg.get("option_id"))
            if side is None or effect is None or option_id is None:
                return (
                    PRETRADE_DENIAL_PREFIX
                    + f"Leg {index} needs option_id, side, and position_effect."
                )
            if side.lower() == "sell" and effect.lower() == "open":
                opening.append(option_id)
            closing = side.lower() == "buy" and effect.lower() == "close"
            parsed.append(PlacementLeg(option_id=option_id, closing=closing))
        if self.role is not AgentRole.WHEEL:
            denial = self._role(tool_input, parsed, opening)
            if denial is not None:
                return denial
        as_of = self.clock()
        state = self.placements() if self.placements is not None else None
        concurrent = state is not None and (
            bool(state.unresolved) or state.placements_in_flight > 1
        )
        if not opening and not concurrent:
            return None  # a lone buy-to-close: nothing to check (ADR-0048, ADR-0051)
        evidence = self.evidence()
        if state is not None and concurrent:
            denial = self._concurrency(tool_input, parsed, state, evidence, as_of)
            if denial is not None:
                return denial
        if not opening:
            return None
        validations = []
        for option_id in opening:
            instrument = evidence.instrument(option_id)
            leg_input = OpeningLeg(
                option_id=option_id,
                instrument=instrument,
                option_quote=evidence.option_quote(option_id),
                underlying_quote=(
                    evidence.underlying_quote(instrument.underlying)
                    if instrument is not None
                    else None
                ),
            )
            validations.append(validate_opening_leg(leg_input, self.rules, as_of))
        return pretrade_feedback(validations)

    def _role(
        self, tool_input: Mapping[str, object], legs: list[PlacementLeg], opening: list[str]
    ) -> str | None:
        evidence = self.evidence() if opening else None

        def occ(option_id: str | None) -> OccSymbol | None:
            if evidence is None or option_id is None:
                return None
            instrument = evidence.instrument(option_id)
            return instrument.occ_symbol if instrument is not None else None

        role_legs = tuple(
            RoleLeg(
                option_id=leg.option_id,
                closing=leg.closing,
                opening=leg.option_id in opening,
                occ_symbol=occ(leg.option_id) if leg.option_id in opening else None,
            )
            for leg in legs
        )
        fills: list[RunFill] = []
        if self.role is AgentRole.CLOSE and opening:
            for record in self.run_orders():
                intent = record.intent
                if intent is None or intent.side is None:
                    continue
                symbol = intent.occ_symbol or occ(intent.broker_instrument_id)
                if intent.side is OrderSide.BUY_TO_CLOSE:
                    quantity = record.filled_quantity or 0
                else:
                    quantity = intent.quantity or 0
                fills.append(RunFill(side=intent.side, occ_symbol=symbol, quantity=quantity))
        return check_role(
            self.role, role_legs, quantity=_count(tool_input.get("quantity")), run=tuple(fills)
        )

    def _concurrency(
        self,
        tool_input: Mapping[str, object],
        legs: list[PlacementLeg],
        state: PlacementState,
        evidence: RunEvidence,
        as_of: datetime,
    ) -> str | None:
        working = tuple(_working(r) for r in state.unresolved)
        ids = {leg.option_id for leg in legs} | {w.option_id for w in working if w.option_id}
        multipliers: dict[str, int | None] = {}
        for iid in ids:
            inst = evidence.instrument(iid)
            multipliers[iid] = inst.multiplier if inst is not None else None
        return check_concurrency(
            NewPlacement(
                legs=tuple(legs),
                quantity=_count(tool_input.get("quantity")),
                limit_price=_price(tool_input.get("price")),
            ),
            working,
            other_placements_in_flight=max(state.placements_in_flight - 1, 0),
            multipliers=multipliers,
            account=evidence.account(),
            account_max_age=self.rules.account_state_max_age_seconds,
            as_of=as_of,
        )
