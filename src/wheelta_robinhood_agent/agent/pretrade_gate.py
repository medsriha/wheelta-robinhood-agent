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
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from wheelta_robinhood_agent.agent.facts_tool import RunEvidence
from wheelta_robinhood_agent.domain.pretrade import (
    PRETRADE_DENIAL_PREFIX,
    OpeningLeg,
    PretradeRules,
    pretrade_feedback,
    validate_opening_leg,
)


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class PretradeGate:
    """Validates the sell-to-open legs of one `place_option_order` input (ADR-0048)."""

    evidence: Callable[[], RunEvidence]
    rules: PretradeRules
    clock: Callable[[], datetime]

    def __call__(self, tool_input: Mapping[str, object]) -> str | None:
        """A denial reason with the failed checks and values, or None when all legs pass."""
        legs = tool_input.get("legs")
        if not isinstance(legs, list) or not legs:
            return (
                PRETRADE_DENIAL_PREFIX
                + "The order has no legs to validate; list each leg's option_id."
            )
        opening: list[str] = []
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
        if not opening:
            return None
        evidence = self.evidence()
        as_of = self.clock()
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
