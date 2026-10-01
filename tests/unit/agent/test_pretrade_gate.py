"""PretradeGate: which legs of place_option_order are validated, against which evidence
(ADR-0048). The checks themselves are covered in tests/unit/domain/test_domain_pretrade.py."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from wheelta_robinhood_agent.agent.facts_tool import RunEvidence
from wheelta_robinhood_agent.agent.mapped_evidence import MappedEvidence
from wheelta_robinhood_agent.agent.pretrade_gate import PretradeGate
from wheelta_robinhood_agent.config.facts_rules import pretrade_rules_from
from wheelta_robinhood_agent.config.rules import load_rules
from wheelta_robinhood_agent.domain.facts_compute import OptionInstrument, UnderlyingQuote
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.pretrade import PRETRADE_DENIAL_PREFIX, PretradeRules
from wheelta_robinhood_agent.domain.run_record import Quote

D = Decimal
T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
TC = UUID(int=900)
RULES = pretrade_rules_from(load_rules())


def evidence(bid: str = "1.00", with_quotes: bool = True) -> RunEvidence:
    """Instrument inst-1 (XYZ 73 put, DTE 20) and, optionally, fresh quotes for it."""
    inst = OptionInstrument(
        evidence_id=UUID(int=1),
        as_of=T0 - timedelta(seconds=30),
        source_tool_call_ids=(TC,),
        occ_symbol=OccSymbol.parse("XYZ   261015P00073000"),
        broker_instrument_id="inst-1",
        underlying="XYZ",
        multiplier=100,
    )
    items = [MappedEvidence(instruments=(inst,))]
    if with_quotes:
        quote = Quote(
            quote_id=UUID(int=2),
            broker_instrument_id="inst-1",
            bid=D(bid),
            ask=D(bid) + D("0.10"),
            delta=D("-0.20"),
            as_of=T0 - timedelta(seconds=10),
            source_tool_call_ids=(TC,),
        )
        spot = UnderlyingQuote(
            evidence_id=UUID(int=3),
            as_of=T0 - timedelta(seconds=10),
            source_tool_call_ids=(TC,),
            symbol="XYZ",
            price=D("80.00"),
        )
        items.append(MappedEvidence(option_quotes=(quote,), underlying_quotes=(spot,)))
    return RunEvidence(tuple(items))


class Loads:
    def __init__(self, ev: RunEvidence) -> None:
        self.ev = ev
        self.count = 0

    def __call__(self) -> RunEvidence:
        self.count += 1
        return self.ev


def gate(ev: RunEvidence | None = None, rules: PretradeRules = RULES) -> tuple[PretradeGate, Loads]:
    loads = Loads(ev if ev is not None else evidence())
    return PretradeGate(evidence=loads, rules=rules, clock=lambda: T0), loads


def order(*legs: dict[str, Any]) -> dict[str, Any]:
    return {"account_number": "AGENTIC_ACCOUNT", "legs": list(legs), "price": "1.00"}


def sto(option_id: str = "inst-1") -> dict[str, Any]:
    return {"option_id": option_id, "side": "sell", "position_effect": "open"}


def test_the_shipped_rules_map_to_the_owner_values() -> None:
    assert RULES.min_cushion_ratio == D("0.04")
    assert RULES.min_annualized_yield_ratio == D("0.25")
    assert (RULES.min_dte, RULES.max_dte) == (3, 45)
    assert (RULES.min_abs_delta, RULES.max_abs_delta) == (D("0.15"), D("0.30"))


def test_a_passing_sell_to_open_is_allowed() -> None:
    g, loads = gate()
    assert g(order(sto())) is None
    assert loads.count == 1


def test_a_failing_sell_to_open_is_denied_with_feedback() -> None:
    g, _ = gate(evidence(bid="0.50"))
    reason = g(order(sto()))
    assert reason is not None and reason.startswith(PRETRADE_DENIAL_PREFIX)
    assert "annualized_yield 0.1250 is below filters.min_annualized_yield_ratio 0.25" in reason


def test_side_and_effect_are_case_insensitive() -> None:
    g, _ = gate(evidence(bid="0.50"))
    leg = {"option_id": "inst-1", "side": " SELL ", "position_effect": "Open"}
    assert g(order(leg)) is not None


def test_buy_to_close_is_not_validated() -> None:
    g, loads = gate(evidence(bid="0.50"))
    assert g(order({"option_id": "inst-1", "side": "buy", "position_effect": "close"})) is None
    assert loads.count == 0  # no evidence read for a close


def test_a_roll_validates_only_its_opening_leg() -> None:
    g, _ = gate(evidence(bid="0.50"))
    close = {"option_id": "held", "side": "buy", "position_effect": "close"}
    reason = g(order(close, sto()))
    assert reason is not None and "option_id inst-1" in reason and "option_id held" not in reason


def test_an_unknown_contract_is_denied() -> None:
    g, _ = gate()
    reason = g(order(sto("never-read")))
    assert reason is not None and "leg unknown contract (option_id never-read)" in reason


def test_unquoted_contract_is_denied() -> None:
    g, _ = gate(evidence(with_quotes=False))
    reason = g(order(sto()))
    assert reason is not None and "no validated option quote recorded in this run" in reason


def test_malformed_orders_are_denied_without_reading_evidence() -> None:
    for tool_input in (
        {"account_number": "AGENTIC_ACCOUNT"},
        order(),
        {"legs": None},
        {"legs": ["inst-1"]},
        order({"option_id": "inst-1", "side": "sell"}),
        order({"option_id": "", "side": "sell", "position_effect": "open"}),
        order({"option_id": 7, "side": "sell", "position_effect": "open"}),
    ):
        g, loads = gate()
        reason = g(tool_input)
        assert reason is not None and reason.startswith(PRETRADE_DENIAL_PREFIX), tool_input
        assert loads.count == 0


# ---- concurrency (ADR-0051) -----------------------------------------------------------------


def _unresolved_close(option_id: str = "inst-2") -> Any:
    from wheelta_robinhood_agent.domain.orders import BrokerOrder, OrderIntent, OrderRecord

    return OrderRecord(
        intent=OrderIntent(
            intent_id=UUID(int=10),
            run_id=UUID(int=11),
            place_tool_call_id=UUID(int=12),
            account_scope_id="acct:1234",
            occ_symbol=None,
            broker_instrument_id=option_id,
            side_raw="buy_to_close",
            quantity=1,
            order_type_raw="limit",
            time_in_force_raw="gfd",
            limit_price=D("0.50"),
            requested_at=T0,
        ),
        broker_order=BrokerOrder(
            order_id=UUID(int=13),
            account_scope_id="acct:1234",
            broker_order_id="ord-2",
            intent_id=UUID(int=10),
            first_observed_at=T0,
        ),
    )


def btc(option_id: str = "inst-1") -> dict[str, Any]:
    return {"option_id": option_id, "side": "buy", "position_effect": "close"}


def concurrent_gate(unresolved: tuple[Any, ...], in_flight: int = 1) -> tuple[PretradeGate, Loads]:
    from wheelta_robinhood_agent.agent.pretrade_gate import PlacementState

    loads = Loads(evidence())
    state = PlacementState(unresolved=unresolved, placements_in_flight=in_flight)
    return PretradeGate(
        evidence=loads, rules=RULES, clock=lambda: T0, placements=lambda: state
    ), loads


def test_a_lone_close_loads_no_evidence() -> None:
    g, loads = concurrent_gate(())
    assert g(order(btc())) is None
    assert loads.count == 0


def test_a_second_placement_in_flight_is_denied() -> None:
    g, _ = concurrent_gate((), in_flight=2)
    reason = g(order(btc()))
    assert reason is not None and "one at a time" in reason


def test_an_open_while_a_close_works_is_denied_before_leg_checks() -> None:
    g, _ = concurrent_gate((_unresolved_close(),))
    reason = g(order(sto()))
    assert reason is not None and "Only buy-to-close orders" in reason and "ord-2" in reason


def test_a_concurrent_close_needs_verified_funds() -> None:
    """No account snapshot and no instrument for the working order: funds unverifiable."""
    g, _ = concurrent_gate((_unresolved_close(),))
    reason = g({**order(btc()), "quantity": "1"})
    assert reason is not None and "cannot be verified" in reason
    assert "no validated account snapshot" in reason


def test_a_working_close_is_recognised_whatever_the_case() -> None:
    """The recorded side is the agent's raw text; "BUY_to_Close" is still a close."""
    record = _unresolved_close()
    intent = record.intent.model_copy(update={"side_raw": "BUY_to_Close"})
    g, _ = concurrent_gate((record.model_copy(update={"intent": intent}),))
    reason = g({**order(btc()), "quantity": "1"})
    assert reason is not None and "Only buy-to-close" not in reason
