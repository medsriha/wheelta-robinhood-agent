"""Pure pre-trade validation of sell-to-open legs (ADR-0048).

`validate_opening_leg(leg, rules, as_of)` checks one sell-to-open leg against four trading
rules, from this run's validated evidence only:

- DTE: `filters.min_dte <= DTE <= filters.max_dte`, DTE in America/New_York calendar days
  (definitions.annualization), as in `compute_decision_facts`.
- delta: `filters.min_abs_delta <= |live delta| <= filters.max_abs_delta`.
- cushion: `(underlying - strike) / underlying` for a put, `(strike - underlying) /
  underlying` for a call, at least `filters.min_cushion_ratio` (definitions.cushion).
- annualized yield: live BID / collateral per share x 365 / DTE, at least
  `filters.min_annualized_yield_ratio` (CSP collateral = strike, CC collateral = live share
  price; the contract multiplier cancels). The same basis as the facts formula
  `annualized_yield_on_collateral` (ADR-0014).

Every comparison is inclusive ("no less than", "no more than"). A quote older than its
`data_quality.freshness` max age, or a missing input, makes the dependent check `missing`,
which blocks the order like a failure: nothing is estimated. A bound set to "TBD" makes its
check `missing`; "none" and "agent_discretion" bounds are not enforced by code. Buy-to-close
legs are never validated here (filters notes: filters apply to sell-to-open legs only).
"""

from collections.abc import Iterable
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, DivisionByZero, InvalidOperation, Overflow
from enum import StrEnum

from wheelta_robinhood_agent.domain.base import Dec, DomainModel, NonEmptyStr
from wheelta_robinhood_agent.domain.enums import OptionRight
from wheelta_robinhood_agent.domain.facts_compute import (
    OptionInstrument,
    UnderlyingQuote,
    annualized_ratio,
    cushion_ratio,
    dte_days,
)
from wheelta_robinhood_agent.domain.facts_rules import (
    CountSetting,
    DecimalSetting,
    FactsRuleMarker,
)
from wheelta_robinhood_agent.domain.run_record import Quote
from wheelta_robinhood_agent.domain.sanity import is_fresh, require_aware

PRETRADE_DENIAL_PREFIX = "Pre-trade validation failed (ADR-0048); the order was NOT placed. "
_SHOWN = Decimal("0.0001")
_CTX = Context(
    prec=28, rounding=ROUND_HALF_EVEN, traps=[InvalidOperation, DivisionByZero, Overflow]
)
_R_OPTION_AGE = "data_quality.freshness.option_quote_max_age_seconds"
_R_EQUITY_AGE = "data_quality.freshness.equity_quote_max_age_seconds"


class PretradeRules(DomainModel):
    """The rule values pre-trade validation reads (keys mirror `rules/trading_rules.toml`).

    `config.facts_rules.pretrade_rules_from` maps the loaded rules into this model.
    """

    min_dte: CountSetting
    max_dte: CountSetting
    min_abs_delta: DecimalSetting
    max_abs_delta: DecimalSetting
    min_cushion_ratio: DecimalSetting
    min_annualized_yield_ratio: DecimalSetting
    option_quote_max_age_seconds: CountSetting
    equity_quote_max_age_seconds: CountSetting


class CheckName(StrEnum):
    DTE = "dte"
    DELTA = "abs_delta"
    CUSHION = "cushion"
    ANNUALIZED_YIELD = "annualized_yield"


class CheckStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - a check outcome, not a credential
    FAIL = "fail"
    MISSING = "missing"  # an input or rule is missing or stale; blocks like a failure


class LegCheck(DomainModel):
    """One check's outcome. `detail` states the value, the bound, and the inputs used."""

    check: CheckName
    status: CheckStatus
    value: Dec | None = None
    detail: NonEmptyStr


class OpeningLeg(DomainModel):
    """A sell-to-open leg and the latest validated evidence for it recorded in this run."""

    option_id: NonEmptyStr
    instrument: OptionInstrument | None
    option_quote: Quote | None
    underlying_quote: UnderlyingQuote | None


class LegValidation(DomainModel):
    option_id: NonEmptyStr
    contract: NonEmptyStr | None
    checks: tuple[LegCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.status is CheckStatus.PASS for c in self.checks)


def _shown(value: Decimal) -> str:
    return str(value.quantize(_SHOWN, context=_CTX))


def _compare(
    check: CheckName,
    value: Decimal,
    shown: str,
    lower: tuple[str, DecimalSetting | CountSetting] | None,
    upper: tuple[str, DecimalSetting | CountSetting] | None,
    inputs: str,
) -> LegCheck:
    """Inclusive bounds; a TBD bound is `missing`, a none/agent_discretion bound is skipped."""
    unset: list[str] = []
    broken: list[str] = []
    for bound, below in ((lower, True), (upper, False)):
        if bound is None:
            continue
        key, setting = bound
        if setting is FactsRuleMarker.UNSET:
            unset.append(key)
            continue
        if isinstance(setting, FactsRuleMarker):
            continue
        limit = Decimal(setting)
        if below and value < limit:
            broken.append(f"below {key} {setting}")
        if not below and value > limit:
            broken.append(f"above {key} {setting}")
    if unset:
        detail = f"{check.value} {shown} cannot be checked: {', '.join(unset)} is TBD"
        return LegCheck(check=check, status=CheckStatus.MISSING, value=value, detail=detail)
    if broken:
        detail = f"{check.value} {shown} is {' and '.join(broken)} ({inputs})"
        return LegCheck(check=check, status=CheckStatus.FAIL, value=value, detail=detail)
    return LegCheck(
        check=check, status=CheckStatus.PASS, value=value, detail=f"{check.value} {shown}"
    )


def _missing(check: CheckName, why: str) -> LegCheck:
    return LegCheck(check=check, status=CheckStatus.MISSING, detail=f"{check.value}: {why}")


def _stale_reason(
    observed: datetime | None, as_of: datetime, setting: CountSetting, key: str, what: str
) -> str | None:
    """None when fresh; otherwise why the observation cannot be used (fails closed)."""
    if observed is None:
        return f"no validated {what} recorded in this run"
    if setting is FactsRuleMarker.NONE:
        fresh = observed <= as_of
    elif isinstance(setting, FactsRuleMarker):
        return f"{what} freshness unknown: {key} is not set"
    else:
        fresh = is_fresh(observed, as_of, int(setting))
    return None if fresh else f"the {what} is older than {key}; re-quote"


def validate_opening_leg(leg: OpeningLeg, rules: PretradeRules, as_of: datetime) -> LegValidation:
    """Check one sell-to-open leg against DTE, delta, cushion, and annualized yield rules."""
    require_aware(as_of, "as_of")
    inst = leg.instrument
    if inst is None:
        why = "no validated instrument for this option_id in this run; read it first"
        return LegValidation(
            option_id=leg.option_id,
            contract=None,
            checks=tuple(_missing(c, why) for c in CheckName),
        )
    occ = inst.occ_symbol
    put = occ.right is OptionRight.PUT
    contract = f"{occ.root} {occ.expiration.isoformat()} {occ.right.value} {occ.strike}"
    strike = occ.strike
    dte = dte_days(occ.expiration, as_of)
    checks = [
        _compare(
            CheckName.DTE,
            Decimal(dte),
            str(dte),
            ("filters.min_dte", rules.min_dte),
            ("filters.max_dte", rules.max_dte),
            f"expiration {occ.expiration.isoformat()}",
        )
    ]

    quote = leg.option_quote
    quote_why = _stale_reason(
        quote.as_of if quote is not None else None,
        as_of,
        rules.option_quote_max_age_seconds,
        _R_OPTION_AGE,
        "option quote",
    )
    spot_quote = leg.underlying_quote
    spot_why = _stale_reason(
        spot_quote.as_of if spot_quote is not None else None,
        as_of,
        rules.equity_quote_max_age_seconds,
        _R_EQUITY_AGE,
        f"{inst.underlying} quote",
    )
    bid = quote.bid if quote is not None and quote_why is None else None
    spot = spot_quote.price if spot_quote is not None and spot_why is None else None

    if quote is None or quote_why is not None:
        checks.append(_missing(CheckName.DELTA, str(quote_why)))
    elif quote.delta is None:
        checks.append(_missing(CheckName.DELTA, "the live option quote reports no delta"))
    else:
        abs_delta = abs(quote.delta)
        checks.append(
            _compare(
                CheckName.DELTA,
                abs_delta,
                str(abs_delta),
                ("filters.min_abs_delta", rules.min_abs_delta),
                ("filters.max_abs_delta", rules.max_abs_delta),
                f"live delta {quote.delta}",
            )
        )

    if spot is None:
        checks.append(_missing(CheckName.CUSHION, str(spot_why)))
    else:
        cushion = cushion_ratio(occ.right, strike, spot)
        checks.append(
            _compare(
                CheckName.CUSHION,
                cushion,
                _shown(cushion),
                ("filters.min_cushion_ratio", rules.min_cushion_ratio),
                None,
                f"underlying {spot}, strike {strike}",
            )
        )

    collateral = strike if put else spot
    if bid is None:
        checks.append(_missing(CheckName.ANNUALIZED_YIELD, str(quote_why)))
    elif collateral is None:
        checks.append(_missing(CheckName.ANNUALIZED_YIELD, str(spot_why)))
    elif dte <= 0:
        checks.append(_missing(CheckName.ANNUALIZED_YIELD, "DTE is 0 or less; undefined"))
    else:
        annualized = annualized_ratio(bid, collateral, dte)
        basis = "strike" if put else "share price"
        checks.append(
            _compare(
                CheckName.ANNUALIZED_YIELD,
                annualized,
                _shown(annualized),
                ("filters.min_annualized_yield_ratio", rules.min_annualized_yield_ratio),
                None,
                f"live bid {bid}, collateral {basis} {collateral}, DTE {dte}",
            )
        )
    return LegValidation(option_id=leg.option_id, contract=contract, checks=tuple(checks))


def pretrade_feedback(validations: Iterable[LegValidation]) -> str | None:
    """The denial the agent receives when any leg fails, or None when every leg passes.

    Names each failed or missing check with its value, bound, and inputs, and the checks that
    passed, so the agent can adjust the proposed trade and try again.
    """
    failed = [v for v in validations if not v.passed]
    if not failed:
        return None
    parts: list[str] = []
    for v in failed:
        label = v.contract or "unknown contract"
        bad = [c.detail for c in v.checks if c.status is not CheckStatus.PASS]
        ok = [c.detail for c in v.checks if c.status is CheckStatus.PASS]
        text = f"leg {label} (option_id {v.option_id}): failed: {'; '.join(bad)}"
        if ok:
            text += f"; passed: {', '.join(ok)}"
        parts.append(text)
    return (
        PRETRADE_DENIAL_PREFIX
        + " | ".join(parts)
        + ". Adjust the trade: choose a contract that meets these rules, or re-quote when a "
        "value was missing or stale, then repeat the order procedure from re-quote. The limit "
        "price does not change these values (yield uses the live bid)."
    )
