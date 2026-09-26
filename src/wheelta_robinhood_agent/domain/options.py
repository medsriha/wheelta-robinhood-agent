"""OCC option symbols (CLAUDE.md §4: options are identified by OCC symbol + instrument ID)."""

import re
from datetime import date
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from wheelta_robinhood_agent.domain.enums import OptionRight

# Root (1-6 chars; adjusted roots may carry a digit), YYMMDD, C/P, strike x 1000 in 8 digits.
_OCC_RE = re.compile(
    r"^(?P<root>[A-Z][A-Z0-9]{0,5}) *(?P<date>\d{6})(?P<right>[CP])(?P<strike>\d{8})$"
)
_STRIKE_SCALE = Decimal(1000)


class OccSymbol(BaseModel):
    """A parsed OCC option symbol. `str()` gives the canonical 21-character padded form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str
    expiration: date
    right: OptionRight
    strike: Decimal

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not re.fullmatch(r"[A-Z][A-Z0-9]{0,5}", self.root):
            raise ValueError(f"invalid OCC root: {self.root!r}")
        if not self.strike > 0:
            raise ValueError(f"strike must be positive: {self.strike}")
        return self

    @classmethod
    def parse(cls, symbol: str) -> Self:
        """Parse a padded (`AAPL  260116C00190000`) or compact (`AAPL260116C00190000`) symbol.

        Raises ValueError for anything else; lowercase input and surrounding whitespace are
        not accepted, because a loosely formatted string is not an identifier.
        """
        match = _OCC_RE.fullmatch(symbol)
        if match is None:
            raise ValueError(f"not an OCC option symbol: {symbol!r}")
        raw_date = match["date"]
        expiration = date(2000 + int(raw_date[:2]), int(raw_date[2:4]), int(raw_date[4:]))
        return cls(
            root=match["root"],
            expiration=expiration,
            right=OptionRight.CALL if match["right"] == "C" else OptionRight.PUT,
            strike=Decimal(int(match["strike"])) / _STRIKE_SCALE,
        )

    def __str__(self) -> str:
        right = "C" if self.right is OptionRight.CALL else "P"
        strike_units = self.strike * _STRIKE_SCALE
        if strike_units != strike_units.to_integral_value() or not 0 < strike_units < 10**8:
            raise ValueError(f"strike not representable in OCC format: {self.strike}")
        return f"{self.root:<6}{self.expiration:%y%m%d}{right}{int(strike_units):08d}"
