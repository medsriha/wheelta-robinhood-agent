from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.domain.enums import DataQuality, StrategyKind
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.positions import (
    PositionBook,
    PositionBookEntry,
    PositionInstrument,
    RollEvent,
    count_rolls,
)

T0 = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
PID = UUID(int=7)


def _inst() -> PositionInstrument:
    return PositionInstrument(
        occ_symbol=OccSymbol.parse("AAPL  261016P00190000"),
        broker_instrument_id="inst-1",
        short_quantity=1,
    )


def _roll(order: UUID | None = None, **kw: object) -> RollEvent:
    base: dict[str, object] = {
        "roll_event_id": uuid4(),
        "position_id": PID,
        "close_fill_ids": (uuid4(),),
        "replacement_order_id": order or uuid4(),
        "replacement_first_fill_id": uuid4(),
        "replacement_instrument": _inst(),
        "counted_at": T0,
    }
    base.update(kw)
    return RollEvent.model_validate(base)


def _entry(**kw: object) -> PositionBookEntry:
    base: dict[str, object] = {
        "position_id": PID,
        "position_ref": "position:1",
        "underlying": "AAPL",
        "strategy": StrategyKind.CASH_SECURED_PUT,
        "current_instruments": (_inst(),),
        "entry_fill_ids": (uuid4(),),
        "entry_date": date(2026, 9, 1),
        "entry_weighted_credit": Decimal("1.20"),
        "thesis": "Durable business.",
        "roll_count": 0,
        "history_quality": DataQuality.OK,
    }
    base.update(kw)
    return PositionBookEntry.model_validate(base)


def _gap() -> Gap:
    return Gap(field="roll_count", kind=DataQuality.MISSING, detail="imported without history")


def test_count_rolls_dedupes_replacement_orders() -> None:
    order = uuid4()
    assert count_rolls([_roll(order), _roll(order), _roll()]) == 2
    assert count_rolls([]) == 0


def test_entry_with_rolls() -> None:
    order = uuid4()
    e = _entry(roll_events=(_roll(order), _roll(order)), roll_count=1)
    assert e.roll_count == 1
    with pytest.raises(ValidationError, match="distinct replacement"):
        _entry(roll_events=(_roll(),), roll_count=0)


def test_imported_short_has_unknown_roll_count() -> None:
    e = _entry(
        entry_fill_ids=(),
        entry_date=None,
        entry_weighted_credit=None,
        thesis=None,
        roll_count=None,
        history_quality=DataQuality.MISSING,
        gaps=(_gap(),),
    )
    assert e.roll_count is None
    with pytest.raises(ValidationError, match="unknown roll count"):
        _entry(
            entry_fill_ids=(),
            entry_date=None,
            entry_weighted_credit=None,
            roll_count=0,
        )
    with pytest.raises(ValidationError, match="entry fill evidence"):
        _entry(entry_fill_ids=(), roll_count=None, gaps=(_gap(),))


def test_unknown_roll_count_needs_gap() -> None:
    with pytest.raises(ValidationError, match="gap"):
        _entry(roll_count=None, history_quality=DataQuality.MISSING)


def test_ok_history_forbids_gaps() -> None:
    with pytest.raises(ValidationError, match="quality 'ok'"):
        _entry(roll_count=None, gaps=(_gap(),))


def test_roll_events_must_belong_to_lineage() -> None:
    with pytest.raises(ValidationError, match="lineage"):
        _entry(roll_events=(_roll(position_id=uuid4()),), roll_count=1)


def test_roll_event_validation() -> None:
    with pytest.raises(ValidationError):
        _roll(close_fill_ids=())
    f = uuid4()
    with pytest.raises(ValidationError, match="also be a close"):
        _roll(close_fill_ids=(f,), replacement_first_fill_id=f)
    with pytest.raises(ValidationError, match="duplicate"):
        _roll(close_fill_ids=(f, f))


def test_book_uniqueness() -> None:
    e = _entry()
    PositionBook(as_of=T0, entries=(e,))
    with pytest.raises(ValidationError, match="duplicate"):
        PositionBook(as_of=T0, entries=(e, e))
