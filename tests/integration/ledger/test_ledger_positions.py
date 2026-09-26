"""Position lineages and the PositionBook projection rebuilt from events."""

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import count

import psycopg
import pytest

from wheelta_robinhood_agent.domain.enums import (
    AppEnv,
    AttemptStatus,
    DataQuality,
    StrategyKind,
    ToolTier,
)
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.options import OccSymbol
from wheelta_robinhood_agent.domain.orders import OrderIntent
from wheelta_robinhood_agent.domain.positions import PositionBookEntry, PositionInstrument
from wheelta_robinhood_agent.ledger.errors import DedupConflict
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.ledger.orders import (
    link_intent,
    observe_fill,
    observe_status,
    record_broker_order,
    record_order_intent,
)
from wheelta_robinhood_agent.ledger.positions import (
    FillRole,
    close_position,
    link_fill,
    open_position,
    position_book,
    record_assignment,
    record_gap,
    record_reconciliation,
    record_roll,
)
from wheelta_robinhood_agent.ledger.runs import open_run_slot
from wheelta_robinhood_agent.ledger.tool_calls import record_tool_call_requested

Conn = psycopg.Connection[tuple[object, ...]]
SLOT = datetime(2026, 9, 25, 14, tzinfo=UTC)
T0 = SLOT + timedelta(minutes=5)
AS_OF = SLOT + timedelta(minutes=50)
ACCOUNT = "acct-scope-1"
PUT_OCT = PositionInstrument(
    occ_symbol=OccSymbol.parse("AAPL  261016P00180000"),
    broker_instrument_id="inst-oct",
    short_quantity=2,
)
PUT_NOV = PositionInstrument(
    occ_symbol=OccSymbol.parse("AAPL  261120P00175000"),
    broker_instrument_id="inst-nov",
    short_quantity=2,
)
_sdk = count(1)
_broker = count(1)


@pytest.fixture
def run_id(conn: Conn) -> uuid.UUID:
    return open_run_slot(conn, AppEnv.LOCAL, SLOT).run_id


def _read(conn: Conn, run_id: uuid.UUID) -> uuid.UUID:
    return record_tool_call_requested(
        conn,
        run_id=run_id,
        sdk_tool_use_id=f"toolu_{next(_sdk)}",
        stage="agent",
        server="robinhood",
        tool="get_option_order",
        tier=ToolTier.R,
        arguments_redacted={},
        requested_at=T0,
    ).tool_call_id


def _order(conn: Conn, run_id: uuid.UUID) -> uuid.UUID:
    return record_broker_order(
        conn, run_id=run_id, account_scope_id=ACCOUNT, broker_order_id=f"B-{next(_broker)}"
    )


def _exec_fill(
    conn: Conn,
    run_id: uuid.UUID,
    order_id: uuid.UUID,
    exec_id: str,
    quantity: int,
    price: str,
    at: datetime = T0,
) -> uuid.UUID:
    return observe_fill(
        conn,
        order_id,
        run_id=run_id,
        quantity=quantity,
        price=Decimal(price),
        broker_execution_id=exec_id,
        observed_at=at,
        source_tool_call_id=_read(conn, run_id),
    ).event_id


def _open_with_entry(conn: Conn, run_id: uuid.UUID) -> uuid.UUID:
    position_id = open_position(
        conn,
        run_id=run_id,
        account_scope_id=ACCOUNT,
        underlying="AAPL",
        strategy=StrategyKind.CASH_SECURED_PUT,
        instruments=[PUT_OCT],
        observed_at=T0,
        thesis="Range-bound into earnings",
        invalidation_conditions=["close below 170"],
        entry_event_refs=["event:earnings-2026-10-29"],
    )
    entry_order = _order(conn, run_id)
    for exec_id, price in (("E-1", "1.20"), ("E-2", "1.30")):
        fill = _exec_fill(conn, run_id, entry_order, exec_id, 1, price)
        link_fill(
            conn,
            position_id,
            run_id=run_id,
            fill_event_id=fill,
            role=FillRole.ENTRY,
            observed_at=T0,
        )
    return position_id


def _entry(conn: Conn, position_id: uuid.UUID) -> PositionBookEntry:
    book = position_book(conn, ACCOUNT, as_of=AS_OF)
    return next(e for e in book.entries if e.position_id == position_id)


def test_entry_facts_come_from_linked_fills(conn: Conn, run_id: uuid.UUID) -> None:
    position_id = _open_with_entry(conn, run_id)
    entry = _entry(conn, position_id)
    assert entry.entry_weighted_credit == Decimal("1.25")
    assert entry.entry_date == date(2026, 9, 25)
    assert len(entry.entry_fill_ids) == 2
    assert entry.roll_count == 0
    assert entry.history_quality is DataQuality.OK
    assert entry.thesis == "Range-bound into earnings"
    assert entry.invalidation_conditions == ("close below 170",)
    assert entry.entry_event_refs == ("event:earnings-2026-10-29",)
    assert entry.current_instruments == (PUT_OCT,)
    assert entry.position_ref == f"position:{position_id}"
    # A rebuild gives the same projection.
    assert _entry(conn, position_id) == entry


def test_roll_counted_once_despite_price_steps_and_partial_fills(
    conn: Conn, run_id: uuid.UUID
) -> None:
    position_id = _open_with_entry(conn, run_id)
    close_order = _order(conn, run_id)
    close_fill = _exec_fill(conn, run_id, close_order, "C-1", 2, "0.40")
    link_fill(
        conn,
        position_id,
        run_id=run_id,
        fill_event_id=close_fill,
        role=FillRole.CLOSE,
        observed_at=T0,
    )

    # First replacement order partially fills, then is cancelled and re-priced.
    replacement_1 = _order(conn, run_id)
    first_fill = _exec_fill(conn, run_id, replacement_1, "R-1", 1, "1.10")
    rolled = record_roll(
        conn,
        position_id,
        run_id=run_id,
        roll_key="decision:roll-aapl",
        close_fill_ids=[close_fill],
        replacement_order_id=replacement_1,
        replacement_first_fill_id=first_fill,
        replacement_instrument=PUT_NOV,
        observed_at=T0 + timedelta(minutes=1),
    )
    # The re-read of the same execution is one observation.
    assert _exec_fill(conn, run_id, replacement_1, "R-1", 1, "1.10") == first_fill
    replacement_2 = _order(conn, run_id)
    step_fill = _exec_fill(conn, run_id, replacement_2, "R-2", 1, "1.05")
    again = record_roll(
        conn,
        position_id,
        run_id=run_id,
        roll_key="decision:roll-aapl",
        close_fill_ids=[close_fill],
        replacement_order_id=replacement_2,
        replacement_first_fill_id=step_fill,
        replacement_instrument=PUT_NOV,
        observed_at=T0 + timedelta(minutes=2),
    )
    for fill in (first_fill, step_fill):
        link_fill(
            conn,
            position_id,
            run_id=run_id,
            fill_event_id=fill,
            role=FillRole.REPLACEMENT,
            observed_at=T0,
        )
    assert rolled.created and not again.created

    entry = _entry(conn, position_id)
    assert entry.roll_count == 1
    assert len(entry.roll_events) == 1
    assert entry.roll_events[0].replacement_order_id == replacement_1
    assert entry.current_instruments == (PUT_NOV,)
    # Entry facts are unchanged by the roll.
    assert entry.entry_weighted_credit == Decimal("1.25")


def test_partial_close_without_replacement_does_not_count(conn: Conn, run_id: uuid.UUID) -> None:
    position_id = _open_with_entry(conn, run_id)
    close_fill = _exec_fill(conn, run_id, _order(conn, run_id), "C-1", 1, "0.40")
    link_fill(
        conn,
        position_id,
        run_id=run_id,
        fill_event_id=close_fill,
        role=FillRole.CLOSE,
        observed_at=T0,
    )
    record_reconciliation(
        conn,
        position_id,
        run_id=run_id,
        dedup_key="partial-close",
        detail="one of two contracts closed",
        observed_at=T0,
        current_instruments=[PUT_OCT.model_copy(update={"short_quantity": 1})],
    )
    entry = _entry(conn, position_id)
    assert entry.roll_count == 0
    assert entry.roll_events == ()
    assert entry.current_instruments[0].short_quantity == 1
    with pytest.raises(DedupConflict):
        link_fill(
            conn,
            position_id,
            run_id=run_id,
            fill_event_id=close_fill,
            role=FillRole.ENTRY,
            observed_at=T0,
        )


def test_imported_short_without_history_has_null_entry_and_unknown_rolls(
    conn: Conn, run_id: uuid.UUID
) -> None:
    position_id = open_position(
        conn,
        run_id=run_id,
        account_scope_id=ACCOUNT,
        underlying="AAPL",
        strategy=StrategyKind.CASH_SECURED_PUT,
        instruments=[PUT_OCT],
        observed_at=T0,
        source_tool_call_ids=[_read(conn, run_id)],
        imported=True,
    )
    entry = _entry(conn, position_id)
    assert entry.entry_fill_ids == ()
    assert entry.entry_date is None
    assert entry.entry_weighted_credit is None
    assert entry.thesis is None
    assert entry.roll_count is None
    assert entry.history_quality is DataQuality.MISSING
    assert [g.field for g in entry.gaps] == ["entry_history"]


def test_gap_makes_rolls_unknown_until_superseded(conn: Conn, run_id: uuid.UUID) -> None:
    position_id = _open_with_entry(conn, run_id)
    gap = record_gap(
        conn,
        position_id,
        run_id=run_id,
        dedup_key="corporate-action",
        gap=Gap(field="roll_count", kind=DataQuality.CONTRADICTORY, detail="split adjustment"),
        observed_at=T0,
    )
    entry = _entry(conn, position_id)
    assert entry.roll_count is None
    assert entry.history_quality is DataQuality.CONTRADICTORY
    fix = record_reconciliation(
        conn,
        position_id,
        run_id=run_id,
        dedup_key="corporate-action-resolved",
        detail="operator reconciled the split",
        observed_at=T0 + timedelta(minutes=1),
        corrects_event_id=gap.event_id,
    )
    stored = conn.execute(
        "SELECT corrects_event_id FROM position_events WHERE event_id = %s", (fix.event_id,)
    ).fetchone()
    assert stored == (gap.event_id,)
    entry = _entry(conn, position_id)
    assert entry.roll_count == 0
    assert entry.gaps == ()


def test_book_has_all_active_lineages_across_runs(conn: Conn, run_id: uuid.UUID) -> None:
    old = _open_with_entry(conn, run_id)
    record_assignment(
        conn,
        old,
        run_id=run_id,
        dedup_key="assign-1",
        share_lot_refs=["lot-1"],
        observed_at=T0,
        source_tool_call_ids=[_read(conn, run_id)],
    )
    closed = _open_with_entry(conn, run_id)
    close_position(
        conn, closed, run_id=run_id, observed_at=T0, source_tool_call_ids=[_read(conn, run_id)]
    )
    later_runs = [
        open_run_slot(conn, AppEnv.LOCAL, SLOT + timedelta(days=d)).run_id for d in range(1, 6)
    ]
    newer = _open_with_entry(conn, later_runs[-1])
    open_position(
        conn,
        run_id=run_id,
        account_scope_id="other-account",
        underlying="MSFT",
        strategy=StrategyKind.CASH_SECURED_PUT,
        instruments=[PUT_OCT],
        observed_at=T0,
        imported=True,
    )

    book = position_book(conn, ACCOUNT, as_of=AS_OF)
    assert [e.position_id for e in book.entries] == [old, newer]
    assert book.entries[0].share_lot_refs == ("lot-1",)


def test_book_lists_unresolved_owned_orders(conn: Conn, run_id: uuid.UUID) -> None:
    def intent() -> OrderIntent:
        place = record_tool_call_requested(
            conn,
            run_id=run_id,
            sdk_tool_use_id=f"toolu_{next(_sdk)}",
            stage="agent",
            server="robinhood",
            tool="place_option_order",
            tier=ToolTier.X,
            arguments_redacted={},
            requested_at=T0,
        ).tool_call_id
        value = OrderIntent(
            intent_id=new_id(),
            run_id=run_id,
            place_tool_call_id=place,
            account_scope_id=ACCOUNT,
            occ_symbol=None,
            broker_instrument_id=None,
            side_raw="sell_to_open",
            quantity=1,
            order_type_raw="limit",
            time_in_force_raw="gfd",
            limit_price=Decimal("1.00"),
            requested_at=T0,
        )
        record_order_intent(conn, value)
        return value

    working_intent = intent()
    working = _order(conn, run_id)
    link_intent(
        conn,
        working,
        working_intent.intent_id,
        run_id=run_id,
        observed_at=T0,
        source_tool_call_ids=[working_intent.place_tool_call_id],
    )
    observe_status(
        conn,
        working,
        run_id=run_id,
        status=AttemptStatus.PLACED,
        observed_at=T0,
        source_tool_call_id=_read(conn, run_id),
    )
    pending = intent()

    book = position_book(conn, ACCOUNT, as_of=AS_OF)
    assert book.unresolved_owned_order_ids == (working,)
    assert [g.evidence_ids for g in book.gaps] == [(pending.intent_id,)]
