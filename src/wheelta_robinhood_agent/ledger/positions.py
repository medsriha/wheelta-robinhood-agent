"""Position lineages, position events, and the PositionBook projection.

INTERFACES.md "PositionBook". The book covers every active lineage in the account scope (not
the last N runs) and is rebuilt from `positions` identities plus `position_events`:

- `opened` carries the lineage's underlying, strategy, instruments, thesis, invalidation
  conditions, and events known at entry; `imported=true` marks a short found without known
  history: its entry facts stay null and its roll count unknown (None), explained by a gap.
- `fill_linked` references a fill event in `order_events` with a role (entry/close/
  replacement). Entry facts are derived from linked entry fills, never from model claims.
- `roll` is appended once per roll operation (`roll_key`, e.g. the decision), at the
  replacement's first fill. Price-stepped replacement orders and later partial fills of the
  same roll are deduplicated by that key, so the roll is counted once. A partial close with
  no replacement fill appends no roll event and does not count.
- `assignment` links share lots; `gap` records ambiguity; `reconciliation` may replace the
  current instruments and, with `corrects_event_id`, supersede an earlier event (e.g. a gap).
- `closed` ends the lineage; reopening is a new lineage.
"""

import uuid
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import dict_row

from wheelta_robinhood_agent.domain.enums import DataQuality, StrategyKind
from wheelta_robinhood_agent.domain.events import PositionEventType
from wheelta_robinhood_agent.domain.evidence import Gap
from wheelta_robinhood_agent.domain.positions import (
    PositionBook,
    PositionBookEntry,
    PositionInstrument,
    RollEvent,
    count_rolls,
)
from wheelta_robinhood_agent.ledger.errors import DedupConflict, IdentityConflict, UnknownEntity
from wheelta_robinhood_agent.ledger.events import AppendedEvent, EventTable, append_event
from wheelta_robinhood_agent.ledger.ids import new_id
from wheelta_robinhood_agent.ledger.orders import effective_events, owned_unresolved_orders

Conn = psycopg.Connection[tuple[object, ...]]
Row = dict[str, Any]

MARKET_TZ: Final = ZoneInfo("America/New_York")
# Gap fields that make the roll history (and so the roll count) unknown.
HISTORY_GAP_FIELDS: Final = frozenset({"entry_history", "roll_count"})
_QUALITY_ORDER: Final = (
    DataQuality.CONTRADICTORY,
    DataQuality.MISSING,
    DataQuality.STALE,
    DataQuality.LOW_CONFIDENCE,
)


class FillRole(StrEnum):
    ENTRY = "entry"
    CLOSE = "close"
    REPLACEMENT = "replacement"


def _rows(conn: Conn, query: str, params: Sequence[object]) -> list[Row]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(query.encode(), params).fetchall()


def _instruments_json(instruments: Sequence[PositionInstrument]) -> list[object]:
    return [i.model_dump(mode="json") for i in instruments]


def _account_of(conn: Conn, position_id: uuid.UUID) -> str:
    row = conn.execute(
        "SELECT account_scope_id FROM positions WHERE position_id = %s", (position_id,)
    ).fetchone()
    if row is None:
        raise UnknownEntity(f"positions has no row {position_id}")
    return str(row[0])


def _fill_row(conn: Conn, fill_event_id: uuid.UUID) -> Row:
    rows = _rows(
        conn,
        "SELECT e.entity_id AS order_id, o.account_scope_id FROM order_events e "
        "JOIN orders o ON o.order_id = e.entity_id "
        "WHERE e.event_id = %s AND e.event_type = 'fill_observed'",
        (fill_event_id,),
    )
    if not rows:
        raise UnknownEntity(f"no fill_observed order event {fill_event_id}")
    return rows[0]


# ---------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------
def open_position(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    account_scope_id: str,
    underlying: str,
    strategy: StrategyKind,
    instruments: Sequence[PositionInstrument],
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
    thesis: str | None = None,
    invalidation_conditions: Sequence[str] = (),
    entry_event_refs: Sequence[str] = (),
    imported: bool = False,
) -> uuid.UUID:
    """Create a new lineage and its `opened` event in one transaction.

    `imported=True` records a short found at the broker without known history (no thesis,
    no entry fills): the projection keeps entry facts null and the roll count unknown.
    """
    if not account_scope_id or not underlying:
        raise ValueError("account_scope_id and underlying must be non-empty")
    if not instruments:
        raise ValueError("a lineage opens with at least one instrument")
    position_id = new_id()
    with conn.transaction():
        conn.execute(
            "INSERT INTO positions (position_id, account_scope_id, first_run_id) "
            "VALUES (%s, %s, %s)",
            (position_id, account_scope_id, run_id),
        )
        append_event(
            conn,
            EventTable.POSITION,
            entity_id=position_id,
            run_id=run_id,
            event_type=PositionEventType.OPENED.value,
            observed_at=observed_at,
            dedup_key="opened",
            source_tool_call_ids=source_tool_call_ids,
            payload={
                "underlying": underlying,
                "strategy": strategy.value,
                "instruments": _instruments_json(instruments),
                "thesis": thesis,
                "invalidation_conditions": list(invalidation_conditions),
                "entry_event_refs": list(entry_event_refs),
                "imported": imported,
            },
        )
    return position_id


def link_fill(
    conn: Conn,
    position_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    fill_event_id: uuid.UUID,
    role: FillRole,
    observed_at: datetime,
) -> AppendedEvent:
    """Link a recorded fill (an order_events `fill_observed` row) to this lineage."""
    with conn.transaction():
        fill = _fill_row(conn, fill_event_id)
        if fill["account_scope_id"] != _account_of(conn, position_id):
            raise IdentityConflict("fill and position belong to different account scopes")
        appended = append_event(
            conn,
            EventTable.POSITION,
            entity_id=position_id,
            run_id=run_id,
            event_type=PositionEventType.FILL_LINKED.value,
            observed_at=observed_at,
            dedup_key=f"fill:{fill_event_id}",
            payload={"fill_event_id": str(fill_event_id), "role": role.value},
            extra={"order_id": fill["order_id"]},
        )
        if not appended.created:
            stored = conn.execute(
                "SELECT payload->>'role' FROM position_events WHERE event_id = %s",
                (appended.event_id,),
            ).fetchone()
            if stored is None or stored[0] != role.value:
                raise DedupConflict(f"fill {fill_event_id} is linked with another role")
    return appended


def record_roll(
    conn: Conn,
    position_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    roll_key: str,
    close_fill_ids: Sequence[uuid.UUID],
    replacement_order_id: uuid.UUID,
    replacement_first_fill_id: uuid.UUID,
    replacement_instrument: PositionInstrument,
    observed_at: datetime,
    current_instruments: Sequence[PositionInstrument] | None = None,
) -> AppendedEvent:
    """Record a roll at the replacement's first fill; idempotent per `roll_key`.

    A later call with the same key (another price-stepped replacement order, a further
    partial fill, a re-read) returns the stored event and does not count again.
    `current_instruments` is the post-roll holding; default: the replacement only.
    """
    if not roll_key:
        raise ValueError("roll_key must be non-empty")
    if not close_fill_ids:
        raise ValueError("a roll needs at least one close fill")
    with conn.transaction():
        account = _account_of(conn, position_id)
        for fill_id in close_fill_ids:
            if _fill_row(conn, fill_id)["account_scope_id"] != account:
                raise IdentityConflict("close fill belongs to another account scope")
        replacement = _fill_row(conn, replacement_first_fill_id)
        if replacement["order_id"] != replacement_order_id:
            raise IdentityConflict("the replacement first fill is not on the replacement order")
        instruments = (
            current_instruments if current_instruments is not None else [replacement_instrument]
        )
        return append_event(
            conn,
            EventTable.POSITION,
            entity_id=position_id,
            run_id=run_id,
            event_type=PositionEventType.ROLL.value,
            observed_at=observed_at,
            dedup_key=f"roll:{roll_key}",
            payload={
                "roll_key": roll_key,
                "close_fill_ids": [str(i) for i in close_fill_ids],
                "replacement_order_id": str(replacement_order_id),
                "replacement_first_fill_id": str(replacement_first_fill_id),
                "replacement_instrument": replacement_instrument.model_dump(mode="json"),
                "current_instruments": _instruments_json(instruments),
            },
            extra={"order_id": replacement_order_id},
        )


def record_assignment(
    conn: Conn,
    position_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    dedup_key: str,
    share_lot_refs: Sequence[str],
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID],
    current_instruments: Sequence[PositionInstrument] | None = None,
) -> AppendedEvent:
    """Link assignment share lots (broker tax-lot refs) to the lineage."""
    if not source_tool_call_ids:
        raise ValueError("an assignment needs broker evidence")
    payload: dict[str, object] = {"share_lot_refs": list(share_lot_refs)}
    if current_instruments is not None:
        payload["current_instruments"] = _instruments_json(current_instruments)
    return append_event(
        conn,
        EventTable.POSITION,
        entity_id=position_id,
        run_id=run_id,
        event_type=PositionEventType.ASSIGNMENT.value,
        observed_at=observed_at,
        dedup_key=f"assignment:{dedup_key}",
        source_tool_call_ids=source_tool_call_ids,
        payload=payload,
    )


def record_gap(
    conn: Conn,
    position_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    dedup_key: str,
    gap: Gap,
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
) -> AppendedEvent:
    """Record ambiguity (corporate action, ambiguous matching, ...). Field `entry_history`
    or `roll_count` makes the roll count unknown until the gap is superseded."""
    return append_event(
        conn,
        EventTable.POSITION,
        entity_id=position_id,
        run_id=run_id,
        event_type=PositionEventType.GAP.value,
        observed_at=observed_at,
        dedup_key=f"gap:{dedup_key}",
        source_tool_call_ids=source_tool_call_ids,
        payload=gap.model_dump(mode="json"),
    )


def record_reconciliation(
    conn: Conn,
    position_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    dedup_key: str,
    detail: str,
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID] = (),
    corrects_event_id: uuid.UUID | None = None,
    current_instruments: Sequence[PositionInstrument] | None = None,
) -> AppendedEvent:
    """Reconcile the lineage: optionally replace current instruments (e.g. after a partial
    close) and/or supersede an earlier event via `corrects_event_id`."""
    if not detail:
        raise ValueError("a reconciliation needs a detail")
    payload: dict[str, object] = {"detail": detail}
    if current_instruments is not None:
        payload["current_instruments"] = _instruments_json(current_instruments)
    return append_event(
        conn,
        EventTable.POSITION,
        entity_id=position_id,
        run_id=run_id,
        event_type=PositionEventType.RECONCILIATION.value,
        observed_at=observed_at,
        dedup_key=f"reconciliation:{dedup_key}",
        source_tool_call_ids=source_tool_call_ids,
        corrects_event_id=corrects_event_id,
        payload=payload,
    )


def close_position(
    conn: Conn,
    position_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    observed_at: datetime,
    source_tool_call_ids: Sequence[uuid.UUID],
) -> AppendedEvent:
    """Final closure (expiry, full buy-to-close, assignment completion) ends the lineage."""
    if not source_tool_call_ids:
        raise ValueError("a closure needs broker evidence")
    return append_event(
        conn,
        EventTable.POSITION,
        entity_id=position_id,
        run_id=run_id,
        event_type=PositionEventType.CLOSED.value,
        observed_at=observed_at,
        dedup_key="closed",
        source_tool_call_ids=source_tool_call_ids,
    )


# ---------------------------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------------------------
def _fill_facts(conn: Conn, fill_ids: Sequence[uuid.UUID]) -> list[Row]:
    if not fill_ids:
        return []
    return _rows(
        conn,
        "SELECT event_id, entity_id AS order_id, broker_execution_id, fill_quantity, "
        "fill_price, observed_at, payload FROM order_events "
        "WHERE event_id = ANY(%s) AND event_type = 'fill_observed'",
        (list(fill_ids),),
    )


def _fill_time(fill: Row) -> datetime:
    executed = fill["payload"].get("executed_at")
    return datetime.fromisoformat(executed) if executed else fill["observed_at"]


def _entry_credit(fills: list[Row]) -> tuple[Decimal | None, Gap | None]:
    """Quantity-weighted entry price from linked entry fills (INTERFACES.md PositionBook).

    Executions contribute quantity x price. For an order reported only as cumulative totals,
    the latest total and its reported (average) price are used once. Missing prices or an
    order reported both ways yield None with a gap; nothing is estimated.
    """
    by_order: dict[uuid.UUID, list[Row]] = {}
    for f in fills:
        by_order.setdefault(f["order_id"], []).append(f)
    quantity = Decimal(0)
    notional = Decimal(0)
    evidence = tuple(f["event_id"] for f in fills)
    for order_fills in by_order.values():
        executions = [f for f in order_fills if f["broker_execution_id"] is not None]
        cumulative = [f for f in order_fills if f["broker_execution_id"] is None]
        if executions and cumulative:
            return None, Gap(
                field="entry_weighted_credit",
                kind=DataQuality.CONTRADICTORY,
                detail="entry order reported both as executions and cumulative totals",
                evidence_ids=evidence,
            )
        used = executions or [max(cumulative, key=lambda f: f["observed_at"])]
        for f in used:
            if f["fill_price"] is None:
                return None, Gap(
                    field="entry_weighted_credit",
                    kind=DataQuality.MISSING,
                    detail="an entry fill has no reported price",
                    evidence_ids=evidence,
                )
            quantity += f["fill_quantity"]
            notional += f["fill_quantity"] * f["fill_price"]
    if quantity <= 0:
        return None, Gap(
            field="entry_weighted_credit",
            kind=DataQuality.MISSING,
            detail="entry fills report no filled quantity",
            evidence_ids=evidence,
        )
    return notional / quantity, None


def _history_quality(gaps: Sequence[Gap]) -> DataQuality:
    kinds = {g.kind for g in gaps}
    return next((k for k in _QUALITY_ORDER if k in kinds), DataQuality.OK)


def _entry(conn: Conn, position_id: uuid.UUID, events: list[Row]) -> PositionBookEntry | None:
    live = effective_events(events)
    if any(e["event_type"] == PositionEventType.CLOSED for e in live):
        return None
    opened = next(e for e in live if e["event_type"] == PositionEventType.OPENED)
    info = opened["payload"]
    gaps: list[Gap] = []

    instruments = info["instruments"]
    entry_fill_ids: list[uuid.UUID] = []
    share_lots: dict[str, None] = {}
    rolls: list[RollEvent] = []
    for e in live:
        kind, payload = e["event_type"], e["payload"]
        if kind == PositionEventType.FILL_LINKED and payload["role"] == FillRole.ENTRY:
            entry_fill_ids.append(uuid.UUID(payload["fill_event_id"]))
        elif kind == PositionEventType.ROLL:
            rolls.append(
                RollEvent(
                    roll_event_id=e["event_id"],
                    position_id=position_id,
                    close_fill_ids=tuple(uuid.UUID(i) for i in payload["close_fill_ids"]),
                    replacement_order_id=uuid.UUID(payload["replacement_order_id"]),
                    replacement_first_fill_id=uuid.UUID(payload["replacement_first_fill_id"]),
                    replacement_instrument=PositionInstrument.model_validate(
                        payload["replacement_instrument"]
                    ),
                    counted_at=e["observed_at"],
                )
            )
        elif kind == PositionEventType.ASSIGNMENT:
            share_lots.update(dict.fromkeys(payload["share_lot_refs"]))
        elif kind == PositionEventType.GAP:
            gaps.append(Gap.model_validate(payload))
        if kind != PositionEventType.OPENED and "current_instruments" in payload:
            instruments = payload["current_instruments"]

    if info["imported"] and not entry_fill_ids:
        gaps.append(
            Gap(
                field="entry_history",
                kind=DataQuality.MISSING,
                detail="short imported without known history; entry facts and rolls unknown",
            )
        )
    elif not entry_fill_ids:
        gaps.append(
            Gap(
                field="entry_fill_ids",
                kind=DataQuality.MISSING,
                detail="no entry fill is linked to this lineage",
            )
        )

    entry_date: date | None = None
    credit: Decimal | None = None
    if entry_fill_ids:
        fills = _fill_facts(conn, entry_fill_ids)
        entry_date = min(_fill_time(f) for f in fills).astimezone(MARKET_TZ).date()
        credit, credit_gap = _entry_credit(fills)
        if credit_gap is not None:
            gaps.append(credit_gap)

    history_unknown = not entry_fill_ids or any(g.field in HISTORY_GAP_FIELDS for g in gaps)
    return PositionBookEntry(
        position_id=position_id,
        position_ref=f"position:{position_id}",
        underlying=info["underlying"],
        strategy=StrategyKind(info["strategy"]),
        current_instruments=tuple(PositionInstrument.model_validate(i) for i in instruments),
        entry_fill_ids=tuple(entry_fill_ids),
        entry_date=entry_date,
        entry_weighted_credit=credit,
        thesis=info["thesis"],
        invalidation_conditions=tuple(info["invalidation_conditions"]),
        entry_event_refs=tuple(info["entry_event_refs"]),
        share_lot_refs=tuple(share_lots),
        roll_events=tuple(rolls),
        roll_count=None if history_unknown else count_rolls(rolls),
        history_quality=_history_quality(gaps),
        gaps=tuple(gaps),
    )


def position_book(conn: Conn, account_scope_id: str, *, as_of: datetime) -> PositionBook:
    """Rebuild the PositionBook for every active lineage in the account scope.

    `as_of` is supplied by the caller (no clock reads here). Unresolved owned broker orders
    are listed by order ID; intents with no verified broker order yet cannot be (they have
    no order ID) and are reported as a book-level gap naming the intent IDs.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    positions = _rows(
        conn,
        "SELECT position_id FROM positions WHERE account_scope_id = %s "
        "ORDER BY recorded_at, position_id",
        (account_scope_id,),
    )
    ids = [p["position_id"] for p in positions]
    grouped: dict[uuid.UUID, list[Row]] = {pid: [] for pid in ids}
    if ids:
        for e in _rows(
            conn,
            "SELECT * FROM position_events WHERE entity_id = ANY(%s) ORDER BY entity_id, sequence",
            (ids,),
        ):
            grouped[e["entity_id"]].append(e)
    entries = [entry for pid in ids if (entry := _entry(conn, pid, grouped[pid])) is not None]

    unresolved = owned_unresolved_orders(conn, account_scope_id)
    order_ids = tuple(r.broker_order.order_id for r in unresolved if r.broker_order is not None)
    pending = tuple(
        r.intent.intent_id for r in unresolved if r.broker_order is None and r.intent is not None
    )
    book_gaps = (
        (
            Gap(
                field="unresolved_owned_order_ids",
                kind=DataQuality.MISSING,
                detail=f"{len(pending)} placement intent(s) without a verified broker order",
                evidence_ids=pending,
            ),
        )
        if pending
        else ()
    )
    return PositionBook(
        as_of=as_of,
        entries=tuple(entries),
        unresolved_owned_order_ids=order_ids,
        gaps=book_gaps,
    )


__all__ = [
    "FillRole",
    "PositionEventType",
    "close_position",
    "link_fill",
    "open_position",
    "position_book",
    "record_assignment",
    "record_gap",
    "record_reconciliation",
    "record_roll",
]
