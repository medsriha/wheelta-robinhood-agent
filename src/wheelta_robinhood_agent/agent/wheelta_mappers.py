"""Verified result mappers for the Wheelta MCP (ADR-0041).

Shapes are from our own capture of `wheelta-mcp` on 2026-09-29
(`tests/fixtures/wheelta/`): every tool returns `structuredContent`.

- `wheelta_board_query` (rows mode) -> one `BoardScreen` per row that names a contract
  (`symbol`, `contract.strike`, `contract.expiration`, `contract.bid`), with the board's
  `freshness.buildId` and `asOf`. A screen is build-time data, never a quote: code uses it
  only to set a Robinhood candidate's origin and to compute selection.board_comparison.
  The other selected columns (WheelIQ score, yield, cushion, ...) are delivered beside it as
  `screen_context`: labelled, non-citable context for ranking, never evidence. A row without
  a contract identity, and grouped (`groupBy`) results, are context only, with a gap.
  `freshness.buildState` other than `ready` raises (a building board is a tool error, 503).
- `wheelta_macro_snapshot` (ADR-0045) -> one `MacroRegime` and one `MacroIndicator` per
  series: citable, so a Mignon can back a macro number; no decision fact uses them.
- Every other Wheelta tool is delivered context-only (`result_boundary.CONTEXT_ONLY_TOOLS`):
  research, macro, calendar, quotes, and board detail inform judgment but cannot back a
  number or a decision.

Numbers are decimal strings in the delivered context; money in screens is `Decimal`.
"""

import uuid
from collections.abc import Callable, Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Annotated, Any, Final

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue

from wheelta_robinhood_agent.agent.mapped_evidence import (
    EvidenceMapper,
    MacroIndicator,
    MacroRegime,
    MacroRegimeInput,
    MappedEvidence,
    MappingRequest,
)
from wheelta_robinhood_agent.domain.enums import OptionRight
from wheelta_robinhood_agent.domain.facts_compute import BoardScreen
from wheelta_robinhood_agent.domain.options import OccSymbol

BOARD_QUERY_TOOL: Final = "wheelta_board_query"
_IDENTITY: Final = ("symbol", "contract.strike", "contract.expiration", "contract.bid")
IDENTITY_GAP: Final = (
    "wheelta_board_query: rows without symbol, contract.strike, contract.expiration, and "
    "contract.bid are context only; select those columns to screen contracts"
)
GROUPS_GAP: Final = "wheelta_board_query: a grouped summary names no contract; context only"
CONTEXT_GAP: Final = (
    "wheelta_board_query: screen_context is a build-time screen, not evidence; re-quote every "
    "contract from Robinhood before relying on it"
)


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"not a number: {value!r}") from None
    if not number.is_finite():
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("expected an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("board timestamps must carry an offset")
    return parsed


def _context_value(value: object) -> JsonValue:
    """A context value as delivered: numbers as decimal strings, other scalars as-is."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(_decimal(value))
    if isinstance(value, list):
        return [_context_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _context_value(v) for k, v in value.items()}
    raise ValueError(f"unexpected value type {type(value).__name__}")


def _screen(
    row: Mapping[str, Any],
    *,
    build_id: str,
    as_of: datetime,
    request: MappingRequest,
    new_id: Callable[[], uuid.UUID],
) -> BoardScreen:
    kind = row.get("contract.type", "put")
    if kind != "put":
        raise ValueError(f"the board lists puts only, got {kind!r}")
    symbol = row["symbol"]
    if not isinstance(symbol, str):
        raise ValueError("symbol must be a string")
    expiration = row["contract.expiration"]
    if not isinstance(expiration, str):
        raise ValueError("contract.expiration must be YYYY-MM-DD")
    occ = OccSymbol(
        root=symbol,
        expiration=date.fromisoformat(expiration),
        right=OptionRight.PUT,
        strike=_decimal(row["contract.strike"]),
    )
    listed = row.get("contract.occSymbol")
    if listed is not None and OccSymbol.parse(str(listed)) != occ:
        raise ValueError("contract.occSymbol disagrees with the row's contract fields")
    score = row.get("wheelIq.score")
    row_id = row.get("rowId")
    return BoardScreen(
        evidence_id=new_id(),
        as_of=as_of,
        source_tool_call_ids=(request.tool_call_id,),
        build_id=build_id,
        row_id=row_id if isinstance(row_id, str) and row_id else f"{symbol}:{expiration}",
        underlying=symbol,
        occ_symbol=occ,
        bid=_decimal(row["contract.bid"]),
        wheel_iq_score=None if score is None else _decimal(score),
    )


def map_board_query(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`wheelta_board_query` -> board screens plus non-citable screen context (module doc)."""
    payload = request.payload
    if not isinstance(payload, dict):
        raise ValueError("board_query returns an object")
    freshness = payload["freshness"]
    if not isinstance(freshness, dict) or freshness.get("buildState") != "ready":
        raise ValueError("the board is not ready")
    build_id = freshness["buildId"]
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("freshness.buildId is required")
    as_of = _timestamp(freshness["asOf"])
    board = {
        "build_id": build_id,
        "as_of": as_of.isoformat(),
        "next_refresh_at": _context_value(freshness.get("nextRefreshAt")),
        "matched": _context_value(payload.get("matched")),
        "note": _context_value(payload.get("note")),
    }
    if payload.get("mode") == "groups":
        groups = payload.get("groups") or []
        if not isinstance(groups, list):
            raise ValueError("groups must be a list")
        context: list[dict[str, JsonValue]] = [
            {"board": board, "grouped_by": _context_value(payload.get("groupedBy"))},
            *({"group": _context_value(g)} for g in groups),
        ]
        return MappedEvidence(screen_context=tuple(context), gaps=(GROUPS_GAP,))
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    screens: list[BoardScreen] = []
    context_rows: list[dict[str, JsonValue]] = []
    gaps: list[str] = [CONTEXT_GAP]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row is an object")
        context_rows.append({str(k): _context_value(v) for k, v in row.items()})
        if any(row.get(k) is None for k in _IDENTITY):
            if IDENTITY_GAP not in gaps:
                gaps.append(IDENTITY_GAP)
            continue
        screens.append(_screen(row, build_id=build_id, as_of=as_of, request=request, new_id=new_id))
    occs = [str(s.occ_symbol) for s in screens]
    if len(occs) != len(set(occs)):
        raise ValueError("duplicate contract in one board result")
    return MappedEvidence(
        board_screens=tuple(screens),
        screen_context=({"board": board}, *context_rows),
        gaps=tuple(gaps),
    )


# --------------------------------------------------------------------------------------------
# wheelta_macro_snapshot (ADR-0045)
# --------------------------------------------------------------------------------------------

MACRO_SNAPSHOT_TOOL: Final = "wheelta_macro_snapshot"
MACRO_DIRECTION_GAP: Final = (
    "wheelta_macro_snapshot: direction and the regime summary/guidance are Wheelta's own "
    "reading, not a forecast"
)


def _observed(value: object) -> tuple[date, datetime | None]:
    """`updatedAt`/`observedAt`: a date (`2026-08-01`) or an offset timestamp."""
    if not isinstance(value, str):
        raise ValueError("expected a date or timestamp string")
    if len(value) == 10:
        return date.fromisoformat(value), None
    at = _timestamp(value)
    return at.date(), at


Num = Annotated[Decimal, BeforeValidator(_decimal)]


class _External(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _RegimeInput(_External):
    id: Annotated[str, Field(min_length=1)]
    value: Num
    observed_at: str = Field(alias="observedAt")


class _Regime(_External):
    tag: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    summary: str
    guidance: str
    inputs: tuple[_RegimeInput, ...]


class _Indicator(_External):
    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    category: Annotated[str, Field(min_length=1)]
    unit: Annotated[str, Field(min_length=1)]
    value: Num
    change: Num | None
    change_pct: Num | None = Field(alias="changePct")
    change_period: str = Field(alias="changePeriod")
    direction: str
    source: Annotated[str, Field(min_length=1)]
    updated_at: str = Field(alias="updatedAt")


class _MacroSnapshot(_External):
    as_of: str = Field(alias="asOf")
    regime: _Regime
    indicators: tuple[_Indicator, ...]


def map_macro_snapshot(request: MappingRequest, new_id: Callable[[], uuid.UUID]) -> MappedEvidence:
    """`wheelta_macro_snapshot` -> one `MacroRegime` and one `MacroIndicator` per series.

    `changePct` is a ratio (checked against `change` / previous value on the capture) and
    becomes `change_ratio`. Values keep Wheelta's units. `as_of` is the boundary's
    `retrieved_at`; the snapshot time and each observation date are carried explicitly.
    Descriptions (explanatory prose) are dropped. A repeated series raises.
    """
    parsed = _MacroSnapshot.model_validate(request.payload)
    snapshot_as_of = _timestamp(parsed.as_of)
    regime = MacroRegime(
        evidence_id=new_id(),
        as_of=request.retrieved_at,
        source_tool_call_ids=(request.tool_call_id,),
        snapshot_as_of=snapshot_as_of,
        tag=parsed.regime.tag,
        label=parsed.regime.label,
        summary=parsed.regime.summary,
        guidance=parsed.regime.guidance,
        inputs=tuple(
            MacroRegimeInput(series_id=i.id, value=i.value, observed_on=_observed(i.observed_at)[0])
            for i in parsed.regime.inputs
        ),
    )
    indicators: list[MacroIndicator] = []
    for ind in parsed.indicators:
        observed_on, observed_at = _observed(ind.updated_at)
        indicators.append(
            MacroIndicator(
                evidence_id=new_id(),
                as_of=request.retrieved_at,
                source_tool_call_ids=(request.tool_call_id,),
                series_id=ind.id,
                name=ind.name,
                category=ind.category,
                unit=ind.unit,
                value=ind.value,
                change=ind.change,
                change_ratio=ind.change_pct,
                change_period=ind.change_period,
                direction=ind.direction,
                source=ind.source,
                observed_on=observed_on,
                observed_at=observed_at,
            )
        )
    ids = [i.series_id for i in indicators]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate macro series")
    return MappedEvidence(
        macro_regimes=(regime,),
        macro_indicators=tuple(indicators),
        gaps=(MACRO_DIRECTION_GAP,),
    )


# Tool name -> mapper; `result_boundary.VERIFIED_MAPPERS` keys these by the Wheelta server.
WHEELTA_MAPPERS: Mapping[str, EvidenceMapper] = MappingProxyType(
    {BOARD_QUERY_TOOL: map_board_query, MACRO_SNAPSHOT_TOOL: map_macro_snapshot}
)
