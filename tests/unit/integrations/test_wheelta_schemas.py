"""Wheelta result parsing: freshness, board status, board_building state, error codes.

Payloads mirror /Users/deepset/wheelta-mcp (tests/fixtures/board.json meta, captured
2026-08-25; tool-error text from src/wheelta_mcp/errors.py `describe`).
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from wheelta_robinhood_agent.integrations.wheelta.schemas import (
    BoardBuilding,
    BoardFreshness,
    BoardQueryResult,
    BoardStatus,
    WheeltaError,
    WheeltaErrorCode,
    parse_error_envelope,
    parse_tool_error,
)

STATUS = {
    "asOf": "2026-08-25T19:35:00Z",
    "buildId": "build-2026-08-25-1530",
    "buildState": "ready",
    "universeRows": 26,
    "distinctSymbols": 10,
    "nextRefreshAt": "2026-08-26T11:45:00Z",
    "dteBuckets": [{"bucket": "short", "minDte": 14, "maxDte": 21, "rows": 9}],
    "funnel": {"screened": 812},
    "scoreRange": [41.2, 88.7],
    "sectors": ["Energy", "Technology"],
    "someFutureField": 1,
}


def test_board_status_parses_and_normalizes_utc() -> None:
    status = BoardStatus.model_validate(STATUS)
    assert status.as_of == datetime(2026, 8, 25, 19, 35, tzinfo=UTC)
    assert status.build_id == "build-2026-08-25-1530"
    assert status.next_refresh_at == datetime(2026, 8, 26, 11, 45, tzinfo=UTC)
    assert status.freshness.build_id == status.build_id
    assert status.sectors == ("Energy", "Technology")


def test_offset_timestamps_become_utc() -> None:
    f = BoardFreshness.model_validate({"asOf": "2026-08-25T15:35:00-04:00"})
    assert f.as_of == datetime(2026, 8, 25, 19, 35, tzinfo=UTC)
    assert f.as_of.tzinfo is UTC
    assert f.build_id is None


@pytest.mark.parametrize("as_of", ["", "2026-08-25T19:35:00", "yesterday"])
def test_bad_or_naive_as_of_rejected(as_of: str) -> None:
    with pytest.raises(ValidationError):
        BoardFreshness.model_validate({"asOf": as_of})


def test_missing_required_status_field_rejected() -> None:
    with pytest.raises(ValidationError):
        BoardStatus.model_validate({k: v for k, v in STATUS.items() if k != "universeRows"})


def test_board_query_rows_are_decimal_screens() -> None:
    result = BoardQueryResult.model_validate(
        {
            "mode": "rows",
            "matched": 1,
            "universeRows": 26,
            "returned": 1,
            "freshness": {"asOf": "2026-08-25T19:35:00Z", "buildId": "b1"},
            "rows": [{"symbol": "MSFT", "contract.bid": 0.43, "contract.greeks.delta": -0.29}],
        }
    )
    assert result.freshness.build_id == "b1"
    assert result.screen_rows is not None
    row = result.screen_rows[0]
    assert row["contract.bid"] == Decimal("0.43")
    assert row["contract.greeks.delta"] == Decimal("-0.29")
    assert row["symbol"] == "MSFT"


BUILDING_TEXT = (
    "The screener board is being rebuilt — this is the normal first-of-day replacing build, "
    "not an outage. Retry after 45s. The API deliberately refuses to serve the previous day's "
    "rows, so there is no stale result to fall back on. Macro, candles, calendar and research "
    "tools are unaffected.\n\n(requestId: req-bb)"
)


def test_board_building_tool_error_is_a_state() -> None:
    out = parse_tool_error(BUILDING_TEXT)
    assert out == BoardBuilding(retry_after_seconds=45, request_id="req-bb")
    assert out.state == "board_building"


def test_board_building_envelope_is_a_state() -> None:
    body = {"error": {"code": "board_building", "message": "building", "requestId": "r1"}}
    assert parse_error_envelope(body, 30) == BoardBuilding(retry_after_seconds=30, request_id="r1")


def test_rate_limited_carries_retry_after() -> None:
    text = (
        "Rate limited by the Wheelta API. The budget is shared across every process using "
        "this API key, so it may not be this session that spent it. Retry after 12s."
    )
    out = parse_tool_error(text)
    assert isinstance(out, WheeltaError)
    assert (out.code, out.retry_after_seconds, out.retryable) == (
        WheeltaErrorCode.RATE_LIMITED,
        12,
        True,
    )


def test_envelope_codes_and_retry_after() -> None:
    body = {"error": {"code": "rate_limited", "message": "slow down", "requestId": "r2"}}
    out = parse_error_envelope(body, 60)
    assert isinstance(out, WheeltaError)
    assert (out.code, out.retry_after_seconds, out.request_id) == (
        WheeltaErrorCode.RATE_LIMITED,
        60,
        "r2",
    )
    with pytest.raises(ValidationError):
        parse_error_envelope(body, 3601)


def test_unauthorized_not_retryable() -> None:
    out = parse_tool_error(
        "The Wheelta API rejected this server's credentials. WHEELTA_SERVICE_API_KEY is missing"
    )
    assert isinstance(out, WheeltaError)
    assert out.code is WheeltaErrorCode.UNAUTHORIZED
    assert not out.retryable


def test_fallback_text_maps_code() -> None:
    out = parse_tool_error("Wheelta API error (413 request_too_large): too big")
    assert isinstance(out, WheeltaError) and out.code is WheeltaErrorCode.REQUEST_TOO_LARGE


@pytest.mark.parametrize(
    "text",
    [
        "Unknown column 'contract.spreadPctx'. Call wheelta_board_fields for the full catalog.",
        "Wheelta API error (418 teapot): no",
        "",
    ],
)
def test_unknown_text_fails_closed(text: str) -> None:
    out = parse_tool_error(text)
    assert isinstance(out, WheeltaError)
    assert out.code is WheeltaErrorCode.UNRECOGNIZED
    assert not out.retryable


def test_unknown_envelope_code_is_unrecognized() -> None:
    out = parse_error_envelope({"error": {"code": "brand_new", "message": "?"}})
    assert isinstance(out, WheeltaError) and out.code is WheeltaErrorCode.UNRECOGNIZED


def test_malformed_envelope_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_error_envelope({"detail": "nope"})
