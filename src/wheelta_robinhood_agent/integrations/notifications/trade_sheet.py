"""POST filled option orders to the owner's Google Sheet trade log (ADR-0073).

The destination is a Google Apps Script web app bound to the sheet
(`scripts/trade_sheet_apps_script.gs`), which writes the Trade Log rows. The script skips
broker order IDs it has already written, so a repeated post is harmless and a transport error
is retried once. Apps Script answers POST with a redirect to the script's output, so redirects
are followed, and it always answers 200, so success is the body's `ok`.

The URL is a secret: never logged, never in a result. Never raises.
"""

from collections.abc import Mapping, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr

MAX_ATTEMPTS = 2


class TradeSheetResult(BaseModel):
    """Outcome of one post. `statuses` holds the script's per-order outcome, in order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delivered: bool
    attempts: int
    statuses: tuple[str, ...] = ()
    error: str | None = None


def post_trade_fills(
    fills: Sequence[Mapping[str, object]],
    *,
    client: httpx.Client,
    url: SecretStr,
    timeout_seconds: float,
) -> TradeSheetResult:
    error: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.post(
                url.get_secret_value(),
                json={"fills": list(fills)},
                timeout=timeout_seconds,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            error = type(exc).__name__
            continue
        if not response.is_success:
            return TradeSheetResult(
                delivered=False, attempts=attempt, error=f"http_{response.status_code}"
            )
        try:
            body = response.json()
        except ValueError:
            return TradeSheetResult(delivered=False, attempts=attempt, error="non_json_response")
        if not isinstance(body, dict) or body.get("ok") is not True:
            reason = body.get("error") if isinstance(body, dict) else None
            return TradeSheetResult(
                delivered=False,
                attempts=attempt,
                error=f"script_error: {reason}"[:200] if reason else "script_error",
            )
        results = body.get("results")
        statuses = (
            tuple(str(r.get("status")) for r in results if isinstance(r, dict))
            if isinstance(results, list)
            else ()
        )
        return TradeSheetResult(delivered=True, attempts=attempt, statuses=statuses)
    return TradeSheetResult(delivered=False, attempts=MAX_ATTEMPTS, error=error)
