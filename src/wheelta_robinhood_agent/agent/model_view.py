"""The model view: what the model reads of a validated envelope (ADR-0037).

The ledger keeps every validated envelope in full; the facts tool, the post-run audit, and
the broker ledger read that. The model receives `model_view(envelope)` instead, a
deterministic, versioned rendering that drops repetition but no fact the model can use, and
the DELIVERED row records exactly that rendering. `run_loader` resolves a delivered view
through the validated envelope of the same call and rebuilds the view to confirm it matches.

Kept apart from `result_boundary` so `hooks` and `proxy` can import it without a cycle.
"""

import json
import re
from collections.abc import Mapping
from typing import Any, Final, cast

from pydantic import JsonValue

EVIDENCE_KEY: Final = "evidence"
EVIDENCE_REF_KEY: Final = "evidence_ref"
EVIDENCE_VIEW_KEY: Final = "evidence_view"
EVIDENCE_VIEW_VERSION: Final = "compact.v1"
# ADR-0052: the top-level envelope key carrying a review/place/cancel call's `order_call:`
# ref. The view keeps top-level keys, so the model always receives it.
ORDER_CALL_REF_KEY: Final = "order_call_ref"
# Code-issued row identities and internal cross-references. The model cites only the
# envelope's `evidence_ref`, `order_call_ref`, and `candidate:` refs, and passes broker IDs to
# tools; it has no use for these, and nothing it outputs may contain them.
_INTERNAL_ID_KEYS: Final = frozenset(
    {
        "evidence_id",
        "quote_id",
        "snapshot_id",
        "instrument_evidence_id",
        "evidence_ids",
        "csp_cash_base_evidence_ids",
        "positions_ref",
        "open_orders_ref",
        "tax_lots_ref",
    }
)
# Provenance lists that always name the envelope's own call (`_check_provenance`); dropped
# only when they do, so a list naming any other call is kept.
_SOURCE_CALL_KEYS: Final = frozenset({"source_tool_call_ids", "tool_call_ids"})
# Columns placed first in a table so each row starts with what identifies it.
_LEADING_COLUMNS: Final[tuple[str, ...]] = (
    "candidate_ref",
    "broker_instrument_id",
    "symbol",
    "underlying",
)
_DECIMAL_STRING: Final = re.compile(r"^-?\d+\.\d+$")


def model_view(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """The envelope as the model receives it (ADR-0037). Pure and deterministic.

    Only an envelope carrying mapped evidence changes; any other comes back as a copy.
    The view keeps every fact the model can use and removes only repetition:

    - evidence categories with no items are omitted;
    - code-internal row IDs (`_INTERNAL_ID_KEYS`) are omitted, as are provenance lists and
      `retrieved_at` values that only repeat the envelope's own `tool_call_id` and
      `retrieved_at`;
    - decimal strings lose trailing zeros (`"0.040000"` -> `"0.04"`; same value);
    - a category with two or more items becomes a table: `common` holds each field whose
      value is the same in every row, `columns` names the rest, and each of `rows` lists
      those values in `columns` order. A nested object of plain values (an OCC symbol)
      becomes dotted columns (`occ_symbol.strike`). Rows keep their original order.
    """
    envelope = dict(envelope)
    data = envelope.get("data")
    if not isinstance(data, dict) or EVIDENCE_REF_KEY not in data:
        return envelope
    evidence = data.get(EVIDENCE_KEY)
    if not isinstance(evidence, dict) or EVIDENCE_VIEW_KEY in data:
        return envelope
    call_id = envelope.get("tool_call_id")
    retrieved_at = envelope.get("retrieved_at")
    view: dict[str, JsonValue] = {}
    for category, items in evidence.items():
        if items == [] or items is None:
            continue
        cleaned = _strip(items, call_id, retrieved_at)
        if isinstance(cleaned, list) and len(cleaned) >= 2:
            view[category] = tabulate(cleaned)
        else:
            view[category] = cleaned
    return {
        **envelope,
        "data": {
            **{k: v for k, v in data.items() if k != EVIDENCE_KEY},
            EVIDENCE_VIEW_KEY: EVIDENCE_VIEW_VERSION,
            EVIDENCE_KEY: view,
        },
    }


def is_model_view(envelope: Mapping[str, Any]) -> bool:
    """Whether `envelope` is a model view (it names its view version)."""
    data = envelope.get("data")
    return isinstance(data, dict) and EVIDENCE_VIEW_KEY in data


def _strip(value: JsonValue, call_id: JsonValue, retrieved_at: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if key in _INTERNAL_ID_KEYS:
                continue
            if key in _SOURCE_CALL_KEYS and item == [call_id]:
                continue
            if key == "retrieved_at" and item == retrieved_at:
                continue
            out[key] = _strip(item, call_id, retrieved_at)
        return out
    if isinstance(value, list):
        return [_strip(item, call_id, retrieved_at) for item in value]
    if isinstance(value, str) and _DECIMAL_STRING.fullmatch(value):
        return value.rstrip("0").rstrip(".")
    return value


def _is_plain(value: JsonValue) -> bool:
    return not isinstance(value, dict | list)


def flatten(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """One level of dotted keys for nested objects whose values are all plain."""
    out: dict[str, JsonValue] = {}
    for key, value in row.items():
        if isinstance(value, dict) and value and all(_is_plain(v) for v in value.values()):
            for sub, item in value.items():
                out[f"{key}.{sub}"] = item
        else:
            out[key] = value
    return out


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True)


def tabulate(items: list[JsonValue]) -> JsonValue:
    """`items` as {common, columns, rows}; unchanged unless every item is an object with the
    same keys (after flattening, or else before it)."""
    if not all(isinstance(i, dict) for i in items):
        return items
    rows = [cast(dict[str, JsonValue], i) for i in items]
    flat = [flatten(r) for r in rows]
    if len({frozenset(r) for r in flat}) != 1:
        flat = rows
        if len({frozenset(r) for r in flat}) != 1:
            return items
    keys = [k for k in _LEADING_COLUMNS if k in flat[0]]
    keys += sorted(k for k in flat[0] if k not in _LEADING_COLUMNS)
    common = {k: flat[0][k] for k in keys if len({_canonical(r[k]) for r in flat}) == 1}
    columns = [k for k in keys if k not in common]
    return {
        "common": dict(sorted(common.items())),
        "columns": cast(JsonValue, columns),
        "rows": [[r[k] for k in columns] for r in flat],
    }
