"""The arguments code lets a Mignon send to Tavily (ADR-0058).

Pure: no I/O, no clock. The `agent/` `PreToolUse` hook calls `effective_input` and sends the
result as `updatedInput`; on `TavilyInputError` it denies the call with the error as the
reason, so the Mignon can correct its arguments.

Each tool accepts only the arguments listed here, within bounds, and code sets the rest:

- no images, image descriptions, favicons, or raw page content in a search: they add size and
  no citable text (a search result is a lead; `tavily_extract` reads the page);
- extract returns markdown without images or favicons, at most `MAX_EXTRACT_URLS` URLs a call
  (Tavily bills 1 credit per 5 URLs, basic depth);
- `topic` has one value on the server (`general`, captured 2026-09-30), so it is not sent.

Bounds are operational (result size and credits), not trading values.
"""

import re
from collections.abc import Mapping
from typing import Any, Final

from wheelta_robinhood_agent.integrations.websearch.registry import EXTRACT_TOOL, SEARCH_TOOL

MAX_QUERY_CHARS: Final = 400
MAX_SEARCH_RESULTS: Final = 10
DEFAULT_SEARCH_RESULTS: Final = 5
MAX_DOMAINS: Final = 50
MAX_EXTRACT_URLS: Final = 5
MAX_URL_CHARS: Final = 2048

SEARCH_DEPTHS: Final = frozenset({"basic", "advanced"})
EXTRACT_DEPTHS: Final = frozenset({"basic", "advanced"})
TIME_RANGES: Final = frozenset({"day", "week", "month", "year"})

SEARCH_FIXED: Final[Mapping[str, Any]] = {
    "include_images": False,
    "include_image_descriptions": False,
    "include_raw_content": False,
    "include_favicon": False,
}
EXTRACT_FIXED: Final[Mapping[str, Any]] = {
    "format": "markdown",
    "include_images": False,
    "include_favicon": False,
}
SEARCH_ARGS: Final = frozenset(
    {
        "query",
        "max_results",
        "search_depth",
        "time_range",
        "start_date",
        "end_date",
        "include_domains",
        "exclude_domains",
        "exact_match",
    }
)
EXTRACT_ARGS: Final = frozenset({"urls", "query", "extract_depth"})

_DATE_RE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_DOMAIN_RE: Final = re.compile(r"(?=.{1,253}$)[a-z0-9-]+(\.[a-z0-9-]+)+")
_URL_RE: Final = re.compile(r"https?://[^\s/?#]+\.[^\s/?#]+[^\s]*")


class TavilyInputError(ValueError):
    """The arguments are outside what code sends to Tavily. The hook denies the call."""


def effective_input(tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments sent upstream for one Tavily call. Raises TavilyInputError."""
    if tool == SEARCH_TOOL:
        return _search(arguments)
    if tool == EXTRACT_TOOL:
        return _extract(arguments)
    raise TavilyInputError(f"{tool} is not a Tavily tool code sends")


def _check_keys(
    tool: str, arguments: Mapping[str, Any], allowed: frozenset[str], fixed: Mapping[str, Any]
) -> None:
    """Only `allowed` arguments, plus a code-set argument given exactly its fixed value (the
    server's schema lists them, so a Mignon may repeat one)."""
    extra = sorted(
        k for k in arguments if k not in allowed and not (k in fixed and arguments[k] == fixed[k])
    )
    if extra:
        raise TavilyInputError(
            f"{tool} does not accept {extra} (code sets them); allowed arguments: {sorted(allowed)}"
        )


def _query(arguments: Mapping[str, Any], *, required: bool) -> str | None:
    value = arguments.get("query")
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TavilyInputError("query must be non-empty text")
    if len(value) > MAX_QUERY_CHARS:
        raise TavilyInputError(f"query must be at most {MAX_QUERY_CHARS} characters")
    return value


def _choice(arguments: Mapping[str, Any], arg: str, allowed: frozenset[str]) -> str | None:
    value = arguments.get(arg)
    if value is None:
        return None
    if value not in allowed:
        raise TavilyInputError(f"{arg} must be one of {sorted(allowed)}")
    return str(value)


def _domains(arguments: Mapping[str, Any], arg: str) -> list[str] | None:
    value = arguments.get(arg)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > MAX_DOMAINS:
        raise TavilyInputError(f"{arg} must be a list of at most {MAX_DOMAINS} domains")
    domains: list[str] = []
    for item in value:
        domain = item.strip().lower() if isinstance(item, str) else ""
        if not _DOMAIN_RE.fullmatch(domain):
            raise TavilyInputError(f"{arg} entries must be bare domains like 'sec.gov'")
        domains.append(domain)
    return domains


def _search(arguments: Mapping[str, Any]) -> dict[str, Any]:
    _check_keys(SEARCH_TOOL, arguments, SEARCH_ARGS, {**SEARCH_FIXED, "topic": "general"})
    out: dict[str, Any] = {"query": _query(arguments, required=True)}
    count = arguments.get("max_results", DEFAULT_SEARCH_RESULTS)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= MAX_SEARCH_RESULTS
    ):
        raise TavilyInputError(f"max_results must be an integer from 1 to {MAX_SEARCH_RESULTS}")
    out["max_results"] = count
    for arg, allowed in (("search_depth", SEARCH_DEPTHS), ("time_range", TIME_RANGES)):
        choice = _choice(arguments, arg, allowed)
        if choice is not None:
            out[arg] = choice
    for arg in ("start_date", "end_date"):
        value = arguments.get(arg)
        if value is None:
            continue
        if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
            raise TavilyInputError(f"{arg} must be a date written YYYY-MM-DD")
        out[arg] = value
    for arg in ("include_domains", "exclude_domains"):
        domains = _domains(arguments, arg)
        if domains is not None:
            out[arg] = domains
    exact = arguments.get("exact_match")
    if exact is not None:
        if not isinstance(exact, bool):
            raise TavilyInputError("exact_match must be true or false")
        out["exact_match"] = exact
    return {**out, **SEARCH_FIXED}


def _extract(arguments: Mapping[str, Any]) -> dict[str, Any]:
    _check_keys(EXTRACT_TOOL, arguments, EXTRACT_ARGS, EXTRACT_FIXED)
    urls = arguments.get("urls")
    if not isinstance(urls, list) or not 1 <= len(urls) <= MAX_EXTRACT_URLS:
        raise TavilyInputError(f"urls must be a list of 1 to {MAX_EXTRACT_URLS} URLs")
    for url in urls:
        if not isinstance(url, str) or len(url) > MAX_URL_CHARS or not _URL_RE.fullmatch(url):
            raise TavilyInputError("urls entries must be absolute http(s) URLs")
    if len(set(urls)) != len(urls):
        raise TavilyInputError("urls must not repeat a URL")
    out: dict[str, Any] = {"urls": list(urls)}
    query = _query(arguments, required=False)
    if query is not None:
        out["query"] = query
    depth = _choice(arguments, "extract_depth", EXTRACT_DEPTHS)
    if depth is not None:
        out["extract_depth"] = depth
    return {**out, **EXTRACT_FIXED}
