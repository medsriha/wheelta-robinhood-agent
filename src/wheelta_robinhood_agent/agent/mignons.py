"""Orchestrator and research Mignons: who may call which tool (ADR-0025).

The main agent is the **orchestrator**. It reads account state, re-quotes before an order,
computes decision facts, maintains the workspace, places orders in armed live mode, and
spawns **Mignons** through the built-in `Agent` tool (Tier D). Mignons research and hand back
a typed `MignonReport` (domain/mignon_report.py). Web pages are read only by Mignons, so the
session that holds order tools never ingests fetched web content directly.

`ROLE_TOOLS` is the static allowlist per role, in qualified tool names. Access is enforced in
three layers (CLAUDE.md §8):

1. visibility: each Mignon's `AgentDefinition.tools` (the CLI hides every other tool from it,
   verified 2026-09-27 against CLI 2.1.283);
2. `allowed_tools` + `dontAsk` for the session (the union of the roles);
3. the PreToolUse hook (`agent/hooks.py`): a call is allowed only if its tool is in the
   caller's role, identified by the hook input's `agent_id`/`agent_type`.

Known limit: built-ins a Mignon uses must be in the session's `tools`, so the orchestrator
also *sees* WebSearch/WebFetch; the hook denies them on the main thread.

Model choice (ADR-0025 amendment): each Mignon type is offered once per model in the owner's
allowlist (`Settings.mignon_models`), as agent name `<type>--<model id>`
(`agent_name`/`parse_agent_name`). The orchestrator picks a model by picking the
`subagent_type`; each definition carries its exact pinned ID. The Agent tool's own `model`
input takes only aliases the CLI resolves itself (`sonnet`, `opus`, …), so it stays denied.

CLI behaviour this relies on (tests/e2e/test_e2e_mignons_cli.py):

- `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` makes `Agent` synchronous, so PostToolUse(Agent)
  receives the completed report and can replace it. Without it the CLI launches agents
  asynchronously and delivers reports outside any hook.
- `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1` removes the CLI's general-purpose agent types
  (which carry every tool), leaving only the Mignons defined here.
- `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` is a second limit behind the hook's own count.
- `Task` is the CLI's alias of `Agent`: disallowing it disables `Agent` too, so it is not in
  `disallowed_tools`; it has no tier and the hook denies it.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from claude_agent_sdk.types import AgentDefinition

from wheelta_robinhood_agent.agent.web_cache import LOCAL_SERVER_NAME, WEB_CACHE_TOOL_NAME
from wheelta_robinhood_agent.config.rules import TradingRules
from wheelta_robinhood_agent.config.settings import MODEL_ID_PATTERN
from wheelta_robinhood_agent.domain.enums import MignonType
from wheelta_robinhood_agent.integrations.robinhood.registry import SERVER_NAME as ROBINHOOD
from wheelta_robinhood_agent.integrations.wheelta.registry import SERVER_NAME as WHEELTA

DELEGATION_TOOL: Final = "Agent"
AGENT_NAME_SEP: Final = "--"
WEB_TOOLS: Final = ("WebSearch", "WebFetch")
# Agent tool inputs the hook accepts; `model`, `cwd`, `run_in_background`, `name` etc. are
# denied so the orchestrator cannot change a Mignon's model, directory, or mode.
AGENT_INPUT_KEYS: Final = frozenset({"description", "prompt", "subagent_type"})


class Role(StrEnum):
    ORCHESTRATOR = "orchestrator"
    MARKET = MignonType.MARKET.value
    COMPANY = MignonType.COMPANY.value
    MACRO = MignonType.MACRO.value


def _rh(*tools: str) -> tuple[str, ...]:
    return tuple(f"mcp__{ROBINHOOD}__{t}" for t in tools)


def _wh(*tools: str) -> tuple[str, ...]:
    return tuple(f"mcp__{WHEELTA}__wheelta_{t}" for t in tools)


_WEB_CACHE = f"mcp__{LOCAL_SERVER_NAME}__{WEB_CACHE_TOOL_NAME}"
# Spelled out, not imported from facts_tool (hooks -> mignons -> facts_tool -> result_boundary
# -> hooks would cycle); tests pin it to FACTS_TOOL_NAME.
_FACTS = f"mcp__{LOCAL_SERVER_NAME}__get_decision_facts"

ROLE_TOOLS: Mapping[Role, frozenset[str]] = MappingProxyType(
    {
        Role.ORCHESTRATOR: frozenset(
            {
                DELEGATION_TOOL,
                _FACTS,
                # Account state (every account-scoped read) and the pre-order re-quote.
                *_rh(
                    "get_portfolio",
                    "get_realized_pnl",
                    "get_pnl_trade_history",
                    "get_limited_margin_upgrade_info",
                    "get_option_level_upgrade_info",
                    "get_equity_tradability",
                    "get_equity_tax_lots",
                    "get_equity_positions",
                    "get_equity_orders",
                    "get_option_positions",
                    "get_option_orders",
                    "get_equity_quotes",
                    "get_option_instruments",
                    "get_option_quotes",
                ),
                # Workspace reads and writes (Tier S) and the live order tools (Tier X).
                *_rh(
                    "get_scans",
                    "get_watchlists",
                    "get_watchlist_items",
                    "get_option_watchlist",
                    "get_alerts",
                    "get_alert_log",
                    "create_scan",
                    "update_scan_filters",
                    "update_scan_config",
                    "create_watchlist",
                    "update_watchlist",
                    "add_to_watchlist",
                    "remove_from_watchlist",
                    "follow_watchlist",
                    "unfollow_watchlist",
                    "add_option_to_watchlist",
                    "remove_option_from_watchlist",
                    "create_alert",
                    "update_alert",
                    "delete_alert",
                    "mark_alerts_read",
                    "review_option_order",
                    "place_option_order",
                    "cancel_option_order",
                ),
            }
        ),
        Role.MARKET: frozenset(
            {
                *_rh(
                    "search",
                    "get_equity_quotes",
                    "get_equity_historicals",
                    "get_equity_price_book",
                    "get_equity_technical_indicators",
                    "get_option_chains",
                    "get_option_instruments",
                    "get_option_quotes",
                    "get_option_historicals",
                    "get_indexes",
                    "get_index_quotes",
                    "get_index_historicals",
                    "get_scanner_filter_specs",
                    "get_scanner_datapoints",
                    "preview_scan",
                    "get_popular_watchlists",
                ),
                *_wh(
                    "board_status",
                    "board_fields",
                    "board_query",
                    "board_row",
                    "assignment_rates",
                    "candles",
                    "quotes",
                    "correlations",
                ),
            }
        ),
        Role.COMPANY: frozenset(
            {
                *WEB_TOOLS,
                _WEB_CACHE,
                *_rh(
                    "search",
                    "get_equity_quotes",
                    "get_equity_fundamentals",
                    "get_equity_analyst_ratings",
                    "get_financials",
                    "get_sec_filing_index",
                    "get_sec_filing",
                    "get_sec_filing_facts",
                    "get_sec_filing_facts_catalog",
                    "get_earnings_calendar",
                    "get_earnings_results",
                    "get_politician_trades",
                ),
                *_wh("company_research", "calendar_events"),
            }
        ),
        Role.MACRO: frozenset(
            {
                *WEB_TOOLS,
                _WEB_CACHE,
                *_rh("get_indexes", "get_index_quotes", "get_index_historicals"),
                *_rh("get_earnings_calendar"),
                *_wh("macro_snapshot", "macro_series", "calendar_events", "correlations"),
            }
        ),
    }
)

MIGNON_DESCRIPTIONS: Mapping[MignonType, str] = MappingProxyType(
    {
        MignonType.MARKET: (
            "Market research: live quotes, option chains and quotes, historicals, technicals, "
            "the Robinhood scanner preview, and the Wheelta candidate board. Returns a "
            "MignonReport with code-issued candidate and evidence refs."
        ),
        MignonType.COMPANY: (
            "Company research: fundamentals, financials, SEC filings, earnings, analyst "
            "ratings, Wheelta company research and calendar, and trusted web sources. Returns "
            "a MignonReport."
        ),
        MignonType.MACRO: (
            "Macro research: Wheelta macro regime and series, index data, market calendar, "
            "correlations, and trusted web sources. Returns a MignonReport."
        ),
    }
)


# Orchestrator-facing guidance per known model ID: price per 1M input/output tokens from
# Anthropic's model table (cached 2026-06-24, docs/REFERENCES.md) and typical use. An ID
# without an entry is offered with no guidance.
MODEL_GUIDANCE: Mapping[str, str] = MappingProxyType(
    {
        "claude-haiku-4-5": "$1/$5 per 1M tokens; fastest and cheapest; simple lookups",
        "claude-sonnet-5": "$2/$10 per 1M tokens; routine screening and structured research",
        "claude-opus-4-8": "$5/$25 per 1M tokens; deep analysis and judgment",
        "claude-opus-5": "$5/$25 per 1M tokens; newest Opus; the hardest analysis",
    }
)


def agent_name(mignon: MignonType, model: str) -> str:
    """The `subagent_type` of one Mignon type on one pinned model."""
    return f"{mignon.value}{AGENT_NAME_SEP}{model}"


def parse_agent_name(name: object) -> tuple[MignonType, str] | None:
    """(type, model) from an agent name, or None when it is not `<Mignon type>--<model id>`."""
    if not isinstance(name, str) or AGENT_NAME_SEP not in name:
        return None
    kind, model = name.split(AGENT_NAME_SEP, 1)
    try:
        mignon = MignonType(kind)
    except ValueError:
        return None
    return (mignon, model) if MODEL_ID_PATTERN.fullmatch(model) else None


def role_of(agent_type: object, models: Iterable[str]) -> Role | None:
    """The Mignon role named by a hook's `agent_type` on an allowed model, else None."""
    parsed = parse_agent_name(agent_type)
    if parsed is None or parsed[1] not in frozenset(models):
        return None
    return Role(parsed[0].value)


@dataclass(frozen=True, slots=True)
class MignonLimits:
    """Integer limits from `rules.mignons`; None from `mignon_limits` disables Mignons."""

    max_per_run: int
    max_concurrent: int
    max_turns_per_mignon: int


def mignon_limits(rules: TradingRules) -> MignonLimits | None:
    """The run's Mignon limits, or None when any is TBD/none/agent_discretion or zero.

    A marker cannot be enforced as a count, so Mignons are then disabled (fail closed)."""
    m = rules.mignons
    values = (m.max_per_run, m.max_concurrent, m.max_turns_per_mignon)
    if not all(isinstance(v, int) and v >= 1 for v in values):
        return None
    per_run, concurrent, turns = (int(v) for v in values)
    return MignonLimits(per_run, concurrent, turns)


def cli_env(limits: MignonLimits | None) -> dict[str, str]:
    """CLI variables the Mignon design depends on (module docstring). Always set, so the
    general-purpose agent types are never offered even when Mignons are disabled."""
    env = {
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
    }
    if limits is not None:
        env["CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS"] = str(limits.max_concurrent)
    return env


def role_allowed(role: Role, allowed_tools: Iterable[str]) -> tuple[str, ...]:
    """The role's tools that are allowed this run, sorted (withheld sources drop out)."""
    return tuple(sorted(ROLE_TOOLS[role] & frozenset(allowed_tools)))


def build_agent_definitions(
    *,
    prompts: Mapping[MignonType, str],
    allowed_tools: Iterable[str],
    models: Sequence[str],
    limits: MignonLimits,
) -> dict[str, AgentDefinition]:
    """One `AgentDefinition` per Mignon type and allowed model: the type's rendered prompt and
    allowed tools, that exact model ID, and the turn cap. A type with no allowed tool is not
    offered."""
    allowed = tuple(allowed_tools)
    definitions: dict[str, AgentDefinition] = {}
    for mignon in MignonType:
        tools = role_allowed(Role(mignon.value), allowed)
        if not tools:
            continue
        for model in models:
            # The CLI lists every definition's description (and tools) to the orchestrator;
            # the type and model guidance are in its prompt once, so this stays a pointer.
            definitions[agent_name(mignon, model)] = AgentDefinition(
                description=(
                    f"{mignon.value} on {model}: see Mignon types and Models in your tool table."
                ),
                prompt=prompts[mignon],
                tools=list(tools),
                model=model,
                maxTurns=limits.max_turns_per_mignon,
                background=False,
            )
    return definitions
