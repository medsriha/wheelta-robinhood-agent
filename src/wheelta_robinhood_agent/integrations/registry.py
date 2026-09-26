"""Shared tool-registry types for MCP integrations (CLAUDE.md §8)."""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from wheelta_robinhood_agent.domain.enums import ToolTier


class ToolSpec(BaseModel):
    """One registered MCP tool. A tool not in a registry has no tier and is denied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    tier: ToolTier
    # Tier X only: true for the three option-order tools callable in armed live mode (ADR-0006).
    live_order_tool: bool = False


class ToolRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server: str
    tools: tuple[ToolSpec, ...]
    # False until the names/schemas are captured from our own tools/list (CLAUDE.md §9).
    verified: bool

    def qualified(self, name: str) -> str:
        """The Agent SDK's fully qualified name: `mcp__<server>__<tool>`."""
        return f"mcp__{self.server}__{name}"

    def by_tier(self, tier: ToolTier) -> tuple[ToolSpec, ...]:
        return tuple(t for t in self.tools if t.tier is tier)

    def get(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)


class DiscoveryDiff(BaseModel):
    """Discovered tools versus the registry after the server connects (CLAUDE.md §8).

    `unknown` tools are denied; `missing` expected tools fail the stage and alert.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unknown: frozenset[str]
    missing: frozenset[str]

    @property
    def ok(self) -> bool:
        return not self.missing


def diff_discovered(registry: ToolRegistry, discovered: Iterable[str]) -> DiscoveryDiff:
    """Compare unqualified tool names from the SDK init message with the registry.

    Only tools a mode would expose (not EXCLUDED, not denied X) count as expected; a missing
    denied tool is harmless.
    """
    found = frozenset(discovered)
    registered = frozenset(t.name for t in registry.tools)
    expected = frozenset(
        t.name
        for t in registry.tools
        if t.tier in (ToolTier.R, ToolTier.S) or (t.tier is ToolTier.X and t.live_order_tool)
    )
    return DiscoveryDiff(unknown=found - registered, missing=expected - found)


def make_registry(
    server: str,
    tiers: dict[ToolTier, Iterable[str]],
    *,
    verified: bool,
    live_order_tools: Iterable[str] = (),
) -> ToolRegistry:
    live = frozenset(live_order_tools)
    tools = tuple(
        ToolSpec(name=name, tier=tier, live_order_tool=name in live)
        for tier, names in tiers.items()
        for name in names
    )
    names = [t.name for t in tools]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate tool names in the {server} registry")
    if not live <= {t.name for t in tools if t.tier is ToolTier.X}:
        raise ValueError("live order tools must be registered as Tier X")
    return ToolRegistry(server=server, tools=tools, verified=verified)
