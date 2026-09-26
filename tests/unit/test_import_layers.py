"""Enforce the layering rules in CLAUDE.md §3 and each module README."""

import ast
from pathlib import Path

PACKAGE = "wheelta_robinhood_agent"
ROOT = Path(__file__).resolve().parents[2] / "src" / PACKAGE

# Which top-level packages each package may import (module READMEs).
ALLOWED: dict[str, set[str]] = {
    "domain": set(),
    "config": {"domain"},
    "observability": {"domain", "config"},
    "integrations": {"domain", "config", "observability"},
    "ledger": {"domain", "config", "observability"},
    "agent": {"domain", "config", "integrations", "ledger", "observability"},
    "orchestrator": {"domain", "config", "integrations", "ledger", "observability", "agent"},
}
DOMAIN_THIRD_PARTY = {"pydantic"}


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_layering() -> None:
    violations = []
    for package, allowed in ALLOWED.items():
        for path in (ROOT / package).rglob("*.py"):
            for name in _imports(path):
                parts = name.split(".")
                if parts[0] == PACKAGE and len(parts) > 1:
                    target = parts[1]
                    if target != package and target not in allowed:
                        violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not violations, violations


def test_integrations_subpackages_do_not_import_each_other() -> None:
    violations = []
    for path in (ROOT / "integrations").rglob("*.py"):
        rel = path.relative_to(ROOT / "integrations").parts
        own = rel[0] if len(rel) > 1 else None
        for name in _imports(path):
            parts = name.split(".")
            if parts[:2] == [PACKAGE, "integrations"] and len(parts) > 3:
                sub = parts[2]
                if own is not None and sub != own and sub != "registry":
                    violations.append(f"{path} imports {name}")
    assert not violations, violations


def test_domain_imports_only_stdlib_and_pydantic() -> None:
    import sys

    violations = []
    for path in (ROOT / "domain").rglob("*.py"):
        for name in _imports(path):
            top = name.split(".")[0]
            if top in (PACKAGE, *DOMAIN_THIRD_PARTY) or top in sys.stdlib_module_names:
                continue
            violations.append(f"{path.name} imports {name}")
    assert not violations, violations
