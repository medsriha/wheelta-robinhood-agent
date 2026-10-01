"""Operator script: trace every decision of a recorded run, read-only (docs/OPERATIONS.md).

Shows, for one run, each decision's rationale and thesis, every cited reference resolved to
the recorded tool call or fact set, the code-computed facts and quotes per leg, each order
attempt with its review/place/cancel calls (simulated in a dry run, ADR-0038), the audit
findings attached to it, and the full tool-call timeline. Or lists recent runs to pick dry
and live runs to compare. Never writes to the ledger and never calls a broker.

Uses DATABASE_URL and APP_ENV from the usual settings (`.env` locally).

Usage:
  uv run python scripts/trace_run.py --list [N]            recent runs that started a session
  uv run python scripts/trace_run.py RUN_ID [--json]       one run's decision trace
  uv run python scripts/trace_run.py --slot 2026-09-28T15:05:00Z [--agent close|sell] [--json]

A due tick has two runs (ADR-0057): the Buy-to-Close run (`close`) and the Sell Options run
(`sell`). `--slot` without `--agent` traces the slot's last run that started a session.
"""

import argparse
import sys
import uuid
from datetime import datetime

from wheelta_robinhood_agent.agent.trace_loader import (
    RunListing,
    list_runs,
    load_decision_trace,
    run_id_at,
)
from wheelta_robinhood_agent.config.settings import SettingsError, load_settings
from wheelta_robinhood_agent.domain.enums import AgentRole
from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.errors import UnknownEntity
from wheelta_robinhood_agent.observability.decision_trace import render_trace_markdown


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("run_id", nargs="?", type=uuid.UUID, help="run to trace")
    target.add_argument("--slot", type=datetime.fromisoformat, help="trace the run at this slot")
    target.add_argument("--list", nargs="?", const=20, type=int, metavar="N", help="list runs")
    parser.add_argument(
        "--agent", type=AgentRole, choices=list(AgentRole), help="with --slot: which agent's run"
    )
    parser.add_argument("--json", action="store_true", help="print the trace as JSON")
    return parser.parse_args(argv)


def _value(item: object) -> str:
    return str(getattr(item, "value", item)) if item is not None else "?"


def _listing(rows: tuple[RunListing, ...]) -> str:
    head = (
        f"{'slot (UTC)':17} {'agent':5} {'status':14} {'mode':4} {'venue':9} "
        "dec att placed viol unv run_id"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"{r.slot:%Y-%m-%d %H:%M} {_value(r.agent):5} {_value(r.status):14} "
            f"{_value(r.effective_execution_mode):4} {_value(r.order_venue):9} "
            f"{r.decisions:3} {r.attempts:3} {r.placed:6} {r.violations:4} {r.unverifiable:3} "
            f"{r.run_id}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        settings = load_settings()
    except SettingsError as exc:
        sys.stderr.write(f"settings invalid: {exc}\n")
        return 1
    with connect(settings.DATABASE_URL) as conn:
        if args.list is not None:
            sys.stdout.write(_listing(list_runs(conn, settings.APP_ENV, args.list)))
            return 0
        try:
            run_id = args.run_id or run_id_at(conn, settings.APP_ENV, args.slot, args.agent)
            trace = load_decision_trace(conn, run_id)
        except UnknownEntity as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
    sys.stdout.write(
        trace.model_dump_json(indent=2) + "\n" if args.json else render_trace_markdown(trace)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
