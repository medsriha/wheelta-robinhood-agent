"""Operator script: run one read-only SQL query against the ledger, for debugging.

The session is forced read-only (`default_transaction_read_only`), so INSERT/UPDATE/DELETE/DDL
fail in Postgres. Locally it uses DATABASE_URL; with `--prod`, LEDGER_READONLY_URL, the
production ledger as the `ledger_reader` role (SELECT only, no `oauth_credentials`). Never
prints a connection URL.

Usage:
  uv run --env-file .env python scripts/ledger_query.py [--prod] "SELECT ..." [--limit N] [--full]
  uv run --env-file .env python scripts/ledger_query.py [--prod] --schema [TABLE]
"""

import argparse
import sys

import psycopg

from wheelta_robinhood_agent.config.settings import (
    SettingsError,
    load_database_url,
    load_ledger_readonly_url,
)
from wheelta_robinhood_agent.ledger.db import connect
from wheelta_robinhood_agent.ledger.errors import LedgerUnavailable

CELL_LIMIT = 300

_SCHEMA_SQL = """
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public' AND (%(table)s::text IS NULL OR table_name = %(table)s)
ORDER BY table_name, ordinal_position
"""


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("sql", nargs="?", help="one SELECT statement")
    target.add_argument("--schema", nargs="?", const="", metavar="TABLE", help="list columns")
    parser.add_argument("--prod", action="store_true", help="production, as ledger_reader")
    parser.add_argument("--limit", type=int, default=200, help="max rows printed (default 200)")
    parser.add_argument("--full", action="store_true", help=f"don't cut cells at {CELL_LIMIT}")
    return parser.parse_args(argv)


def _cell(value: object, full: bool) -> str:
    text = "" if value is None else str(value).replace("\t", " ").replace("\n", " ")
    return text if full or len(text) <= CELL_LIMIT else text[:CELL_LIMIT] + "…"


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        url = load_ledger_readonly_url() if args.prod else load_database_url()
    except SettingsError as exc:
        sys.stderr.write(f"settings invalid: {exc}\n")
        return 1
    try:
        with connect(url, application_name="ledger_query") as conn:
            conn.execute("SET default_transaction_read_only = on")
            if args.schema is not None:
                cur = conn.execute(_SCHEMA_SQL, {"table": args.schema or None})
            else:
                cur = conn.execute(args.sql)
            if cur.description is None:
                sys.stdout.write("(no result set)\n")
                return 0
            sys.stdout.write("\t".join(col.name for col in cur.description) + "\n")
            rows = cur.fetchmany(args.limit + 1)
            for row in rows[: args.limit]:
                sys.stdout.write("\t".join(_cell(v, args.full) for v in row) + "\n")
            if len(rows) > args.limit:
                sys.stdout.write(f"(more than {args.limit} rows; raise --limit or narrow it)\n")
    except LedgerUnavailable as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    except psycopg.Error as exc:
        sys.stderr.write(f"query failed: {type(exc).__name__}: {exc.diag.message_primary}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
