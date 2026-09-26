"""Ledger-backed implementations of the hook dependencies (CLAUDE.md §3: SQL stays in ledger/).

- `ledger_result_writer`: the `ResultWriter` for `LedgerToolEventRecorder` (results rows).
- `LedgerWorkspaceOwnership` / `LedgerWorkspaceCounter`: ownership and cap counts for Tier S.

Every Tier S target spec is still unverified (hooks.ROBINHOOD_WORKSPACE_TARGETS), so the hook
denies Tier S calls before consulting these. They fail closed where the ledger has no query:
there is no by-name lookup yet, so `by_name` raises and the hook denies and stops the session.
"""

import uuid
from dataclasses import dataclass
from typing import Final

import psycopg
from pydantic import JsonValue

from wheelta_robinhood_agent.agent.hooks import OwnedWorkspaceObject, WorkspaceKind
from wheelta_robinhood_agent.domain.orders import WorkspaceObjectKind
from wheelta_robinhood_agent.ledger import evidence as ledger_evidence
from wheelta_robinhood_agent.ledger import workspace as ledger_workspace

Conn = psycopg.Connection[tuple[object, ...]]

_KINDS: Final[dict[WorkspaceKind, WorkspaceObjectKind]] = {
    WorkspaceKind.WATCHLIST: WorkspaceObjectKind.WATCHLIST,
    WorkspaceKind.SCAN: WorkspaceObjectKind.SCAN,
    WorkspaceKind.ALERT: WorkspaceObjectKind.ALERT,
}


def ledger_result_writer(
    conn: Conn,
    *,
    run_id: uuid.UUID,
    tool_call_id: uuid.UUID,
    kind: str,
    payload: JsonValue,
) -> uuid.UUID:
    """Insert one `results` row (the payload is already redacted by the hook)."""
    return ledger_evidence.insert_result(
        conn,
        run_id=run_id,
        kind=ledger_evidence.ResultKind(kind),
        payload=payload,
        tool_call_id=tool_call_id,
    )


@dataclass(frozen=True)
class LedgerWorkspaceOwnership:
    conn: Conn
    account_scope_id: str
    prefix: str

    def by_id(self, kind: WorkspaceKind, object_id: str) -> OwnedWorkspaceObject | None:
        state = ledger_workspace.owned_object(
            self.conn, self.account_scope_id, _KINDS[kind], object_id
        )
        if state is None or not state.owned(self.prefix) or state.current_name is None:
            return None
        return OwnedWorkspaceObject(kind=kind, object_id=object_id, name=state.current_name)

    def by_name(self, kind: WorkspaceKind, name: str) -> OwnedWorkspaceObject | None:
        raise LookupError("no ledger lookup of workspace objects by name exists yet")


@dataclass(frozen=True)
class LedgerWorkspaceCounter:
    conn: Conn
    account_scope_id: str
    prefix: str
    run_id: uuid.UUID

    def _counts(self) -> ledger_workspace.WorkspaceCounts:
        return ledger_workspace.owned_counts(
            self.conn, self.account_scope_id, prefix=self.prefix, run_id=self.run_id
        )

    def mutations_this_run(self) -> int | None:
        return self._counts().mutations_in_run

    def owned_count(self, kind: WorkspaceKind) -> int | None:
        counts = self._counts()
        key = _KINDS[kind]
        # Ambiguous creates count against the cap (fail closed, ledger/workspace.py).
        return counts.active_owned.get(key, 0) + counts.ambiguous_creates.get(key, 0)

    def items_in(self, object_id: str) -> int | None:
        return None  # no item count is recorded yet: unknown, so the hook denies
