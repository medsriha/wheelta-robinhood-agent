"""Typed ledger errors (CLAUDE.md §14). None of them carries the database URL."""


class LedgerError(Exception):
    """Base class for ledger failures. A ledger failure fails the stage closed (CLAUDE.md §2.7)."""


class LedgerUnavailable(LedgerError):
    """The database could not be reached or the connection failed."""


class MigrationError(LedgerError):
    """A migration file is malformed, missing, changed after being applied, or failed."""


class IdentityConflict(LedgerError):
    """An identity row exists with attributes that contradict the requested insert."""


class DedupConflict(LedgerError):
    """An event with the same source dedup key exists but records a different event type."""


class UnknownEntity(LedgerError):
    """The referenced identity row does not exist."""
