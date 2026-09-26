"""Code-issued ledger IDs (INTERFACES.md: IDs are UUIDv7 unless specified otherwise)."""

import os
import time
import uuid


def new_id() -> uuid.UUID:
    """A UUIDv7 (RFC 9562): 48-bit Unix milliseconds, version 7, variant 10, 74 random bits.

    Time-ordered IDs keep B-tree inserts local. Ordering is not a causality claim; events use
    per-entity sequences and source timestamps for that (INTERFACES.md).
    """
    unix_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (unix_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= ((rand >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= rand & ((1 << 62) - 1)
    return uuid.UUID(int=value)
