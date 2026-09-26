"""Single-flight advisory lock contention between sessions."""

from collections.abc import Callable

import psycopg

from wheelta_robinhood_agent.domain.enums import AppEnv
from wheelta_robinhood_agent.ledger.lock import single_flight

Conn = psycopg.Connection[tuple[object, ...]]


def test_second_session_is_refused_until_release(conn_factory: Callable[[], Conn]) -> None:
    a, b = conn_factory(), conn_factory()
    with single_flight(a, AppEnv.STAGING) as got_a:
        assert got_a is True
        with single_flight(b, AppEnv.STAGING) as got_b:
            assert got_b is False
        with single_flight(b, AppEnv.PRODUCTION) as got_other_env:
            assert got_other_env is True
    with single_flight(b, AppEnv.STAGING) as got_b_after:
        assert got_b_after is True


def test_lock_is_released_when_the_holder_disconnects(conn_factory: Callable[[], Conn]) -> None:
    a, b = conn_factory(), conn_factory()
    with single_flight(a, AppEnv.LOCAL) as got_a:
        assert got_a
        a.close()
        with single_flight(b, AppEnv.LOCAL) as got_b:
            assert got_b is True


def test_not_acquired_does_not_release_the_holders_lock(conn_factory: Callable[[], Conn]) -> None:
    a, b, c = conn_factory(), conn_factory(), conn_factory()
    with single_flight(a, AppEnv.LOCAL) as got_a:
        assert got_a
        with single_flight(b, AppEnv.LOCAL) as got_b:
            assert not got_b
        with single_flight(c, AppEnv.LOCAL) as got_c:
            assert not got_c
