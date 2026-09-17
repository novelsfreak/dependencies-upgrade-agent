"""
Day 6's concurrency proof: two connections (standing in for two
workers) racing for the same repo's advisory lock must never both
succeed, and the loser must succeed once the winner releases.

Requires a live Postgres (see docker-compose.yml). No Docker sandbox
needed -- this tests the locking primitive directly, in milliseconds.
"""
import os

import psycopg
import pytest

from core.repo_lock import try_lock_repo, unlock_repo

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def two_connections():
    conn_a = psycopg.connect(DSN, autocommit=False)
    conn_b = psycopg.connect(DSN, autocommit=False)
    yield conn_a, conn_b
    conn_a.close()
    conn_b.close()


def test_second_connection_blocked_while_first_holds_lock(two_connections):
    conn_a, conn_b = two_connections
    repo_key = "repo:test-lock-contention"

    assert try_lock_repo(conn_a, repo_key) is True
    assert try_lock_repo(conn_b, repo_key) is False, "second connection should not get the lock"

    unlock_repo(conn_a, repo_key)
    assert try_lock_repo(conn_b, repo_key) is True, "lock should be free once the holder releases"
    unlock_repo(conn_b, repo_key)


def test_different_repos_dont_contend(two_connections):
    conn_a, conn_b = two_connections
    assert try_lock_repo(conn_a, "repo:one") is True
    assert try_lock_repo(conn_b, "repo:two") is True
    unlock_repo(conn_a, "repo:one")
    unlock_repo(conn_b, "repo:two")


def test_lock_released_when_connection_closes():
    """
    The core promise from the plan: a killed worker's session-scoped
    lock must not survive it -- closing the connection is what a
    kill -9 does to the TCP session, without giving us a chance to run
    a finally block at all.
    """
    repo_key = "repo:test-lock-dies-with-connection"
    holder = psycopg.connect(DSN, autocommit=False)
    assert try_lock_repo(holder, repo_key) is True
    holder.close()

    checker = psycopg.connect(DSN, autocommit=False)
    try:
        assert try_lock_repo(checker, repo_key) is True, "lock should be free after holder's connection died"
        unlock_repo(checker, repo_key)
    finally:
        checker.close()
