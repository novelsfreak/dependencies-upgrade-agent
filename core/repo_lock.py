# core/repo_lock.py
#
# Postgres advisory lock, per repo, so two runs against the same repo
# never touch the shared install-phase cache volume (npm-cache,
# uv-cache) at the same time. Session-scoped, not transaction-scoped:
# it's meant to survive across the several statements/commits inside
# handle_building, and it's released automatically if the holding
# connection dies -- which is exactly the behavior we want from a
# killed worker (see HeartbeatGuard's docstring for the same theme).
#
# Deliberately NOT held across handle_building AND handle_testing:
# those are two separately-claimable states, so the run that acquires
# this lock in handle_building might have its TESTING step picked up by
# a *different* worker's connection later. Trying to release a lock
# from a connection that never held it is a silent no-op in Postgres,
# not an error -- which would leak this lock on the original worker's
# connection until that worker process exits. Scoping it to
# handle_building's own install+build (the only phase that actually
# touches the shared cache) avoids that failure mode entirely, at the
# cost of not literally matching "held across build and test" -- see
# the note in core/states.py where this is used.
from __future__ import annotations

import psycopg


class LockContention(Exception):
    """
    Couldn't get the per-repo lock -- another run against this same repo
    is currently in its install/build phase. This is NOT a failure:
    the worker loop must requeue with backoff (backpressure), not bump
    attempt or record it as an error the way a real build failure would.
    """
    def __init__(self, repo_key: str):
        self.repo_key = repo_key
        super().__init__(f"could not acquire lock for {repo_key}")


def try_lock_repo(conn: psycopg.Connection, repo_key: str) -> bool:
    row = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (repo_key,)).fetchone()
    conn.commit()
    return bool(row[0])


def unlock_repo(conn: psycopg.Connection, repo_key: str) -> None:
    conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (repo_key,))
    conn.commit()
