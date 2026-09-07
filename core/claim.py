"""
Claim, heartbeat, and release — the three operations that let N workers
share one queue of runs without stepping on each other, and let a run
survive a worker being kill -9'd mid-flight.

Nothing here is clever. That's the point: the cleverness is entirely in
the SQL, and the SQL is small enough to read in one sitting.
"""
from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row

# States a worker is allowed to pick up and act on. AWAITING_CI and
# AWAITING_REVIEW are deliberately excluded -- they only move on a webhook,
# never on a worker's own initiative. A run sitting in one of those states
# with an expired lease is not a bug; it's just waiting for the outside
# world.
ACTIONABLE_STATES = [
    "CREATED",
    "PATCHING",
    "BUILDING",
    "TESTING",
    "PATCH_READY",
    "PR_OPEN",
]

LEASE_DURATION = "2 minutes"


def claim(conn: psycopg.Connection, worker_id: str) -> dict[str, Any] | None:
    """
    Atomically pick one actionable, currently-unowned (or lease-expired)
    run, mark it owned by worker_id, and return it. Returns None if there
    is nothing to do right now.

    Safe to call from any number of concurrent workers against the same
    table: SKIP LOCKED means they fan out across distinct rows instead of
    queuing behind each other.
    """
    
    sql = """
        UPDATE runs SET
            lease_owner = %(worker_id)s,
            lease_expires_at = now() + %(lease)s::interval,
            updated_at = now()
        WHERE id = (
            SELECT id FROM runs
            WHERE state = ANY(%(actionable)s)
              AND next_attempt_at <= now()
              AND (lease_expires_at IS NULL OR lease_expires_at < now())
            ORDER BY next_attempt_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING *;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql,
            {
                "worker_id": worker_id,
                "lease": LEASE_DURATION,
                "actionable": ACTIONABLE_STATES,
            },
        )
        row = cur.fetchone()
    conn.commit()
    return row


def heartbeat(conn: psycopg.Connection, run_id: int, worker_id: str) -> bool:
    """
    Push a run's lease forward. Only succeeds if worker_id still owns the
    lease -- if someone else has since claimed it (because our lease
    expired while we were slow), this returns False and the caller MUST
    stop work immediately. Continuing after a failed heartbeat means two
    workers are now doing the same thing.
    """
    sql = """
        UPDATE runs SET
            lease_expires_at = now() + %(lease)s::interval,
            updated_at = now()
        WHERE id = %(run_id)s
          AND lease_owner = %(worker_id)s
        RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            {"run_id": run_id, "worker_id": worker_id, "lease": LEASE_DURATION},
        )
        row = cur.fetchone()
    conn.commit()
    return row is not None


def release(
    conn: psycopg.Connection,
    run_id: int,
    new_state: str,
    checkpoint_delta: dict[str, Any] | None = None,
    next_attempt_at_sql: str = "now()",
) -> None:
    """
    Hand a run back: clear the lease, move it to new_state, and merge
    checkpoint_delta into the existing checkpoint jsonb (shallow merge --
    keys in the delta overwrite keys already there).

    next_attempt_at_sql is a raw SQL expression (e.g. "now() + interval
    '4 seconds'") used for backoff. It is never user input -- it is always
    a literal string we constructed in Python, never anything derived from
    a run's data -- so building it into the query text is safe here.
    """
    checkpoint_delta = checkpoint_delta or {}
    sql = f"""
        UPDATE runs SET
            state = %(new_state)s,
            lease_owner = NULL,
            lease_expires_at = NULL,
            checkpoint = checkpoint || %(delta)s::jsonb,
            next_attempt_at = {next_attempt_at_sql},
            updated_at = now()
        WHERE id = %(run_id)s;
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            {
                "new_state": new_state,
                "delta": json.dumps(checkpoint_delta),
                "run_id": run_id,
            },
        )
    conn.commit()
