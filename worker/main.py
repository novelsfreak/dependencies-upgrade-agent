"""
The worker loop: claim a run, dispatch to its handler, release it.

This file deliberately knows nothing about what any individual state
*means*. It only knows: claim, look up, call, release-or-backoff. All
state-specific behavior lives in core/states.py.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid

import psycopg

from core.claim import claim, release
from core.heartbeat_guard import LeaseLostError
from core.states import HANDLERS

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")
POLL_INTERVAL_SECONDS = 60.0
MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 300  # 5 minutes -- the cap Day 7 calls out as missing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("worker")


def backoff_seconds(attempt: int) -> int:
    """
    Exponential backoff, capped. 2^1=2, 2^2=4, 2^3=8 ... capped at
    MAX_BACKOFF_SECONDS so a run that's failed 20 times doesn't schedule
    itself for next year (the exact bug the plan calls out on Day 7).
    """
    return min(2 ** attempt, MAX_BACKOFF_SECONDS)


def handle_failure(conn: psycopg.Connection, run: dict, error: Exception) -> None:
    """
    A handler raised. Bump attempt, compute backoff, and either send the
    run back to retry from its current state or, past MAX_ATTEMPTS, park
    it in FAILED so it stops being retried forever.
    """
    attempt = run["attempt"] + 1
    log.warning(
        "run %s failed on attempt %s in state %s: %s",
        run["id"], attempt, run["state"], error,
    )

    if attempt >= MAX_ATTEMPTS:
        release(
            conn,
            run["id"],
            "FAILED",
            checkpoint_delta={"last_error": str(error)},
        )
        log.error("run %s exhausted %s attempts, marked FAILED", run["id"], MAX_ATTEMPTS)
        return

    delay = backoff_seconds(attempt)
    # Retry from the SAME state the run was in when it failed -- the
    # handler is expected to be safe to re-run (fresh temp dir per
    # attempt, no in-place mutation). We don't move it backward or
    # forward, just try the same step again after a delay.
    release(
        conn,
        run["id"],
        run["state"],
        checkpoint_delta={"last_error": str(error)},
        next_attempt_at_sql=f"now() + interval '{delay} seconds'",
    )
    # attempt itself isn't bumped by release() -- do it as a small
    # separate update rather than teaching release() about a column
    # it otherwise has no reason to know about.
    conn.execute(
        "UPDATE runs SET attempt = %s WHERE id = %s", (attempt, run["id"])
    )
    conn.commit()


def run_worker(worker_id: str | None = None) -> None:
    worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    log.info("starting worker %s", worker_id)

    with psycopg.connect(DSN, autocommit=False) as conn:
        while True:
            run = claim(conn, worker_id)

            if run is None:
                log.info("Looking for a new Work");
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            log.info("worker %s claimed run %s (state=%s)", worker_id, run["id"], run["state"])

            try:
                handler = HANDLERS[run["state"]]
                next_state, checkpoint_delta = handler(run, conn, worker_id)
                release(conn, run["id"], next_state, checkpoint_delta)
                log.info(
                    "run %s moved %s -> %s", run["id"], run["state"], next_state
                )
            except KeyError:
                # A state with no handler reached the worker loop. Per
                # core/states.py, this should be structurally impossible
                # for AWAITING_CI-style states (claim() can't select
                # them). If it happens anyway, that's a real bug in
                # claim()'s ACTIONABLE_STATES list, not something to
                # paper over here.
                log.error(
                    "run %s is in state %s with no handler -- this should be "
                    "unreachable, check ACTIONABLE_STATES in core/claim.py",
                    run["id"], run["state"],
                )
                raise
            except LeaseLostError as e:
                # We were superseded mid-step by another worker. This is
                # NOT a failure to retry -- the run no longer belongs to
                # us. Do NOT call release() or touch this row in any way;
                # whatever worker holds the lease now is responsible for
                # it. Just log and go back to polling for other work.
                log.warning("run %s: %s -- ceding, not touching the row", run["id"], e)
            except Exception as e:
                handle_failure(conn, run, e)


if __name__ == "__main__":
    run_worker(worker_id=sys.argv[1] if len(sys.argv) > 1 else None)
