"""
The worker loop: claim a run, dispatch to its handler, release it.

This file deliberately knows nothing about what any individual state
*means*. It only knows: claim, look up, call, release-or-backoff. All
state-specific behavior lives in core/states.py.
"""
from __future__ import annotations

import logging
import subprocess
import os
import sys
import time
import uuid

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from core.claim import claim, release
from core.heartbeat_guard import LeaseLostError
from core.repo_lock import LockContention
from core.states import HANDLERS

LOCK_CONTENTION_BACKOFF_SECONDS = 30

# Load .env BEFORE anything reads os.environ below or in core/states.py.
# This was missing until now -- publisher.py had its own load_dotenv()
# call, but the worker process (which runs handle_patch_ready, needing
# GITHUB_TOKEN for the git push step) never loaded it, so GITHUB_TOKEN
# and GITHUB_REPO were genuinely absent from this process's environment
# regardless of what was sitting in .env on disk.
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE)

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")
POLL_INTERVAL_SECONDS = 1.0
MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 300  # 5 minutes -- the cap Day 7 calls out as missing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("worker")

def sweep_orphaned_containers(conn: psycopg.Connection) -> None:
    """
    On worker boot: find containers matching upgrade-* whose run is not
    currently leased (lease expired, or run gone entirely), and kill them.
    Prevents orphaned containers from a crashed worker quietly eating
    host resources over a long soak.

    The "still leased" check is done in SQL (lease_expires_at > now())
    rather than comparing to a Python-side now() -- avoids a tz-aware
    vs. naive datetime mismatch, and one round trip per container.
    """
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}", "--filter", "name=upgrade-"],
        capture_output=True, text=True,
    )
    container_names = [n for n in result.stdout.splitlines() if n]

    for name in container_names:
        # name shape: upgrade-{run_id}-{attempt}-{phase}
        parts = name.split("-")
        try:
            run_id = int(parts[1])
        except (IndexError, ValueError):
            log.warning(
                "sweep: container name %s doesn't match upgrade-{run_id}-{attempt}-{phase}, skipping",
                name,
            )
            continue

        with conn.cursor(row_factory=dict_row) as cur:
            row = cur.execute(
                "SELECT id FROM runs WHERE id = %(id)s AND lease_expires_at > now()",
                {"id": run_id},
            ).fetchone()

        if row is None:
            log.info(
                "sweep: killing orphaned container %s (run %s not currently leased)",
                name, run_id,
            )
            subprocess.run(["docker", "kill", name], capture_output=True)

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
        sweep_orphaned_containers(conn)
        conn.commit()

        while True:
            run = claim(conn, worker_id)

            if run is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            log.info("worker %s claimed run %s (state=%s)", worker_id, run["id"], run["state"])

            try:
                handler = HANDLERS[run["state"]]
                next_state, checkpoint_delta = handler(run, conn, worker_id)

                if checkpoint_delta.get("_released"):
                    # This handler (currently only handle_patch_ready)
                    # already committed its own state change, as part of
                    # a transaction that also had to include something
                    # else atomically (an outbox insert). Calling
                    # release() here too would run a second, redundant
                    # UPDATE outside that transaction and undo the
                    # point of writing both changes together. Trust the
                    # handler's own commit and just log it.
                    log.info(
                        "run %s moved %s -> %s (handler self-released)",
                        run["id"], run["state"], next_state,
                    )
                else:
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
            except LockContention as e:
                # Another run against this same repo is mid install/build
                # right now. Pure backpressure, not a failure: requeue in
                # the SAME state with a short delay, don't bump attempt,
                # don't record last_error -- this isn't a defect in the
                # run, it's just contention that will clear on its own.
                log.info("run %s: %s -- requeuing in %ss", run["id"], e, LOCK_CONTENTION_BACKOFF_SECONDS)
                release(
                    conn, run["id"], run["state"],
                    next_attempt_at_sql=f"now() + interval '{LOCK_CONTENTION_BACKOFF_SECONDS} seconds'",
                )
            except Exception as e:
                handle_failure(conn, run, e)


if __name__ == "__main__":
    run_worker(worker_id=sys.argv[1] if len(sys.argv) > 1 else None)
