"""
Publisher: a separate process/loop from the worker. Polls the outbox
for unpublished rows and actually calls GitHub.

Runs independently of worker/main.py on purpose -- a slow or flaky
GitHub API must never block a worker from claiming and progressing
other runs, and killing/restarting this process must never be able to
produce a duplicate PR (see find_open_pr in core/github_client.py and
the deterministic branch naming in core/states.py's handle_patch_ready).
"""
from __future__ import annotations

import logging
import os
import time

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from pathlib import Path

from core.github_client import create_pr, find_open_pr

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE)

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")
POLL_INTERVAL_SECONDS = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("publisher")


def fetch_unpublished(conn: psycopg.Connection) -> dict | None:
    """
    Claim one unpublished outbox row using the SAME SKIP LOCKED pattern
    as core/claim.py -- if you ever run more than one publisher process,
    they must not both try to publish the same row concurrently.
    """
    sql = """
        UPDATE outbox SET attempts = attempts + 1
        WHERE id = (
            SELECT id FROM outbox
            WHERE published_at IS NULL
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING *;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        row = cur.fetchone()
    conn.commit()
    return row


def mark_published(conn: psycopg.Connection, outbox_id: int) -> None:
    conn.execute(
        "UPDATE outbox SET published_at = now() WHERE id = %s", (outbox_id,)
    )
    conn.commit()


def publish_open_pr(conn: psycopg.Connection, row: dict) -> None:
    payload = row["payload"]
    token = os.environ["GITHUB_TOKEN"]
    repo = payload["repo"]
    head_branch = payload["head_branch"]

    # THE idempotency check that protects against the publisher itself
    # retrying after a partial success: if a PR already exists on this
    # deterministic branch, someone (possibly a earlier, crashed run of
    # this very publisher) already created it. Don't create a second one
    # -- just mark this row done and move on.
    existing = find_open_pr(token, repo, head_branch)
    if existing:
        log.info(
            "outbox row %s: PR already exists (#%s) for branch %s, marking published",
            row["id"], existing["number"], head_branch,
        )
        mark_published(conn, row["id"])
        return

    pr = create_pr(
        token, repo,
        head_branch=head_branch,
        base_branch=payload["base_branch"],
        title=payload["title"],
        body=payload["body"],
    )
    log.info("outbox row %s: created PR #%s (%s)", row["id"], pr["number"], pr["html_url"])
    mark_published(conn, row["id"])


PUBLISHERS = {
    "open_pr": publish_open_pr,
}


def run_publisher() -> None:
    log.info("starting publisher")
    with psycopg.connect(DSN, autocommit=False) as conn:
        while True:
            row = fetch_unpublished(conn)
            if row is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            log.info("publishing outbox row %s (kind=%s, attempt %s)", row["id"], row["kind"], row["attempts"])
            try:
                handler = PUBLISHERS[row["kind"]]
                handler(conn, row)
            except Exception as e:
                # No backoff/attempt-cap here yet -- outbox.attempts is
                # tracked (see fetch_unpublished) but week 1 doesn't act
                # on it. A permanently-failing publish (e.g. bad token)
                # would retry every POLL_INTERVAL_SECONDS forever. Worth
                # a cap later; not the focus of today's proof.
                log.error("outbox row %s failed: %s", row["id"], e)


if __name__ == "__main__":
    run_publisher()
