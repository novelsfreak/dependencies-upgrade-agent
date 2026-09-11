"""
POST /webhooks/github

Deliberately does almost nothing. Per the Day 6 plan: verify the
signature, write a durable row, flip a state if we can map the event to
a run, and return. All real work (re-cloning, re-building) happens
later, off this request, in the normal worker poll loop.

Why so little work here: GitHub's own webhook timeout window is only
10 seconds -- if this handler doesn't respond within that window,
GitHub records the delivery as a FAILURE, and GitHub does not
automatically redeliver failed deliveries. A slow handler risks losing
events entirely, not just retrying them. So the only things allowed in
this request path are: verify signature (cheap, local computation), one
INSERT, one UPDATE, return. Nothing that touches the network, the
filesystem, or a subprocess.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

import psycopg
from fastapi import APIRouter, Header, HTTPException, Request

log = logging.getLogger("webhooks")
router = APIRouter()

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


def verify_signature(secret: str, payload_body: bytes, signature_header: str | None) -> bool:
    """
    Recompute the HMAC-SHA256 over the RAW request body (not the parsed
    JSON -- re-serializing JSON can produce different bytes than what
    was actually sent, e.g. different key ordering or whitespace, which
    would make a legitimate signature fail to verify) and compare
    against what GitHub sent.

    hmac.compare_digest, not ==: a plain == comparison on strings/bytes
    short-circuits at the first differing byte, which leaks timing
    information an attacker could use to guess the correct signature
    one byte at a time (a timing attack). compare_digest runs in
    constant time regardless of where the mismatch is.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)


def record_event(source: str, external_id: str, payload: dict) -> int | None:
    """
    Insert into inbound_events. ON CONFLICT DO NOTHING on the (source,
    external_id) unique constraint -- if GitHub redelivers the exact
    same delivery ID, this is a silent no-op, which is exactly the
    dedupe behavior we want. Returns the new row's id, or None if it
    was a duplicate (nothing was inserted).
    """
    with psycopg.connect(DSN, autocommit=True) as conn:
        row = conn.execute(
            """
            INSERT INTO inbound_events (source, external_id, payload)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (source, external_id) DO NOTHING
            RETURNING id
            """,
            (source, external_id, json.dumps(payload)),
        ).fetchone()
    return row[0] if row else None


def find_run_by_branch(conn: psycopg.Connection, branch: str) -> dict | None:
    """
    Map an incoming event to a run via the branch name recorded in its
    checkpoint (handle_patch_ready wrote checkpoint->>'branch' when it
    pushed). This is a jsonb containment/path query, not a join through
    a dedicated column -- fine for week 1's volume, worth a real column
    + index if this table grows large.
    """
    from psycopg.rows import dict_row
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM runs WHERE checkpoint->>'branch' = %s ORDER BY id DESC LIMIT 1",
            (branch,),
        )
        return cur.fetchone()


def handle_check_suite_completed(conn: psycopg.Connection, payload: dict, event_id: int) -> None:
    if payload.get("action") != "completed":
        # GitHub also sends check_suite for "requested" and
        # "rerequested" -- conclusion is None on those since the suite
        # hasn't finished yet. Without this guard, the very first
        # notification (requested, before any result exists) would fall
        # through to the `else` below and get misread as a failure,
        # bouncing the run back to PATCHING before CI even ran.
        return

    conclusion = payload.get("check_suite", {}).get("conclusion")
    branch = payload.get("check_suite", {}).get("head_branch")

    if not branch:
        log.warning("check_suite.completed with no head_branch, ignoring")
        return

    run = find_run_by_branch(conn, branch)
    if not run:
        log.info("check_suite.completed for branch %s, no matching run found", branch)
        return

    if run["state"] != "AWAITING_CI":
        # Could be a redelivery processed already, or a check_suite for
        # an unrelated commit on the same branch. Don't blindly flip
        # state regardless of what we currently think this run is doing.
        log.info(
            "run %s got check_suite.completed but is in state %s, not AWAITING_CI -- skipping",
            run["id"], run["state"],
        )
        return

    if conclusion == "success":
        next_state = "MERGED_READY"
    else:
        next_state = "PATCHING"

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE runs SET state = %s, updated_at = now(), "
            "attempt = CASE WHEN %s = 'PATCHING' THEN attempt + 1 ELSE attempt END, "
            "next_attempt_at = now() "
            "WHERE id = %s",
            (next_state, next_state, run["id"]),
        )
        cur.execute(
            "UPDATE inbound_events SET run_id = %s, processed_at = now() WHERE id = %s",
            (run["id"], event_id),
        )
    conn.commit()
    log.info("run %s: check_suite %s -> state moved to %s", run["id"], conclusion, next_state)


EVENT_HANDLERS = {
    "check_suite": handle_check_suite_completed,
}


@router.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
):
    raw_body = await request.body()

    secret = os.environ.get("GITHUB_WEBHOOK_SECRET")
    if not secret:
        # Fail loudly at startup-adjacent time, not by silently
        # accepting unverified webhooks. A misconfigured deployment
        # should refuse traffic, not quietly disable security.
        raise HTTPException(status_code=500, detail="GITHUB_WEBHOOK_SECRET not configured")

    if not verify_signature(secret, raw_body, x_hub_signature_256):
        # Reject anything that fails verification. No partial trust,
        # no logging-and-continuing -- an unverified payload could be
        # forged, so it does not get to influence any run's state.
        raise HTTPException(status_code=401, detail="invalid signature")

    if not x_github_delivery:
        raise HTTPException(status_code=400, detail="missing X-GitHub-Delivery header")

    payload = json.loads(raw_body)
    event_id = record_event("github", x_github_delivery, payload)

    if event_id is None:
        # Duplicate delivery (GitHub redelivered the same event, or a
        # genuine retry after we returned a timeout on a prior attempt
        # that actually succeeded). Per the plan: "On conflict, return
        # 200 and do nothing." Returning 200 here, not re-processing,
        # is what prevents a redelivered event from double-applying a
        # state transition.
        return {"status": "duplicate, ignored"}

    if x_github_event in EVENT_HANDLERS:
        with psycopg.connect(DSN, autocommit=False) as conn:
            EVENT_HANDLERS[x_github_event](conn, payload, event_id)

    return {"status": "accepted"}
