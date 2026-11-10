"""
Week 4 Day 2: api/webhooks.py's handle_check_suite_completed now
branches three ways on a failed check suite instead of two -- success
still goes to MERGED_READY, but failure on an agent-driven run goes to
REVISING (carrying a revision_trigger) instead of PATCHING, since
PATCHING would just reapply the identical deterministic manifest bump
and land back exactly where it started. A deterministic (non-agent)
run's failure still goes to PATCHING, unchanged from Week 1.

Only the routing decision is tested here -- signature verification and
the FastAPI route itself are pre-existing, unchanged, and were already
untested before this (a real gap, not introduced here).
"""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from api.webhooks import (
    _is_actionable_comment,
    handle_check_suite_completed,
    handle_issue_comment,
    handle_pull_request_review_comment,
)

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def conn():
    c = psycopg.connect(DSN, autocommit=False, row_factory=dict_row)
    yield c
    c.close()


def _seed_run(conn: psycopg.Connection, branch: str, use_agent: bool, pr_number: int | None = None) -> int:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id FROM repos WHERE url = 'test://webhook-fixture'")
        repo = cur.fetchone()
        if not repo:
            cur.execute(
                "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
                "('test://webhook-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
            )
            repo = cur.fetchone()
        repo_id = repo["id"]

        cur.execute("SELECT id FROM dependencies WHERE repo_id = %s AND name = 'wh-dep'", (repo_id,))
        dep = cur.fetchone()
        if not dep:
            cur.execute(
                "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
                "VALUES (%s, 'wh-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
                (repo_id,),
            )
            dep = cur.fetchone()
        dep_id = dep["id"]

        cur.execute(
            "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
            "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
            (dep_id,),
        )
        candidate_id = cur.fetchone()["id"]

        checkpoint = {
            "branch": branch, "repo_url": "https://github.com/o/r", "use_agent": use_agent,
        }
        if pr_number is not None:
            checkpoint["pr_number"] = pr_number
        cur.execute(
            "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
            "VALUES (%s, 'AWAITING_CI', %s::jsonb, now()) RETURNING id",
            (candidate_id, json.dumps(checkpoint)),
        )
        run_id = cur.fetchone()["id"]
    conn.commit()
    return run_id


def _cleanup(conn: psycopg.Connection, run_id: int) -> None:
    conn.execute("DELETE FROM inbound_events WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.commit()


def _fake_event_id(conn: psycopg.Connection) -> int:
    row = conn.execute(
        "INSERT INTO inbound_events (source, external_id, payload) VALUES "
        "('github', %s, '{}'::jsonb) RETURNING id",
        (f"test-delivery-{os.urandom(4).hex()}",),
    ).fetchone()
    conn.commit()
    return row["id"]


def test_failed_check_suite_on_agent_run_goes_to_revising(conn):
    branch = "agent/upgrade/npm/wh-dep-2.0.0"
    run_id = _seed_run(conn, branch, use_agent=True)
    event_id = _fake_event_id(conn)
    payload = {
        "action": "completed",
        "check_suite": {"conclusion": "failure", "head_branch": branch, "head_sha": "deadbeef"},
    }

    try:
        handle_check_suite_completed(conn, payload, event_id)
        row = conn.execute("SELECT state, checkpoint FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert row["state"] == "REVISING"
        assert row["checkpoint"]["revision_trigger"] == {"kind": "ci_failure", "head_sha": "deadbeef"}
        # use_agent must survive the merge -- checkpoint || delta, not a replace
        assert row["checkpoint"]["use_agent"] is True
    finally:
        _cleanup(conn, run_id)


def test_failed_check_suite_on_deterministic_run_still_goes_to_patching(conn):
    branch = "agent/upgrade/npm/wh-dep-2.0.0-det"
    run_id = _seed_run(conn, branch, use_agent=False)
    event_id = _fake_event_id(conn)
    payload = {
        "action": "completed",
        "check_suite": {"conclusion": "failure", "head_branch": branch, "head_sha": "deadbeef"},
    }

    try:
        handle_check_suite_completed(conn, payload, event_id)
        row = conn.execute("SELECT state, attempt FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert row["state"] == "PATCHING"
        assert row["attempt"] == 1
    finally:
        _cleanup(conn, run_id)


def test_successful_check_suite_goes_to_merged_ready_regardless_of_use_agent(conn):
    branch = "agent/upgrade/npm/wh-dep-2.0.0-ok"
    run_id = _seed_run(conn, branch, use_agent=True)
    event_id = _fake_event_id(conn)
    payload = {
        "action": "completed",
        "check_suite": {"conclusion": "success", "head_branch": branch, "head_sha": "deadbeef"},
    }

    try:
        handle_check_suite_completed(conn, payload, event_id)
        row = conn.execute("SELECT state FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert row["state"] == "MERGED_READY"
    finally:
        _cleanup(conn, run_id)


# --- Week 4 Day 3: human review comments -----------------------------------


def test_is_actionable_comment_requires_allowlist_member_and_mention(monkeypatch):
    monkeypatch.setattr("api.webhooks.COMMENT_ALLOWLIST", {"trusteduser"})
    monkeypatch.setattr("api.webhooks.AGENT_MENTION_TRIGGER", "@upgrade-agent")

    both = {"user": {"login": "TrustedUser"}, "body": "hey @upgrade-agent please look at this"}
    assert _is_actionable_comment(both) is True  # case-insensitive login match

    wrong_author = {"user": {"login": "randomperson"}, "body": "@upgrade-agent do something"}
    assert _is_actionable_comment(wrong_author) is False

    no_mention = {"user": {"login": "trusteduser"}, "body": "just a normal comment"}
    assert _is_actionable_comment(no_mention) is False


def test_is_actionable_comment_fails_closed_with_empty_allowlist(monkeypatch):
    # An unconfigured allowlist must reject everyone, not allow anyone
    # -- the same fail-closed default as a missing webhook secret.
    monkeypatch.setattr("api.webhooks.COMMENT_ALLOWLIST", set())
    comment = {"user": {"login": "anyone"}, "body": "@upgrade-agent do something"}
    assert _is_actionable_comment(comment) is False


def test_issue_comment_from_allowlisted_user_moves_run_to_revising(conn, monkeypatch):
    monkeypatch.setattr("api.webhooks.COMMENT_ALLOWLIST", {"reviewer1"})
    monkeypatch.setattr("api.webhooks.AGENT_MENTION_TRIGGER", "@upgrade-agent")

    run_id = _seed_run(conn, "agent/upgrade/npm/wh-dep-ic", use_agent=True, pr_number=42)
    event_id = _fake_event_id(conn)
    payload = {
        "action": "created",
        "repository": {"full_name": "o/r"},
        "issue": {"number": 42, "pull_request": {"url": "https://api.github.com/..."}},
        "comment": {
            "user": {"login": "reviewer1"},
            "body": "@upgrade-agent don't change the retry config, we rely on that behaviour",
        },
    }

    try:
        handle_issue_comment(conn, payload, event_id)
        row = conn.execute("SELECT state, checkpoint FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert row["state"] == "REVISING"
        trigger = row["checkpoint"]["revision_trigger"]
        assert trigger["kind"] == "review_comment"
        assert "don't change the retry config" in trigger["detail"]
        assert trigger["author"] == "reviewer1"
    finally:
        _cleanup(conn, run_id)


def test_issue_comment_on_plain_issue_is_ignored(conn, monkeypatch):
    monkeypatch.setattr("api.webhooks.COMMENT_ALLOWLIST", {"reviewer1"})
    run_id = _seed_run(conn, "agent/upgrade/npm/wh-dep-ic2", use_agent=True, pr_number=43)
    event_id = _fake_event_id(conn)
    payload = {
        "action": "created",
        "repository": {"full_name": "o/r"},
        "issue": {"number": 43},  # no "pull_request" key -- a plain issue comment
        "comment": {"user": {"login": "reviewer1"}, "body": "@upgrade-agent hello"},
    }

    try:
        handle_issue_comment(conn, payload, event_id)
        row = conn.execute("SELECT state FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert row["state"] == "AWAITING_CI"  # untouched
    finally:
        _cleanup(conn, run_id)


def test_issue_comment_from_non_allowlisted_user_is_ignored(conn, monkeypatch):
    monkeypatch.setattr("api.webhooks.COMMENT_ALLOWLIST", {"reviewer1"})
    run_id = _seed_run(conn, "agent/upgrade/npm/wh-dep-ic3", use_agent=True, pr_number=44)
    event_id = _fake_event_id(conn)
    payload = {
        "action": "created",
        "repository": {"full_name": "o/r"},
        "issue": {"number": 44, "pull_request": {}},
        "comment": {"user": {"login": "randomer"}, "body": "@upgrade-agent hello"},
    }

    try:
        handle_issue_comment(conn, payload, event_id)
        row = conn.execute("SELECT state FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert row["state"] == "AWAITING_CI"
    finally:
        _cleanup(conn, run_id)


def test_pull_request_review_comment_includes_diff_hunk(conn, monkeypatch):
    monkeypatch.setattr("api.webhooks.COMMENT_ALLOWLIST", {"reviewer1"})
    monkeypatch.setattr("api.webhooks.AGENT_MENTION_TRIGGER", "@upgrade-agent")

    run_id = _seed_run(conn, "agent/upgrade/npm/wh-dep-rc", use_agent=True, pr_number=50)
    event_id = _fake_event_id(conn)
    payload = {
        "action": "created",
        "repository": {"full_name": "o/r"},
        "pull_request": {"number": 50},
        "comment": {
            "user": {"login": "reviewer1"},
            "body": "@upgrade-agent this retry logic shouldn't change",
            "path": "src/retry.js",
            "diff_hunk": "@@ -10,3 +10,3 @@\n-const retries = 3;\n+const retries = 5;",
        },
    }

    try:
        handle_pull_request_review_comment(conn, payload, event_id)
        row = conn.execute("SELECT state, checkpoint FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert row["state"] == "REVISING"
        detail = row["checkpoint"]["revision_trigger"]["detail"]
        assert "src/retry.js" in detail
        assert "-const retries = 3;" in detail
        assert "this retry logic shouldn't change" in detail
    finally:
        _cleanup(conn, run_id)
