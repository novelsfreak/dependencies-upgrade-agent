"""
Week 4 Day 6: classify_run's mechanical categorization of a completed
run's final state, against seeded (not live) run/run_messages rows --
deterministic, no Docker/Groq needed to prove the classification logic
itself is correct.
"""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from core.taxonomy import classify_run

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def seeded_run():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo = conn.execute("SELECT id FROM repos WHERE url = 'test://taxonomy-fixture'").fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://taxonomy-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep = conn.execute("SELECT id FROM dependencies WHERE repo_id = %s AND name = 'tax-dep'", (repo_id,)).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'tax-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]

    def make(state: str, checkpoint: dict) -> int:
        return conn.execute(
            "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
            "VALUES (%s, %s, %s::jsonb, now()) RETURNING id",
            (candidate_id, state, json.dumps(checkpoint)),
        ).fetchone()["id"]

    yield conn, make

    conn.execute(
        "DELETE FROM run_messages WHERE run_id IN (SELECT id FROM runs WHERE candidate_id = %s)",
        (candidate_id,),
    )
    conn.execute("DELETE FROM runs WHERE candidate_id = %s", (candidate_id,))
    conn.close()


def _tool_msg(conn, run_id, seq, text):
    conn.execute(
        "INSERT INTO run_messages (run_id, seq, role, content) VALUES (%s, %s, 'tool', %s::jsonb)",
        (run_id, seq, json.dumps({"role": "tool", "tool_call_id": f"c{seq}", "content": text})),
    )


def test_patch_ready_is_success(seeded_run):
    conn, make = seeded_run
    run_id = make("PATCH_READY", {})
    category, _ = classify_run(conn, run_id)
    assert category == "succeeded"


def test_patch_ready_with_auto_triage_is_zero_token_success(seeded_run):
    conn, make = seeded_run
    run_id = make("PATCH_READY", {"triage_classification": "AUTO", "triage_reason": "patch bump"})
    category, evidence = classify_run(conn, run_id)
    assert category == "auto_zero_token_success"
    assert "patch bump" in evidence


def test_escalated_max_turns_with_no_progress(seeded_run):
    conn, make = seeded_run
    run_id = make("ESCALATED", {"escalated_reason": "max_turns exceeded"})
    _tool_msg(conn, run_id, 0, "no matches for 'foo'")
    category, _ = classify_run(conn, run_id)
    assert category == "ran_out_of_turns_making_progress"


def test_escalated_max_turns_with_progress_still_flagged_as_progress(seeded_run):
    conn, make = seeded_run
    run_id = make("ESCALATED", {"escalated_reason": "max_turns exceeded"})
    _tool_msg(conn, run_id, 0, "patch applied and committed. Current diff from before your patch:\n\ndiff...")
    category, evidence = classify_run(conn, run_id)
    assert category == "ran_out_of_turns_making_progress"
    assert "real patch landed" in evidence


def test_escalated_repeated_tool_call_is_looping(seeded_run):
    conn, make = seeded_run
    run_id = make("ESCALATED", {"escalated_reason": "repeated identical tool call: search"})
    category, _ = classify_run(conn, run_id)
    assert category == "ran_out_of_turns_looping"


def test_patch_kept_failing_to_apply(seeded_run):
    conn, make = seeded_run
    run_id = make("ESCALATED", {"escalated_reason": "max_turns exceeded"})
    for i in range(5):
        _tool_msg(conn, run_id, i, "patch did not apply: hunk header is missing line numbers")
    _tool_msg(conn, run_id, 5, "no matches for 'x'")
    category, evidence = classify_run(conn, run_id)
    assert category == "patch_kept_failing_to_apply"
    assert "5/6" in evidence


def test_give_up_is_genuinely_impossible(seeded_run):
    conn, make = seeded_run
    run_id = make("ESCALATED", {"escalated_reason": "agent gave up: peer dependency conflict with react 18"})
    category, evidence = classify_run(conn, run_id)
    assert category == "genuinely_impossible"
    assert "peer dependency" in evidence


def test_failed_with_lockfile_error_is_infra_failure(seeded_run):
    conn, make = seeded_run
    run_id = make("FAILED", {"last_error": "npm ci failed: lockfile mismatch"})
    category, _ = classify_run(conn, run_id)
    assert category == "infra_failure"


def test_failed_with_groq_rate_limit_exhaustion_is_infra_failure(seeded_run):
    # Real error text captured verbatim from bench.py's own live Week 5
    # Day 6 benchmark run (run 5837), which exhausted MAX_ATTEMPTS on a
    # genuine Groq daily-token-cap 429 -- previously fell through to
    # "uncategorized", indistinguishable from an actual unrecognized bug.
    conn, make = seeded_run
    real_error = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`openai/gpt-oss-120b` in organization `org_01m2pe6s70epn98x82t8hbjm59` "
        "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used "
        "196555, Requested 4767. Please try again in 9m31.104s.', 'type': 'tokens', "
        "'code': 'rate_limit_exceeded'}}"
    )
    run_id = make("FAILED", {"last_error": real_error})
    category, evidence = classify_run(conn, run_id)
    assert category == "infra_failure"


def test_failed_with_no_last_error_is_infra_failure(seeded_run):
    conn, make = seeded_run
    run_id = make("FAILED", {})
    category, evidence = classify_run(conn, run_id)
    assert category == "infra_failure"
    assert "no last_error" in evidence


def test_modified_test_to_pass_is_flagged(seeded_run):
    conn, make = seeded_run
    run_id = make("FAILED", {"last_error": "tests still failing after 5 attempts"})
    _tool_msg(
        conn, run_id, 0,
        "patch applied and committed. Current diff from before your patch:\n\n"
        "diff --git a/test/run.test.js b/test/run.test.js\n-assert.equal(x, 1)\n+assert.equal(x, 2)",
    )
    category, evidence = classify_run(conn, run_id)
    assert category == "modified_test_to_pass"


def test_skipped_run_is_uncategorized_with_deny_reason(seeded_run):
    conn, make = seeded_run
    run_id = make("SKIPPED", {"triage_reason": "'tax-dep' is on the deny list"})
    category, evidence = classify_run(conn, run_id)
    assert category == "uncategorized"
    assert "deny list" in evidence


def test_missing_run_reports_not_found(seeded_run):
    conn, _make = seeded_run
    category, evidence = classify_run(conn, 99999999)
    assert category == "uncategorized"
    assert "not found" in evidence
