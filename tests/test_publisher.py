"""
Week 4 Day 3: publisher.py must write the real PR number back onto the
run's checkpoint -- without it, api/webhooks.py's comment handlers
(issue_comment, pull_request_review_comment) have no way to map a
GitHub comment back to the run it belongs to, since neither payload
carries a branch name.
"""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from publisher import record_pr_number

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def seeded_run():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo = conn.execute("SELECT id FROM repos WHERE url = 'test://publisher-fixture'").fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://publisher-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep = conn.execute("SELECT id FROM dependencies WHERE repo_id = %s AND name = 'pub-dep'", (repo_id,)).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'pub-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'minor', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]
    run_id = conn.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
        "VALUES (%s, 'PR_OPEN', %s::jsonb, now()) RETURNING id",
        (candidate_id, json.dumps({"branch": "agent/upgrade/npm/pub-dep-2.0.0"})),
    ).fetchone()["id"]

    yield conn, run_id

    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.close()


def test_record_pr_number_merges_into_existing_checkpoint(seeded_run):
    conn, run_id = seeded_run

    record_pr_number(conn, run_id, 77)

    row = conn.execute("SELECT checkpoint FROM runs WHERE id = %s", (run_id,)).fetchone()
    assert row["checkpoint"]["pr_number"] == 77
    assert row["checkpoint"]["branch"] == "agent/upgrade/npm/pub-dep-2.0.0"  # untouched
