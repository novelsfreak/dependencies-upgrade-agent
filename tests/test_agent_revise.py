"""
Week 4 Day 2: the mechanical (non-LLM) summary that stands in for a
prior revision round's raw conversation.
"""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from agent.revise import summarize_prior_conversation

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def conn():
    c = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo = c.execute("SELECT id FROM repos WHERE url = 'test://revise-fixture'").fetchone()
    repo_id = repo["id"] if repo else c.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://revise-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep = c.execute("SELECT id FROM dependencies WHERE repo_id = %s AND name = 'revise-dep'", (repo_id,)).fetchone()
    dep_id = dep["id"] if dep else c.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'revise-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = c.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]
    run_id = c.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
        "VALUES (%s, 'AWAITING_CI', '{}'::jsonb, now()) RETURNING id",
        (candidate_id,),
    ).fetchone()["id"]

    yield c, run_id

    c.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
    c.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    c.close()


def _msg(seq, role, content):
    return (seq, role, json.dumps(content))


def test_summary_extracts_applied_patches_and_last_build_test(conn):
    c, run_id = conn
    rows = [
        _msg(0, "system", {"role": "system", "content": "sys"}),
        _msg(1, "user", {"role": "user", "content": "brief"}),
        _msg(2, "assistant", {"role": "assistant", "content": None, "tool_calls": []}),
        _msg(3, "tool", {"role": "tool", "tool_call_id": "c1", "content":
             "patch applied and committed. Current diff from before your patch:\n\n"
             "diff --git a/x.js b/x.js\n-old\n+new"}),
        _msg(4, "tool", {"role": "tool", "tool_call_id": "c2", "content": "build ok. {'status': 'ok'}"}),
        _msg(5, "tool", {"role": "tool", "tool_call_id": "c3", "content": "tests failed: some assertion"}),
    ]
    for seq, role, content in rows:
        c.execute(
            "INSERT INTO run_messages (run_id, revision, seq, role, content) VALUES (%s, 0, %s, %s, %s::jsonb)",
            (run_id, seq, role, content),
        )

    summary = summarize_prior_conversation(c, run_id, revision=0)

    assert "1 patch(es)" in summary
    assert "diff --git a/x.js b/x.js" in summary
    assert "build ok" in summary
    assert "tests failed: some assertion" in summary


def test_summary_reports_no_patch_when_none_applied(conn):
    c, run_id = conn
    c.execute(
        "INSERT INTO run_messages (run_id, revision, seq, role, content) VALUES (%s, 0, 0, 'system', %s::jsonb)",
        (run_id, json.dumps({"role": "system", "content": "sys"})),
    )
    summary = summarize_prior_conversation(c, run_id, revision=0)
    assert "No patch was successfully applied" in summary
