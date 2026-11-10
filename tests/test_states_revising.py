"""
Week 4 Day 2: handle_revising. Clones against a LOCAL git repo (a
plain file path, not github.com) rather than a real GitHub remote --
git itself doesn't care, and it keeps this deterministic and fast
instead of depending on network/auth. The github_client calls this
handler makes (fetching real CI failure detail) are only exercised
when checkpoint["revision_trigger"] lacks a "detail" key and carries a
head_sha -- the tests below always provide "detail" directly, so no
network call happens; that live path is proven separately.
"""
from __future__ import annotations

import json
import os
import subprocess

import psycopg
import pytest
from psycopg.rows import dict_row

from core.states import MAX_REVISIONS, handle_revising

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def local_repo(tmp_path):
    repo_dir = tmp_path / "origin"
    repo_dir.mkdir()
    (repo_dir / "src.js").write_text("const x = 1;\n")
    subprocess.run(["git", "init", "-q"], cwd=repo_dir)
    subprocess.run(["git", "add", "-A"], cwd=repo_dir)
    subprocess.run(
        ["git", "-c", "user.email=a@a.com", "-c", "user.name=a", "commit", "-q", "-m", "init"], cwd=repo_dir
    )
    branch = "agent/upgrade/npm/dep-2.0.0"
    subprocess.run(["git", "branch", branch], cwd=repo_dir)
    return str(repo_dir), branch


@pytest.fixture
def seeded_run(local_repo):
    repo_url, branch = local_repo
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo = conn.execute("SELECT id FROM repos WHERE url = 'test://revising-fixture'").fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://revising-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep = conn.execute("SELECT id FROM dependencies WHERE repo_id = %s AND name = 'rev-dep'", (repo_id,)).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'rev-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]

    checkpoint = {
        "repo_url": repo_url, "branch": branch, "dep_name": "rev-dep", "target_version": "2.0.0",
        "manifest_path": "package.json", "ecosystem": "npm", "use_agent": True,
        "revision_trigger": {"kind": "ci_failure", "detail": "npm test failed: 1 assertion did not pass"},
    }
    run_id = conn.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
        "VALUES (%s, 'REVISING', %s::jsonb, now()) RETURNING id",
        (candidate_id, json.dumps(checkpoint)),
    ).fetchone()["id"]

    run = conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
    yield conn, run

    conn.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.close()


def test_handle_revising_clones_branch_and_seeds_new_revision(seeded_run):
    conn, run = seeded_run

    next_state, delta = handle_revising(dict(run), conn, "test-worker")

    assert next_state == "AGENT_PATCHING"
    assert delta["revision_count"] == 1
    assert delta["use_agent"] is True
    assert "revision_trigger" not in delta  # consumed, not carried forward

    repo_dir = delta["repo_dir"]
    assert os.path.isfile(os.path.join(repo_dir, "src.js"))
    branch_check = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, capture_output=True, text=True,
    )
    assert branch_check.stdout.strip() == "agent/upgrade/npm/dep-2.0.0"

    msgs = conn.execute(
        "SELECT role, content FROM run_messages WHERE run_id = %s AND revision = 1 ORDER BY seq", (run["id"],)
    ).fetchall()
    assert [m["role"] for m in msgs] == ["system", "user"]
    brief = msgs[1]["content"]["content"]
    assert "npm test failed: 1 assertion did not pass" in brief
    assert "REVISION of work already done" in brief
    assert "No patch was successfully applied" in brief  # nothing was in revision 0 yet


def test_handle_revising_escalates_past_max_revisions(seeded_run):
    conn, run = seeded_run
    run = dict(run)
    run["checkpoint"] = {**run["checkpoint"], "revision_count": MAX_REVISIONS}

    next_state, delta = handle_revising(run, conn, "test-worker")

    assert next_state == "ESCALATED"
    assert "revision_count exceeded" in delta["escalated_reason"]


def test_handle_revising_requires_branch_and_repo_url(seeded_run):
    conn, run = seeded_run
    run = dict(run)
    run["checkpoint"] = {**run["checkpoint"]}
    del run["checkpoint"]["branch"]

    with pytest.raises(RuntimeError, match="requires checkpoint"):
        handle_revising(run, conn, "test-worker")
