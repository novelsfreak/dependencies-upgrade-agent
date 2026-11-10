"""
Week 4 Day 5: handle_created's real triage routing, and
handle_building's AUTO(minor) -> AGENT fallback on a genuine build
failure. No Docker needed -- _run_subprocess_step is monkeypatched to
return a canned result, exercising the REAL adapter.parse_build and the
REAL new except/fallback branch against it, exactly like real Docker
output would.
"""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from adapters.base import StepResult
from core.states import BuildFailed, handle_building, handle_created

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def seeded_run(tmp_path):
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo = conn.execute("SELECT id FROM repos WHERE url = 'test://triage-fixture'").fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://triage-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep = conn.execute("SELECT id FROM dependencies WHERE repo_id = %s AND name = 'tri-dep'", (repo_id,)).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'tri-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '1.1.0', 'minor', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]

    def make(checkpoint: dict) -> dict:
        run_id = conn.execute(
            "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
            "VALUES (%s, 'CREATED', %s::jsonb, now()) RETURNING id",
            (candidate_id, json.dumps(checkpoint)),
        ).fetchone()["id"]
        return conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()

    yield conn, make

    conn.execute(
        "DELETE FROM runs WHERE candidate_id = %s", (candidate_id,)
    )
    conn.close()


def _stub_changelog(monkeypatch, text: str):
    monkeypatch.setattr("agent.changelog.fetch_changelog", lambda *a, **kw: (text, "stub-source"))


def test_handle_created_routes_patch_bump_to_auto(seeded_run, monkeypatch):
    conn, make = seeded_run
    _stub_changelog(monkeypatch, "Just fixed a typo, nothing else.")
    run = make({"dep_name": "tri-dep", "current_version": "1.0.0", "target_version": "1.0.1", "semver_jump": "patch"})

    next_state, delta = handle_created(dict(run), conn, "test-worker")

    assert next_state == "PATCHING"
    assert delta["use_agent"] is False
    assert delta["triage_classification"] == "AUTO"
    assert "auto_fallback_to_agent" not in delta  # patch bumps don't get the fallback


def test_handle_created_routes_minor_bump_to_auto_with_fallback(seeded_run, monkeypatch):
    conn, make = seeded_run
    _stub_changelog(monkeypatch, "Added an optional new parameter, fully backwards compatible.")
    run = make({"dep_name": "tri-dep", "current_version": "1.0.0", "target_version": "1.1.0", "semver_jump": "minor"})

    next_state, delta = handle_created(dict(run), conn, "test-worker")

    assert next_state == "PATCHING"
    assert delta["use_agent"] is False
    assert delta["auto_fallback_to_agent"] is True


def test_handle_created_routes_major_bump_to_agent(seeded_run, monkeypatch):
    conn, make = seeded_run
    _stub_changelog(monkeypatch, "No markers here at all.")
    run = make({"dep_name": "tri-dep", "current_version": "1.0.0", "target_version": "2.0.0", "semver_jump": "major"})

    next_state, delta = handle_created(dict(run), conn, "test-worker")

    assert next_state == "PATCHING"
    assert delta["use_agent"] is True
    assert delta["triage_classification"] == "AGENT"


def test_handle_created_routes_breaking_marker_to_agent_even_for_patch(seeded_run, monkeypatch):
    conn, make = seeded_run
    _stub_changelog(monkeypatch, "BREAKING: removed a deprecated internal helper.")
    run = make({"dep_name": "tri-dep", "current_version": "1.0.0", "target_version": "1.0.1", "semver_jump": "patch"})

    next_state, delta = handle_created(dict(run), conn, "test-worker")

    assert delta["use_agent"] is True


def test_handle_created_routes_deny_listed_dep_to_skipped(seeded_run, monkeypatch):
    conn, make = seeded_run
    monkeypatch.setenv("UPGRADE_DENY_LIST", "tri-dep")
    _stub_changelog(monkeypatch, "harmless")
    run = make({"dep_name": "tri-dep", "current_version": "1.0.0", "target_version": "1.0.1", "semver_jump": "patch"})

    next_state, delta = handle_created(dict(run), conn, "test-worker")

    assert next_state == "SKIPPED"
    assert delta["triage_classification"] == "SKIP"


def test_handle_created_honors_explicit_use_agent_override(seeded_run, monkeypatch):
    conn, make = seeded_run

    def _boom(*a, **kw):
        raise AssertionError("triage must not run when use_agent was already explicitly set")
    monkeypatch.setattr("agent.changelog.fetch_changelog", _boom)

    run = make({"dep_name": "tri-dep", "current_version": "1.0.0", "target_version": "1.0.1", "use_agent": True})
    next_state, delta = handle_created(dict(run), conn, "test-worker")

    assert next_state == "PATCHING"
    assert delta == {}  # untouched -- the existing checkpoint value stands as-is


# --- handle_building's AUTO(minor) -> AGENT fallback ------------------------


def test_auto_minor_build_failure_falls_back_to_agent_patching(seeded_run, monkeypatch, tmp_path):
    conn, make = seeded_run
    (tmp_path / "package.json").write_text("{}")
    run = make({
        "ecosystem": "npm", "repo_dir": str(tmp_path), "work_dir": str(tmp_path),
        "auto_fallback_to_agent": True,
    })

    failing_result = StepResult(status="failed", error_count=1, errors=[])
    monkeypatch.setattr(
        "core.states._run_subprocess_step",
        lambda *a, **kw: (0, "", "log.txt", 100) if a[4] == "install" else (1, "build broke", "log.txt", 100),
    )
    monkeypatch.setattr(
        "adapters.npm.NpmAdapter.parse_build",
        lambda self, exit_code, stdout, stderr: StepResult(status="ok", error_count=0, errors=[])
        if exit_code == 0 else failing_result,
    )

    next_state, delta = handle_building(dict(run), conn, "test-worker")

    assert next_state == "AGENT_PATCHING"
    assert delta["use_agent"] is True
    assert delta["auto_fallback_to_agent"] is False


def test_build_failure_without_fallback_flag_still_raises(seeded_run, monkeypatch, tmp_path):
    conn, make = seeded_run
    (tmp_path / "package.json").write_text("{}")
    # No auto_fallback_to_agent -- an AGENT-classified or plain run must
    # keep its existing behavior (raise, let the normal attempt/backoff
    # machinery in worker/main.py handle it) completely unchanged.
    run = make({"ecosystem": "npm", "repo_dir": str(tmp_path), "work_dir": str(tmp_path)})

    failing_result = StepResult(status="failed", error_count=1, errors=[])
    monkeypatch.setattr(
        "core.states._run_subprocess_step",
        lambda *a, **kw: (0, "", "log.txt", 100) if a[4] == "install" else (1, "build broke", "log.txt", 100),
    )
    monkeypatch.setattr(
        "adapters.npm.NpmAdapter.parse_build",
        lambda self, exit_code, stdout, stderr: StepResult(status="ok", error_count=0, errors=[])
        if exit_code == 0 else failing_result,
    )

    with pytest.raises(BuildFailed):
        handle_building(dict(run), conn, "test-worker")


def test_infra_error_does_not_fall_back_even_with_flag_set(seeded_run, monkeypatch, tmp_path):
    conn, make = seeded_run
    (tmp_path / "package.json").write_text("{}")
    run = make({
        "ecosystem": "npm", "repo_dir": str(tmp_path), "work_dir": str(tmp_path),
        "auto_fallback_to_agent": True,
    })

    # exit code 137 = OOM-killed -- _classify_infra_failure reclassifies
    # "failed" to "infra_error" for exactly this code. An infra problem
    # isn't something the agent can fix either; it still wants the
    # normal retry, not a fallback to AGENT_PATCHING.
    monkeypatch.setattr(
        "core.states._run_subprocess_step",
        lambda *a, **kw: (0, "", "log.txt", 100) if a[4] == "install" else (137, "", "log.txt", 100),
    )
    monkeypatch.setattr(
        "adapters.npm.NpmAdapter.parse_build",
        lambda self, exit_code, stdout, stderr: StepResult(status="ok", error_count=0, errors=[])
        if exit_code == 0 else StepResult(status="failed", error_count=1, errors=[]),
    )

    with pytest.raises(BuildFailed) as exc_info:
        handle_building(dict(run), conn, "test-worker")
    assert exc_info.value.step_result.status == "infra_error"
