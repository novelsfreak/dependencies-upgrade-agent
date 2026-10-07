"""
Week 6 Days 1-2: the sub-agent harness and the test-fixer instance,
tested against a MOCKED Groq client (same reasoning as
tests/test_agent_stop_conditions.py: deterministic, no live-rate-limit
waits) but REAL Postgres and REAL git/subprocess mechanics -- a
sub-agent is a real `runs` row driven by the real claim/release
machinery, not a simulation of one.
"""
from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row

import agent.subagent as subagent_module
from agent.subagent import CannotFix, SubAgentResult, fix_failing_tests, run_sub_agent_loop, spawn_sub_agent

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


def fake_response(tool_calls=None, content=None, finish_reason="tool_calls", tokens=(50, 50)):
    choice = SimpleNamespace(message=FakeMessage(tool_calls=tool_calls, content=content), finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1])
    return SimpleNamespace(choices=[choice], usage=usage)


def _patch_client(monkeypatch, responses):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = responses
    monkeypatch.setattr(subagent_module, "_get_client", lambda: fake_client)
    monkeypatch.setattr(subagent_module, "heartbeat", lambda conn, run_id, worker_id: True)
    return fake_client


@pytest.fixture
def seeded_parent(tmp_path):
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo_url = "test://subagent-fixture"
    repo = conn.execute("SELECT id FROM repos WHERE url = %s", (repo_url,)).fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES (%s, 'npm', 'npm run build', 'npm test') RETURNING id",
        (repo_url,),
    ).fetchone()["id"]
    dep = conn.execute(
        "SELECT id FROM dependencies WHERE repo_id = %s AND name = 'subagent-dep'", (repo_id,)
    ).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'subagent-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]

    (tmp_path / "a.test.js").write_text("assert(1 === 1);\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.com", "-c", "user.name=t", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True
    )

    parent_checkpoint = {"repo_dir": str(tmp_path), "ecosystem": "npm"}
    parent_id = conn.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
        "VALUES (%s, 'AGENT_PATCHING', %s::jsonb, now()) RETURNING id",
        (candidate_id, json.dumps(parent_checkpoint)),
    ).fetchone()["id"]
    parent_run = conn.execute("SELECT * FROM runs WHERE id = %s", (parent_id,)).fetchone()

    yield conn, parent_run, tmp_path

    conn.execute("DELETE FROM run_messages WHERE run_id IN (SELECT id FROM runs WHERE parent_run_id = %s OR id = %s)",
                 (parent_id, parent_id))
    conn.execute("DELETE FROM runs WHERE parent_run_id = %s", (parent_id,))
    conn.execute("DELETE FROM runs WHERE id = %s", (parent_id,))
    conn.execute("DELETE FROM candidates WHERE id = %s", (candidate_id,))
    conn.execute("DELETE FROM dependencies WHERE id = %s", (dep_id,))
    conn.close()


def _make_subagent_run(conn, parent_run, tmp_path, extra_checkpoint=None):
    checkpoint = {
        "repo_dir": str(tmp_path), "ecosystem": "npm",
        "test_file": "a.test.js", "test_filter": "a.test.js",
        "failure_output": "assert failed", "current_diff": "", "changelog_section": "",
        "budget_turns": 5,
    }
    checkpoint.update(extra_checkpoint or {})
    run_id = conn.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, parent_run_id, task_type, task_target, next_attempt_at) "
        "VALUES (%s, 'SUBAGENT_PATCHING', %s::jsonb, %s, 'fix_test', 'a.test.js', now()) RETURNING id",
        (parent_run["candidate_id"], json.dumps(checkpoint), parent_run["id"]),
    ).fetchone()["id"]
    return conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()


def test_run_sub_agent_loop_reports_cannot_fix_when_called(monkeypatch, seeded_parent):
    conn, parent_run, tmp_path = seeded_parent
    run = _make_subagent_run(conn, parent_run, tmp_path)
    responses = [fake_response(tool_calls=[FakeToolCall("c1", "cannot_fix", '{"reason": "no idea how"}')])]
    _patch_client(monkeypatch, responses)

    next_state, delta = run_sub_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "SUBAGENT_DONE"
    assert delta["subagent_status"] == "cannot_fix"
    assert delta["subagent_reason"] == "no idea how"


def test_run_sub_agent_loop_max_turns_reports_cannot_fix(monkeypatch, seeded_parent):
    conn, parent_run, tmp_path = seeded_parent
    run = _make_subagent_run(conn, parent_run, tmp_path, {"budget_turns": 2})
    responses = [
        fake_response(tool_calls=[FakeToolCall(f"c{i}", "search", '{"pattern": "x"}')]) for i in range(10)
    ]
    _patch_client(monkeypatch, responses)

    next_state, delta = run_sub_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "SUBAGENT_DONE"
    assert delta["subagent_status"] == "cannot_fix"
    assert "max_turns" in delta["subagent_reason"]


def test_run_sub_agent_loop_stopping_without_a_passing_test_is_cannot_fix(monkeypatch, seeded_parent):
    conn, parent_run, tmp_path = seeded_parent
    run = _make_subagent_run(conn, parent_run, tmp_path)
    # Model just announces it's done, with no apply_patch and no run_tests call at all.
    responses = [fake_response(tool_calls=None, content="I think it's fine now", finish_reason="stop")]
    _patch_client(monkeypatch, responses)

    next_state, delta = run_sub_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "SUBAGENT_DONE"
    assert delta["subagent_status"] == "cannot_fix"
    assert "confirmed passing test run" in delta["subagent_reason"]


def test_run_sub_agent_loop_uses_restricted_tool_schema(monkeypatch, seeded_parent):
    conn, parent_run, tmp_path = seeded_parent
    run = _make_subagent_run(conn, parent_run, tmp_path)
    responses = [fake_response(tool_calls=[FakeToolCall("c1", "cannot_fix", '{"reason": "stuck"}')])]
    fake_client = _patch_client(monkeypatch, responses)

    run_sub_agent_loop(dict(run), conn, "test-worker")

    sent_tools = fake_client.chat.completions.create.call_args.kwargs["tools"]
    names = {t["function"]["name"] for t in sent_tools}
    assert names == {"read_file", "search", "apply_patch", "run_tests", "cannot_fix"}


def test_spawn_sub_agent_is_idempotent_on_replay(monkeypatch, seeded_parent):
    conn, parent_run, tmp_path = seeded_parent
    responses = [fake_response(tool_calls=[FakeToolCall("c1", "cannot_fix", '{"reason": "stuck first time"}')])]
    fake_client = _patch_client(monkeypatch, responses)

    inputs = {
        "repo_dir": str(tmp_path), "ecosystem": "npm", "test_file": "a.test.js", "test_filter": "a.test.js",
        "failure_output": "assert failed", "current_diff": "", "changelog_section": "",
    }
    result1 = spawn_sub_agent(dict(parent_run), conn, "test-worker", "fix_test", "a.test.js", inputs, budget_turns=5)
    assert result1.status == "cannot_fix"
    assert result1.reason == "stuck first time"
    call_count_after_first = fake_client.chat.completions.create.call_count

    # Replaying with the exact same (parent, task_type, target) must
    # NOT create a second row or re-invoke the model -- it's already
    # terminal, so the persisted result is returned as-is.
    result2 = spawn_sub_agent(dict(parent_run), conn, "test-worker", "fix_test", "a.test.js", inputs, budget_turns=5)
    assert result2.status == "cannot_fix"
    assert result2.run_id == result1.run_id
    assert fake_client.chat.completions.create.call_count == call_count_after_first

    rows = conn.execute(
        "SELECT count(*) c FROM runs WHERE parent_run_id = %s AND task_type = 'fix_test' AND task_target = 'a.test.js'",
        (parent_run["id"],),
    ).fetchone()
    assert rows["c"] == 1


def test_fix_failing_tests_reruns_full_suite_only_when_something_was_patched(monkeypatch, seeded_parent):
    conn, parent_run, tmp_path = seeded_parent

    def fake_spawn(parent_run, conn, worker_id, task_type, target, inputs, **kw):
        status = "patch" if target == "a.test.js" else "cannot_fix"
        return SubAgentResult(status=status, reason=None if status == "patch" else "stuck",
                               turns=3, cost_cents=0.5, run_id=999)

    monkeypatch.setattr(subagent_module, "spawn_sub_agent", fake_spawn)

    calls = []

    def fake_handle_testing(run, conn, worker_id):
        calls.append(run["id"])
        return "PATCH_READY", {}

    monkeypatch.setattr("core.states.handle_testing", fake_handle_testing)

    outcome = fix_failing_tests(
        dict(parent_run), conn, "test-worker",
        failing_tests=[{"test_file": "a.test.js", "failure_output": "x"},
                       {"test_file": "b.test.js", "failure_output": "y"}],
    )

    assert outcome["results"]["a.test.js"].status == "patch"
    assert outcome["results"]["b.test.js"].status == "cannot_fix"
    assert outcome["full_suite_status"] == "ok"
    assert calls == [parent_run["id"]]  # the full suite WAS re-run, exactly once


def test_fix_failing_tests_skips_full_suite_rerun_when_nothing_was_patched(monkeypatch, seeded_parent):
    conn, parent_run, tmp_path = seeded_parent

    def fake_spawn(parent_run, conn, worker_id, task_type, target, inputs, **kw):
        return SubAgentResult(status="cannot_fix", reason="stuck", turns=3, cost_cents=0.5, run_id=999)

    monkeypatch.setattr(subagent_module, "spawn_sub_agent", fake_spawn)

    calls = []
    monkeypatch.setattr("core.states.handle_testing", lambda run, conn, worker_id: calls.append(1))

    outcome = fix_failing_tests(
        dict(parent_run), conn, "test-worker",
        failing_tests=[{"test_file": "a.test.js", "failure_output": "x"}],
    )
    assert outcome["full_suite_status"] == "not_attempted"
    assert calls == []
