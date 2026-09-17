"""
Day 7's stop conditions, tested against a MOCKED Groq client rather than
live calls. Two reasons: these need to be deterministic (a live model
might not loop, or might loop differently, on any given run), and the
account's free-tier rate limit (8000 TPM, observed directly) makes a
live run of this take minutes per turn from 429 backoff alone --
exactly the opposite of what a fast, reliable CI check needs.

Requires a live Postgres (run_messages persistence is real) but zero
network calls to Groq or Docker.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row

import agent.loop as loop_module
from agent.tools import GiveUp

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self, exclude_none: bool = False) -> dict:
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


@pytest.fixture
def seeded_run(tmp_path):
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo_url = "test://stop-conditions-fixture"
    repo = conn.execute("SELECT id FROM repos WHERE url = %s", (repo_url,)).fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES (%s, 'npm', 'npm run build', 'npm test') RETURNING id",
        (repo_url,),
    ).fetchone()["id"]
    dep = conn.execute(
        "SELECT id FROM dependencies WHERE repo_id = %s AND name = 'stopcond-dep'", (repo_id,)
    ).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'stopcond-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]

    (tmp_path / "package.json").write_text('{"name": "fixture", "dependencies": {}}')
    checkpoint = {
        "repo_dir": str(tmp_path), "work_dir": str(tmp_path),
        "dep_name": "stopcond-dep", "current_version": "1.0.0", "target_version": "2.0.0",
        "manifest_path": "package.json", "ecosystem": "npm", "use_agent": True,
    }
    run_id = conn.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
        "VALUES (%s, 'AGENT_PATCHING', %s::jsonb, now()) RETURNING id",
        (candidate_id, json.dumps(checkpoint)),
    ).fetchone()["id"]

    run = conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
    yield conn, run

    conn.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.close()


def _patch_client(monkeypatch, responses):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = responses
    monkeypatch.setattr(loop_module, "_get_client", lambda: fake_client)
    monkeypatch.setattr(loop_module, "heartbeat", lambda conn, run_id, worker_id: True)


def test_max_turns_escalates(monkeypatch, seeded_run):
    conn, run = seeded_run
    monkeypatch.setattr(loop_module, "MAX_TURNS", 2)

    responses = [
        fake_response(tool_calls=[FakeToolCall(f"c{i}", "list_files", '{"glob": "*.json"}')])
        for i in range(10)
    ]
    _patch_client(monkeypatch, responses)

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "ESCALATED"
    assert "max_turns" in delta["escalated_reason"]


def test_max_cost_cents_escalates(monkeypatch, seeded_run):
    conn, run = seeded_run
    monkeypatch.setattr(loop_module, "MAX_COST_CENTS", 0.001)  # trivially exceeded by turn 2

    responses = [
        fake_response(tool_calls=[FakeToolCall(f"c{i}", "list_files", '{"glob": "*.json"}')], tokens=(1000, 1000))
        for i in range(10)
    ]
    _patch_client(monkeypatch, responses)

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "ESCALATED"
    assert "max_cost_cents" in delta["escalated_reason"]
    assert delta["escalated_cost_cents"] < 100  # nowhere near a dollar


def test_repeated_identical_tool_call_escalates_after_warning(monkeypatch, seeded_run):
    conn, run = seeded_run
    # Same tool, same args, every turn -- exactly the "stuck" scenario
    # loop detection exists for.
    responses = [
        fake_response(tool_calls=[FakeToolCall(f"c{i}", "list_files", '{"glob": "*.json"}')])
        for i in range(10)
    ]
    _patch_client(monkeypatch, responses)

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "ESCALATED"
    assert "repeated identical tool call" in delta["escalated_reason"]

    # The warning (3rd occurrence) should be visible in persisted
    # messages before the escalation on the 4th.
    rows = conn.execute(
        "SELECT content FROM run_messages WHERE run_id = %s AND role = 'tool' ORDER BY seq", (run["id"],)
    ).fetchall()
    warned = any("identical arguments" in r["content"].get("content", "") for r in rows)
    assert warned


def test_give_up_escalates_with_reason(monkeypatch, seeded_run):
    conn, run = seeded_run
    responses = [fake_response(tool_calls=[FakeToolCall("c1", "give_up", '{"reason": "peer dependency conflict"}')])]
    _patch_client(monkeypatch, responses)

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "ESCALATED"
    assert "peer dependency conflict" in delta["escalated_reason"]


def test_clean_completion_hands_off_to_building(monkeypatch, seeded_run):
    conn, run = seeded_run
    responses = [fake_response(tool_calls=None, content="Done, build and tests pass.", finish_reason="stop")]
    _patch_client(monkeypatch, responses)

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "BUILDING"


def test_malformed_tool_arguments_do_not_crash_the_loop(monkeypatch, seeded_run):
    conn, run = seeded_run
    bad_call = FakeToolCall("c1", "read_file", "{not valid json")
    good_call = FakeToolCall("c2", "list_files", '{"glob": "*.json"}')
    responses = [
        fake_response(tool_calls=[bad_call]),
        fake_response(tool_calls=None, content="ok", finish_reason="stop"),
    ]
    _patch_client(monkeypatch, responses)

    next_state, _delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "BUILDING"
