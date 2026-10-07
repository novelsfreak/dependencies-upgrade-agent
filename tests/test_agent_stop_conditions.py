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
import subprocess
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

    (tmp_path / "package.json").write_text('{"name": "fixture", "dependencies": {}}\n')
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=fixture@example.com", "-c", "user.name=fixture",
         "add", "-A"], cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=fixture@example.com", "-c", "user.name=fixture",
         "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True,
    )
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
    return fake_client


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


_PACKAGE_JSON_PATCH = (
    "--- a/package.json\n"
    "+++ b/package.json\n"
    "@@ -1 +1 @@\n"
    '-{"name": "fixture", "dependencies": {}}\n'
    '+{"name": "fixture-patched", "dependencies": {}}\n'
)


def test_token_ceiling_with_applied_patch_hands_off_to_building(monkeypatch, seeded_run):
    """
    Week 4 Day 7: the real Day 6 finding -- a run that already applied
    a correct, committed patch should never have that work discarded
    into ESCALATED just because the conversation ran out of budget
    before the model itself could confirm it. Forcing prompt_tokens=0
    on every response disables chars_per_token recalibration, so the
    proactive ceiling check below is driven purely by real message
    content size (a large read_file result), not live Groq usage.
    """
    conn, run = seeded_run
    repo_dir = run["checkpoint"]["repo_dir"]
    big_file = os.path.join(repo_dir, "big.txt")
    with open(big_file, "w") as f:
        for _ in range(200):
            f.write("x" * 200 + "\n")

    responses = [
        fake_response(
            tool_calls=[FakeToolCall("c1", "apply_patch", json.dumps({"diff": _PACKAGE_JSON_PATCH}))],
            tokens=(0, 50),
        ),
        fake_response(
            tool_calls=[FakeToolCall("c2", "read_file", json.dumps({"path": "big.txt"}))],
            tokens=(0, 50),
        ),
        # Should never be reached -- the ceiling check trips before a
        # 3rd request goes out.
        fake_response(content="should not get here", finish_reason="stop", tokens=(0, 50)),
    ]
    _patch_client(monkeypatch, responses)

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "BUILDING"


def test_token_ceiling_without_applied_patch_escalates(monkeypatch, seeded_run):
    conn, run = seeded_run
    repo_dir = run["checkpoint"]["repo_dir"]
    big_file = os.path.join(repo_dir, "big.txt")
    with open(big_file, "w") as f:
        for _ in range(200):
            f.write("x" * 200 + "\n")

    responses = [
        fake_response(
            tool_calls=[FakeToolCall(f"c{i}", "read_file", json.dumps({"path": "big.txt"}))],
            tokens=(0, 50),
        )
        for i in range(5)
    ]
    _patch_client(monkeypatch, responses)

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")
    assert next_state == "ESCALATED"
    assert "per-request cap" in delta["escalated_reason"]


def test_fresh_conversation_persists_a_repo_map_when_repo_url_is_a_real_repo(monkeypatch, seeded_run):
    conn, run = seeded_run
    # This fixture's checkpoint has no "repo_url" key (see seeded_run
    # above) -- add one pointing at the real `repos` row the fixture
    # already created, which is what makes agent/loop.py's repo_map
    # lookup find a match.
    conn.execute(
        "UPDATE runs SET checkpoint = checkpoint || '{\"repo_url\": \"test://stop-conditions-fixture\"}'::jsonb "
        "WHERE id = %s", (run["id"],),
    )
    run = conn.execute("SELECT * FROM runs WHERE id = %s", (run["id"],)).fetchone()

    responses = [fake_response(tool_calls=None, content="done", finish_reason="stop")]
    _patch_client(monkeypatch, responses)

    loop_module.run_agent_loop(dict(run), conn, "test-worker")

    rows = conn.execute(
        "SELECT content FROM run_messages WHERE run_id = %s AND tokens_by_segment->>'repo_map' IS NOT NULL",
        (run["id"],),
    ).fetchall()
    assert len(rows) == 1
    assert "Repo map:" in rows[0]["content"]["content"]
    assert "package.json" in rows[0]["content"]["content"]


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
