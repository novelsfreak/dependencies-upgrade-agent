"""
Week 4 Day 1: an agent run must survive a crash mid-conversation and
resume coherently, rather than silently restarting from turn 0 (which
would throw away real turns and real Groq spend every time a worker
gets kill -9'd, loses its lease, or gets requeued by LockContention).

Same mocked-client approach as test_agent_stop_conditions.py, for the
same reason: deterministic, and doesn't burn real rate-limited tokens
just to prove control flow. The `seeded_run` fixture, fakes, and
`_patch_client` helper are reused from there rather than duplicated.
"""
from __future__ import annotations

import json

import agent.loop as loop_module
from tests.test_agent_stop_conditions import (
    _patch_client,
    fake_response,
    seeded_run,  # noqa: F401 -- pytest fixture, referenced by name only
)


def test_resume_loads_existing_conversation_without_rebuilding_brief(monkeypatch, seeded_run):
    conn, run = seeded_run

    # Simulate one turn already completed and persisted by a prior
    # (crashed) attempt: system + task brief + one clean assistant/tool
    # exchange, exactly what run_agent_loop itself would have written.
    conn.execute(
        "INSERT INTO run_messages (run_id, seq, role, content) VALUES "
        "(%s, 0, 'system', %s::jsonb), (%s, 1, 'user', %s::jsonb), "
        "(%s, 2, 'assistant', %s::jsonb), (%s, 3, 'tool', %s::jsonb)",
        (
            run["id"], json.dumps({"role": "system", "content": "sys"}),
            run["id"], json.dumps({"role": "user", "content": "brief"}),
            run["id"], json.dumps({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "list_files", "arguments": "{}"}}],
            }),
            run["id"], json.dumps({"role": "tool", "tool_call_id": "c1", "content": "package.json"}),
        ),
    )

    # Week 4 Day 4: fetch_changelog IS legitimately called on resume now
    # (run_build/run_tests need changelog text available for symbol
    # correlation on ANY turn, not just turn 1) -- stubbed harmlessly
    # here rather than asserted-never-called, which is what this test
    # used to check before that changed. What still must NOT happen on
    # resume is rebuilding the persisted system+brief messages
    # themselves; that's what the assertions below actually prove.
    monkeypatch.setattr("agent.changelog.fetch_changelog", lambda *a, **kw: ("stub changelog", "stub-source"))

    responses = [fake_response(tool_calls=None, content="done", finish_reason="stop")]
    fake_client = _patch_client(monkeypatch, responses)

    next_state, _delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")

    assert next_state == "BUILDING"
    # Week 5 Day 2: `messages` is now rebuilt fresh from Postgres via
    # ContextAssembler.build() every iteration rather than mutated
    # in-place, so what MagicMock captures IS a true snapshot at call
    # time (4 pre-existing messages) -- no longer the "it kept growing
    # after the call because it's a live reference" workaround this
    # assertion needed before that refactor. Still clear proof the real
    # history was loaded rather than a fresh 2-message system+brief start.
    sent_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert len(sent_messages) == 4
    assert sent_messages[0]["content"] == "sys"
    assert sent_messages[3]["tool_call_id"] == "c1"


def test_resume_repairs_dangling_tool_call(monkeypatch, seeded_run):
    conn, run = seeded_run

    # Simulate a crash between EXECUTING a tool and PERSISTING its
    # result: the assistant's tool_calls turn is on disk, but there is
    # no matching tool-role response for it. This is exactly what a
    # kill -9 between those two steps leaves behind.
    conn.execute(
        "INSERT INTO run_messages (run_id, seq, role, content) VALUES "
        "(%s, 0, 'system', %s::jsonb), (%s, 1, 'user', %s::jsonb), (%s, 2, 'assistant', %s::jsonb)",
        (
            run["id"], json.dumps({"role": "system", "content": "sys"}),
            run["id"], json.dumps({"role": "user", "content": "brief"}),
            run["id"], json.dumps({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "dangling1", "type": "function",
                                 "function": {"name": "list_files", "arguments": '{"glob": "*.json"}'}}],
            }),
        ),
    )

    responses = [fake_response(tool_calls=None, content="done", finish_reason="stop")]
    fake_client = _patch_client(monkeypatch, responses)

    next_state, _delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")

    assert next_state == "BUILDING"
    rows = conn.execute(
        "SELECT content FROM run_messages WHERE run_id = %s AND role = 'tool' ORDER BY seq", (run["id"],)
    ).fetchall()
    assert len(rows) == 1
    repaired = rows[0]["content"]
    assert repaired["tool_call_id"] == "dangling1"
    assert "package.json" in repaired["content"]  # the real list_files call actually ran

    # The repaired tool result must have reached the model as real
    # conversation history on the next call, not been silently dropped.
    sent_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert any(m.get("tool_call_id") == "dangling1" for m in sent_messages)


def test_resume_preserves_turn_count_toward_max_turns(monkeypatch, seeded_run):
    conn, run = seeded_run
    monkeypatch.setattr(loop_module, "MAX_TURNS", 2)

    # Two full turns already spent (2 assistant messages, each answered)
    # -- resuming a 3rd time must count as turn 3, not restart at turn 1,
    # or a run that's genuinely exhausted its budget would get an extra
    # free turn on every crash+resume.
    rows = [
        (0, "system", {"role": "system", "content": "sys"}),
        (1, "user", {"role": "user", "content": "brief"}),
        (2, "assistant", {"role": "assistant", "content": None,
                           "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "list_files", "arguments": "{}"}}]}),
        (3, "tool", {"role": "tool", "tool_call_id": "c1", "content": "ok"}),
        (4, "assistant", {"role": "assistant", "content": None,
                           "tool_calls": [{"id": "c2", "type": "function",
                                            "function": {"name": "list_files", "arguments": "{}"}}]}),
        (5, "tool", {"role": "tool", "tool_call_id": "c2", "content": "ok"}),
    ]
    for seq, role, content in rows:
        conn.execute(
            "INSERT INTO run_messages (run_id, seq, role, content) VALUES (%s, %s, %s, %s::jsonb)",
            (run["id"], seq, role, json.dumps(content)),
        )

    # No responses queued at all: if turn count didn't carry over, the
    # loop would try to make a 3rd-ever API call and this would raise
    # StopIteration instead of the assertion below firing cleanly.
    fake_client = _patch_client(monkeypatch, [])

    next_state, delta = loop_module.run_agent_loop(dict(run), conn, "test-worker")

    assert next_state == "ESCALATED"
    assert "max_turns" in delta["escalated_reason"]
    fake_client.chat.completions.create.assert_not_called()
