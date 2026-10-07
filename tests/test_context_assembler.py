"""
Week 5 Day 2: ContextAssembler.build() is now the only place that
reconstructs `messages[]` from run_messages. Day 2 itself renders
everything verbatim (compaction/eviction land Day 3/4) -- these tests
lock in that contract, plus the "fail loudly" fixed-budget check.
"""
from __future__ import annotations

import os

import psycopg
import pytest
from psycopg.rows import dict_row

from agent.context.assembler import BudgetExceeded, build
from agent.messages import persist_message

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def seeded_run():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo_id = conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://assembler-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep_id = conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'asm-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]
    run_id = conn.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
        "VALUES (%s, 'AGENT_PATCHING', '{}'::jsonb, now()) RETURNING id",
        (candidate_id,),
    ).fetchone()["id"]

    yield conn, run_id

    conn.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.execute("DELETE FROM candidates WHERE id = %s", (candidate_id,))
    conn.execute("DELETE FROM dependencies WHERE id = %s", (dep_id,))
    conn.execute("DELETE FROM repos WHERE id = %s", (repo_id,))
    conn.close()


def test_build_renders_verbatim_history(seeded_run):
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    persist_message(conn, run_id, 1, "user", {"role": "user", "content": "brief"}, segment="brief")
    persist_message(conn, run_id, 2, "tool", {"role": "tool", "content": "output"}, segment="tool_results")

    messages = build(conn, run_id)
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "brief"},
        {"role": "tool", "content": "output"},
    ]


def test_build_is_revision_scoped(seeded_run):
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "round0"}, segment="system")
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "round1"}, segment="system", revision=1)

    assert build(conn, run_id, revision=0)[0]["content"] == "round0"
    assert build(conn, run_id, revision=1)[0]["content"] == "round1"


def test_build_raises_loudly_when_a_fixed_segment_overflows_its_own_budget(seeded_run):
    conn, run_id = seeded_run
    # A system prompt this large should never happen in practice -- the
    # plan's own words apply here: that's a bug, not something an
    # eviction policy should paper over.
    oversized_system_prompt = "x " * 10000
    persist_message(
        conn, run_id, 0, "system", {"role": "system", "content": oversized_system_prompt}, segment="system"
    )

    with pytest.raises(BudgetExceeded):
        build(conn, run_id)


def test_build_is_byte_stable_across_repeated_calls_when_nothing_changed(seeded_run):
    """
    Week 6 Day 4/7: prompt caching (Groq's own, fully automatic --
    console.groq.com/docs/prompt-caching) is a PREFIX cache requiring
    an exact byte match. If assemble_context's output drifted between
    two calls over identical persisted state (dict key ordering,
    whitespace, an accidental timestamp), every cache hit would be lost
    silently. This is the precondition the rest of Day 4 depends on.
    """
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    persist_message(conn, run_id, 1, "user", {"role": "user", "content": "brief"}, segment="brief")
    persist_message(conn, run_id, 2, "tool", {"role": "tool", "content": "output"}, segment="tool_results")

    import json
    first = json.dumps(build(conn, run_id), sort_keys=False)
    second = json.dumps(build(conn, run_id), sort_keys=False)
    assert first == second


def test_build_does_not_raise_for_a_compactable_segment_over_budget(seeded_run):
    # Only FIXED segments fail loudly today -- tool_results growing
    # past what a fixed budget would allow is expected pre-Day-3
    # (compaction doesn't exist yet) and is what agent/loop.py's own
    # ceiling check already handles by escalating cleanly.
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    big_tool_output = "y " * 5000
    persist_message(
        conn, run_id, 1, "tool", {"role": "tool", "content": big_tool_output}, segment="tool_results"
    )

    messages = build(conn, run_id)  # must not raise
    assert len(messages) == 2
