"""
Week 6 Day 7: chaos.py's new context-era invariant checks, tested
directly against synthetic seeded Postgres state -- no live 15-minute
chaos soak needed to have real confidence these fire correctly (that
soak is still the thing that finds problems these don't anticipate,
but these run in under a second and need no Docker).
"""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from agent.loop import MAX_COST_CENTS, REQUEST_TOKEN_CEILING
from agent.messages import persist_message
from chaos import (
    check_no_context_window_exceeded,
    check_no_dangling_tool_calls,
    check_no_orphaned_subagents,
    check_no_run_exceeded_max_cost,
)

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def seeded():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo = conn.execute("SELECT id FROM repos WHERE url = 'test://chaos-invariants-fixture'").fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://chaos-invariants-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep = conn.execute(
        "SELECT id FROM dependencies WHERE repo_id = %s AND name = 'chaos-dep'", (repo_id,)
    ).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'chaos-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
        (repo_id,),
    ).fetchone()["id"]
    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, '2.0.0', 'major', 'new') RETURNING id",
        (dep_id,),
    ).fetchone()["id"]

    created_run_ids: list[int] = []

    def make(state: str, checkpoint: dict | None = None, parent_run_id: int | None = None) -> int:
        run_id = conn.execute(
            "INSERT INTO runs (candidate_id, state, checkpoint, parent_run_id, next_attempt_at) "
            "VALUES (%s, %s, %s::jsonb, %s, now()) RETURNING id",
            (candidate_id, state, json.dumps(checkpoint or {}), parent_run_id),
        ).fetchone()["id"]
        created_run_ids.append(run_id)
        return run_id

    yield conn, make, created_run_ids

    # Reverse creation order: a child is always created (via make(...,
    # parent_run_id=parent)) AFTER its parent, so deleting in reverse
    # guarantees children go before parents -- deleting a parent while
    # a child's parent_run_id still points at it violates that FK.
    for run_id in reversed(created_run_ids):
        conn.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
        conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.execute("DELETE FROM candidates WHERE id = %s", (candidate_id,))
    conn.execute("DELETE FROM dependencies WHERE id = %s", (dep_id,))
    conn.execute("DELETE FROM repos WHERE id = %s", (repo_id,))
    conn.close()


def test_context_window_check_passes_when_under_ceiling(seeded):
    conn, make, _ = seeded
    run_id = make("PATCH_READY")
    persist_message(conn, run_id, 0, "assistant", {"role": "assistant", "content": "ok"},
                     segment="assistant", tokens_in=500, tokens_out=50)
    assert check_no_context_window_exceeded(conn) == []


def test_context_window_check_flags_a_real_overshoot(seeded):
    conn, make, _ = seeded
    run_id = make("ESCALATED")
    persist_message(conn, run_id, 0, "assistant", {"role": "assistant", "content": "ok"},
                     segment="assistant", tokens_in=REQUEST_TOKEN_CEILING + 500, tokens_out=50)
    problems = check_no_context_window_exceeded(conn)
    assert any(str(run_id) in p for p in problems)


def test_dangling_tool_calls_flagged_only_on_a_really_terminal_run(seeded):
    conn, make, _ = seeded
    dangling_assistant = {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
    }
    # Still mid-flight (AGENT_PATCHING) -- a dangling call here is
    # normal, waiting on the crash-repair path, not a bug.
    inflight_run = make("AGENT_PATCHING")
    persist_message(conn, inflight_run, 0, "assistant", dangling_assistant, segment="assistant")
    assert check_no_dangling_tool_calls(conn, [str(inflight_run)]) == []

    # Terminal (ESCALATED) with the SAME dangling shape -- nothing will
    # ever repair this one. Real bug.
    terminal_run = make("ESCALATED")
    persist_message(conn, terminal_run, 0, "assistant", dangling_assistant, segment="assistant")
    problems = check_no_dangling_tool_calls(conn, [str(terminal_run)])
    assert len(problems) == 1
    assert str(terminal_run) in problems[0]


def test_orphaned_subagent_detected_when_parent_is_terminal(seeded):
    conn, make, _ = seeded
    parent = make("ESCALATED")
    make("SUBAGENT_PATCHING", parent_run_id=parent)  # never finished, parent already done

    problems = check_no_orphaned_subagents(conn)
    assert len(problems) == 1
    assert str(parent) in problems[0]


def test_subagent_not_flagged_while_parent_still_alive(seeded):
    conn, make, _ = seeded
    parent = make("AGENT_PATCHING")
    make("SUBAGENT_PATCHING", parent_run_id=parent)

    assert check_no_orphaned_subagents(conn) == []


def test_subagent_not_flagged_once_it_reaches_subagent_done(seeded):
    conn, make, _ = seeded
    parent = make("ESCALATED")
    make("SUBAGENT_DONE", parent_run_id=parent)

    assert check_no_orphaned_subagents(conn) == []


def test_max_cost_check_passes_within_tolerance(seeded):
    conn, make, _ = seeded
    run_id = make("PATCH_READY")
    persist_message(conn, run_id, 0, "assistant", {"role": "assistant", "content": "ok"},
                     segment="assistant", tokens_in=100, tokens_out=100, cost_cents=MAX_COST_CENTS * 0.5)
    assert check_no_run_exceeded_max_cost(conn, [str(run_id)]) == []


def test_max_cost_check_flags_a_real_overshoot(seeded):
    conn, make, _ = seeded
    run_id = make("ESCALATED")
    persist_message(conn, run_id, 0, "assistant", {"role": "assistant", "content": "ok"},
                     segment="assistant", tokens_in=100, tokens_out=100, cost_cents=MAX_COST_CENTS * 3)
    problems = check_no_run_exceeded_max_cost(conn, [str(run_id)])
    assert len(problems) == 1
    assert str(run_id) in problems[0]


def test_max_cost_check_includes_subagent_spend(seeded):
    conn, make, _ = seeded
    parent = make("PATCH_READY")
    child = make("SUBAGENT_DONE", parent_run_id=parent)
    # Parent's OWN spend is small, but its sub-agent's pushes the
    # combined total (agent/messages.py's total_cost_cents sums both)
    # over budget -- the exact "including sub-agent spend" case.
    persist_message(conn, parent, 0, "assistant", {"role": "assistant", "content": "ok"},
                     segment="assistant", tokens_in=10, tokens_out=10, cost_cents=1.0)
    persist_message(conn, child, 0, "assistant", {"role": "assistant", "content": "ok"},
                     segment="assistant", tokens_in=10, tokens_out=10, cost_cents=MAX_COST_CENTS * 3)

    problems = check_no_run_exceeded_max_cost(conn, [str(parent)])
    assert len(problems) == 1
    assert str(parent) in problems[0]
