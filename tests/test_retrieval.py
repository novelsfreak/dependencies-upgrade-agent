"""Week 6 Day 5: search-before-read compliance measurement."""
from __future__ import annotations

import os

import psycopg
import pytest
from psycopg.rows import dict_row

from agent.context.retrieval import search_before_read_compliance
from agent.messages import persist_message

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def seeded_run():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo_id = conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://retrieval-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep_id = conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'retrieval-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
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


def _assistant_call(conn, run_id, seq, call_id, name, args):
    persist_message(conn, run_id, seq, "assistant", {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}],
    }, segment="assistant")


def _tool_result(conn, run_id, seq, call_id, content):
    persist_message(conn, run_id, seq, "tool", {"role": "tool", "tool_call_id": call_id, "content": content},
                     segment="tool_results")


def test_no_read_file_calls_returns_none_rate(seeded_run):
    conn, run_id = seeded_run
    result = search_before_read_compliance(conn, run_id)
    assert result["read_file_calls"] == 0
    assert result["compliance_rate"] is None


def test_read_preceded_by_search_counts_as_compliant(seeded_run):
    conn, run_id = seeded_run
    _assistant_call(conn, run_id, 0, "c1", "search", '{"pattern": "foo"}')
    _tool_result(conn, run_id, 1, "c1", "src/a.js:10:foo() called here (suggested read_file range: 1-40)")
    _assistant_call(conn, run_id, 2, "c2", "read_file", '{"path": "src/a.js"}')
    _tool_result(conn, run_id, 3, "c2", "1: foo()")

    result = search_before_read_compliance(conn, run_id)
    assert result["read_file_calls"] == 1
    assert result["preceded_by_search"] == 1
    assert result["compliance_rate"] == 1.0


def test_speculative_read_with_no_prior_search_is_noncompliant(seeded_run):
    conn, run_id = seeded_run
    _assistant_call(conn, run_id, 0, "c1", "read_file", '{"path": "src/a.js"}')
    _tool_result(conn, run_id, 1, "c1", "1: foo()")

    result = search_before_read_compliance(conn, run_id)
    assert result["read_file_calls"] == 1
    assert result["preceded_by_search"] == 0
    assert result["compliance_rate"] == 0.0


def test_mixed_compliance_computes_real_ratio(seeded_run):
    conn, run_id = seeded_run
    _assistant_call(conn, run_id, 0, "c1", "search", '{"pattern": "foo"}')
    _tool_result(conn, run_id, 1, "c1", "src/a.js:10:foo() (suggested read_file range: 1-40)")
    _assistant_call(conn, run_id, 2, "c2", "read_file", '{"path": "src/a.js"}')  # compliant
    _tool_result(conn, run_id, 3, "c2", "1: foo()")
    _assistant_call(conn, run_id, 4, "c3", "read_file", '{"path": "src/unrelated.js"}')  # speculative
    _tool_result(conn, run_id, 5, "c3", "1: bar()")

    result = search_before_read_compliance(conn, run_id)
    assert result["read_file_calls"] == 2
    assert result["preceded_by_search"] == 1
    assert result["compliance_rate"] == 0.5
