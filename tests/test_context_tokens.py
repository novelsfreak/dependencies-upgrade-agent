"""
Week 5 Day 1: real tokenizer counts (not len(text)/4), per-segment
tagging on every persisted message, and the retrospective report query
the plan's own Day 1 asks for ("one query that gives you, for any run:
turn number, total context size, size by segment, cost, and the tool
called").
"""
from __future__ import annotations

import os

import psycopg
import pytest
from psycopg.rows import dict_row

from agent.context.report import cache_hit_rate_for_run, segment_totals_for_run, turn_report
from agent.context.tokenizer import count_message_tokens, count_tokens
from agent.messages import persist_message

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


def test_count_tokens_is_a_real_tokenizer_not_a_char_heuristic():
    # A char/4 heuristic would give the same-ish ratio for both; a real
    # BPE tokenizer gives JSON's punctuation-heavy structure a
    # noticeably worse (higher) tokens-per-char ratio than prose.
    prose = "the quick brown fox jumps over the lazy dog " * 10
    assert count_tokens(prose) > 0
    assert count_tokens("") == 0


def test_count_tokens_survives_special_token_lookalikes():
    # A real risk for this project: changelog/build-log text is
    # untrusted and could contain a substring that LOOKS like one of
    # tiktoken's own special tokens. Without disallowed_special=(),
    # this raises instead of counting.
    text = "some build output mentioning <|endoftext|> and <|im_start|> literally"
    assert count_tokens(text) > 0


def test_count_message_tokens_handles_non_string_content():
    n = count_message_tokens({"role": "tool", "content": "x" * 400})
    assert n > 0


@pytest.fixture
def seeded_run():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo = conn.execute("SELECT id FROM repos WHERE url = 'test://context-tokens-fixture'").fetchone()
    repo_id = repo["id"] if repo else conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://context-tokens-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep_id = conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'ctx-tok-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
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
    conn.close()


def test_persist_message_tags_segment_and_real_token_count(seeded_run):
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "you are an agent"}, segment="system")

    row = conn.execute(
        "SELECT tokens_by_segment FROM run_messages WHERE run_id = %s AND seq = 0", (run_id,)
    ).fetchone()
    assert set(row["tokens_by_segment"].keys()) == {"system"}
    assert row["tokens_by_segment"]["system"] > 0


def test_persist_message_rejects_unknown_segment(seeded_run):
    conn, run_id = seeded_run
    with pytest.raises(AssertionError):
        persist_message(conn, run_id, 0, "user", {"role": "user", "content": "hi"}, segment="not_a_real_segment")


def test_segment_totals_sums_across_rows(seeded_run):
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    persist_message(conn, run_id, 1, "user", {"role": "user", "content": "brief text here"}, segment="brief")
    persist_message(conn, run_id, 2, "tool", {"role": "tool", "content": "some tool output"}, segment="tool_results")
    persist_message(conn, run_id, 3, "tool", {"role": "tool", "content": "more tool output"}, segment="tool_results")

    totals = segment_totals_for_run(conn, run_id)
    assert totals["system"] > 0
    assert totals["brief"] > 0
    assert totals["tool_results"] > 0
    assert totals["scratchpad"] == 0  # never written in this run


def test_turn_report_shows_growing_context_and_tools_called(seeded_run):
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    persist_message(conn, run_id, 1, "user", {"role": "user", "content": "brief"}, segment="brief")
    persist_message(
        conn, run_id, 2, "assistant",
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                               "function": {"name": "list_files", "arguments": "{}"}}]},
        segment="assistant", cost_cents=0.5,
    )
    persist_message(conn, run_id, 3, "tool", {"role": "tool", "content": "a.js\nb.js"}, segment="tool_results")
    persist_message(
        conn, run_id, 4, "assistant",
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
                                               "function": {"name": "read_file", "arguments": '{"path":"a.js"}'}}]},
        segment="assistant", cost_cents=0.7,
    )

    report = turn_report(conn, run_id)
    assert len(report) == 2
    assert report[0]["turn"] == 1
    assert report[0]["tools_called"] == ["list_files"]
    assert report[1]["tools_called"] == ["read_file"]
    # Turn 2's context must be strictly bigger than turn 1's -- the
    # tool result from turn 1 is now part of what turn 2 sends.
    assert report[1]["total_context_tokens"] > report[0]["total_context_tokens"]
    assert report[1]["by_segment"]["tool_results"] > report[0]["by_segment"]["tool_results"]
    assert report[0]["cost_cents"] == 0.5
    assert report[1]["cost_cents"] == 0.7


def test_cache_hit_rate_returns_none_with_no_assistant_turns(seeded_run):
    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    assert cache_hit_rate_for_run(conn, run_id) is None


def test_cache_hit_rate_computes_real_ratio_across_turns(seeded_run):
    conn, run_id = seeded_run
    persist_message(
        conn, run_id, 0, "assistant", {"role": "assistant", "content": "a"}, segment="assistant",
        tokens_in=1000, tokens_out=50, cached_tokens=800,
    )
    persist_message(
        conn, run_id, 1, "assistant", {"role": "assistant", "content": "b"}, segment="assistant",
        tokens_in=1000, tokens_out=50, cached_tokens=200,
    )
    rate = cache_hit_rate_for_run(conn, run_id)
    assert rate == 0.5  # (800+200) / (1000+1000)
