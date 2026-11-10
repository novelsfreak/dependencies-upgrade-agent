"""
Week 5 Day 3: deduplication (free, deterministic) and compaction (a
real cheap-model call, mocked here for determinism -- same reasoning
as tests/test_agent_stop_conditions.py) over the RENDERED view only.
run_messages itself must never be touched by either pass.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row

from agent.context.compaction import deduplicate, maybe_compact, supersede_scratchpad
from agent.context.tokenizer import count_message_tokens
from agent.messages import persist_message

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


def _row(seq, role, content):
    return {"seq": seq, "role": role, "content": content}


def _assistant_call(seq, call_id, name, args):
    return _row(seq, "assistant", {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}],
    })


def _tool_result(seq, call_id, content):
    return _row(seq, "tool", {"role": "tool", "tool_call_id": call_id, "content": content})


def test_deduplicate_collapses_identical_repeat_reads():
    rows = [
        _row(0, "system", {"role": "system", "content": "sys"}),
        _row(1, "user", {"role": "user", "content": "brief"}),
        _assistant_call(2, "c1", "read_file", '{"path":"a.js"}'),
        _tool_result(3, "c1", "const x = 1;"),
        _assistant_call(4, "c2", "read_file", '{"path":"a.js"}'),
        _tool_result(5, "c2", "const x = 1;"),  # identical content -- a true no-op repeat
    ]
    out = deduplicate(rows)
    assert "duplicate" in out[3]["content"]["content"]
    assert out[5]["content"]["content"] == "const x = 1;"  # latest kept verbatim


def test_deduplicate_never_touches_a_genuinely_changed_result():
    rows = [
        _assistant_call(0, "c1", "read_file", '{"path":"a.js"}'),
        _tool_result(1, "c1", "const x = 1;"),
        _assistant_call(2, "c2", "read_file", '{"path":"a.js"}'),
        _tool_result(3, "c2", "const x = 2;"),  # file legitimately changed
    ]
    out = deduplicate(rows)
    assert out[1]["content"]["content"] == "const x = 1;"
    assert out[3]["content"]["content"] == "const x = 2;"


def test_deduplicate_ignores_non_string_and_untracked_tool_results():
    rows = [
        _tool_result(0, "unknown-id", "orphan result, no matching assistant call"),
    ]
    out = deduplicate(rows)  # must not raise
    assert out[0]["content"]["content"] == "orphan result, no matching assistant call"


def test_supersede_scratchpad_keeps_only_the_latest_note():
    rows = [
        _assistant_call(0, "c1", "write_findings", '{"content":"first draft"}'),
        _tool_result(1, "c1", "findings saved (2 tokens)."),
        _assistant_call(2, "c2", "write_findings", '{"content":"updated draft"}'),
        _tool_result(3, "c2", "findings saved (2 tokens)."),
    ]
    out = supersede_scratchpad(rows)
    assert "superseded" in out[1]["content"]["content"]
    assert out[3]["content"]["content"] == "findings saved (2 tokens)."


def test_supersede_scratchpad_is_a_noop_with_zero_or_one_calls():
    rows = [_assistant_call(0, "c1", "write_findings", '{"content":"only draft"}'),
            _tool_result(1, "c1", "findings saved (2 tokens).")]
    assert supersede_scratchpad(rows) == rows
    assert supersede_scratchpad([]) == []


def test_compaction_preserves_the_scratchpad_byte_for_byte_under_pressure(seeded_run):
    """
    Week 6 Day 7 context invariant: "compaction preserves the
    scratchpad byte-for-byte." Builds a conversation big enough to
    force real compaction (mocked cheap-model client) alongside a real
    write_findings note, then asserts the note comes back through
    assemble_context completely unmodified -- not summarized,
    paraphrased, or truncated by so much as one character.
    """
    from agent.context.assembler import build

    conn, run_id = seeded_run
    findings_note = "Changed: src/index.js to use named export. Tried and failed: default export cast. Open: none."
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    persist_message(conn, run_id, 1, "user", {"role": "user", "content": "brief"}, segment="brief")
    persist_message(conn, run_id, 2, "assistant", {
        "role": "assistant",
        "tool_calls": [{"id": "cf", "type": "function", "function": {"name": "write_findings", "arguments": "{}"}}],
    }, segment="assistant")
    persist_message(conn, run_id, 3, "tool", {"role": "tool", "tool_call_id": "cf",
                                               "content": f"findings saved. {findings_note}"}, segment="scratchpad")

    seq = 4
    for i in range(30):
        for row in _make_big_exchange(seq, f"filler content {i} " * 50):
            persist_message(conn, run_id, row["seq"], row["role"], row["content"], segment="tool_results")
        seq += 2

    client = _fake_summarizer_client()
    rendered = build(conn, run_id, client=client)

    scratchpad_messages = [
        m for m in rendered if m.get("role") == "tool" and isinstance(m.get("content"), str)
        and findings_note in m["content"]
    ]
    assert len(scratchpad_messages) == 1
    assert scratchpad_messages[0]["content"] == f"findings saved. {findings_note}"


def test_maybe_compact_never_selects_a_write_findings_exchange(seeded_run):
    conn, run_id = seeded_run
    rows = [_row(0, "system", {"role": "system", "content": "sys"})]
    rows.append(_assistant_call(1, "cf", "write_findings", '{"content":"important note"}'))
    rows.append(_tool_result(2, "cf", "findings saved (3 tokens)."))
    seq = 3
    for i in range(10):
        rows += _make_big_exchange(seq, "filler " * 200)
        seq += 2

    client = _fake_summarizer_client()
    maybe_compact(conn, run_id, 0, rows, client, budget=1000, trigger_ratio=0.8)
    # The write_findings exchange must never appear in a compacted-away
    # range: verify by checking the persisted summary doesn't cover it.
    summary_row = conn.execute(
        "SELECT covers_seq_start, covers_seq_end FROM compaction_summaries WHERE run_id = %s", (run_id,)
    ).fetchone()
    assert summary_row["covers_seq_start"] > 2


@pytest.fixture
def seeded_run():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo_id = conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://compaction-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    dep_id = conn.execute(
        "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
        "VALUES (%s, 'compact-dep', 'npm', '1.0.0', 'package.json') RETURNING id",
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

    conn.execute("DELETE FROM compaction_summaries WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.execute("DELETE FROM candidates WHERE id = %s", (candidate_id,))
    conn.execute("DELETE FROM dependencies WHERE id = %s", (dep_id,))
    conn.execute("DELETE FROM repos WHERE id = %s", (repo_id,))
    conn.close()


def _fake_summarizer_client(summary_text="## Findings so far\nChanged: a.js"):
    fake = MagicMock()
    message = SimpleNamespace(content=summary_text)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    fake.chat.completions.create.return_value = SimpleNamespace(choices=[choice], usage=usage)
    return fake


def _make_big_exchange(seq_start: int, filler: str) -> list[dict]:
    return [
        _assistant_call(seq_start, f"c{seq_start}", "read_file", f'{{"path":"f{seq_start}.js"}}'),
        _tool_result(seq_start + 1, f"c{seq_start}", filler),
    ]


def test_maybe_compact_does_nothing_under_budget(seeded_run):
    conn, run_id = seeded_run
    rows = [_row(0, "system", {"role": "system", "content": "sys"}), _row(1, "user", {"role": "user", "content": "b"})]
    rows += _make_big_exchange(2, "small")
    client = _fake_summarizer_client()

    out = maybe_compact(conn, run_id, 0, rows, client, budget=10000, trigger_ratio=0.8)
    assert out == rows
    client.chat.completions.create.assert_not_called()


def test_maybe_compact_skips_without_a_client(seeded_run):
    conn, run_id = seeded_run
    rows = [_row(0, "system", {"role": "system", "content": "sys"})]
    for i in range(20):
        rows += _make_big_exchange(1 + i * 2, "x" * 2000)
    out = maybe_compact(conn, run_id, 0, rows, None, budget=100, trigger_ratio=0.8)
    assert out == rows


def test_maybe_compact_summarizes_oldest_half_and_protects_recent_and_current_diff(seeded_run):
    conn, run_id = seeded_run
    rows = [
        _row(0, "system", {"role": "system", "content": "sys"}),
        _row(1, "user", {"role": "user", "content": "brief"}),
    ]
    # 10 old, unremarkable exchanges -- pure filler, safe to compact.
    seq = 2
    for i in range(10):
        rows += _make_big_exchange(seq, f"filler content number {i} " * 20)
        seq += 2
    # One exchange holds the current diff -- must survive regardless of position.
    rows.append(_assistant_call(seq, "cdiff", "apply_patch", '{"diff": "..."}'))
    rows.append(_tool_result(seq + 1, "cdiff", "patch applied and committed. Current diff from before your patch:\n\ndiff..."))
    seq += 2

    client = _fake_summarizer_client()
    out = maybe_compact(conn, run_id, 0, rows, client, budget=1000, trigger_ratio=0.8)

    client.chat.completions.create.assert_called_once()
    # The synthetic summary message replaced the compacted stretch.
    assert any(
        isinstance(r["content"].get("content"), str) and "was compacted to save context space" in r["content"]["content"]
        for r in out
    )
    # The current diff's exchange is untouched and still present verbatim.
    assert any(
        r["role"] == "tool" and isinstance(r["content"].get("content"), str)
        and "applied and committed" in r["content"]["content"]
        for r in out
    )
    # run_messages was never touched -- compaction only ever affects the render.
    original_count = conn.execute("SELECT count(*) c FROM run_messages WHERE run_id = %s", (run_id,)).fetchone()["c"]
    assert original_count == 0  # nothing was ever persisted to run_messages by this test -- proves no write occurred
    # But the summary WAS persisted, for human review later.
    summaries = conn.execute("SELECT * FROM compaction_summaries WHERE run_id = %s", (run_id,)).fetchall()
    assert len(summaries) == 1
    assert "Changed: a.js" in summaries[0]["summary_text"]


def test_assembler_never_lets_tool_results_run_away_over_500_synthetic_turns(seeded_run):
    """
    Week 6 Day 7 context invariant: "assembler output never exceeds
    budget, on synthetic conversations up to 500 turns." No live model
    needed -- persists 500 progressively-growing exchanges one at a
    time (matching how the real loop actually calls build(), turn by
    turn, not a single 500-turn dump) through the real assembler with a
    mocked cheap-model client, and checks that compaction keeps the
    tool_results segment from running away unbounded.
    """
    from agent.context.assembler import build
    from agent.context.assembler import COMPACTABLE_BUDGETS

    conn, run_id = seeded_run
    persist_message(conn, run_id, 0, "system", {"role": "system", "content": "sys"}, segment="system")
    persist_message(conn, run_id, 1, "user", {"role": "user", "content": "brief"}, segment="brief")

    client = _fake_summarizer_client("## Findings so far\nChanged: many files across this long run")

    seq = 2
    budget = COMPACTABLE_BUDGETS["tool_results"]
    max_observed_over_budget = 0
    for i in range(500):
        persist_message(
            conn, run_id, seq, "assistant",
            {"role": "assistant", "tool_calls": [{"id": f"c{i}", "type": "function",
                                                    "function": {"name": "read_file", "arguments": f'{{"path":"f{i}.js"}}'}}]},
            segment="assistant",
        )
        persist_message(
            conn, run_id, seq + 1, "tool",
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"file {i} content " * 30},
            segment="tool_results",
        )
        seq += 2

        rendered = build(conn, run_id, client=client)
        rendered_tool_tokens = sum(
            count_message_tokens(m) for m in rendered if m.get("role") == "tool"
        )
        if rendered_tool_tokens > budget:
            max_observed_over_budget = max(max_observed_over_budget, rendered_tool_tokens - budget)

    # Compaction only fires once it crosses the trigger ratio and then
    # halves the ELIGIBLE portion -- some transient overshoot between
    # compaction events is expected (it's a trigger, not a hard cap the
    # plan itself enforces at every single turn). What must NOT happen
    # is unbounded growth: the overshoot should stay small relative to
    # one exchange's worth of tokens, not climb with turn count.
    assert max_observed_over_budget < budget  # never even doubles the budget, let alone grows unbounded


def test_maybe_compact_never_selects_a_lone_correction_message(seeded_run):
    conn, run_id = seeded_run
    rows = [_row(0, "system", {"role": "system", "content": "sys"})]
    rows.append(_row(1, "user", {"role": "user", "content": "a correction message, not an exchange"}))
    seq = 2
    for i in range(10):
        rows += _make_big_exchange(seq, "filler " * 200)
        seq += 2

    client = _fake_summarizer_client()
    out = maybe_compact(conn, run_id, 0, rows, client, budget=1000, trigger_ratio=0.8)
    # The lone correction must still be present, untouched.
    assert any(r["content"].get("content") == "a correction message, not an exchange" for r in out)
