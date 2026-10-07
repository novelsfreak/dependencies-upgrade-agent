# agent/messages.py
#
# Persistence for the agent conversation, one row per turn. Written as
# the loop goes, not batched at the end -- a crash mid-run should lose
# at most the in-flight turn, not the whole conversation (Week 4 builds
# resume-from-here on top of this; this just makes sure the data exists).
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import tuple_row

from agent.context.segments import SEGMENTS
from agent.context.tokenizer import count_message_tokens


def persist_message(
    conn: psycopg.Connection,
    run_id: int,
    seq: int,
    role: str,
    content: Any,
    segment: str = "tool_results",
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_cents: float | None = None,
    cached_tokens: int | None = None,
    revision: int = 0,
) -> None:
    """
    Week 5 Day 1: every persisted turn is tagged with which of the
    plan's own context segments it belongs to (agent/context/segments.py),
    and its real tokenizer-counted size (not len(text)/4 -- see
    agent/context/tokenizer.py) is stored alongside it. This is the
    instrument the rest of week 5 is built on: without a per-turn,
    per-segment record, "which tokens carried weight" is a guess, not a
    measurement.
    """
    import json

    assert segment in SEGMENTS, f"unknown segment {segment!r}, add it to agent/context/segments.py"
    tokens_by_segment = {segment: count_message_tokens(content)}

    conn.execute(
        """
        INSERT INTO run_messages
            (run_id, revision, seq, role, content, tokens_in, tokens_out, cost_cents, tokens_by_segment, cached_tokens)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (run_id, revision, seq) DO NOTHING
        """,
        (run_id, revision, seq, role, json.dumps(content), tokens_in, tokens_out, cost_cents,
         json.dumps(tokens_by_segment), cached_tokens),
    )
    conn.commit()


def load_messages(conn: psycopg.Connection, run_id: int, revision: int = 0) -> list[dict]:
    """Week 4 Day 1: rebuilds a resumed run's conversation from here.
    Week 4 Day 2: scoped to one revision round -- REVISING starts a
    fresh, compacted conversation per round rather than replaying every
    prior round's raw history forever (see db/migrations/004), so a
    resume must only ever look at the CURRENT round's messages, not
    every round this run has ever had.

    Explicit tuple_row for the same reason total_cost_cents needs it --
    this positional r[0]/r[1]/r[2] indexing silently breaks under a
    dict_row-configured connection (caught for real by
    tests/test_agent_resume.py, the first thing to actually call this
    since it was written in Week 3 with nothing exercising it yet)."""
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT seq, role, content FROM run_messages WHERE run_id = %s AND revision = %s ORDER BY seq",
            (run_id, revision),
        )
        rows = cur.fetchall()
    return [{"seq": r[0], "role": r[1], "content": r[2]} for r in rows]


def total_cost_cents(conn: psycopg.Connection, run_id: int) -> float:
    # Explicit tuple_row, not whatever row_factory this connection
    # happens to default to (production leaves it unset -> tuples
    # already, but a caller configured for dict_row would otherwise
    # break on row[0] here).
    #
    # Week 5 Day 3: includes compaction_summaries.cost_cents too --
    # compaction is a real, billed cheap-model call, and MAX_COST_CENTS
    # enforcement (agent/loop.py) is meaningless if it can't see that
    # spend.
    #
    # Week 6 Day 1: also includes every direct child run's own spend
    # (run_messages + ITS compaction_summaries) -- "budget drawn from
    # the parent's remaining allowance" (the plan's own words) means a
    # parent's MAX_COST_CENTS check has to see what its sub-agents
    # spent, not just its own conversation. One level deep only:
    # sub-agents don't spawn sub-agents in this design.
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT COALESCE(SUM(cost_cents), 0) FROM run_messages WHERE run_id = %s",
            (run_id,),
        )
        messages_total = float(cur.fetchone()[0])
        cur.execute(
            "SELECT COALESCE(SUM(cost_cents), 0) FROM compaction_summaries WHERE run_id = %s",
            (run_id,),
        )
        compaction_total = float(cur.fetchone()[0])
        cur.execute(
            "SELECT COALESCE(SUM(rm.cost_cents), 0) FROM run_messages rm "
            "JOIN runs r ON r.id = rm.run_id WHERE r.parent_run_id = %s",
            (run_id,),
        )
        children_messages_total = float(cur.fetchone()[0])
        cur.execute(
            "SELECT COALESCE(SUM(cs.cost_cents), 0) FROM compaction_summaries cs "
            "JOIN runs r ON r.id = cs.run_id WHERE r.parent_run_id = %s",
            (run_id,),
        )
        children_compaction_total = float(cur.fetchone()[0])
    return messages_total + compaction_total + children_messages_total + children_compaction_total
