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


def persist_message(
    conn: psycopg.Connection,
    run_id: int,
    seq: int,
    role: str,
    content: Any,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_cents: float | None = None,
) -> None:
    import json

    conn.execute(
        """
        INSERT INTO run_messages (run_id, seq, role, content, tokens_in, tokens_out, cost_cents)
        VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
        ON CONFLICT (run_id, seq) DO NOTHING
        """,
        (run_id, seq, role, json.dumps(content), tokens_in, tokens_out, cost_cents),
    )
    conn.commit()


def load_messages(conn: psycopg.Connection, run_id: int) -> list[dict]:
    """Not used by the Week 3 loop itself (a fresh run always starts at
    seq 0) -- exists now because Week 4's resume-on-crash needs exactly
    this, and the table already makes it trivial."""
    rows = conn.execute(
        "SELECT seq, role, content FROM run_messages WHERE run_id = %s ORDER BY seq",
        (run_id,),
    ).fetchall()
    return [{"seq": r[0], "role": r[1], "content": r[2]} for r in rows]


def total_cost_cents(conn: psycopg.Connection, run_id: int) -> float:
    # Explicit tuple_row, not whatever row_factory this connection
    # happens to default to (production leaves it unset -> tuples
    # already, but a caller configured for dict_row would otherwise
    # break on row[0] here).
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT COALESCE(SUM(cost_cents), 0) FROM run_messages WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    return float(row[0])
