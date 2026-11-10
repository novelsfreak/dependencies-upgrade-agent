# agent/context/report.py
#
# Week 5 Day 1 task 4: "One query that gives you, for any run: turn
# number, total context size, size by segment, cost, and the tool
# called." Two entry points: segment_totals_for_run (a cheap running
# total, used for the per-turn log line agent/loop.py emits live) and
# turn_report (the retrospective, per-turn breakdown Day 1's own
# "hockey stick" investigation needs -- find where context size started
# climbing and which segment drove it).
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from agent.context.segments import SEGMENTS


def segment_totals_for_run(conn: psycopg.Connection, run_id: int, revision: int = 0) -> dict[str, int]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT tokens_by_segment FROM run_messages WHERE run_id = %s AND revision = %s",
            (run_id, revision),
        )
        rows = cur.fetchall()
    totals = {s: 0 for s in SEGMENTS}
    for r in rows:
        for seg, count in (r["tokens_by_segment"] or {}).items():
            totals[seg] = totals.get(seg, 0) + count
    return totals


def cache_hit_rate_for_run(conn: psycopg.Connection, run_id: int, revision: int = 0) -> float | None:
    """
    Week 6 Day 4: sum(cached_tokens)/sum(tokens_in) across every
    assistant turn -- "Cache Hit Rate = cached_tokens / prompt_tokens"
    is Groq's own definition (console.groq.com/docs/prompt-caching),
    applied here across a whole run rather than eyeballed per-turn from
    log lines. Returns None (not 0.0) when there's nothing to divide by
    yet, so callers can tell "no data" apart from "confirmed zero".
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT COALESCE(SUM(cached_tokens), 0) AS cached, COALESCE(SUM(tokens_in), 0) AS total "
            "FROM run_messages WHERE run_id = %s AND revision = %s AND role = 'assistant'",
            (run_id, revision),
        )
        row = cur.fetchone()
    if not row["total"]:
        return None
    return row["cached"] / row["total"]


def turn_report(conn: psycopg.Connection, run_id: int, revision: int = 0) -> list[dict]:
    """
    One entry per assistant turn: turn number, the total context size
    and per-segment breakdown AS IT WAS at the moment that turn's
    request was actually sent (i.e. accumulated from every row before
    this one, not including the assistant's own response), that turn's
    cost, and which tool(s) it called.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT seq, role, content, tokens_by_segment, cost_cents "
            "FROM run_messages WHERE run_id = %s AND revision = %s ORDER BY seq",
            (run_id, revision),
        )
        rows = cur.fetchall()

    report: list[dict] = []
    running: dict[str, int] = {s: 0 for s in SEGMENTS}
    turn = 0
    for row in rows:
        if row["role"] == "assistant":
            turn += 1
            tool_names = [tc["function"]["name"] for tc in (row["content"].get("tool_calls") or [])]
            report.append({
                "turn": turn,
                "total_context_tokens": sum(running.values()),
                "by_segment": dict(running),
                "cost_cents": float(row["cost_cents"]) if row["cost_cents"] is not None else None,
                "tools_called": tool_names,
            })
        for seg, count in (row["tokens_by_segment"] or {}).items():
            running[seg] = running.get(seg, 0) + count
    return report
