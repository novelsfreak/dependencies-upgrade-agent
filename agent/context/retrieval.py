# agent/context/retrieval.py
#
# Week 6 Day 5: measuring compliance with "search before you read" --
# the plan's own instruction is to prompt for it AND measure whether
# the model actually does it, not just hope. A read_file call counts as
# "speculative" (not preceded by a search that surfaced that same
# path) if no earlier search result in the same conversation mentioned
# that file.
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row


def _tool_call_index(rows: list[dict]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row["role"] != "assistant":
            continue
        for tc in (row["content"].get("tool_calls") or []):
            index[tc["id"]] = (tc["function"]["name"], tc["function"]["arguments"])
    return index


def search_before_read_compliance(conn: psycopg.Connection, run_id: int, revision: int = 0) -> dict:
    """
    Returns {"read_file_calls": N, "preceded_by_search": M, "compliance_rate": M/N}
    (rate is None when there were zero read_file calls -- nothing to
    measure, not zero compliance).
    """
    import re

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT seq, role, content FROM run_messages WHERE run_id = %s AND revision = %s ORDER BY seq",
            (run_id, revision),
        )
        rows = cur.fetchall()

    call_index = _tool_call_index(rows)
    path_re = re.compile(r'"path"\s*:\s*"([^"]+)"')

    seen_search_paths: set[str] = set()
    total_reads = 0
    preceded = 0

    for row in rows:
        if row["role"] != "tool":
            continue
        name_args = call_index.get(row["content"].get("tool_call_id"))
        if not name_args:
            continue
        name, args = name_args
        content = row["content"].get("content")
        if name == "search" and isinstance(content, str):
            for line in content.splitlines():
                # search results are "path:line:text (suggested...)" --
                # the path is everything before the first ':'.
                if ":" in line:
                    seen_search_paths.add(line.split(":", 1)[0])
        elif name == "read_file":
            m = path_re.search(args)
            if m:
                total_reads += 1
                if m.group(1) in seen_search_paths:
                    preceded += 1

    return {
        "read_file_calls": total_reads,
        "preceded_by_search": preceded,
        "compliance_rate": (preceded / total_reads) if total_reads else None,
    }
