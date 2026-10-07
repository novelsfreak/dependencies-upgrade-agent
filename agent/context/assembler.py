# agent/context/assembler.py
#
# Week 5 Day 2: the architectural change the rest of week 5 depends on.
# Before today, agent/loop.py owned an in-memory `messages` list it
# grew with .append() and trusted to stay in sync with whatever got
# persisted. From today, Postgres (run_messages) is the only truth --
# ContextAssembler.build() rebuilds the array fresh from it every turn,
# and the loop just asks for the current view instead of owning one.
#
# Day 2 itself changes nothing about WHAT gets rendered -- this is
# still a full, verbatim replay, identical to what load_messages()
# already did (the plan's own "done when": "the same 20 runs produce
# identical results to before, with the assembler in the path"). Day
# 3's compaction and Day 4's working-set eviction are what start
# actually shrinking the output; they change what happens INSIDE
# build(), not its contract, which is what makes them additive rather
# than another rewrite of loop.py's control flow.
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from agent.context import compaction, working_set
from agent.pricing import CHEAP_MODEL


class BudgetExceeded(Exception):
    """
    Raised when a FIXED segment (system, tools, brief, repo_map) alone
    is already over its own declared budget -- the plan's own words:
    "if assembly can't fit under budget, that's a bug -- fail loudly
    rather than sending an oversized request and finding out from a
    400." Fixed segments are supposed to already fit by construction
    (the system prompt is a constant; the brief is built to a budget);
    if one doesn't, that's a real bug upstream, not something an
    eviction policy can paper over the way it can for the working set
    or tool results.
    """


# Real, MEASURED budgets for THIS project's actual per-request ceiling
# (agent/loop.py's REQUEST_TOKEN_CEILING=7600, ~7300 after the output
# reserve) -- not the plan's own illustrative table sized for a 200k
# context window, which this project doesn't operate at. Day 1's live
# proof (run 4438) measured system ~350, tools ~1210, and an
# UNTRUNCATED brief with a full changelog embed at ~3100 tokens -- 42%
# of the entire real budget before one tool call. Week 5 Day 3 fixed
# that specific finding (agent/changelog.py's brief_excerpt caps what
# the brief embeds; build_tools still gets the FULL text for Day 4's
# correlation), so "brief" is tightened here to match the new real
# size with headroom, not the plan's generic number.
FIXED_BUDGETS = {
    "system": 600,
    "tools": 1500,
    "brief": 1200,
    "repo_map": 500,  # declared now, unused until week 6 day 5 builds one
}

# Compactable segments don't fail loudly when over budget -- Day 3's
# compaction (and Day 4's working-set eviction) are what actively keep
# them near this line; agent/loop.py's own ceiling check is the
# already-existing backstop if they don't quite make it. 80% is the
# plan's own trigger point, applied to this project's real remaining
# room after the fixed segments above and the output reserve.
COMPACTABLE_BUDGETS = {
    "tool_results": 3500,
}
COMPACTION_TRIGGER_RATIO = 0.8

# Week 5 Day 4: real, working-set-specific budget (see
# agent/context/working_set.py) -- files the agent has open are a
# separate, LRU-evicted concern from the general tool_results bucket.
WORKING_SET_BUDGET = working_set.WORKING_SET_BUDGET


def build(conn: psycopg.Connection, run_id: int, revision: int = 0, client=None) -> list[dict]:
    """
    The one place in the codebase that reconstructs the conversation
    array from run_messages. agent/loop.py calls this fresh every turn
    instead of keeping its own list.

    Passes now run over the loaded rows before rendering, in order:
    1. deduplicate (Day 3, always, free, deterministic)
    2. working-set eviction (Day 4, always, free, deterministic --
       targets read_file results specifically, by path, before the
       more general pass below sees them)
    3. scratchpad supersession (Day 5, always, free, deterministic --
       keeps only the latest write_findings note verbatim)
    4. compaction (Day 3, only when `client` is given, since it costs
       a real, cheap-model API call -- everything else that's still
       over budget after the more targeted passes above; write_findings
       exchanges are protected from this one, see _protected_indices)
    None of these ever write back to run_messages; they only change
    what THIS call returns.
    """
    rows = _load_rows(conn, run_id, revision)
    _check_fixed_budgets(rows)

    rows = compaction.deduplicate(rows)
    rows = working_set.apply_working_set_eviction(rows, budget=WORKING_SET_BUDGET)
    rows = compaction.supersede_scratchpad(rows)
    rows = compaction.maybe_compact(
        conn, run_id, revision, rows, client,
        budget=COMPACTABLE_BUDGETS["tool_results"], trigger_ratio=COMPACTION_TRIGGER_RATIO, model=CHEAP_MODEL,
    )
    return [r["content"] for r in rows]


def _load_rows(conn: psycopg.Connection, run_id: int, revision: int) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT seq, role, content, tokens_by_segment FROM run_messages "
            "WHERE run_id = %s AND revision = %s ORDER BY seq",
            (run_id, revision),
        )
        return cur.fetchall()


def _check_fixed_budgets(rows: list[dict]) -> None:
    for row in rows:
        for segment, count in (row["tokens_by_segment"] or {}).items():
            budget = FIXED_BUDGETS.get(segment)
            if budget is not None and count > budget:
                raise BudgetExceeded(
                    f"seq {row['seq']} ({row['role']}, segment={segment!r}) is {count} tokens, "
                    f"over its fixed budget of {budget} -- this segment is supposed to already fit "
                    f"by construction; investigate what generated it rather than raising the budget."
                )
