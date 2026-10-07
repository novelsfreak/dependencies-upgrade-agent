# agent/context/working_set.py
#
# Week 5 Day 4: files the agent has read are tracked as an explicit,
# LRU-evicted "working set", distinct from the general tool_results
# compaction Day 3 built. A file's current content is a different kind
# of thing than a build log -- the model actively reasons against it
# turn after turn, and silently dropping it while the model still
# believes it's there is the specific failure mode the plan calls out:
# "a confident, well-reasoned edit to content that is no longer there."
from __future__ import annotations

import re

from agent.context.tokenizer import count_message_tokens

# Real, scaled to this project's actual per-request ceiling (see
# agent/context/assembler.py's FIXED_BUDGETS/COMPACTABLE_BUDGETS
# comment for the full accounting) -- not the plan's 45k, sized for a
# 200k-window model.
WORKING_SET_BUDGET = 2500

_READ_FILE_PATH_RE = re.compile(r'"path"\s*:\s*"([^"]+)"')
_SOURCE_FILE_RE = re.compile(r"[\w./-]+\.(?:js|ts|jsx|tsx|mjs|cjs|py|json)\b")


def _tool_call_index(rows: list[dict]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row["role"] != "assistant":
            continue
        for tc in (row["content"].get("tool_calls") or []):
            index[tc["id"]] = (tc["function"]["name"], tc["function"]["arguments"])
    return index


def _path_from_args(args: str) -> str | None:
    m = _READ_FILE_PATH_RE.search(args)
    return m.group(1) if m else None


def _protected_paths(rows: list[dict]) -> set[str]:
    """
    Never evict a file named in the most recent build/test failure or
    touched by the most recent applied patch's diff -- the plan's own
    exclusion ("never evict a file with an unapplied pending edit, or a
    file named in the current error").
    """
    last_error_text = None
    last_diff_text = None
    for row in rows:
        if row["role"] != "tool" or not isinstance(row["content"].get("content"), str):
            continue
        text = row["content"]["content"]
        if "build failed" in text or "tests failed" in text:
            last_error_text = text
        if "applied and committed" in text:
            last_diff_text = text

    protected: set[str] = set()
    for text in (last_error_text, last_diff_text):
        if not text:
            continue
        for m in _SOURCE_FILE_RE.finditer(text):
            protected.add(m.group(0).removeprefix("a/").removeprefix("b/"))
    return protected


def apply_working_set_eviction(rows: list[dict], budget: int = WORKING_SET_BUDGET) -> list[dict]:
    """
    Finds the most recent read_file result for each distinct path (an
    earlier read of the same path is superseded -- not "open" anymore).
    If the combined size of currently-open files is over budget, evicts
    the least-recently-read ones first, replacing their content with an
    explicit, model-visible announcement -- never silently, since a
    silently-vanished file is what produces a confident edit against
    content that's no longer there.
    """
    call_index = _tool_call_index(rows)
    protected = _protected_paths(rows)

    latest_read_seq: dict[str, int] = {}
    for row in rows:
        if row["role"] != "tool":
            continue
        name_args = call_index.get(row["content"].get("tool_call_id"))
        if not name_args or name_args[0] != "read_file":
            continue
        path = _path_from_args(name_args[1])
        if path:
            latest_read_seq[path] = row["seq"]  # a later read of the same path wins

    if not latest_read_seq:
        return rows

    seq_to_path = {seq: path for path, seq in latest_read_seq.items()}
    out = [dict(r) for r in rows]
    by_seq = {r["seq"]: r for r in out}
    sizes = {seq: count_message_tokens(by_seq[seq]["content"]) for seq in seq_to_path}
    total = sum(sizes.values())
    if total <= budget:
        return out

    # Least-recently-read first (oldest seq among the LATEST reads).
    lru_order = sorted(seq_to_path.items(), key=lambda kv: kv[0])
    for seq, path in lru_order:
        if total <= budget:
            break
        if path in protected:
            continue
        row = by_seq[seq]
        freed = sizes[seq]
        row["content"] = dict(row["content"])
        row["content"]["content"] = (
            f"[{path} ({freed} tokens) was removed from context to save space. "
            f"Call read_file to bring it back if you need it again.]"
        )
        total -= freed
    return out
