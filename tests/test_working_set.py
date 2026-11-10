"""
Week 5 Day 4: files the agent has read are an explicit, LRU-evicted
working set, distinct from Day 3's general tool_results compaction --
eviction is always ANNOUNCED, and a file named in the current error or
the most recent applied diff is never evicted.
"""
from __future__ import annotations

from agent.context.working_set import apply_working_set_eviction


def _row(seq, role, content):
    return {"seq": seq, "role": role, "content": content}


def _read(seq, call_id, path, filler="x"):
    return [
        _row(seq, "assistant", {
            "role": "assistant",
            "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "read_file", "arguments": f'{{"path":"{path}"}}'}}],
        }),
        _row(seq + 1, "tool", {"role": "tool", "tool_call_id": call_id, "content": filler}),
    ]


def test_no_eviction_under_budget():
    rows = _read(0, "c1", "a.js", "small content")
    out = apply_working_set_eviction(rows, budget=10000)
    assert out == rows


def test_evicts_least_recently_read_file_first_and_announces_it():
    rows = []
    rows += _read(0, "c1", "old.js", "y " * 2000)      # read first -- LRU, big
    rows += _read(2, "c2", "new.js", "small content")  # read most recently, tiny
    out = apply_working_set_eviction(rows, budget=100)

    old_content = out[1]["content"]["content"]
    new_content = out[3]["content"]["content"]
    assert "was removed from context" in old_content
    assert "old.js" in old_content
    assert new_content == "small content"  # most recently read survives


def test_never_evicts_a_file_named_in_the_current_error():
    rows = []
    rows += _read(0, "c1", "broken.js", "y " * 2000)
    rows += _read(2, "c2", "other.js", "z " * 2000)
    rows.append(_row(4, "tool", {
        "role": "tool", "tool_call_id": "cbuild",
        "content": "build failed: TS2554 in broken.js -- expected 2 arguments, got 1",
    }))
    out = apply_working_set_eviction(rows, budget=1500)

    broken_content = out[1]["content"]["content"]
    assert broken_content == "y " * 2000  # protected -- never evicted despite being LRU


def test_a_second_read_of_the_same_path_supersedes_the_first():
    rows = []
    rows += _read(0, "c1", "a.js", "old version")
    rows += _read(2, "c2", "a.js", "new version, after a patch changed it")
    out = apply_working_set_eviction(rows, budget=10000)
    # Nothing to evict (well under budget) -- both entries remain as-is,
    # but only the LATEST read's seq should ever count as "open" for a.js.
    assert out[3]["content"]["content"] == "new version, after a patch changed it"


def test_returns_rows_unchanged_when_no_read_file_calls_exist():
    rows = [_row(0, "system", {"role": "system", "content": "sys"})]
    out = apply_working_set_eviction(rows, budget=10)
    assert out == rows
