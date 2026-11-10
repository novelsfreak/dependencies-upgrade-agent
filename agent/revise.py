# agent/revise.py
#
# Week 4 Day 2: turns one revision round's raw conversation into a
# short, mechanical account of what it actually did -- not an LLM
# summary (that would cost real tokens just to start the NEXT
# conversation, working against the whole point of compacting), a
# deterministic scan over what's already persisted.
from __future__ import annotations

import psycopg

from agent.messages import load_messages

_MAX_SUMMARY_CHARS = 4000


def summarize_prior_conversation(conn: psycopg.Connection, run_id: int, revision: int) -> str:
    """
    Extracts, from one revision round's persisted messages: every
    successfully-applied patch's diff (the diff IS the why -- it
    carries its own context lines), and the last build/test outcome if
    either tool was called. Mechanical and cheap, on purpose -- see the
    module docstring.
    """
    messages = load_messages(conn, run_id, revision=revision)

    patches: list[str] = []
    last_build: str | None = None
    last_test: str | None = None

    for m in messages:
        if m["role"] != "tool":
            continue
        text = m["content"].get("content")
        if not isinstance(text, str):
            continue
        if text.startswith("patch applied and committed"):
            # Keep the diff itself, drop the tool's own boilerplate
            # prefix -- the diff is the informative part.
            _, _, diff = text.partition("Current diff from before your patch:")
            if diff.strip():
                patches.append(diff.strip())
        elif text.startswith(("build ok", "build failed")):
            last_build = text
        elif text.startswith(("tests ok", "tests failed")):
            last_test = text

    if patches:
        lines = [f"{len(patches)} patch(es) were already applied and committed to this branch:"]
        for i, p in enumerate(patches, 1):
            lines.append(f"\n--- patch {i} ---\n{p}")
    else:
        lines = ["No patch was successfully applied and committed in the prior attempt."]

    if last_build:
        lines.append(f"\nLast build result: {last_build}")
    if last_test:
        lines.append(f"\nLast test result: {last_test}")

    summary = "\n".join(lines)
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "\n... (summary truncated)"
    return summary
