# core/taxonomy.py
#
# Week 4 Day 6: the plan's own failure categories, applied to one
# completed run's final state. Mechanical classification over what's
# already persisted (checkpoint, run_messages) -- no LLM call needed to
# classify a run about an LLM.
from __future__ import annotations

import re

import psycopg
from psycopg.rows import dict_row

CATEGORIES = [
    "succeeded",
    "ran_out_of_turns_making_progress",
    "ran_out_of_turns_looping",
    "patch_kept_failing_to_apply",
    "fixed_build_broke_tests",
    "modified_test_to_pass",  # flagged loudly -- see the plan's own callout
    "misread_changelog",
    "genuinely_impossible",
    "infra_failure",
    "auto_zero_token_success",
    "uncategorized",
]

_TEST_FILE_RE = re.compile(r"\btest[s]?[/\\][\w.-]+\.(test|spec)\.[jt]sx?\b|__tests__", re.IGNORECASE)


def classify_run(conn: psycopg.Connection, run_id: int) -> tuple[str, str]:
    """
    Returns (category, evidence) -- evidence is a short human-readable
    quote/summary of what actually drove the classification, so a
    person reviewing the taxonomy by hand (the plan's own instruction:
    "open every failed run") can verify or override it in a glance
    rather than re-deriving it from scratch.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT state, checkpoint FROM runs WHERE id = %s", (run_id,))
        run = cur.fetchone()
    if run is None:
        return "uncategorized", f"run {run_id} not found"

    state = run["state"]
    checkpoint = run["checkpoint"] or {}

    if state in ("PATCH_READY", "PR_OPEN", "AWAITING_CI", "MERGED_READY"):
        if checkpoint.get("triage_classification") == "AUTO":
            return "auto_zero_token_success", checkpoint.get("triage_reason", "AUTO path, no agent turns")
        return "succeeded", "reached PATCH_READY or later with a real passing build+test"

    if state == "ESCALATED":
        reason = checkpoint.get("escalated_reason", "")
        if "max_turns" in reason:
            return _classify_max_turns_escalation(conn, run_id, reason)
        if "repeated identical tool call" in reason:
            return "ran_out_of_turns_looping", reason
        if "agent gave up" in reason:
            return _classify_give_up(reason)
        if "max_cost_cents" in reason or "per-request cap" in reason or "413" in reason:
            return "infra_failure", reason
        return "uncategorized", reason or "escalated with no reason recorded"

    if state == "FAILED":
        return _classify_failed(conn, run_id, checkpoint)

    if state == "SKIPPED":
        return "uncategorized", checkpoint.get("triage_reason", "deny-listed")

    return "uncategorized", f"run still in-flight (state={state})"


def _classify_max_turns_escalation(conn: psycopg.Connection, run_id: int, reason: str) -> tuple[str, str]:
    # A patch that kept failing to apply is visible directly in the
    # persisted tool results -- count how much of the conversation was
    # actually apply_patch rejections vs. anything else.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT content FROM run_messages WHERE run_id = %s AND role = 'tool' ORDER BY seq", (run_id,)
        )
        tool_msgs = cur.fetchall()

    rejected = sum(
        1 for m in tool_msgs
        if isinstance(m["content"].get("content"), str) and "did not apply" in m["content"]["content"]
    )
    if tool_msgs and rejected / len(tool_msgs) > 0.3:
        return "patch_kept_failing_to_apply", f"{rejected}/{len(tool_msgs)} tool results were rejected patches"

    applied = any(
        isinstance(m["content"].get("content"), str) and "applied and committed" in m["content"]["content"]
        for m in tool_msgs
    )
    if applied:
        return "ran_out_of_turns_making_progress", "at least one real patch landed before turns ran out"
    return "ran_out_of_turns_making_progress", reason


def _classify_give_up(reason: str) -> tuple[str, str]:
    lowered = reason.lower()
    if "peer dep" in lowered or "conflict" in lowered or "incompatible" in lowered:
        return "genuinely_impossible", reason
    return "genuinely_impossible", reason


def _classify_failed(conn: psycopg.Connection, run_id: int, checkpoint: dict) -> tuple[str, str]:
    last_error = checkpoint.get("last_error", "")
    if not last_error:
        return "infra_failure", "FAILED with no last_error recorded -- likely a real bug, not a model problem"

    # A test-suite diff appearing in an applied patch, touching a test
    # file, is the plan's own loudly-flagged case: the model made the
    # build/tests pass by changing what they assert instead of the
    # actual code.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT content FROM run_messages WHERE run_id = %s AND role = 'tool' ORDER BY seq", (run_id,)
        )
        tool_msgs = cur.fetchall()
    for m in tool_msgs:
        content = m["content"].get("content")
        if isinstance(content, str) and "applied and committed" in content and _TEST_FILE_RE.search(content):
            return "modified_test_to_pass", "an applied patch's diff touches a test file"

    if "npm ci" in last_error or "lockfile" in last_error.lower() or "ENOTFOUND" in last_error:
        return "infra_failure", last_error[:200]
    # Week 5 Day 7 real finding: bench.py's live run 5837/5838 both
    # exhausted MAX_ATTEMPTS on a genuine Groq daily-token-cap 429 (a
    # known, already-understood infra constraint -- see PROJECT_SUMMARY
    # -- not a repeatable model-behavior defect) and landed here as
    # "uncategorized" with the raw error text, indistinguishable from an
    # actual unrecognized bug. A rate-limit exhaustion is exactly as
    # much "infra, not model" as a lockfile mismatch.
    if "rate_limit_exceeded" in last_error or "rate limit" in last_error.lower() or "429" in last_error:
        return "infra_failure", last_error[:200]
    if "tests failed" in last_error and "build ok" in checkpoint.get("build_result", {}).__repr__():
        return "fixed_build_broke_tests", last_error[:200]
    return "uncategorized", last_error[:200]
