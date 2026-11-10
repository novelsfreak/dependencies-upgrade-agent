# agent/subagent.py
#
# Week 6 Days 1-3: sub-agents. A sub-agent IS a run (db/migrations/007)
# -- same claim/lease/heartbeat/crash-resume machinery every other run
# already gets, not a parallel system built from scratch. This module
# is what runs INSIDE a "SUBAGENT_PATCHING" state (core/states.py's
# handle_subagent_patching) and what a PARENT calls to spawn/drive one.
#
# Design choice worth stating explicitly: the plan's Day 3 imagines
# sub-agents each proposing a patch to be integrated (and possibly
# conflicting) afterward. Here, sub-agents run SEQUENTIALLY against the
# SAME shared repo_dir as their parent and commit directly via
# apply_patch -- exactly like the main agent already does. This isn't
# a shortcut: with a shared, sequentially-updated repo_dir, "integrate
# sub-agent B's patch against the tree as A left it" is not a separate
# step to build, because B's own read_file/apply_patch calls already
# see A's committed changes the moment B starts (there is no un-applied
# patch object waiting to be merged). The conflict shape the plan
# describes -- two INDEPENDENTLY-generated patches touching the same
# file -- cannot arise by construction here. What still fully applies,
# and is still implemented: re-running the FULL test suite once after
# every sub-agent finishes (fix_failing_tests below) -- Day 3's own
# point that "four tests passing individually does not mean they pass
# together" has nothing to do with how patches got merged.
from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from agent.context.assembler import build as assemble_context
from agent.loop import (
    _execute_tool_call,
    _get_client,
    _has_applied_patch,
    _LoopDetected,
    _pending_tool_calls,
    _segment_for_tool,
)
from agent.messages import load_messages, persist_message
from agent.pricing import DEFAULT_MODEL, compute_cost_cents
from agent.prompts import SUBAGENT_SYSTEM_PROMPT, build_subagent_brief
from agent.schemas import SUBAGENT_TOOL_SCHEMAS
from agent.tools import GiveUp, apply_patch, read_file, search
from core.claim import claim_specific, heartbeat, release
from core.heartbeat_guard import LeaseLostError

log = logging.getLogger("agent.subagent")

SUBAGENT_MAX_TURNS_DEFAULT = 10
SUBAGENT_MAX_TOKENS_PER_TURN = 1500  # a 10-turn, 4-tool task never needs the main loop's headroom
DEFAULT_PARALLEL_CAP = 4


class CannotFix(GiveUp):
    """
    The sub-agent equivalent of GiveUp -- a distinct name because it's
    a distinct, narrower situation: "I could not fix this ONE test,"
    not "this whole upgrade is stuck." Subclasses GiveUp, not Exception,
    for a real mechanical reason: _execute_tool_call (reused from
    agent/loop.py) has a hardcoded `except GiveUp: raise` / `except
    Exception: content = "tool error: ..."` split -- any exception that
    isn't GiveUp (or a subclass) gets silently swallowed into a normal
    tool-result string instead of propagating, which would make
    cannot_fix indistinguishable from any other tool error and never
    actually stop the loop. Confirmed by direct reproduction: with this
    as a plain Exception, calling cannot_fix produced a "tool error:
    ..." message and the loop just kept going.
    """


@dataclasses.dataclass
class SubAgentResult:
    status: str  # "patch" | "cannot_fix" | "escalated"
    reason: str | None
    turns: int
    cost_cents: float
    run_id: int


def _build_subagent_tools(run: dict, conn: psycopg.Connection, worker_id: str) -> dict:
    checkpoint = run.get("checkpoint") or {}
    repo_dir = Path(checkpoint["repo_dir"])
    test_filter = checkpoint.get("test_filter")

    def _run_tests() -> str:
        """
        Reuses the exact same sandboxed step runner handle_testing does
        (_run_subprocess_step -- disposable container, no network,
        heartbeated) rather than a host subprocess call, so a sub-agent
        never regresses the sandboxing this project's whole security
        model depends on. Disclosed simplification: this runs the FULL
        test command, not a per-file-filtered one -- there's no
        cross-ecosystem-generic CLI convention for "run just this one
        test" the way there is for install/build/test commands
        themselves, so the sub-agent is told which file is "its" test
        and reads that file's result out of the full output, rather
        than this project pretending a filter exists that it doesn't.
        """
        from adapters.base import add_source_context
        from core.states import BuildFailed, _classify_infra_failure, _run_subprocess_step
        from adapters import ADAPTERS

        adapter = ADAPTERS[checkpoint["ecosystem"]]
        exit_code, stdout, log_path, duration_ms = _run_subprocess_step(
            run, conn, worker_id, adapter.test_cmd(), "subagent_test.log", phase="subagent-test",
        )
        test_result = adapter.parse_test(exit_code, stdout, "")
        test_result.log_ref = log_path
        test_result.duration_ms = duration_ms
        add_source_context(test_result.errors, repo_dir)
        _classify_infra_failure(test_result, exit_code)

        if test_result.status != "ok":
            relevant = [e for e in test_result.errors if test_filter and test_filter in (e.file or "")]
            shown = relevant or test_result.errors
            return f"tests failed: {shown[:6]} (showing errors relevant to {test_filter!r} if any matched)"
        return "tests ok."

    def _cannot_fix(reason: str) -> str:
        raise CannotFix(reason)

    return {
        "read_file": lambda path, start=1, end=200: read_file(repo_dir, path, start, end),
        "search": lambda pattern, glob="**/*": search(repo_dir, pattern, glob),
        "apply_patch": lambda diff: apply_patch(repo_dir, diff),
        "run_tests": lambda: _run_tests(),
        "cannot_fix": lambda reason: _cannot_fix(reason),
    }


def _finish(run_id: int, status: str, reason: str | None, turn: int, cost_cents: float) -> tuple[str, dict]:
    return "SUBAGENT_DONE", {
        "subagent_status": status,
        "subagent_reason": reason,
        "subagent_turns": turn,
        "subagent_cost_cents": cost_cents,
    }


def run_sub_agent_loop(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    Deliberately leaner than agent.loop.run_agent_loop: a 10-turn,
    4-tool task never grows large enough to need the main loop's
    proactive token-ceiling estimation or LLM-based compaction (dedup
    and working-set eviction still apply for free -- assemble_context
    always runs them, client=None only skips the compaction pass).
    Crash-resume (load_messages / _pending_tool_calls) is the SAME
    mechanism the main loop uses, because it's the same underlying
    problem: a worker can die between executing a tool and persisting
    its result here exactly as easily as anywhere else.
    """
    checkpoint = run["checkpoint"]
    run_id = run["id"]
    model = checkpoint.get("agent_model", DEFAULT_MODEL)
    max_turns = checkpoint.get("budget_turns", SUBAGENT_MAX_TURNS_DEFAULT)

    tools = _build_subagent_tools(run, conn, worker_id)
    client = _get_client()
    call_hash_counts: dict[str, int] = {}
    start_time = time.monotonic()

    def _persist(seq: int, role: str, content: dict, segment: str = "tool_results", **kw) -> None:
        persist_message(conn, run_id, seq, role, content, segment=segment, revision=0, **kw)

    existing = load_messages(conn, run_id, revision=0)
    if existing:
        seq = len(existing)
        turn = sum(1 for m in existing if m["role"] == "assistant")
    else:
        brief = build_subagent_brief(
            test_file=checkpoint.get("test_file", checkpoint.get("test_filter", "")),
            failure_output=checkpoint.get("failure_output", ""),
            current_diff=checkpoint.get("current_diff", ""),
            changelog_section=checkpoint.get("changelog_section", ""),
        )
        seq = 0
        _persist(seq, "system", {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT}, segment="system"); seq += 1
        _persist(seq, "user", {"role": "user", "content": brief}, segment="brief"); seq += 1
        turn = 0

    messages = assemble_context(conn, run_id, 0)
    pending = _pending_tool_calls(messages)
    if pending:
        for call in pending:
            try:
                tr = _execute_tool_call(
                    call["id"], call["function"]["name"], call["function"]["arguments"], tools, call_hash_counts,
                )
            except CannotFix as e:
                return _finish(run_id, "cannot_fix", e.reason, turn, 0.0)
            except _LoopDetected as e:
                return _finish(run_id, "cannot_fix", f"repeated identical tool call: {e.tool_name}", turn, 0.0)
            _persist(seq, "tool", tr, segment=_segment_for_tool(call["function"]["name"]))
            seq += 1

    total_cost = 0.0
    while True:
        turn += 1
        if turn > max_turns:
            return _finish(run_id, "cannot_fix", "max_turns exceeded", turn, total_cost)

        if not heartbeat(conn, run_id, worker_id):
            raise LeaseLostError(f"lease lost during sub-agent loop turn {turn}")

        messages = assemble_context(conn, run_id, 0, client=client)

        resp = client.chat.completions.create(
            model=model, messages=messages, tools=SUBAGENT_TOOL_SCHEMAS, max_tokens=SUBAGENT_MAX_TOKENS_PER_TURN,
        )
        choice = resp.choices[0]
        usage = resp.usage
        cost = compute_cost_cents(model, usage.prompt_tokens, usage.completion_tokens)
        total_cost += cost
        cached_tokens = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)

        assistant_message = choice.message.model_dump(exclude_none=True)
        _persist(
            seq, "assistant", assistant_message, segment="assistant",
            tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens, cost_cents=cost,
            cached_tokens=cached_tokens,
        )
        seq += 1

        if choice.finish_reason != "tool_calls":
            # The model believes it's done -- not trusted on its own.
            # It only counts as a real fix if it actually applied a
            # patch AND its own run_tests call (not just its claim)
            # came back passing.
            all_messages = [m["content"] for m in load_messages(conn, run_id, revision=0)]
            tests_passed = any(
                m.get("role") == "tool" and isinstance(m.get("content"), str) and "tests ok" in m["content"]
                for m in all_messages
            )
            if _has_applied_patch(all_messages) and tests_passed:
                return _finish(run_id, "patch", None, turn, total_cost)
            return _finish(
                run_id, "cannot_fix",
                "sub-agent stopped without both an applied patch and a confirmed passing test run",
                turn, total_cost,
            )

        for call in choice.message.tool_calls:
            try:
                tr = _execute_tool_call(
                    call.id, call.function.name, call.function.arguments, tools, call_hash_counts,
                )
            except CannotFix as e:
                return _finish(run_id, "cannot_fix", e.reason, turn, total_cost)
            except _LoopDetected as e:
                return _finish(run_id, "cannot_fix", f"repeated identical tool call: {e.tool_name}", turn, total_cost)
            _persist(seq, "tool", tr, segment=_segment_for_tool(call.function.name))
            seq += 1


def spawn_sub_agent(
    parent_run: dict,
    conn: psycopg.Connection,
    worker_id: str,
    task_type: str,
    target: str,
    inputs: dict,
    budget_turns: int = SUBAGENT_MAX_TURNS_DEFAULT,
    timeout_seconds: int = 600,
) -> SubAgentResult:
    """
    Idempotent: looks up an existing (parent_run_id, task_type, target)
    row before creating one. If it already reached a terminal state
    (SUBAGENT_DONE), returns the persisted result without re-running --
    "the parent re-spawns only the ones with no result" (the plan's own
    words). If it exists but isn't terminal (a crash left it mid-flight,
    or this is a resumed parent that already created it), drives it to
    completion rather than creating a duplicate row and violating the
    unique (parent_run_id, task_type, task_target) index.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM runs WHERE parent_run_id = %s AND task_type = %s AND task_target = %s",
            (parent_run["id"], task_type, target),
        )
        existing = cur.fetchone()

    if existing is None:
        checkpoint = dict(inputs)
        checkpoint["budget_turns"] = budget_turns
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "INSERT INTO runs (candidate_id, state, checkpoint, parent_run_id, task_type, task_target, "
                "next_attempt_at) VALUES (%s, 'SUBAGENT_PATCHING', %s::jsonb, %s, %s, %s, now()) RETURNING *",
                (parent_run["candidate_id"], _json(checkpoint), parent_run["id"], task_type, target),
            )
            existing = cur.fetchone()
        conn.commit()

    run_id = existing["id"]
    final_state = _advance_until_terminal(conn, run_id, f"{worker_id}-sub-{target}", timeout_seconds)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT checkpoint FROM runs WHERE id = %s", (run_id,))
        checkpoint = cur.fetchone()["checkpoint"]

    if final_state != "SUBAGENT_DONE":
        return SubAgentResult(status="escalated", reason=f"did not finish (state={final_state})",
                               turns=checkpoint.get("subagent_turns", 0),
                               cost_cents=checkpoint.get("subagent_cost_cents", 0.0), run_id=run_id)

    return SubAgentResult(
        status=checkpoint.get("subagent_status", "cannot_fix"),
        reason=checkpoint.get("subagent_reason"),
        turns=checkpoint.get("subagent_turns", 0),
        cost_cents=checkpoint.get("subagent_cost_cents", 0.0),
        run_id=run_id,
    )


def _json(obj: dict) -> str:
    import json
    return json.dumps(obj)


_SUBAGENT_TERMINAL_STATES = {"SUBAGENT_DONE"}


def _advance_until_terminal(conn: psycopg.Connection, run_id: int, worker_id: str, timeout_seconds: int) -> str:
    """
    Drives exactly this one sub-agent run to completion using the SAME
    dispatch/release primitives the real worker uses, but claim_specific
    (not claim()) -- needed because in a single-worker deployment
    nothing else is polling concurrently to advance a just-created
    child run while the PARENT's own handler call (which is what's
    calling this) is itself sitting in an actionable state (e.g.
    AGENT_PATCHING) and occupying the worker loop. Generic claim()
    would happily pick up the parent's own row instead of the target
    child's -- see claim_specific's docstring for the real bug this
    replaced.
    """
    from core.repo_lock import LockContention
    from core.states import HANDLERS
    from worker.main import handle_failure

    start = time.monotonic()
    while True:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT state FROM runs WHERE id = %s", (run_id,))
            state = cur.fetchone()["state"]
        conn.commit()
        if state in _SUBAGENT_TERMINAL_STATES:
            return state
        if time.monotonic() - start > timeout_seconds:
            return "TIMEOUT"

        run = claim_specific(conn, run_id, worker_id)
        if run is None:
            time.sleep(0.2)
            continue
        try:
            handler = HANDLERS[run["state"]]
            next_state, checkpoint_delta = handler(run, conn, worker_id)
            if not checkpoint_delta.get("_released"):
                release(conn, run["id"], next_state, checkpoint_delta)
        except LeaseLostError:
            pass
        except LockContention:
            release(conn, run["id"], run["state"], next_attempt_at_sql="now() + interval '5 seconds'")
        except Exception as e:
            handle_failure(conn, run, e)


def fix_failing_tests(
    parent_run: dict,
    conn: psycopg.Connection,
    worker_id: str,
    failing_tests: list[dict],
    changelog_text: str = "",
    current_diff: str = "",
    parallel_cap: int = DEFAULT_PARALLEL_CAP,
) -> dict:
    """
    Week 6 Day 2's concrete instance: one sub-agent per distinct failing
    test file, each seeing only its own test/failure/diff/changelog
    section -- never the parent conversation, never the other tests.
    `failing_tests` is `[{"test_file": ..., "failure_output": ...}, ...]`.

    Sequential, not truly parallel (see this module's docstring): real
    infra here is a single worker process and a per-organization Groq
    rate limit that would serialize concurrent calls anyway, so
    `parallel_cap` bounds how many distinct sub-agent tasks this call
    will drive per invocation, not concurrency -- kept as a parameter
    (rather than silently processing all of them) so a run with an
    unusually large number of failing tests doesn't spend unbounded
    turns/budget in one call without the caller deciding that's fine.

    Always re-runs the FULL test suite once after every sub-agent
    finishes (Day 3's own point: passing individually isn't passing
    together) via the exact same sandboxed handle_testing step the
    normal TESTING state uses -- called here purely for its real
    verification side effect (it does NOT commit any state change;
    only release() does that, and this function never calls it), so
    the parent's own state transition still goes through the normal
    worker dispatch path.
    """
    from agent.changelog import find_relevant_changelog_section
    from core.states import BuildFailed, handle_testing

    results: dict[str, SubAgentResult] = {}
    for failing in failing_tests[:parallel_cap]:
        test_file = failing["test_file"]
        section = find_relevant_changelog_section(changelog_text, test_file) if changelog_text else None
        result = spawn_sub_agent(
            parent_run, conn, worker_id,
            task_type="fix_test", target=test_file,
            inputs={
                "repo_dir": parent_run["checkpoint"]["repo_dir"],
                "ecosystem": parent_run["checkpoint"].get("ecosystem", "npm"),
                "test_file": test_file,
                "test_filter": test_file,
                "failure_output": failing.get("failure_output", ""),
                "current_diff": current_diff,
                "changelog_section": section or "",
            },
        )
        results[test_file] = result

    any_patched = any(r.status == "patch" for r in results.values())
    full_suite_status = "not_attempted"
    full_suite_detail = ""
    if any_patched:
        try:
            handle_testing(parent_run, conn, worker_id)
            full_suite_status = "ok"
        except BuildFailed as e:
            full_suite_status = "failed"
            full_suite_detail = str(e)

    return {
        "results": results,
        "full_suite_status": full_suite_status,
        "full_suite_detail": full_suite_detail,
    }
