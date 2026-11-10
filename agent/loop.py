# agent/loop.py
#
# The ~40-line loop from the plan, adapted to Groq's OpenAI-compatible
# shape (finish_reason == "tool_calls", choice.message.tool_calls, tool
# results as role="tool" messages) rather than Anthropic's content
# blocks -- see the model_dump()s below for exactly what that means in
# practice.
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import groq
import psycopg
from groq import Groq

from adapters import ADAPTERS
from agent.context.assembler import build as assemble_context
from agent.context.report import segment_totals_for_run
from agent.context.tokenizer import count_message_tokens
from agent.messages import load_messages, persist_message, total_cost_cents
from agent.pricing import DEFAULT_MODEL, compute_cost_cents
from agent.prompts import SYSTEM_PROMPT, build_task_brief
from agent.schemas import TOOL_SCHEMAS
from agent.tools import GiveUp, build_tools
from core.claim import heartbeat
from core.heartbeat_guard import LeaseLostError

log = logging.getLogger("agent.loop")

MAX_TURNS = 40
MAX_COST_CENTS = 200
MAX_WALL_CLOCK_SECONDS = 20 * 60
# Generous, not the plan's number as-is: gpt-oss-120b is a reasoning
# model whose internal reasoning tokens are billed as part of the
# completion and count against this budget -- verified empirically, a
# 3-word answer alone consumed 142 tokens on reasoning. Too tight a
# budget here means the model runs out of room before ever emitting a
# tool call or its visible answer.
MAX_TOKENS_PER_TURN = 4000
LOOP_REPEAT_THRESHOLD = 3
# Week 6 Day 7: promoted from a local inside run_agent_loop to a real
# module constant -- chaos.py's check_no_context_window_exceeded needs
# to reference the same real ceiling this loop enforces, and a value
# only a function-local scope knows about can't be checked independently
# from the outside.
REQUEST_TOKEN_CEILING = 7600
MIN_TURN_TOKENS = 300

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


def _call_hash(name: str, args: dict) -> str:
    canonical = json.dumps({"name": name, "args": args}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class _LoopDetected(Exception):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name


def _execute_tool_call(
    call_id: str, name: str, arguments_raw: str, tools: dict, call_hash_counts: dict,
) -> dict:
    """
    Runs one tool call and returns the tool-role message to append.
    Shared by the live turn loop and the resume-repair path below --
    both eventually reduce to the same three plain strings (a tool_call
    id, a name, a raw arguments string), whether they came fresh off
    the SDK's response object (live path: call.id / call.function.name)
    or reconstructed from a persisted JSON dict (repair path: call["id"]
    / call["function"]["name"]). Keeping one implementation means loop
    detection and give_up behave identically regardless of which path
    produced the call.

    Raises GiveUp (propagated, not caught) or _LoopDetected -- both are
    "the loop ends now" signals, not per-call errors, so the caller
    escalates rather than turning them into a tool result.
    """
    try:
        args = json.loads(arguments_raw or "{}")
    except json.JSONDecodeError:
        return {"role": "tool", "tool_call_id": call_id,
                "content": "arguments were not valid JSON -- retry with valid JSON"}

    call_hash = _call_hash(name, args)
    call_hash_counts[call_hash] = call_hash_counts.get(call_hash, 0) + 1
    count = call_hash_counts[call_hash]

    if count > LOOP_REPEAT_THRESHOLD:
        raise _LoopDetected(name)

    if count == LOOP_REPEAT_THRESHOLD:
        content = (
            f"You have called {name} with identical arguments {count} times "
            f"and received the same result each time. Try something different, or call "
            f"give_up if you're stuck."
        )
    else:
        fn = tools.get(name)
        if fn is None:
            content = f"unknown tool: {name}"
        else:
            try:
                content = str(fn(**args))
            except GiveUp:
                raise
            except Exception as e:
                content = f"tool error: {e}"

    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _pending_tool_calls(messages: list[dict]) -> list[dict] | None:
    """
    Reconstructed from persisted messages, the conversation's last entry
    might be an assistant turn whose tool_calls don't all have matching
    tool-role responses yet -- exactly what a crash between "execute the
    tool" and "persist the tool result" leaves behind (see agent/tools.py
    apply_patch's idempotency note; this is the other half of that same
    Week 4 Day 1 fix). Returns the specific tool_calls still missing a
    response, or None if the conversation is fully caught up.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    tool_calls = messages[-1].get("tool_calls")
    if not tool_calls:
        return None
    answered_ids = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
    pending = [tc for tc in tool_calls if tc["id"] not in answered_ids]
    return pending or None


def _has_applied_patch(messages: list[dict]) -> bool:
    """
    Week 4 Day 7. Real finding from Day 6's live batch: the `chalk` run
    had already applied a fully correct ESM-migration patch (verified
    by hand afterward -- npm ci + real build + real test all passed)
    but the proactive token-ceiling check below escalated it straight
    to ESCALATED/infra_failure anyway, discarding a genuinely correct
    fix the model never got a chance to verify itself. A forced stop
    that already has a real commit on the branch has something worth
    independently verifying -- exactly what BUILDING already does for
    every other exit path (see the "model believes it's done" case
    further down) -- so it should never be treated identically to a
    forced stop with nothing to show for it.
    """
    return any(
        m.get("role") == "tool" and isinstance(m.get("content"), str)
        and "applied and committed" in m["content"]
        for m in messages
    )


def _segment_for_tool(tool_name: str) -> str:
    """
    Week 5 Day 4/5: which context segment a tool RESULT belongs to, by
    the tool that produced it -- read_file results are the "working
    set" (agent/context/working_set.py targets these specifically by
    path), write_findings results are the "scratchpad" (Week 5 Day 5,
    survives compaction verbatim), everything else (search, run_build,
    run_tests, list_files, read_log) is the general tool_results bucket
    Day 3's compaction manages.
    """
    if tool_name == "read_file":
        return "working_set"
    if tool_name == "write_findings":
        return "scratchpad"
    return "tool_results"


def _classify_semver_jump(current: str, target: str) -> str:
    from agent.changelog import _parse_version
    if current in ("unknown", "", None):
        return "unknown"
    c, t = _parse_version(current), _parse_version(target)
    if c[0] != t[0]:
        return "major"
    if len(c) > 1 and len(t) > 1 and c[1] != t[1]:
        return "minor"
    return "patch"


def _escalate(conn: psycopg.Connection, run: dict, reason: str, turns: int, start_time: float) -> tuple[str, dict]:
    return "ESCALATED", {
        "escalated_reason": reason,
        "escalated_turns": turns,
        "escalated_cost_cents": total_cost_cents(conn, run["id"]),
        "escalated_wall_clock_seconds": time.monotonic() - start_time,
    }


def run_agent_loop(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    checkpoint = run["checkpoint"]
    run_id = run["id"]
    model = checkpoint.get("agent_model", DEFAULT_MODEL)
    # Week 4 Day 2: which conversation round this is. 0 for the original
    # AGENT_PATCHING pass; handle_revising bumps this in checkpoint each
    # time CI fails or a reviewer comments on an open PR, and seeds a
    # FRESH compacted conversation at that new revision number rather
    # than appending to the old one (see db/migrations/004 and
    # agent/messages.py) -- so resume-from-crash below must only ever
    # look at THIS round's messages.
    revision = checkpoint.get("revision_count", 0)

    # Week 4 Day 4: fetched unconditionally, even on resume -- this is
    # what lets run_build/run_tests correlate a build error's symbol
    # against the changelog (agent/tools.py's _correlate_changelog)
    # regardless of whether this is turn 1 or a resumed turn 20. This
    # is a separate concern from Day 1's resume optimization below,
    # which skips rebuilding the PERSISTED system+brief messages (the
    # expensive, conversation-altering part) -- re-fetching changelog
    # text is just a cheap, side-effect-free network call, not a reason
    # to skip it.
    from agent.changelog import brief_excerpt, fetch_changelog

    dep_name = checkpoint["dep_name"]
    current_version = checkpoint.get("current_version", "unknown")
    target_version = checkpoint["target_version"]
    # Week 5 Day 6: a run can carry a pre-fetched changelog in its
    # checkpoint instead of hitting the network -- what bench.py's
    # frozen fixtures use, so a benchmark run's timing/token numbers
    # reflect this project's own code, not a live changelog host's
    # latency or content drifting between benchmark runs. Absent, this
    # is unchanged from Week 3/4: a real live fetch, every time.
    if "vendored_changelog_text" in checkpoint:
        changelog_text = checkpoint["vendored_changelog_text"]
        changelog_source = checkpoint.get("vendored_changelog_source", "vendored")
    else:
        changelog_text, changelog_source = fetch_changelog(dep_name, current_version, target_version)

    tools = build_tools(run, conn, worker_id, changelog_text=changelog_text)
    client = _get_client()
    call_hash_counts: dict[str, int] = {}
    start_time = time.monotonic()

    # Bound to this call's run_id/revision so every persist below can't
    # forget the revision tag (a forgotten one would silently write into
    # revision 0's history, corrupting the wrong round's conversation).
    def _persist(seq: int, role: str, content: dict, segment: str = "tool_results", **kw) -> None:
        persist_message(conn, run_id, seq, role, content, segment=segment, revision=revision, **kw)

    # Week 4 Day 1: resume from wherever a prior attempt left off,
    # rather than always starting a fresh conversation. A run only
    # reaches AGENT_PATCHING once per PATCHING pass, but AGENT_PATCHING
    # itself can be re-entered many times (worker crash, lease
    # expiry+reclaim, LockContention requeue) -- previously every one
    # of those re-entries silently threw away however many real turns
    # (and real Groq spend) the conversation already had, and started
    # over from turn 0. load_messages returning anything means a prior
    # attempt got at least as far as persisting the system+task-brief
    # turns.
    existing = load_messages(conn, run_id, revision=revision)
    if existing:
        seq = len(existing)
        turn = sum(1 for m in existing if m["role"] == "assistant")
    else:
        adapter = ADAPTERS[checkpoint["ecosystem"]]
        semver_jump = _classify_semver_jump(current_version, target_version)

        task_brief = build_task_brief(
            repo_name=checkpoint.get("repo_url", "local"),
            dep_name=dep_name,
            current_version=current_version,
            target_version=target_version,
            semver_jump=semver_jump,
            manifest_path=checkpoint.get("manifest_path", ""),
            build_cmd=adapter.build_cmd(),
            test_cmd=adapter.test_cmd(),
            changelog_text=brief_excerpt(changelog_text),
            changelog_source=changelog_source,
        )

        seq = 0
        _persist(seq, "system", {"role": "system", "content": SYSTEM_PROMPT}, segment="system"); seq += 1
        _persist(seq, "user", {"role": "user", "content": task_brief}, segment="brief"); seq += 1

        # Week 6 Day 5: a cached repo map -- generated once per repo,
        # reused as long as the repo's structure hasn't changed (see
        # agent/context/repo_map.py). Best-effort: a repo_url that
        # isn't a real `repos` row yet (the stub/local-path paths some
        # test fixtures use) just means no repo_map this run, not a
        # crash -- it was never a hard requirement for the loop to work.
        repo_url = checkpoint.get("repo_url")
        if repo_url:
            # Explicit tuple_row, not whatever this connection's own
            # row_factory happens to be -- same recurring footgun as
            # agent/messages.py's load_messages/total_cost_cents.
            from psycopg.rows import tuple_row
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute("SELECT id FROM repos WHERE url = %s", (repo_url,))
                repo_row = cur.fetchone()
            if repo_row:
                from agent.context.repo_map import get_or_build_repo_map
                repo_dir_for_map = Path(checkpoint.get("repo_dir") or checkpoint.get("work_dir", "."))
                if repo_dir_for_map.is_dir():
                    map_text = get_or_build_repo_map(conn, repo_row[0], repo_dir_for_map)
                    _persist(seq, "user", {"role": "user", "content": f"Repo map:\n{map_text}"}, segment="repo_map")
                    seq += 1
        turn = 0

    # Week 5 Day 2: Postgres is the only truth from here on -- every
    # read of "the conversation so far" goes through the assembler
    # rather than an in-memory list this function grows and hopes stays
    # in sync with what got persisted.
    messages = assemble_context(conn, run_id, revision)

    # A dangling assistant tool_calls turn with no matching tool
    # results is exactly what a crash between executing a tool and
    # persisting its result leaves behind. Re-run only the missing
    # ones -- apply_patch's own content-hash check (agent/tools.py)
    # makes a replayed apply_patch a safe no-op rather than a
    # double-apply; run_build/run_tests were already safe to repeat.
    pending = _pending_tool_calls(messages)
    if pending:
        for call in pending:
            try:
                tr = _execute_tool_call(
                    call["id"], call["function"]["name"], call["function"]["arguments"],
                    tools, call_hash_counts,
                )
            except GiveUp as e:
                return _escalate(conn, run, f"agent gave up: {e.reason}", turn, start_time)
            except _LoopDetected as e:
                return _escalate(conn, run, f"repeated identical tool call: {e.tool_name}", turn, start_time)
            _persist(seq, "tool", tr, segment=_segment_for_tool(call["function"]["name"]))
            seq += 1
    # Groq's 8000 TPM cap is charged against prompt_tokens + max_tokens
    # for a SINGLE request, not just steady-state throughput -- a fixed
    # max_tokens=4000 sent on every call works fine while the
    # conversation is short (the stub-path test never grew past a few
    # turns) but guarantees a hard 413 once prompt_tokens alone passes
    # ~4000, which a real repo's tool outputs (file listings, changelog
    # text, build logs) reaches within a dozen or so turns.
    #
    # A first version of this budgeted off the PREVIOUS turn's
    # usage.prompt_tokens, which is stale by construction: it doesn't
    # account for whatever got appended to `messages` since then (the
    # assistant's own message, and -- usually much bigger -- the tool
    # result that followed it, e.g. a file read or build log). That
    # gap is exactly what still 413'd in real testing, since a single
    # large tool result closed most of the safety margin between one
    # turn and the next. Estimating fresh off the CURRENT messages list
    # every turn closes that gap; the chars-per-token ratio is a rough
    # start that gets calibrated against Groq's own reported
    # prompt_tokens as real data comes in, rather than trusting a fixed
    # guess for the whole run.
    # 7200 (800-token margin) proved safe across two live runs -- both
    # escalated cleanly on the proactive check, neither hit a real 413.
    # Tightened to 7600 (400-token margin) to buy back a bit more of the
    # turn budget: a real run got to 2 of 3 call sites fixed and ran out
    # of room reading the third file's content, one turn short of
    # finishing. The APIStatusError 413 catch below is still there as a
    # backstop if this margin turns out too tight on some other run.
    #
    # Week 5 Day 1: the estimate itself is no longer a fixed
    # chars-per-token guess -- that's precisely the "estimate drifts
    # badly on JSON and code" problem the plan calls out by name.
    # gpt-oss-120b's actual tokenizer is registered in tiktoken
    # (agent/context/tokenizer.py), so `messages` is counted for real.
    # The remaining `overhead_factor` accounts for what raw
    # text-tokenization can't see -- Groq's own chat-template wrapping
    # (role tags, special tokens) -- and is recalibrated every turn
    # against Groq's real usage.prompt_tokens, same self-correcting
    # spirit as the chars-per-token estimate it replaces, just anchored
    # to a real tokenizer instead of a constant.
    overhead_factor = 1.15  # conservative starting guess; recalibrated below

    while True:
        turn += 1
        if turn > MAX_TURNS:
            return _escalate(conn, run, "max_turns exceeded", turn, start_time)

        if time.monotonic() - start_time > MAX_WALL_CLOCK_SECONDS:
            return _escalate(conn, run, "max_wall_clock exceeded", turn, start_time)

        # Checked before the call, not after -- the plan's own
        # instruction: don't spend past budget, stop before the call
        # that would exceed it.
        if total_cost_cents(conn, run_id) > MAX_COST_CENTS:
            return _escalate(conn, run, "max_cost_cents exceeded", turn, start_time)

        if not heartbeat(conn, run_id, worker_id):
            raise LeaseLostError(f"lease lost during agent loop turn {turn}")

        # Week 5 Day 2: rebuilt fresh every iteration from Postgres --
        # everything persisted by the end of the previous iteration
        # (the assistant turn and every tool result it triggered, or a
        # repaired dangling call above) is what this turn's request
        # actually sends. No in-memory list to fall out of sync with.
        # Week 5 Day 3: `client` is passed here (and only here, not the
        # pre-loop build used for dangling-call repair) so compaction --
        # a real, billed cheap-model call -- happens at most once per
        # real turn, and never before a crash-repair has even run.
        messages = assemble_context(conn, run_id, revision, client=client)
        raw_token_count = sum(count_message_tokens(m) for m in messages)
        estimated_prompt_tokens = int(raw_token_count * overhead_factor)

        # Week 5 Day 1 task 2: log tokens per segment every turn -- the
        # instrument the rest of week 5's optimizations are measured
        # against. A DB read, not a recompute from `messages`: the two
        # are kept in sync by construction (every append is persisted
        # before the loop comes back around), and reading the
        # already-tagged rows back is the same query a human would run
        # to investigate this run later (agent/context/report.py).
        segment_totals = segment_totals_for_run(conn, run_id, revision)
        log.info(
            "run %s turn %s: estimated_prompt_tokens=%s (raw=%s x overhead=%.3f) segments=%s",
            run["id"], turn, estimated_prompt_tokens, raw_token_count, overhead_factor, segment_totals,
        )

        if estimated_prompt_tokens > REQUEST_TOKEN_CEILING - MIN_TURN_TOKENS:
            # Even the smallest useful completion budget would push this
            # request over the tier's cap. Not something a retry fixes --
            # the conversation itself, not a transient rate window, is
            # the problem -- so stop cleanly here instead of letting the
            # SDK's own request come back as a 413 lower down.
            if _has_applied_patch(messages):
                # Week 4 Day 7: don't discard real, already-committed
                # work just because the conversation ran out of budget
                # before the model could confirm it itself -- hand off
                # to BUILDING for the same independent re-verification
                # every other exit path already gets.
                return "BUILDING", dict(run["checkpoint"])
            return _escalate(
                conn, run,
                f"conversation grew past this rate-limit tier's per-request cap "
                f"(~{estimated_prompt_tokens} estimated prompt tokens, {REQUEST_TOKEN_CEILING} ceiling)",
                turn, start_time,
            )

        max_tokens_this_turn = max(
            MIN_TURN_TOKENS, min(MAX_TOKENS_PER_TURN, REQUEST_TOKEN_CEILING - estimated_prompt_tokens)
        )

        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SCHEMAS, max_tokens=max_tokens_this_turn,
            )
        except groq.BadRequestError as e:
            # Observed for real against gpt-oss-120b: an OpenAI-family
            # model occasionally reverts to OpenAI Codex's own "*** Begin
            # Patch" apply_patch convention from training instead of
            # this schema's unified-diff `diff` field. Groq validates
            # tool-call arguments server-side and rejects the WHOLE turn
            # with a 400 before any assistant message comes back -- there
            # is no tool call to attach a corrective tool_result to, so
            # the correction goes back as a plain user turn, and the
            # SAME conversation continues in-process rather than
            # crashing and relying on the resume path above to pick it
            # back up next attempt -- cheaper and simpler to just
            # correct and keep going right here.
            error_body = getattr(e, "body", None) or {}
            message = (error_body.get("error") or {}).get("message", str(e))
            correction = {
                "role": "user",
                "content": (
                    f"Your last tool call was rejected: {message}\n\n"
                    f"Use EXACTLY the fields defined in the tool's schema. For apply_patch "
                    f"that is a single `diff` field containing a standard unified diff (the "
                    f"output of `git diff`) -- not the OpenAI Codex \"*** Begin Patch\" format "
                    f"or any other patch convention."
                ),
            }
            _persist(seq, "user", correction, segment="brief")
            seq += 1
            continue
        except groq.APIStatusError as e:
            if getattr(e, "status_code", None) == 413:
                # Belt-and-suspenders: the estimate above should keep
                # every request under the cap, but it's still an
                # estimate (chars-per-token varies with content, and
                # turn 1 has no calibration data yet). If Groq itself
                # says the request was too large anyway, that's the
                # same "this conversation outgrew the tier" situation
                # the proactive check handles -- stop cleanly rather
                # than let the worker retry the identical oversized
                # request from scratch. Same Day 7 reasoning as the
                # proactive check above: if a real patch already
                # landed, verify it instead of discarding it.
                if _has_applied_patch(messages):
                    return "BUILDING", dict(run["checkpoint"])
                return _escalate(conn, run, f"request too large for rate-limit tier (413): {e}", turn, start_time)
            raise

        choice = resp.choices[0]
        usage = resp.usage
        if usage.prompt_tokens:
            # Recalibrate against ground truth -- this conversation's
            # actual overhead factor, not the generic starting guess,
            # governs every estimate from here on.
            overhead_factor = usage.prompt_tokens / max(raw_token_count, 1)
        cost = compute_cost_cents(model, usage.prompt_tokens, usage.completion_tokens)
        # Week 6 Day 4: Groq's prompt caching is fully automatic (no
        # cache_control param the way Anthropic's API needs -- confirmed
        # directly against Groq's own docs, not assumed) and reports
        # its effect on every response via this field. Guarded with
        # getattr, not a dict-style .get: prompt_tokens_details is a
        # real attribute on the SDK's usage object when present, but
        # older/mocked usage objects (this project's own tests) may not
        # have it at all.
        cached_tokens = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)

        assistant_message = choice.message.model_dump(exclude_none=True)
        _persist(
            seq, "assistant", assistant_message, segment="assistant",
            tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens, cost_cents=cost,
            cached_tokens=cached_tokens,
        )
        seq += 1
        if cached_tokens:
            log.info(
                "run %s turn %s: %s/%s prompt tokens served from cache (%.0f%%)",
                run["id"], turn, cached_tokens, usage.prompt_tokens, 100 * cached_tokens / max(usage.prompt_tokens, 1),
            )

        if choice.finish_reason != "tool_calls":
            # The model believes it's done. Not trusted on its own --
            # handing off to the unmodified BUILDING/TESTING states for
            # independent re-verification, same defense-in-depth as
            # every mid-loop run_build/run_tests call already gets.
            return "BUILDING", dict(run["checkpoint"])

        for call in choice.message.tool_calls:
            try:
                tr = _execute_tool_call(
                    call.id, call.function.name, call.function.arguments, tools, call_hash_counts,
                )
            except GiveUp as e:
                return _escalate(conn, run, f"agent gave up: {e.reason}", turn, start_time)
            except _LoopDetected as e:
                return _escalate(conn, run, f"repeated identical tool call: {e.tool_name}", turn, start_time)
            _persist(seq, "tool", tr, segment=_segment_for_tool(call.function.name))
            seq += 1
