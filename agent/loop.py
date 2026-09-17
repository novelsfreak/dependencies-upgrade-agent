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
import time

import groq
import psycopg
from groq import Groq

from adapters import ADAPTERS
from agent.messages import persist_message, total_cost_cents
from agent.pricing import DEFAULT_MODEL, compute_cost_cents
from agent.prompts import SYSTEM_PROMPT, build_task_brief
from agent.schemas import TOOL_SCHEMAS
from agent.tools import GiveUp, build_tools
from core.claim import heartbeat
from core.heartbeat_guard import LeaseLostError

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

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


def _call_hash(name: str, args: dict) -> str:
    canonical = json.dumps({"name": name, "args": args}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


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
    from agent.changelog import fetch_changelog

    checkpoint = run["checkpoint"]
    run_id = run["id"]
    model = checkpoint.get("agent_model", DEFAULT_MODEL)
    adapter = ADAPTERS[checkpoint["ecosystem"]]

    dep_name = checkpoint["dep_name"]
    current_version = checkpoint.get("current_version", "unknown")
    target_version = checkpoint["target_version"]

    changelog_text, changelog_source = fetch_changelog(dep_name, current_version, target_version)
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
        changelog_text=changelog_text,
        changelog_source=changelog_source,
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_brief},
    ]
    seq = 0
    persist_message(conn, run_id, seq, "system", messages[0]); seq += 1
    persist_message(conn, run_id, seq, "user", messages[1]); seq += 1

    tools = build_tools(run, conn, worker_id)
    client = _get_client()

    call_hash_counts: dict[str, int] = {}
    start_time = time.monotonic()
    turn = 0
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
    REQUEST_TOKEN_CEILING = 7600
    MIN_TURN_TOKENS = 300
    chars_per_token = 3.2  # conservative starting guess; recalibrated below

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

        current_chars = sum(len(json.dumps(m, default=str)) for m in messages)
        estimated_prompt_tokens = int(current_chars / chars_per_token)

        if estimated_prompt_tokens > REQUEST_TOKEN_CEILING - MIN_TURN_TOKENS:
            # Even the smallest useful completion budget would push this
            # request over the tier's cap. Not something a retry fixes --
            # the conversation itself, not a transient rate window, is
            # the problem -- so stop cleanly here instead of letting the
            # SDK's own request come back as a 413 lower down.
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
            # SAME conversation continues rather than crashing (which
            # would otherwise cost every turn already spent, since Week 3
            # has no mid-run resume -- that's Week 4).
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
            messages.append(correction)
            persist_message(conn, run_id, seq, "user", correction)
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
                # request from scratch.
                return _escalate(conn, run, f"request too large for rate-limit tier (413): {e}", turn, start_time)
            raise

        choice = resp.choices[0]
        usage = resp.usage
        if usage.prompt_tokens:
            # Recalibrate against ground truth -- this conversation's
            # actual chars-per-token, not the generic starting guess,
            # governs every estimate from here on.
            chars_per_token = current_chars / usage.prompt_tokens
        cost = compute_cost_cents(model, usage.prompt_tokens, usage.completion_tokens)

        assistant_message = choice.message.model_dump(exclude_none=True)
        messages.append(assistant_message)
        persist_message(
            conn, run_id, seq, "assistant", assistant_message,
            tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens, cost_cents=cost,
        )
        seq += 1

        if choice.finish_reason != "tool_calls":
            # The model believes it's done. Not trusted on its own --
            # handing off to the unmodified BUILDING/TESTING states for
            # independent re-verification, same defense-in-depth as
            # every mid-loop run_build/run_tests call already gets.
            return "BUILDING", dict(run["checkpoint"])

        for call in choice.message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                tr = {"role": "tool", "tool_call_id": call.id,
                      "content": "arguments were not valid JSON -- retry with valid JSON"}
                messages.append(tr)
                persist_message(conn, run_id, seq, "tool", tr)
                seq += 1
                continue

            call_hash = _call_hash(call.function.name, args)
            call_hash_counts[call_hash] = call_hash_counts.get(call_hash, 0) + 1
            count = call_hash_counts[call_hash]

            if count > LOOP_REPEAT_THRESHOLD:
                return _escalate(conn, run, f"repeated identical tool call: {call.function.name}", turn, start_time)

            if count == LOOP_REPEAT_THRESHOLD:
                content = (
                    f"You have called {call.function.name} with identical arguments {count} times "
                    f"and received the same result each time. Try something different, or call "
                    f"give_up if you're stuck."
                )
            else:
                try:
                    fn = tools.get(call.function.name)
                    content = f"unknown tool: {call.function.name}" if fn is None else str(fn(**args))
                except GiveUp as e:
                    return _escalate(conn, run, f"agent gave up: {e.reason}", turn, start_time)
                except Exception as e:
                    content = f"tool error: {e}"

            tr = {"role": "tool", "tool_call_id": call.id, "content": content}
            messages.append(tr)
            persist_message(conn, run_id, seq, "tool", tr)
            seq += 1
