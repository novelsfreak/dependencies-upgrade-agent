# agent/context/compaction.py
#
# Week 5 Day 3: two independent, additive passes over the RENDERED view
# of a conversation. Neither ever touches run_messages -- Postgres
# stays the complete, untouched record (the plan's own mental model:
# "the full log is retained; the compacted view keeps the latest
# meaningful value per key and drops superseded entries").
from __future__ import annotations

import hashlib

import psycopg

from agent.context.tokenizer import count_message_tokens
from agent.pricing import CHEAP_MODEL, compute_cost_cents

PROTECTED_RECENT_EXCHANGES = 5

# Week 6 Day 7 real finding: a 500-synthetic-turn stress test caught
# unbounded growth over a long-running conversation -- each compaction
# event replaces the oldest half with ONE synthetic summary message,
# but that message (a lone non-assistant row) was never itself
# eligible for a LATER compaction pass, so a long enough run just
# accumulates more and more of these forever. Tagging them with this
# marker (also used in the synthetic message text below) lets a later
# pass fold several old summaries into one, the same rolling/recursive
# summarization real long-context systems use.
_SYNTHETIC_SUMMARY_MARKER = "was compacted to save context space"


def _tool_call_index(rows: list[dict]) -> dict[str, tuple[str, str]]:
    """tool_call_id -> (name, canonical_args) for every assistant tool_calls entry."""
    index: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row["role"] != "assistant":
            continue
        for tc in (row["content"].get("tool_calls") or []):
            index[tc["id"]] = (tc["function"]["name"], tc["function"]["arguments"])
    return index


def deduplicate(rows: list[dict]) -> list[dict]:
    """
    Cheap win, deterministic, no LLM (the plan's own "do it first"):
    if the SAME tool call (name + args) was already answered earlier in
    this conversation with BYTE-IDENTICAL content, the earlier
    occurrence carries nothing the later one doesn't -- replace its
    content with a short pointer, keep the latest verbatim. A file that
    legitimately changed between two reads of the same path produces a
    different content hash and is correctly left untouched -- this
    only collapses genuine no-op repeats, never real information.
    """
    call_index = _tool_call_index(rows)

    def _key(row: dict) -> tuple[str, str, str] | None:
        if row["role"] != "tool":
            return None
        name_args = call_index.get(row["content"].get("tool_call_id"))
        content = row["content"].get("content")
        if not name_args or not isinstance(content, str):
            return None
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return (name_args[0], name_args[1], content_hash)

    last_seq_for_key: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = _key(row)
        if key is not None:
            last_seq_for_key[key] = row["seq"]  # last occurrence wins

    out = []
    for row in rows:
        key = _key(row)
        if key is not None and last_seq_for_key[key] != row["seq"]:
            name, args = call_index[row["content"]["tool_call_id"]]
            new_row = dict(row)
            new_row["content"] = dict(row["content"])
            new_row["content"]["content"] = (
                f"[duplicate of an identical {name}({args}) call, unchanged -- "
                f"see the later occurrence in this conversation for the real content]"
            )
            out.append(new_row)
        else:
            out.append(row)
    return out


def supersede_scratchpad(rows: list[dict]) -> list[dict]:
    """
    Week 5 Day 5: write_findings is the agent's ONE running note, not a
    log -- each call replaces the last. Only the LATEST call's result
    needs to render verbatim; every earlier one is fully superseded (not
    merely a duplicate -- its content is expected to differ, since the
    note evolves) and is replaced with a short pointer, the same
    "announce, don't silently vanish" style as deduplicate() and
    working_set eviction use.
    """
    call_index = _tool_call_index(rows)
    write_findings_seqs = [
        row["seq"] for row in rows
        if row["role"] == "tool" and (call_index.get(row["content"].get("tool_call_id")) or (None,))[0] == "write_findings"
    ]
    if len(write_findings_seqs) <= 1:
        return rows

    latest_seq = max(write_findings_seqs)
    out = []
    for row in rows:
        if row["role"] == "tool" and row["seq"] in write_findings_seqs and row["seq"] != latest_seq:
            new_row = dict(row)
            new_row["content"] = dict(row["content"])
            new_row["content"]["content"] = "[earlier findings note, superseded -- see the later one]"
            out.append(new_row)
        else:
            out.append(row)
    return out


def _group_exchanges(rows: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """
    Splits rows into the leading fixed prefix (system + brief, seq 0-1)
    and a list of "exchanges" -- one assistant row plus the tool rows
    that answer its tool_calls, or a lone non-assistant row (a
    BadRequestError correction) as its own singleton exchange so it's
    never silently dropped.
    """
    prefix = [r for r in rows if r["role"] in ("system", "user") and r["seq"] < 2]
    rest = rows[len(prefix):]

    exchanges: list[list[dict]] = []
    current: list[dict] | None = None
    for row in rest:
        if row["role"] == "assistant":
            if current is not None:
                exchanges.append(current)
            current = [row]
        elif row["role"] == "tool" and current is not None:
            current.append(row)
        else:
            if current is not None:
                exchanges.append(current)
                current = None
            exchanges.append([row])
    if current is not None:
        exchanges.append(current)
    return prefix, exchanges


def _exchange_tokens(exchange: list[dict]) -> int:
    return sum(count_message_tokens(r["content"]) for r in exchange)


def _protected_indices(exchanges: list[list[dict]]) -> set[int]:
    """
    Never compact: the last N tool-bearing exchanges, the exchange
    holding the most recent applied patch (the current diff), the
    exchange holding the most recent build/test failure (the current
    error) -- the plan's own explicit exclusion list -- and (Week 5 Day
    5) any exchange containing a write_findings call. The scratchpad
    already has its own, cheaper pruning (supersede_scratchpad: keep
    only the latest note, no LLM call needed) and is meant to survive
    compaction verbatim; folding it into an LLM-summarized blob would
    undermine the whole point of a note the AGENT curates.
    """
    protected: set[int] = set()

    tool_bearing = [i for i, ex in enumerate(exchanges) if any(r["role"] == "tool" for r in ex)]
    protected.update(tool_bearing[-PROTECTED_RECENT_EXCHANGES:])

    for i, ex in enumerate(exchanges):
        for row in ex:
            if row["role"] == "assistant" and any(
                tc["function"]["name"] == "write_findings" for tc in (row["content"].get("tool_calls") or [])
            ):
                protected.add(i)

    # Only the MOST RECENT diff/error is "the current" one -- an older
    # applied patch or a since-fixed failure is exactly the kind of
    # superseded history compaction should be allowed to summarize away.
    last_diff_idx = None
    for i, ex in enumerate(exchanges):
        for row in ex:
            content = row["content"].get("content") if row["role"] == "tool" else None
            if isinstance(content, str) and "applied and committed" in content:
                last_diff_idx = i
    if last_diff_idx is not None:
        protected.add(last_diff_idx)

    last_error_idx = None
    for i, ex in enumerate(exchanges):
        for row in ex:
            content = row["content"].get("content") if row["role"] == "tool" else None
            if isinstance(content, str) and ("build failed" in content or "tests failed" in content):
                last_error_idx = i
    if last_error_idx is not None:
        protected.add(last_error_idx)

    return protected


_SUMMARY_SYSTEM_PROMPT = """You are compacting part of an in-progress dependency-upgrade agent's own tool-call history to save context space. The agent will keep working from your summary WITHOUT the raw history below -- be concrete and specific (file names, function/symbol names, error codes, exact things tried and ruled out). Use exactly this shape, and omit a line entirely rather than writing "N/A" or "none":

## Findings so far
Changed: <files/symbols actually modified so far>
Tried and failed: <specific approaches already ruled out, so they are not retried>
Learned: <facts discovered about the codebase or the upgrade>
Build: <most recent build/test status mentioned in the transcript below>
Open: <what's still unresolved>"""


def _render_exchange_as_text(exchange: list[dict]) -> str:
    lines = []
    for row in exchange:
        if row["role"] == "assistant":
            content = row["content"]
            for tc in (content.get("tool_calls") or []):
                lines.append(f"[agent called {tc['function']['name']}({tc['function']['arguments']})]")
            if content.get("content"):
                lines.append(f"[agent said: {content['content']}]")
        elif row["role"] == "tool":
            lines.append(f"[result: {row['content'].get('content', '')}]")
        else:
            lines.append(f"[{row['role']}: {row['content'].get('content', '')}]")
    return "\n".join(lines)


def summarize_exchanges(client, model: str, exchanges: list[list[dict]]) -> tuple[str, int, int, float]:
    transcript = "\n\n".join(_render_exchange_as_text(ex) for ex in exchanges)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        max_tokens=500,
    )
    summary = resp.choices[0].message.content or "## Findings so far\n(summary model returned no content)"
    usage = resp.usage
    cost = compute_cost_cents(model, usage.prompt_tokens, usage.completion_tokens)
    return summary, usage.prompt_tokens, usage.completion_tokens, cost


def _is_compactable_leading_row(row: dict) -> bool:
    """
    An exchange is eligible for compaction if it's a normal
    assistant-led exchange, OR it's an earlier compaction's own
    synthetic summary message (a lone "user"-role row) -- letting old
    summaries be folded into a newer one instead of accumulating
    forever. A genuine one-off correction (e.g. a BadRequestError fix)
    is a lone "user" row too but lacks the marker, so it stays
    protected by omission, same as before.
    """
    if row["role"] == "assistant":
        return True
    content = row["content"].get("content")
    return isinstance(content, str) and _SYNTHETIC_SUMMARY_MARKER in content


def maybe_compact(
    conn: psycopg.Connection,
    run_id: int,
    revision: int,
    rows: list[dict],
    client,
    budget: int,
    trigger_ratio: float,
    model: str = CHEAP_MODEL,
) -> list[dict]:
    """
    If the compactable portion of `rows` (everything past the fixed
    system+brief prefix) is over `trigger_ratio` of `budget`, summarize
    the OLDEST eligible exchanges (excluding the protected ones -- see
    _protected_indices), one at a time, until what's left is back under
    half of `budget` -- not a fixed "half of however many happen to be
    eligible right now," which doesn't converge as the eligible pool
    itself grows turn over turn (see the Day 7 comment below). Persists
    the summary and returns a shorter row list with those exchanges
    replaced by one synthetic message. Returns `rows` unchanged if
    nothing qualifies (too small, everything protected, or no client
    supplied -- e.g. tests that don't want a live model call).
    """
    if client is None:
        return rows

    prefix, exchanges = _group_exchanges(rows)
    total_tokens = sum(_exchange_tokens(ex) for ex in exchanges)
    if total_tokens <= trigger_ratio * budget:
        return rows

    protected = _protected_indices(exchanges)
    eligible = [i for i, ex in enumerate(exchanges) if i not in protected and _is_compactable_leading_row(ex[0])]
    if not eligible:
        return rows

    # Week 6 Day 7 real finding: a 500-synthetic-turn stress test caught
    # this not converging -- "take the oldest HALF of whatever's
    # currently eligible" only shrinks in proportion to the eligible
    # pool's CURRENT size, but that pool grows every turn too, so the
    # untouched remainder grew right along with it and blew through the
    # budget by 3x well before turn 500. Compacting oldest-first until
    # the total is actually back under a real target (half the budget,
    # not just "under the trigger ratio" -- leaving room to grow again
    # before the next trigger) converges regardless of how large the
    # eligible pool has become.
    target_tokens = budget * 0.5
    to_compact: list[list[dict]] = []
    half_set: set[int] = set()
    remaining_total = total_tokens
    for i in eligible:
        if remaining_total <= target_tokens:
            break
        to_compact.append(exchanges[i])
        half_set.add(i)
        remaining_total -= _exchange_tokens(exchanges[i])
    if not to_compact:
        to_compact = [exchanges[eligible[0]]]
        half_set = {eligible[0]}
    tokens_before = sum(_exchange_tokens(ex) for ex in to_compact)

    summary_text, _tin, _tout, cost = summarize_exchanges(client, model, to_compact)
    tokens_after = count_message_tokens({"role": "user", "content": summary_text})

    seqs = [r["seq"] for ex in to_compact for r in ex]
    conn.execute(
        "INSERT INTO compaction_summaries "
        "(run_id, revision, covers_seq_start, covers_seq_end, summary_text, tokens_before, tokens_after, cost_cents) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (run_id, revision, min(seqs), max(seqs), summary_text, tokens_before, tokens_after, cost),
    )
    conn.commit()

    synthetic = {
        "role": "user",
        "content": (
            f"[Earlier tool activity from this conversation (seq {min(seqs)}-{max(seqs)}) "
            f"{_SYNTHETIC_SUMMARY_MARKER}. This is a summary, not the raw transcript:]\n\n"
            f"{summary_text}"
        ),
    }

    new_exchanges: list[list[dict]] = []
    inserted = False
    for i, ex in enumerate(exchanges):
        if i in half_set:
            if not inserted:
                new_exchanges.append([{"role": "user", "content": synthetic, "seq": min(seqs)}])
                inserted = True
            continue
        new_exchanges.append(ex)

    flattened = list(prefix)
    for ex in new_exchanges:
        flattened.extend(ex)
    return flattened
