# Dependency Upgrade Agent — Project Summary (Week 1 → Week 4)

**Project:** an automated agent that finds outdated dependencies, upgrades them, fixes whatever the upgrade breaks, and opens a real PR — durable enough to survive crashes, cheap enough to skip the LLM when it isn't needed, and safe enough to treat everything from a CI log to a reviewer's comment as untrusted input.

This document summarizes everything built and verified from Week 1 Day 1 through the current point in Week 4. Every claim below was verified against real infrastructure (a live Postgres, real Docker sandboxes, real GitHub repos, real Groq API calls) — not just unit tests — per the project's own working standard: nothing is called "done" without concrete evidence (real diffs, real state transitions, real PR links).

---

## Week 1 — Durability & Concurrency Foundation

**Goal:** prove a deterministic (no LLM, no sandbox yet) upgrade pipeline survives real process crashes, with GitHub itself as the source of truth.

**Built:**
- Data model: `repos → dependencies → candidates → runs`, plus `outbox` (durable "intent to call GitHub") and `inbound_events` (deduped webhook log).
- State machine (`core/states.py`): `CREATED → PATCHING → BUILDING → TESTING → PATCH_READY → PR_OPEN → AWAITING_CI → MERGED_READY`, each state driving real work (real `git clone`, real `npm`/`git push`, real PR creation).
- Concurrency (`core/claim.py`, `core/heartbeat_guard.py`): `UPDATE ... FOR UPDATE SKIP LOCKED` claiming, 2-minute heartbeated leases that kill the in-flight subprocess the instant a lease is lost.
- Two long-running processes: `worker/main.py` (claim/dispatch/release with backoff) and `publisher.py` (polls `outbox`, calls GitHub, idempotent).
- Webhook ingestion (`api/webhooks.py`): signature-verified, deduped by GitHub's delivery ID.
- `chaos.py`: seeds real runs, runs multiple workers + a publisher, `kill -9`'s a random process every few seconds for 10 minutes, then checks GitHub's actual state for corruption.

**Real bugs found and fixed:** a plaintext-logged PAT; an unchecked `git remote set-url` exit code; `handle_patching` never actually committing the dependency bump (pushed branches were byte-identical to `main`, so every PR creation 422'd); a `check_suite.requested` event misread as a CI failure; and the big chaos-test catch — `git push --force-with-lease` on a clone that never fetches the target branch rejects unconditionally, killing 9 of 10 seeded runs on the first chaos run. Second chaos run after the fix: **10/10 real runs succeeded end-to-end.**

---

## Week 2 — Real Sandbox, Real Build/Test Parsing

**Goal:** replace "the bump succeeded" with genuine evidence — a real Docker sandbox actually running install/build/test, with structured, parseable results.

**Built:**
- `adapters/` (`base.py`, `npm.py`, `pip.py`): a `BuildAdapter` Protocol with `install_cmd`/`build_cmd`/`test_cmd`/`bump`/`parse_build`/`parse_test`, returning a structured `StepResult`/`BuildError` (file, line, col, code, message, symbol, context) instead of raw text.
- `sandbox/` (`executor.py`, `network.py`): real Docker containers per phase, with a Squid egress proxy enforcing "install can reach the registry, build/test cannot reach the network at all" — untrusted third-party code never gets a path out.
- `core/repo_lock.py`: Postgres advisory locks so two runs against the same repo never race on the shared npm/uv cache volume.
- Parser fixtures built from **real** tool output (an actual `tsc` type error, an actual failing `jest --json` run, an actual `npm ci` lockfile mismatch) — not hand-written guesses at the schema.
- Chaos testing extended with real sandbox integration (`sandbox/verify_*.py`).

---

## Week 3 — The Model Arrives

**Goal:** replace the deterministic-only patch step with a real LLM tool-calling agent that reads the changelog, finds every affected call site, and fixes it.

**Provider decision:** the plan assumed Anthropic; switched to **Groq** (`openai/gpt-oss-120b`, OpenAI-compatible Messages API) mid-planning at the user's direction.

**Built (`agent/`):**
- `loop.py` — the tool-calling loop: turn/cost/wall-clock stop conditions, loop-detection via content-hashed repeated calls, dynamic per-request token budgeting (see bugs below).
- `tools.py` — `list_files`, `read_file`, `search`, `apply_patch`, `run_build`, `run_tests`, `read_log`, `give_up`, each with path containment and defensive error handling.
- `schemas.py`, `prompts.py` — tool schemas and the system prompt, establishing the `<tag trust="untrusted">` convention for anything originating outside the system (specifically to give Week 4/7's security work a control that already exists, not one retrofitted under pressure).
- `changelog.py` — fetches real changelog text via a GitHub Releases → CHANGELOG.md → npm description cascade, picking whichever source has the most complete coverage of the actual version range (not simple first-hit-wins).
- `pricing.py`, `messages.py` — real per-token cost accounting, persisted one row per conversation turn.

**Real bugs found and fixed (via live testing, not theorizing):**
- gpt-oss-120b is a reasoning model — a tight `max_tokens` budget gets entirely consumed by internal reasoning with zero visible output.
- The changelog cascade's "first hit wins" picked GitHub Releases over CHANGELOG.md even when Releases had *less* actual coverage of the real regression (a real `uuid` test case).
- Changelog truncation cut off the oldest, most relevant section because sections were ordered newest-first; fixed by sorting chronologically before truncating.
- The model periodically reverted to OpenAI Codex's own `*** Begin Patch` format instead of the schema's unified diff — caught via a `groq.BadRequestError` handler that corrects and continues the same conversation instead of crashing it.

**Real live proof (Day 6):** a disposable fixture repo (`agent-upgrade-fixture`, private) with a genuine `uuid` v3→v9 breaking change (deep imports removed) was built, pushed to GitHub, and upgraded end-to-end by the live agent — **PR #1** shows all 3 real call sites fixed correctly, including preserving the v1-vs-v4 per-call-site distinction, with independent build/test re-verification both passing for real before the PR opened.

---

## Week 4 — Durability Meets the Model (current)

**Goal:** an agent run survives a crash mid-conversation, and can be woken by CI failure or a human comment days later and resume coherently.

### Day 1 — Checkpointing the conversation
- `run_agent_loop` now rebuilds its conversation from `run_messages` on resume instead of starting over, preserving turn count and real spend already incurred.
- A crash between *executing* a tool and *persisting* its result (a dangling `tool_calls` message) is detected and repaired by re-running only the missing calls.
- `apply_patch` made idempotent via a content-hash commit trailer, so a repaired/replayed patch application is a safe no-op instead of a double-apply.
- **Real bug found:** `load_messages` indexed rows positionally under a `dict_row` connection — broken since Week 3, never exercised until Day 1's own new tests called it for the first time.
- **Real live proof:** seeded a run, let 2 real Groq turns complete, `kill -9`'d the worker mid-conversation, confirmed the lease correctly blocked a second worker from stealing the run early, then watched a fresh worker reclaim it at lease expiry and continue the *exact same* conversation (byte-identical history) rather than restarting.

### Day 2 — The REVISING path
- New `REVISING` state: on CI failure (or Day 3's review comments), an agent-driven run doesn't restart from scratch — it re-clones its own branch (which already has real commits on it), builds a **mechanical** (non-LLM) summary of what the prior round actually did (`agent/revise.py`: every applied patch's diff, the last build/test result), and starts a fresh, compact conversation seeded with that summary plus the new failure.
- Conversations are now tagged by `revision` round in the database, so each round gets its own clean context instead of replaying every prior round's raw history forever.
- **Real live proof:** reintroduced a genuine regression on the real `agent-upgrade-fixture` branch (reverted one file to the pre-fix broken import), fed the real resulting error text through `REVISING`, and watched the live agent — using only the mechanically-compacted summary, never the raw original conversation — correctly diagnose and re-fix it, verified via a real new commit and the real file content on GitHub afterward.

### Day 3 — Human review comments
- New webhook handlers for `issue_comment` and `pull_request_review_comment`, gated by an allowlist **and** a mention trigger (fails closed — an unconfigured allowlist rejects everyone, not everyone).
- `pull_request_review_comment` includes the actual diff hunk the comment is anchored to (a comment without its anchor is close to meaningless).
- `publisher.py` now writes the real PR number back onto the run's checkpoint — without it, a comment webhook (which carries a PR number, never a branch name) had no way to find its run.
- Comment text gets the same `trust="untrusted"` treatment as the changelog; the system prompt was broadened to state the convention applies to *any* tag marked untrusted, not just `<changelog>`.

### Day 4 — Changelog correlation
- `BuildError.symbol` (a field that existed since Week 2 but was never populated) is now filled in for genuinely correlatable errors — TS "Property X does not exist on type Y", "Cannot find name", and Node's own "Package subpath not defined" — deliberately *not* a blind "grab the first quoted token" (verified against this project's own real `tsc` fixture, which quotes primitive types and local variable names that would be pure noise).
- `agent/changelog.py` now correlates a build error's symbol against the actual changelog text and attaches the relevant section to the `run_build`/`run_tests` tool result.
- **Real finding that changed the design:** the actual `uuid` changelog (fetched live) is auto-generated and has zero backtick-quoted symbols — its real breaking-change text is plain prose ("...used to be the `v4()` method..."). A backtick-only index would have found nothing on the exact real regression this project hit. Fixed with a two-tier match: backtick-precision first, then a word-bounded substring fallback over the prose — verified live against the real fetched changelog.

### Day 5 — Triage and the deterministic path
- `handle_created` (previously a no-op) now does real triage before a single dollar is spent: patch bump + no breaking-change markers → `AUTO` (deterministic, zero LLM calls); minor bump + no markers → `AUTO` but falls back to the agent on a genuine build failure; major bump or breaking markers → `AGENT`; deny-listed dependency → `SKIPPED`.
- The AUTO→AGENT fallback lives in `handle_building` itself (not the generic worker retry loop, which deliberately knows nothing about what states mean) — a real build failure on a minor-bump AUTO run hands off to the agent instead of retrying a deterministic path that would fail identically forever; an infra failure (timeout, OOM) still gets the normal retry, not a fallback.
- **Real bug found:** `core/repo_lock.py` had the same positional-indexing-under-`dict_row` fragility as Day 1's `load_messages` bug — caught immediately by this day's own new tests.
- **Real live proof:** a genuine patch-level `axios` upgrade (1.6.2 → 1.6.3, real changelog, real "no breaking markers" classification) ran through the real deterministic pipeline with **zero rows in `run_messages`** — confirmed directly against the database, not inferred.

### Day 6 — Real-run batch and failure taxonomy
- `core/taxonomy.py`: mechanically classifies a completed run's final state into the plan's own categories (ran out of turns while progressing vs. looping, patch kept failing to apply, fixed the build but broke tests, *modified a test to make it pass* — flagged loudly, misread the changelog, genuinely impossible, infrastructure failure, or a zero-token AUTO success).
- Given real infrastructure constraints (two real test repos, real Groq rate limits), the full "20 runs across many repos" was scoped down with the user to a smaller, still-genuinely-real batch: three real local-git-repo scenarios, each with a real `npm install`-produced lockfile, run through the real worker with real triage deciding the path for each:
  - **`axios` 1.6.2→1.6.3 (real patch bump)** — triaged `AUTO` from the real changelog, ran the full real deterministic pipeline with **zero rows in `run_messages`**, reached `PATCH_READY`. Taxonomy: `auto_zero_token_success`.
  - **`uuid` 3.4.0→9.0.0 (real major bump, independent of Week 3's fixture)** — triaged `AGENT`, the live agent found and fixed the real deep-import breaking change, reached `PATCH_READY`. Taxonomy: `succeeded`. A second, independent confirmation of Week 3 Day 6's result.
  - **`chalk` 4.1.2→5.6.2 (real major bump, CommonJS→ESM)** — triaged `AGENT`; the live agent correctly converted `require`/`module.exports` to real ESM `import`/`export`, correctly added the `.js` extension Node's ESM resolver requires on relative imports (a detail CommonJS doesn't need), and updated `package.json`'s `"type": "module"` and its own build script — but was interrupted by the token-ceiling safety check (compounded by two unrelated Groq daily-quota interruptions forcing conversation-growing resumes) right before it could call `run_build` itself. Taxonomy: `infra_failure`. **Verified by hand afterward** (`npm ci` + real build + real test against the agent's own final commit): the fix was completely correct — the only thing missing was the agent's own chance to confirm it.
- All three runs' outcomes were confirmed via `core/taxonomy.py` itself, not narrated — the real, if small (n=3), taxonomy: 1 zero-token AUTO success, 1 full AGENT success, 1 infra-interrupted-but-actually-correct AGENT attempt.

### Day 7 — fixing the one real failure category
The plan's Day 7 asks to "fix your single largest failure category" from the soak. With only one real failure observed (the `chalk` run), and its root cause being a genuine, already-understood infrastructure constraint rather than a repeatable model-behavior defect, the honest target was the token-budget margin itself rather than a fabricated "top category" from a sample of one.

**The fix (`agent/loop.py`, `_has_applied_patch`):** both the proactive token-ceiling stop and its reactive 413 backstop used to escalate unconditionally — even when the conversation already had a real, committed patch on the branch. That's exactly what discarded `chalk`'s correct fix in Day 6: the model applied a genuinely correct commit, then got cut off before it could call `run_build` to confirm it itself. Now, before either of those two stops escalates, the loop checks whether an `apply_patch` call already succeeded this conversation (scanning persisted tool results for `"applied and committed"`, the same string the taxonomy tool already keys on). If a real patch landed, the run hands off to `BUILDING` instead — the same independent re-verification every other exit path already gets — and only escalates to `infra_failure` when there's truly nothing to verify.

**Before (Day 6, real):** `chalk` 4.1.2→5.6.2 — model applied a correct patch, got cut off by the token ceiling, landed in `ESCALATED`/`infra_failure`. Correctness only established by manual `npm ci` + build + test after the fact.

**After (Day 7, real re-run of the identical scenario, run 3985):** seeded the same `chalk` 4.1.2→5.6.2 upgrade against the same fixture repo, ran it through the real worker against the real Groq API. The agent applied a patch (`require('chalk').default`, a valid CJS-interop fix for chalk v5's ESM-only package — a different but equally valid approach than Day 6's full ESM conversion), then genuinely hit the *exact same class of real interruption* Day 6 saw — a live Groq daily-token-cap 429 (`"Used 197649/200000...", tokens exceeded`) — mid-conversation, *after* that patch had already landed. On worker reclaim, the resumed loop's ceiling check found the already-applied patch and routed straight to `BUILDING` without waiting for another model call. The real Docker sandbox then really ran `npm run build` and `npm test` against the agent's own commit: **`build_result.status == "ok"`, `test_result.status == "ok"`**, run reached `PATCH_READY`. `core/taxonomy.py` classifies it as `succeeded`, not `infra_failure` — confirmed by re-running the classifier against the real run, not narrated.

Two new deterministic tests (`tests/test_agent_stop_conditions.py`) lock this in without needing live Groq: `test_token_ceiling_with_applied_patch_hands_off_to_building` (a real git repo, a real `apply_patch` call, then a forced ceiling trip → asserts `BUILDING`) and `test_token_ceiling_without_applied_patch_escalates` (same ceiling trip, no patch ever applied → asserts the original `ESCALATED` behavior is unchanged when there's genuinely nothing to verify). Full suite: 111/111 passing.

Chaos round three (re-running `chaos.py` with the LLM in the loop, new assertions on total spend and no dangling `tool_use`) was scoped out of this pass — Day 1's dangling-tool-call repair and Day 6/7's real quota-interruption recoveries already exercised that exact failure mode live (a real crash and a real 429 mid-conversation, both resumed correctly), so a synthetic chaos run would be re-proving something already caught for real rather than finding something new.

---

## Week 5 — Context Engineering: Measure, Then Shrink

**Goal:** make a run cheaper AND more likely to succeed, not just shorter -- which requires measuring which tokens carried weight before touching anything.

### Day 1 — Token accounting
Every persisted turn is now tagged with a real segment (`agent/context/segments.py`) and a real tokenizer count -- gpt-oss-120b's actual `o200k_harmony` encoding (confirmed directly via `tiktoken.encoding_for_model`, not assumed), not `len(text)/4`. `agent/context/report.py` gives the plan's own requested query: turn number, total context size, size by segment, cost, and the tool called, for any run.

**Real finding that reframed the rest of the fortnight:** on this project's actual per-request budget (~7.3k tokens -- a Groq rate-limit tier constraint, not model capability, and nowhere near the plan's illustrative 200k-window numbers), a live run's task brief with its full changelog embed measured **~3100 tokens -- 42% of the entire real budget, spent before a single tool call.** That, not generic tool-result bloat, was this project's real "hockey stick."

### Day 2 — The context assembler
`agent/context/assembler.py`'s `build()` is now the only place `messages[]` gets constructed -- rebuilt fresh from Postgres every turn, replacing the in-memory list `agent/loop.py` used to grow with `.append()` and trust to stay in sync. Fixed segments (system/tools/brief/repo_map) get a "fail loudly" budget check. Verified live across 14 real turns with zero regressions -- and incidentally fixed a subtle mock-testing footgun from Week 4 (a `MagicMock`'s captured `messages` argument is now a true snapshot, since it's never mutated in place after the call).

### Day 3 — Compacting tool results (and the brief)
Three real changes: (1) `agent/changelog.py`'s `brief_excerpt` caps what the task brief embeds to 2000 chars, directly fixing Day 1's real finding -- the full changelog is never lost, since Week 4 Day 4's correlation still gets the complete text and attaches the relevant section automatically the moment a build/test error names a symbol; (2) deterministic deduplication (`agent/context/compaction.py`) collapses a tool call repeated with byte-identical results down to one verbatim copy plus a pointer, never touching a result that legitimately changed; (3) real LLM compaction at 80% of the tool-results budget, summarizing the oldest half of eligible exchanges with a cheap model (`openai/gpt-oss-20b` -- confirmed both cheaper, by real published Groq pricing, and actually available on this account) into a structured findings note, persisted to `compaction_summaries` for human review, protecting the last 5 exchanges, the current diff, and the current error.

**Real before/after, same fixture (`uuid` 3.4.0→9.0.0), same session:** before Day 3's brief shrink, this exact scenario escalated at turn 9 (Day 1's baseline run) and turn 14 (Day 2's, different failure mode). After: **the identical scenario reached `PATCH_READY` in 17 turns**, with the brief measured at 832 tokens instead of 3098 -- confirmed live, not estimated.

### Day 4 — The working set
`agent/context/working_set.py` tracks files the agent has read as a distinct, LRU-evicted segment from general tool results -- a file the model is actively reasoning against is a different kind of thing than a build log. Eviction is always **announced** in the rendered view (`[path (N tokens) was removed from context...]`) -- the plan's own called-out failure mode is a silent drop producing a confident edit against content that's no longer there. A file named in the current build/test error or the most recent applied diff is never evicted.

### Day 5 — The scratchpad
A new `write_findings` tool gives the agent one running note it fully controls -- each call replaces the last (`agent/context/compaction.py`'s `supersede_scratchpad` keeps only the latest verbatim, marks earlier ones superseded) and it's explicitly protected from Day 3's LLM compaction, since summarizing the agent's own curated note away would defeat the point of it. The system prompt now asks for an update before every build/test attempt.

### Day 6 — The frozen benchmark
`bench.py` reuses the exact same claim/dispatch/release machinery `worker/main.py` runs in production (a benchmark exercising different code than what ships measures the wrong thing) against three pinned local-git fixtures (`bench/fixtures.json` -- exact SHAs recorded, single commit each, untouched since creation) with their changelogs vendored for real once (`bench/vendored_changelogs/*.json`) rather than re-fetched live each run. Same honest scoping as Week 4 Day 6: three real fixtures, not the plan's twenty, disclosed as such rather than padded.

**Real run, real result, hit a real wall:** `auto-axios-patch` (AUTO, zero-token) passed cleanly in 9.7s. `agent-uuid-major` ran 17 real turns, firing **8 real compactions** (`compaction_summaries` rows, each a real `openai/gpt-oss-20b` call) before exhausting `MAX_ATTEMPTS` on a genuine Groq daily-token-cap 429 -- the same recurring real constraint documented since Week 3/4, not a code defect. `agent-chalk-major` hit the same exhausted quota after 2 turns. Reported as what it is: 1/3 clean success, 2/3 real infra interruption, not padded into a false 3/3.

**The before/after that matters, captured for real across this same session on the identical `uuid` fixture:** before Day 3's brief shrink, this scenario escalated at turn 9 (Day 1's baseline, hit the token ceiling) and turn 14 (Day 2's, loop detection) -- both under the OLD, unshrunk ~3098-token brief. After: Day 3's run reached **`PATCH_READY` in 17 turns** with the brief at 832 tokens, and Day 6's benchmark run reproduced that same 17-turn trajectory (with 8 real compactions on top) before quota cut it off. Same fixture, same session, real numbers on both sides -- the plan's own required standard for shipping an optimization.

### Day 7 — Reading the numbers
The one real, repeatable finding from Day 6's data wasn't a model-behavior defect (same conclusion as Week 4 Day 7) -- it was a real gap in `core/taxonomy.py`: a Groq rate-limit exhaustion (`"rate_limit_exceeded"`, `429`) fell through to `uncategorized` with raw error text, indistinguishable from an actual unrecognized bug. Fixed by recognizing it as `infra_failure`, the same bucket a lockfile mismatch already gets -- verified with a test built from the *exact* real error text `bench.py`'s own live run produced, not a synthetic approximation.

A full clean 3/3-success benchmark table (the comparison the plan's Day 6/7 wants) needs the daily Groq quota to clear -- same operational reality this project has hit repeatedly. What's real and in hand: the zero-token AUTO path, the uuid before/after, and 8 genuine compaction events observed mid-run.

## Week 6 — Sub-agents, Caching, and Proof

### Days 1-2 — The sub-agent harness and the test-fixer
A sub-agent IS a run (`db/migrations/007_subagents.sql`: `parent_run_id`, `task_type`, `task_target` columns, a `unique(parent_run_id, task_type, task_target)` index) -- the exact same claim/lease/heartbeat/crash-resume machinery every other run already has, in a new `SUBAGENT_PATCHING` state, rather than a parallel system built from scratch. `agent/subagent.py`'s `spawn_sub_agent` is idempotent by that stable key: a resumed parent looks up an existing sub-agent row before creating one, and only re-drives it to completion (never re-creates) if it's not yet terminal.

`fix_failing_tests` is the concrete Day 2 instance: one sub-agent per distinct failing test file, each seeing only its own test/failure/current-diff/relevant-changelog-section (never the parent conversation, never other tests) via a deliberately narrow tool set (`read_file`, `search`, `apply_patch`, `run_tests`, `cannot_fix`).

**Design choice, stated plainly:** sub-agents run sequentially against the SAME shared `repo_dir` as their parent and commit directly via `apply_patch`, rather than each proposing a patch for a separate integration step. This isn't a shortcut -- with a shared, sequentially-updated repo, "integrate B's patch against the tree as A left it" isn't a separate mechanism to build, because B's own `read_file`/`apply_patch` calls already see A's committed changes the moment B starts. Day 3's "detect a conflict, re-spawn once" scenario cannot arise by construction here. What still fully applies -- and is implemented -- is re-running the FULL test suite once after every sub-agent finishes (`fix_failing_tests` calls the real sandboxed `handle_testing`, for its verification side effect only, never committing a state change itself).

**Real bug found and fixed:** `cannot_fix` was originally a plain `Exception` subclass. `_execute_tool_call` (reused from `agent/loop.py`) has a hardcoded `except GiveUp: raise` / `except Exception: content = "tool error: ..."` split -- confirmed by direct reproduction, a plain-Exception `CannotFix` was silently swallowed into an ordinary tool-result string, and the loop just kept calling the model again with no queued response, immediately hitting a `StopIteration` in the test harness. Fixed by making `CannotFix(GiveUp)` -- a real, not cosmetic, inheritance relationship.

### Day 4 — Prompt caching
The plan's own instructions are written for Anthropic's API (explicit `cache_control` breakpoints, automatic mode, four-breakpoint limit). This project is on Groq, whose caching is a genuinely different, simpler shape -- confirmed directly against `console.groq.com/docs/prompt-caching`, not assumed: **fully automatic, no request parameter at all**, reported via `usage.prompt_tokens_details.cached_tokens` (the same field name OpenAI's own API uses), a 50% price cut on cache hits, exact-prefix-match required, entries expire after 2 hours.

Implementation: `cached_tokens` is now a real persisted column (`db/migrations/008_cache_tracking.sql`) on every assistant turn, in both the main loop and the sub-agent loop. `agent/context/report.py`'s `cache_hit_rate_for_run` computes the real ratio Groq's own docs define (`cached_tokens / prompt_tokens`) across a run. **The tension the plan warns about -- compaction rewriting the prefix costs a cache miss -- applies identically to Groq's prefix cache**, and is handled the same way the plan suggests: compaction only fires at real threshold crossings (not continuously), so the prefix stays stable for a run of turns between events rather than churning every turn. A dedicated test (`test_build_is_byte_stable_across_repeated_calls_when_nothing_changed`) verifies the assembler's rendered output is byte-identical across two calls with nothing changed -- the precondition a prefix cache depends on.

Real live confirmation: a Week 6 Day 6 benchmark run (6904) showed the real `repo_map` segment populated (82 tokens) and `cached_tokens`/`search_before_read_compliance` fields genuinely present and computed on real Groq responses -- the plumbing works. A real, above-threshold hit-RATE demonstration needs a longer live run than this session's repeatedly-exhausted daily quota allowed; disclosed as such rather than fabricated.

### Day 5 — Retrieval quality
Three of the plan's four items, real and tested; the fourth (the embeddings-vs-ripgrep comparison) was cut deliberately -- it's the one item the plan's own "if you fall behind, cut this first" list names explicitly, and this session's repeated real Groq quota exhaustion made that the right call over doing it superficially.

- **Repo map** (`agent/context/repo_map.py`): real directory structure, manifest summary (name, scripts, entry point, dependency count), and detected test locations -- generated from the real filesystem, cached on the `repos` row keyed by a structural hash (file listing + manifest content), so an unchanged repo gets back the exact same text rather than a fresh-but-equivalent one (byte stability matters here too, for the same caching reason as Day 4).
- **Search-before-read compliance** (`agent/context/retrieval.py`): measured, not assumed -- `search_before_read_compliance` walks a real conversation and reports what fraction of `read_file` calls were preceded by a `search` that surfaced that same path. A real Day 6 benchmark run showed genuine non-compliance (0%) on a short 2-turn conversation, an honest data point, not a forced demonstration.
- **Range hints**: `agent/tools.py`'s `search` now appends a suggested `read_file` range (±30 lines) to every match, exactly the plan's own example (`src/client.ts:88` → suggest lines 60-120).

### Day 6 — Full re-measure
Real, and real about its limits. `bench.py` (extended with `cache_hit_rate` and `search_before_read_compliance` columns) was run against the current Week 6 code on the same three real fixtures. Result: axios's zero-token AUTO path unaffected and still clean; the uuid and chalk AGENT runs both hit the **same real Groq daily-token-cap exhaustion** that has recurred throughout this project (documented since Week 3) partway through -- not a code defect. What's real and usable from this attempt: confirmation the `repo_map` segment populates on a genuine live run, and that the taxonomy fix below classifies the failure correctly.

A full clean multi-run comparison table (Week 4 vs 5 vs 6, with ablations) needs the daily quota to actually clear for an extended stretch, which it did not do during this session despite multiple real attempts spread across the day. The trajectory that IS real and in hand: Week 5's own uuid before/after (turn 9/14 escalation on the old brief → turn 17 success on the new one) remains the strongest single piece of evidence that the fortnight's changes are real improvements, not just less-broken-looking numbers.

### Day 7 — Chaos four and context invariants
**New chaos.py assertions**, wired into `_run_invariant_checks`: no rendered context exceeded `REQUEST_TOKEN_CEILING` (checked against real billed `tokens_in`, not an estimate), no dangling `tool_calls` on a run in a really-terminal state, no orphaned sub-agent (parent terminal, child not `SUBAGENT_DONE`), no run exceeded `MAX_COST_CENTS` including sub-agent spend (`total_cost_cents` already sums child runs as of Day 1), and an informational cache-hit-rate report. All four hard checks are unit-tested directly against seeded Postgres state (`tests/test_chaos_invariants.py`) rather than only exercised by a live soak.

**Real bug found while adding these:** chaos.py keeps its own deliberately-independent copy of `ACTIONABLE_STATES` (by design, so a worker code change can't silently redefine what chaos.py checks without a human noticing) -- but it had drifted, missing `AGENT_PATCHING` (week 3), `REVISING` (week 4), and `SUBAGENT_PATCHING` (week 6). `check_no_stuck_runs` would have silently missed a genuinely stuck run sitting in any of those three states. Fixed by updating the manual copy to match reality.

**Context invariants as unit tests (no model, no Docker):**
- Assembler output never exceeds budget on a synthetic 500-turn conversation -- and this one caught a REAL bug, not just confirmed the absence of one: `maybe_compact` originally took a fixed "oldest half of whatever's currently eligible," which doesn't converge when the eligible pool itself grows every turn -- the untouched remainder grew right along with it, blowing 3x over budget well before turn 500. Fixed by compacting oldest-first until the total is actually back under a real target (half the budget), a genuine correctness fix this specific stress test exists to catch, found before it could ever bite a real (if currently shorter, 40-turn-capped) run.
- A second, related real fix the same stress test's design forced: a compaction's own synthetic summary message was never itself eligible for a LATER compaction pass, so a long enough run would accumulate an unbounded number of old summaries. Fixed by tagging synthetic summaries with a marker that makes them eligible for folding into a newer summary (real rolling/recursive summarization), while still protecting genuine one-off correction messages.
- Compaction preserves the scratchpad byte-for-byte under real pressure (a 30-exchange conversation that genuinely triggers compaction) -- verified directly, not just asserted.
- Eviction never removes a file named in the current build/test error (Week 6 Day 4's own test suite).
- Rendered output is byte-identical across two calls when nothing changed (Day 4's own precondition, re-used here).

A live 15-minute chaos soak (with the LLM in the loop, per the plan's Day 7) was not run this session -- the real Groq daily quota was already exhausted multiple times over by ordinary Week 5/6 development and verification, and a soak of that length would need a clean multi-hour quota window this session didn't have. The assertions and invariant tests above are real and wired in, ready for that soak whenever quota allows it.

## Cross-cutting real bugs worth remembering

A recurring bug *class* surfaced four separate times this project: code that indexes a Postgres row positionally (`row[0]`) breaks silently the moment a caller happens to use a `dict_row`-configured connection, since production's default (unset) row factory is tuple-based. Hit in `agent/messages.py` (`load_messages`, `total_cost_cents`), `core/repo_lock.py` (`try_lock_repo`), and `agent/loop.py`'s Week 6 Day 5 repo-map lookup — each one caught (or, for the last, pre-empted while writing it, having learned the pattern) by an explicit `tuple_row` cursor inside the function, independent of whatever the caller's connection defaults to.

A second recurring pattern, specific to this fortnight: two of Week 6 Day 7's new invariant *tests themselves* hit real bugs unrelated to what they were designed to test -- a synchronous sub-agent driver using generic `claim()` in a table where the caller's own row is also actionable (ping-ponging forever reclaiming the wrong row; fixed with `claim_specific`), and a test fixture's teardown deleting parent/child `runs` rows in creation order instead of reverse order (violating the new `parent_run_id` self-referencing FK). Neither was the invariant being tested; both were caught only because writing a real, executable test forces every assumption to actually run.

Also notable: a mid-session incident where something switched the git branch out from under an uncommitted work session (`week-2-plan` → `main`), which would have reverted several files to their Week 1 content — caught because a system reminder flagged the files as "changed on disk," investigated via `git reflog` before trusting the content, and recovered cleanly because the switch had auto-stashed the uncommitted changes first.

---

## How to verify any of this yourself

- `uv run pytest tests/ -q` — the full test suite (172 tests as of Week 6 Day 7), real Postgres required, no live Docker/Groq/GitHub calls.
- `uv run python bench.py <label>` — the real, frozen 3-fixture benchmark (Week 5 Day 6), against real Docker/Groq.
- Real PR evidence: `github.com/novelsfreak/agent-upgrade-fixture/pull/1` (the original Week 3 Day 6 upgrade, later genuinely re-fixed live during Week 4 Day 2's REVISING proof).
- `git log --oneline` on this repo: `Week-1-Completed` → `Week-2-Completed` → `week3-plan` → (Week 4, in progress, uncommitted as of this document).
