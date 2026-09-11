# Week 1 Summary — Dependency Upgrade Agent

**Period:** Sep 8 – Sep 10, 2026
**Goal for the week:** prove the durability/concurrency story for a deterministic (no-LLM, no-sandbox) dependency-upgrade pipeline — claim → patch → build → test → push → PR → await CI — survives real process crashes, with GitHub itself as the source of truth, not just the system's own bookkeeping.

## What exists

**Data model** (`db/migrations/`): `repos → dependencies → candidates → runs`, plus `outbox` (durable "intent to call GitHub" rows) and `inbound_events` (durable, deduped webhook log).

**State machine** (`core/states.py`): `CREATED → PATCHING → BUILDING → TESTING → PATCH_READY → PR_OPEN → AWAITING_CI → MERGED_READY`. Each state (except `AWAITING_CI`, which only moves on a webhook) has a handler doing real work — real `git clone`, real `npm install`/`ci`/`build`/`test`, real `git push`, real PR creation.

**Concurrency** (`core/claim.py`, `core/heartbeat_guard.py`): `UPDATE ... FOR UPDATE SKIP LOCKED` lets N workers fan out without colliding; a 2-minute lease, heartbeated every 30s by a background thread that kills the in-flight subprocess the instant the lease is lost, so a crashed worker can never leave an orphaned process or a double-owned run.

**Two long-running processes**: `worker/main.py` (claim/dispatch/release loop with exponential backoff, capped retries, `FAILED` terminal state) and `publisher.py` (polls `outbox`, calls GitHub, idempotent via `find_open_pr`).

**Webhook ingestion** (`api/`): signature-verified `POST /webhooks/github`, deduped by GitHub's delivery ID, maps `check_suite.completed` events to a run via its branch name.

**Chaos test** (`chaos.py`, the Day 7 milestone): seeds 10 real runs against a real repo, runs 3 workers + 1 publisher, `kill -9`'s a random process every 3–7s for 10 minutes, then checks against GitHub's actual API: no stuck runs, no dropped webhooks, no duplicate PRs, no duplicate outbox publishes.

## Bugs found and fixed this week

| # | Bug | Where | Fix |
|---|---|---|---|
| 1 | Stale cached GitHub credential in macOS Keychain, separate from a working `gh` token | local git config | `gh auth setup-git` |
| 2 | Fine-grained PAT missing **Contents: Read and write** — could read the repo, couldn't push | GitHub token config | Regenerated token with correct permission |
| 3 | `log.info("token" + token)` logged the PAT in plaintext, and ran *before* the missing-token check, so a missing token crashed with a raw `TypeError` instead of a clean error | `core/states.py` (`handle_patch_ready`) | Removed the log line; reordered the check first |
| 4 | `git remote set-url`'s exit code was never checked — a silent failure there would push unauthenticated and look identical to a bad-token 403 | `core/states.py` | Now raises loudly on non-zero exit |
| 5 | `publisher.py` computed its `.env` path as `parent.parent`, copy-pasted from `worker/main.py` (which sits one directory deeper) — `GITHUB_TOKEN` never loaded, silent `KeyError` | `publisher.py` | Fixed to `parent` |
| 6 | **The big one:** `handle_patching` ran `npm install` but never `git add`/`git commit` — pushed branches were byte-identical to `main`, so GitHub rejected PR creation with 422 "No commits between main and branch" | `core/states.py` (`handle_patching`) | Added a commit step after the dependency bump |
| 7 | `handle_check_suite_completed` didn't check `payload["action"]` — a `check_suite.requested` event (fired before any result exists, `conclusion: null`) would be misread as a CI failure and bounce the run back to `PATCHING` before CI even ran | `api/webhooks.py` | Added an early return unless `action == "completed"` |
| 8 | **Found by chaos testing:** `git push --force-with-lease` was used on a clone that never fetches the target branch (only ever `main`), so the lease check has nothing to compare against — it rejects unconditionally whenever the branch already has any commit, identically on every retry. Caused **9 of 10** seeded runs to fail in the first chaos run | `core/states.py` (`handle_patch_ready`) | Switched to plain `--force` — the branch is exclusively owned by this automation, so lease-safety wasn't buying anything, only breaking the intended idempotent-push behavior |

## Chaos test results

- **First run:** formally PASSED all 4 invariants, but 9/10 seeded runs ended up `FAILED` — bug #8 above, not caught by the invariant checks because "a run exhausted retries and failed" is a valid outcome by their design, not a structural-corruption signal.
- **Second run (after the fix):** **10/10 runs succeeded** end-to-end (real patch → build → test → push → PR → `AWAITING_CI`), zero warnings, zero errors, all 4 invariants still pass, cleanup left GitHub clean.

## Status: Week 1 goal met

The durability/concurrency proof chaos testing exists to deliver is now solid: leases, heartbeats, the outbox pattern, webhook dedup, and idempotent branch pushes all survive repeated `kill -9` against real GitHub state, with a 100% functional success rate on the second run.

## Known gaps (explicitly out of scope for Week 1, not regressions)

- **`publisher.py` has no backoff or attempt cap.** A permanently-failing outbox row retries every 2 seconds forever — seen directly this week, one row spun to ~72,000 attempts before its root cause was fixed. Worth a cap as a fast follow.
- **`MERGED_READY` has no handler.** A run lands there after CI succeeds and stops — nothing auto-merges. There's also no `pull_request` webhook handler, so a manual merge on GitHub doesn't feed back into the state machine either.
- No `PLANNING` state, no `REVISING`/CI-failure-fix loop, no sandboxing, no LLM-driven patch step — all explicitly deferred per the state machine's own header comment, and squarely Week 2+ scope per the README's broader vision (breaking-change upgrades, reacting to CI failures and review comments).
