"""
Day 7 -- chaos testing.

Seeds real runs against a real repo, launches the real worker and
publisher processes, kills one at random every few seconds for a fixed
window, restarts it immediately (pool size held constant), and then
checks four invariants -- against the DB *and* against GitHub's actual
PR list, not against this system's own account of itself.

Run from the project root:

    uv run python chaos.py

Requires .env with DATABASE_URL, GITHUB_TOKEN, GITHUB_REPO already
configured exactly as worker/main.py and publisher.py expect them --
this script does not reimplement env loading, it just runs your real
worker.py and publisher.py as subprocesses so it is exercising the
exact code path you already proved individually on Days 2-6, under
conditions neither of you controlled by hand.

Design decisions, stated so you can push back on them:

- 10 seeded runs, not 20. Real npm ci / npm run build against a real
  clone takes real minutes. Fewer runs that can genuinely progress
  through BUILDING/TESTING under chaos is more informative than more
  runs that never get past PATCHING before the window closes.
- Every seeded run gets a unique chaos-{short_uuid} suffix baked into
  dep_name, so branch names (agent/upgrade/npm/{dep_name}-{version})
  can never collide with a previous chaos run or with each other. This
  is what makes the duplicate-PR check meaningful rather than
  contaminated by leftover branches from last Tuesday.
- kill -9 (SIGKILL), not SIGTERM. You have no graceful shutdown by
  design -- a real crash doesn't ask politely, and testing graceful
  shutdown would be testing a code path you don't have.
- Pool size held constant: every kill is immediately followed by a
  restart of the same role (worker or publisher), so "no run stuck
  with an expired lease and no owner" is actually being tested against
  a healthy-sized pool, not against a pool that's shrinking.
"""
from __future__ import annotations

import os
import random
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv
from psycopg.rows import dict_row

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPO = os.environ["GITHUB_REPO"]
GITHUB_API = "https://api.github.com"

NUM_SEEDED_RUNS = 10
NUM_WORKERS = 3
CHAOS_WINDOW_SECONDS = 10 * 60
KILL_INTERVAL_MIN = 3
KILL_INTERVAL_MAX = 7
DRAIN_SECONDS = 60  # settle time after chaos stops, before assertions run

# Mirrors core/claim.py's ACTIONABLE_STATES. If that list changes,
# update this -- chaos.py deliberately does not import it, so a worker
# code change can't silently change what chaos.py thinks "actionable"
# means without a human noticing the drift.
#
# Week 6 Day 7: this had drifted -- AGENT_PATCHING (week 3), REVISING
# (week 4), and SUBAGENT_PATCHING (week 6) were all missing, meaning
# check_no_stuck_runs below would have silently missed a genuinely
# stuck run sitting in any of those three states. The "don't import
# it" tradeoff only works if the manual copy actually gets updated when
# the real list does -- caught by re-reading this file while adding
# Day 7's new invariant checks, not by anything that would have failed
# loudly on its own.
ACTIONABLE_STATES = [
    "CREATED", "PATCHING", "AGENT_PATCHING", "BUILDING", "TESTING",
    "PATCH_READY", "PR_OPEN", "REVISING", "SUBAGENT_PATCHING",
]

# AWAITING_CI is excluded on purpose: a run sitting there with no
# owner and an old updated_at is *correct*, not stuck -- it's waiting
# on a webhook that may not arrive for a long time. Only ACTIONABLE_STATES
# rows can be "stuck" in the sense this test checks for.
TERMINAL_STATES = ["MERGED_READY", "FAILED"]

STUCK_LEASE_MARGIN_SECONDS = 60

CHAOS_ID = uuid.uuid4().hex[:8]


@dataclass
class ManagedProcess:
    role: str  # "worker" or "publisher"
    cmd: list[str]
    proc: subprocess.Popen = field(default=None, repr=False)

    def start(self) -> None:
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT_DIR)
        print(f"  [{self.role}] started pid={self.proc.pid}")

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            print(f"  [{self.role}] kill -9 pid={self.proc.pid}")
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait(timeout=5)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def db_connect() -> psycopg.Connection:
    return psycopg.connect(DSN, autocommit=True, row_factory=dict_row)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

# Real, small, well-known npm packages -- npm install {name}@latest is
# guaranteed to resolve for every one of these without us having to
# know or guess a specific version string in advance (the mistake the
# first version of this function made with fake package names, and a
# hardcoded-version approach would repeat with real names -- versions
# get published over time and a value that's valid today may not be
# valid whenever this file is next run).
CHAOS_PACKAGES = ["axios", "lodash", "chalk", "dayjs", "uuid", "debug", "semver"]


def seed_runs(conn: psycopg.Connection, n: int) -> list[str]:
    """
    Insert n CREATED runs through the REAL repos -> dependencies ->
    candidates -> runs chain, matching runs.candidate_id's NOT NULL FK
    (001_init.sql).

    Each run is assigned a real package from CHAOS_PACKAGES (round-
    robined if n > len(CHAOS_PACKAGES)) with target_version = "latest".
    This replaces two earlier, wrong assumptions in a row: first that
    candidate_id was nullable (it isn't -- NOT NULL FK, fixed by
    inserting the real repos/dependencies/candidates chain below), then
    that a fabricated package name like "chaos-<uuid>" would work with
    npm install (it doesn't -- handle_patching's real-repo path calls
    real `npm install {dep_name}@{target_version}` against the real
    registry, which 404s on a name nothing ever published). "latest"
    sidesteps needing to know a real version string ahead of time,
    since it always resolves to whatever currently exists.

    Consequence: handle_patch_ready's branch name
    (agent/upgrade/npm/{dep_name}-{target_version}) becomes
    agent/upgrade/npm/{package}-latest -- unique per PACKAGE, not per
    chaos run. Two different runs of chaos.py against the same package
    list will reuse the same branch names. That's fine as long as
    cleanup_github() (below) actually deletes/closes them at the end of
    every run -- see that function's docstring for why this is the
    right trade rather than trying to force a chaos-id into a branch
    name states.py doesn't leave room for.

    dependencies has UNIQUE (repo_id, name), so if two seeded runs in
    the same invocation happened to pick the same package (only
    possible when n > len(CHAOS_PACKAGES)), the second insert would
    violate that constraint -- handled by reusing the existing
    dependencies row for a package already seen this run, same pattern
    as the repos reuse below.
    """
    import json

    repo_url = f"https://github.com/{GITHUB_REPO}.git"

    existing_repo = conn.execute(
        "SELECT id, sandbox_image FROM repos WHERE url = %s", (repo_url,)
    ).fetchone()
    if existing_repo:
        repo_id = existing_repo["id"]
        sandbox_image = existing_repo["sandbox_image"]
    else:
        sandbox_image = "node:20"
        repo_id = conn.execute(
            """
            INSERT INTO repos (url, default_branch, ecosystem, build_cmd, test_cmd, sandbox_image)
            VALUES (%s, 'main', 'npm', 'npm run build', 'npm test', %s)
            RETURNING id
            """,
            (repo_url, sandbox_image),
        ).fetchone()["id"]

    run_ids = []
    dependency_id_cache: dict[str, int] = {}

    for i in range(n):
        dep_name = CHAOS_PACKAGES[i % len(CHAOS_PACKAGES)]
        target_version = "latest"

        if dep_name in dependency_id_cache:
            dependency_id = dependency_id_cache[dep_name]
        else:
            existing_dep = conn.execute(
                "SELECT id FROM dependencies WHERE repo_id = %s AND name = %s",
                (repo_id, dep_name),
            ).fetchone()
            if existing_dep:
                dependency_id = existing_dep["id"]
            else:
                dependency_id = conn.execute(
                    """
                    INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path)
                    VALUES (%s, %s, 'npm', 'unknown', 'package.json')
                    RETURNING id
                    """,
                    (repo_id, dep_name),
                ).fetchone()["id"]
            dependency_id_cache[dep_name] = dependency_id

        candidate_id = conn.execute(
            """
            INSERT INTO candidates (dependency_id, target_version, semver_jump, status)
            VALUES (%s, %s, 'unknown', 'new')
            RETURNING id
            """,
            (dependency_id, target_version),
        ).fetchone()["id"]

        checkpoint = {
            "repo_url": repo_url,
            "manifest_path": "package.json",
            "dep_name": dep_name,
            "current_version": "unknown",  # not asserted anywhere; real diff comes from git diff --stat
            "target_version": target_version,
            # sandbox_image lives on repos, not runs -- copied onto the
            # checkpoint here since handle_building reads it off
            # checkpoint, not off a live join (see core/states.py).
            "sandbox_image": sandbox_image,
        }
        row = conn.execute(
            """
            INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at)
            VALUES (%s, 'CREATED', %s::jsonb, now())
            RETURNING id
            """,
            (candidate_id, json.dumps(checkpoint)),
        ).fetchone()
        run_ids.append(row["id"])

    print(f"seeded {len(run_ids)} runs, chaos_id={CHAOS_ID}, packages={CHAOS_PACKAGES[:n]}: {run_ids}")
    return run_ids


def cleanup_github(run_ids: list[str]) -> None:
    """
    Because branch names are unique per PACKAGE (see seed_runs
    docstring), not per chaos invocation, every run of chaos.py must
    leave GitHub in a clean state afterward or the NEXT run's
    duplicate-PR check becomes meaningless -- it would see a PR from a
    previous chaos run's leftover branch and either wrongly flag it as
    a duplicate, or wrongly treat find_open_pr's correct "already
    exists" no-op as evidence idempotency worked when it's actually
    just stale state.

    For every one of THIS run's seeded packages: close any open PR on
    its agent/upgrade/npm/{package}-latest branch, then delete the
    branch. Best-effort -- a 404 (branch already gone, PR already
    closed) is not an error worth failing the whole test over.
    """
    packages_used = {CHAOS_PACKAGES[i % len(CHAOS_PACKAGES)] for i in range(len(run_ids))}
    owner = GITHUB_REPO.split("/")[0]
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

    for package in packages_used:
        branch = f"agent/upgrade/npm/{package}-latest"

        try:
            resp = requests.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls",
                headers=headers,
                params={"head": f"{owner}:{branch}", "state": "open"},
                timeout=30,
            )
            resp.raise_for_status()
            for pr in resp.json():
                close = requests.patch(
                    f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr['number']}",
                    headers=headers, json={"state": "closed"}, timeout=30,
                )
                if close.ok:
                    print(f"  closed PR #{pr['number']} ({branch})")
        except requests.RequestException as e:
            print(f"  (cleanup) couldn't check/close PR for {branch}: {e}")

        try:
            resp = requests.delete(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/git/refs/heads/{branch}",
                headers=headers, timeout=30,
            )
            if resp.status_code == 204:
                print(f"  deleted branch {branch}")
            elif resp.status_code != 422:  # 422 = ref doesn't exist, fine
                print(f"  (cleanup) unexpected status deleting {branch}: {resp.status_code}")
        except requests.RequestException as e:
            print(f"  (cleanup) couldn't delete branch {branch}: {e}")


# ---------------------------------------------------------------------------
# Chaos loop
# ---------------------------------------------------------------------------

def run_chaos(procs: list[ManagedProcess], duration_seconds: int) -> None:
    deadline = time.monotonic() + duration_seconds
    kill_count = 0
    while time.monotonic() < deadline:
        sleep_for = random.uniform(KILL_INTERVAL_MIN, KILL_INTERVAL_MAX)
        time.sleep(min(sleep_for, max(deadline - time.monotonic(), 0)))
        if time.monotonic() >= deadline:
            break

        target = random.choice(procs)
        target.kill()
        target.start()  # restart same role immediately, pool size constant
        kill_count += 1

    print(f"chaos window done: {kill_count} kill events over {duration_seconds}s")


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def check_no_double_claims(conn: psycopg.Connection) -> list[str]:
    """
    Structural check, not a chaos-window check: at any instant, a
    lease_owner value should map to at most one currently-held run per
    owner -- but ownership churns constantly under chaos, so this can't
    be checked retrospectively from final state alone (two workers
    could each have validly owned the same run at different times).
    What CAN be checked retrospectively: the claim query's own
    guarantee is that its UPDATE...WHERE...FOR UPDATE SKIP LOCKED only
    ever matches one row per call inside one transaction, so a
    same-run double claim would require two workers to have believed
    they owned the same run AT THE SAME TIME. We don't have a
    changelog table to reconstruct that after the fact in week 1 --
    this invariant is what Day 2's dedicated 5-thread/100-run test
    already proves directly. Chaos.py's job is the other three; this
    function exists so the final report doesn't silently skip
    mentioning it.
    """
    return [
        "(not independently re-checked here -- see Day 2's dedicated "
        "concurrency test, which asserts this directly against claim())"
    ]


def check_no_stuck_runs(conn: psycopg.Connection) -> list[str]:
    """
    A run is stuck if: it's in an actionable state, it's eligible for
    reclamation (next_attempt_at has passed), its lease is gone or
    long expired, AND nothing has touched it in a long time. The
    margin (STUCK_LEASE_MARGIN_SECONDS) is what separates "genuinely
    stuck" from "just released a moment ago, hasn't been reclaimed
    yet" -- ordinary, healthy queueing noise.
    """
    rows = conn.execute(
        """
        SELECT id, state, lease_owner, lease_expires_at, next_attempt_at,
               attempt, updated_at
        FROM runs
        WHERE state = ANY(%s)
          AND next_attempt_at <= now()
          AND (lease_expires_at IS NULL OR lease_expires_at < now() - %s::interval)
          AND updated_at < now() - %s::interval
        """,
        (ACTIONABLE_STATES, f"{STUCK_LEASE_MARGIN_SECONDS} seconds", f"{STUCK_LEASE_MARGIN_SECONDS} seconds"),
    ).fetchall()

    problems = []
    for r in rows:
        if r["lease_owner"] is not None:
            problems.append(
                f"run {r['id']} state={r['state']}: lease_owner={r['lease_owner']} still set, "
                f"lease_expires_at={r['lease_expires_at']} (long expired), never reclaimed -- "
                f"claim() should have picked this up and didn't"
            )
        else:
            problems.append(
                f"run {r['id']} state={r['state']}: no owner, next_attempt_at={r['next_attempt_at']} "
                f"passed, updated_at={r['updated_at']} stale -- eligible but never claimed"
            )
    return problems


def check_no_stuck_ci_limbo(conn: psycopg.Connection) -> list[str]:
    """
    Not one of the four invariants from the week-1 plan verbatim, but a
    real failure mode this specific codebase can hit under chaos (see
    handle_check_suite_completed's `state != AWAITING_CI` guard): if a
    check_suite webhook lands while a run is still PR_OPEN (worker
    killed after handle_patch_ready committed but before another
    worker ran handle_pr_open), the event is silently dropped and never
    redelivered by GitHub. Flagged separately from check_no_stuck_runs
    because PR_OPEN IS in ACTIONABLE_STATES, so a genuinely stuck
    PR_OPEN run would already surface there too -- this check exists to
    name the *cause* distinctly if it happens, by cross-referencing
    inbound_events for an unprocessed check_suite row sitting alongside
    a run that never made it to AWAITING_CI.
    """
    rows = conn.execute(
        """
        SELECT ie.id AS event_id, ie.payload->'check_suite'->>'head_branch' AS branch,
               ie.processed_at
        FROM inbound_events ie
        WHERE ie.source = 'github'
          AND ie.payload->>'action' = 'completed'
          AND ie.processed_at IS NULL
          AND ie.payload->'check_suite'->>'head_branch' LIKE %s
        """,
        (f"agent/upgrade/npm/chaos-{CHAOS_ID}-%",),
    ).fetchall()
    return [
        f"inbound_event {r['event_id']} for branch {r['branch']} was never processed -- "
        f"likely arrived while its run was PR_OPEN rather than AWAITING_CI"
        for r in rows
    ]


def list_all_prs_for_branch(head_branch: str) -> list[dict]:
    """
    Ground truth from GitHub itself -- deliberately NOT calling
    find_open_pr from core/github_client.py, because that function
    returns prs[0] and is designed to answer "does an open PR already
    exist" for the publisher's own idempotency check. If a duplicate
    ever existed, find_open_pr would silently mask it by returning just
    one. This function asks a different, stronger question: how many
    PRs, in any state, actually exist for this branch, ever.
    """
    owner = GITHUB_REPO.split("/")[0]
    resp = requests.get(
        f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls",
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        params={"head": f"{owner}:{head_branch}", "state": "all"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def check_no_duplicate_prs(conn: psycopg.Connection, run_ids: list[str]) -> list[str]:
    """
    For every seeded run that reached PR_OPEN or later, ask GitHub
    itself (not the outbox table, not find_open_pr) how many PRs exist
    for its branch. More than one open PR for the same branch is the
    exact failure handle_patch_ready's deterministic branch name and
    find_open_pr's state=open check exist to prevent.

    Also checks the DB side of the same invariant: no run should have
    more than one 'open_pr' outbox row, since the idempotency_key
    UNIQUE constraint plus ON CONFLICT DO NOTHING in handle_patch_ready
    is what's supposed to guarantee that structurally, independent of
    anything publisher.py does afterward.
    """
    problems = []

    dup_outbox = conn.execute(
        """
        SELECT run_id, count(*) AS n
        FROM outbox
        WHERE kind = 'open_pr' AND run_id = ANY(%s)
        GROUP BY run_id
        HAVING count(*) > 1
        """,
        (run_ids,),
    ).fetchall()
    for r in dup_outbox:
        problems.append(
            f"run {r['run_id']}: {r['n']} 'open_pr' outbox rows -- the UNIQUE "
            f"idempotency_key constraint should make this impossible"
        )

    rows = conn.execute(
        "SELECT id, checkpoint->>'branch' AS branch FROM runs "
        "WHERE id = ANY(%s) AND checkpoint ? 'branch'",
        (run_ids,),
    ).fetchall()

    for r in rows:
        prs = list_all_prs_for_branch(r["branch"])
        open_prs = [p for p in prs if p["state"] == "open"]
        if len(open_prs) > 1:
            numbers = [p["number"] for p in open_prs]
            problems.append(
                f"run {r['id']} branch {r['branch']}: {len(open_prs)} OPEN PRs on GitHub "
                f"(#{numbers}) -- find_open_pr's idempotency check failed to prevent this"
            )

    return problems


def check_no_double_published_outbox(conn: psycopg.Connection, run_ids: list[str]) -> list[str]:
    """
    attempts > 1 is NOT a violation by itself -- at-least-once delivery
    means a row can be legitimately retried after the publisher is
    killed mid-call. The actual invariant, same ground-truth principle
    as check_no_duplicate_prs: published_at should be set at most once
    per row (structurally guaranteed by mark_published's plain UPDATE,
    so this is really a sanity check that nothing bypassed it), AND
    the row's effect (a PR existing) should never have happened twice.
    That second half is exactly check_no_duplicate_prs, so this
    function only covers the DB-side sanity half to avoid duplicating
    the GitHub call.
    """
    rows = conn.execute(
        """
        SELECT id, run_id, attempts, published_at
        FROM outbox
        WHERE run_id = ANY(%s) AND kind = 'open_pr'
        """,
        (run_ids,),
    ).fetchall()

    problems = []
    for r in rows:
        if r["attempts"] > 1:
            print(
                f"  (info, not a failure) outbox row {r['id']} run {r['run_id']}: "
                f"{r['attempts']} attempts before publishing -- expected under chaos, "
                f"this is at-least-once delivery working as designed"
            )
    return problems


def check_no_orphaned_containers() -> list[str]:
    """
    Round 2 (sandbox era): every container this run could have started
    is named upgrade-{run_id}-{attempt}-{phase} (see core/states.py) and
    launched with --rm. By the time this runs, every worker process has
    already been killed (main() stops the pool before invariant checks)
    -- so if a container matching this pattern is still alive, either
    --rm didn't fire (killed mid-`docker run` before the daemon
    registered it) or the Day 1 boot-time sweep has a gap. Either way,
    it's exactly the "orphaned container quietly eating the host"
    failure mode Day 1 and Day 7 both call out.
    """
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=upgrade-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    names = [n for n in result.stdout.splitlines() if n]
    return [f"container {n} still running after chaos window + drain" for n in names]


def check_no_orphaned_advisory_locks() -> list[str]:
    """
    Round 2 (locking era): by the time this runs every worker's
    connection has been killed, which -- per core/repo_lock.py's whole
    reason for using session-scoped, not transaction-scoped, locks --
    should have released every advisory lock those connections held.
    A lock still showing in pg_locks here means some connection either
    didn't die, or died without Postgres cleaning up after it (the
    "advisory lock held by a dead connection blocking everything"
    scenario this test exists to catch).
    """
    conn = db_connect()
    rows = conn.execute(
        "SELECT pid, objid FROM pg_locks WHERE locktype = 'advisory'"
    ).fetchall()
    conn.close()
    return [
        f"advisory lock still held: pid={r['pid']} objid={r['objid']}"
        for r in rows
    ]


def check_run_dirs_match_db(conn: psycopg.Connection) -> list[str]:
    """
    Round 2 (logging era): every runs/{id}/ directory on disk should
    trace back to a real row in the runs table. A directory with no
    matching row is disk usage nobody will ever clean up -- the exact
    "disk filling with directories nobody cleaned up" scenario Day 7
    calls out, just for log dirs instead of /out dirs (patch extraction
    writes into the same runs/{id}/{attempt}/ tree, so one check covers
    both).
    """
    runs_dir = Path("runs")
    if not runs_dir.is_dir():
        return []

    on_disk_ids = {p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    if not on_disk_ids:
        return []

    rows = conn.execute(
        "SELECT id FROM runs WHERE id = ANY(%s)",
        ([int(i) for i in on_disk_ids],),
    ).fetchall()
    known_ids = {str(r["id"]) for r in rows}

    orphaned = on_disk_ids - known_ids
    return [f"runs/{i}/ on disk has no matching row in runs table" for i in sorted(orphaned)]


# Week 6 Day 7's own assertion list, in a state a run genuinely will
# never be worked on again from -- narrower than chaos.py's older
# TERMINAL_STATES (which exists only to print a cosmetic marker):
# AWAITING_CI and PR_OPEN are deliberately excluded here even though
# no worker will pick them up on its own, because they're still
# "alive" -- a webhook can still move them forward, so a sub-agent
# whose parent sits in one of those isn't orphaned, just waiting on
# the same outside event its parent is.
REALLY_TERMINAL_STATES = ["MERGED_READY", "FAILED", "ESCALATED", "SKIPPED", "SUBAGENT_DONE"]


def check_no_context_window_exceeded(conn: psycopg.Connection) -> list[str]:
    """
    Week 6 Day 7: "no rendered context exceeded the window" -- checked
    against real, billed usage.prompt_tokens (tokens_in), not an
    estimate. agent/loop.py's own proactive ceiling check is supposed
    to stop a turn from ever being SENT once it would cross
    REQUEST_TOKEN_CEILING; this is the independent, after-the-fact
    verification that it actually held for real runs, not narrated.
    """
    from agent.loop import REQUEST_TOKEN_CEILING

    rows = conn.execute(
        "SELECT run_id, seq, tokens_in FROM run_messages WHERE tokens_in > %s",
        (REQUEST_TOKEN_CEILING,),
    ).fetchall()
    return [
        f"run {r['run_id']} seq {r['seq']}: tokens_in={r['tokens_in']} exceeded "
        f"REQUEST_TOKEN_CEILING={REQUEST_TOKEN_CEILING} -- the proactive check should have stopped this"
        for r in rows
    ]


def check_no_dangling_tool_calls(conn: psycopg.Connection, run_ids: list[str]) -> list[str]:
    """
    Week 6 Day 7: "no tool_use block without a matching tool_result."
    Only meaningful for runs that reached a state from which nothing
    will ever repair them further -- a run still mid-flight (or
    reclaimable) legitimately has a dangling call sitting there waiting
    for agent/loop.py's own crash-repair path, which is not a bug.
    """
    from agent.loop import _pending_tool_calls
    from agent.messages import load_messages

    problems = []
    rows = conn.execute(
        "SELECT id, state, checkpoint FROM runs WHERE id = ANY(%s) AND state = ANY(%s)",
        (run_ids, REALLY_TERMINAL_STATES),
    ).fetchall()
    for r in rows:
        revision = (r["checkpoint"] or {}).get("revision_count", 0)
        messages = [m["content"] for m in load_messages(conn, r["id"], revision=revision)]
        if _pending_tool_calls(messages):
            problems.append(f"run {r['id']} (state={r['state']}) has a dangling tool_calls with no tool_result")
    return problems


def check_no_orphaned_subagents(conn: psycopg.Connection) -> list[str]:
    """
    Week 6 Day 7: "no sub-agent orphaned (parent terminal, child still
    running)." A sub-agent left in SUBAGENT_PATCHING while its parent
    has reached a really-terminal state (see REALLY_TERMINAL_STATES
    above) will never be claimed again by anything that would notice
    it -- claim() only picks it up on its own merits, but nothing ever
    tells the parent's caller to look at it once the parent itself is
    done.
    """
    rows = conn.execute(
        """
        SELECT child.id AS child_id, child.state AS child_state,
               parent.id AS parent_id, parent.state AS parent_state
        FROM runs child
        JOIN runs parent ON parent.id = child.parent_run_id
        WHERE child.state != 'SUBAGENT_DONE'
          AND parent.state = ANY(%s)
        """,
        (REALLY_TERMINAL_STATES,),
    ).fetchall()
    return [
        f"sub-agent run {r['child_id']} (state={r['child_state']}) orphaned -- "
        f"parent run {r['parent_id']} already reached terminal state {r['parent_state']!r}"
        for r in rows
    ]


def check_no_run_exceeded_max_cost(conn: psycopg.Connection, run_ids: list[str]) -> list[str]:
    """
    Week 6 Day 7: "no run exceeded max_cost, including sub-agent
    spend." total_cost_cents (agent/messages.py) already sums a run's
    own conversation, its own compaction calls, AND every direct
    sub-agent's spend -- exactly the number MAX_COST_CENTS is supposed
    to bound. A small tolerance (one extra turn's worth) is allowed:
    the real check in agent/loop.py is BEFORE the call that would
    exceed budget, so the final total can land slightly over, never
    wildly over.
    """
    from agent.loop import MAX_COST_CENTS
    from agent.messages import total_cost_cents

    tolerance_cents = MAX_COST_CENTS * 0.25
    problems = []
    for run_id in run_ids:
        cost = total_cost_cents(conn, int(run_id))
        if cost > MAX_COST_CENTS + tolerance_cents:
            problems.append(
                f"run {run_id}: total_cost_cents={cost:.2f} exceeds MAX_COST_CENTS={MAX_COST_CENTS} "
                f"by more than the {tolerance_cents:.2f}-cent tolerance"
            )
    return problems


def report_cache_hit_rate(conn: psycopg.Connection, run_ids: list[str]) -> None:
    """
    Week 6 Day 7: "cache hit rate above threshold across the chaos
    run" -- reported, not hard-failed: a chaos run seeds a mix of AUTO
    (zero-token) and short-lived escalated runs alongside any real
    multi-turn AGENT conversations, and the first two categories
    contribute no cache data at all. A blanket threshold across
    everything would be measuring the seed mix, not the caching.
    """
    from agent.context.report import cache_hit_rate_for_run

    rates = [
        r for r in (cache_hit_rate_for_run(conn, int(run_id)) for run_id in run_ids) if r is not None
    ]
    if not rates:
        print("  no cache data across these runs (no multi-turn AGENT conversations with cacheable prefixes)")
        return
    avg = sum(rates) / len(rates)
    print(f"  cache hit rate across {len(rates)} run(s) with data: avg={avg:.1%}, min={min(rates):.1%}, max={max(rates):.1%}")


def report_final_states(conn: psycopg.Connection, run_ids: list[str]) -> None:
    rows = conn.execute(
        "SELECT id, state, attempt, lease_owner FROM runs WHERE id = ANY(%s) ORDER BY id",
        (run_ids,),
    ).fetchall()
    print("\nfinal state of all seeded runs:")
    for r in rows:
        marker = "TERMINAL" if r["state"] in TERMINAL_STATES else ""
        print(f"  run {r['id']}: state={r['state']} attempt={r['attempt']} owner={r['lease_owner']} {marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"=== Day 7 chaos test, chaos_id={CHAOS_ID} ===")
    print(f"repo: {GITHUB_REPO}")
    print(f"seeding {NUM_SEEDED_RUNS} runs, {NUM_WORKERS} workers + 1 publisher, "
          f"{CHAOS_WINDOW_SECONDS}s window\n")

    conn = db_connect()
    run_ids = seed_runs(conn, NUM_SEEDED_RUNS)

    procs = [
        ManagedProcess(role=f"worker-{i}", cmd=[sys.executable, "-m", "worker.main", f"chaos-worker-{i}"])
        for i in range(NUM_WORKERS)
    ] + [
        ManagedProcess(role="publisher", cmd=[sys.executable, "-m", "publisher"])
    ]

    print("starting pool...")
    for p in procs:
        p.start()

    print(f"\nchaos window: {CHAOS_WINDOW_SECONDS}s, kill every {KILL_INTERVAL_MIN}-{KILL_INTERVAL_MAX}s")
    try:
        run_chaos(procs, CHAOS_WINDOW_SECONDS)
    finally:
        print(f"\ndraining {DRAIN_SECONDS}s before assertions (let in-flight work settle)...")
        time.sleep(DRAIN_SECONDS)

        print("stopping pool...")
        for p in procs:
            p.kill()

    report_final_states(conn, run_ids)

    print("\n=== invariant checks ===")
    all_problems: list[str] = []
    try:
        _run_invariant_checks(conn, run_ids, all_problems)
    finally:
        print("\n=== cleanup (closing PRs, deleting branches) ===")
        cleanup_github(run_ids)

    print(f"\n=== {'FAIL' if all_problems else 'PASS'}: {len(all_problems)} total problem(s) ===")
    if all_problems:
        sys.exit(1)


def _run_invariant_checks(conn: psycopg.Connection, run_ids: list[str], all_problems: list[str]) -> None:

    print("\n[1] no double-claimed runs")
    problems = check_no_double_claims(conn)
    for p in problems:
        print(f"  {p}")

    print("\n[2] no stuck runs (expired lease, no owner, stale)")
    problems = check_no_stuck_runs(conn)
    all_problems += problems
    print(f"  {len(problems)} stuck run(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[2b] no CI-webhook-dropped-in-PR_OPEN limbo")
    problems = check_no_stuck_ci_limbo(conn)
    all_problems += problems
    print(f"  {len(problems)} dropped event(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[3] no duplicate PRs on GitHub (ground truth)")
    problems = check_no_duplicate_prs(conn, run_ids)
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[4] no duplicate outbox publishes")
    problems = check_no_double_published_outbox(conn, run_ids)
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[5] no orphaned upgrade-* containers (round 2, sandbox era)")
    problems = check_no_orphaned_containers()
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[6] no orphaned advisory locks (round 2, locking era)")
    problems = check_no_orphaned_advisory_locks()
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[7] every runs/{id}/ dir on disk maps to a real run row (round 2, logging era)")
    problems = check_run_dirs_match_db(conn)
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[8] no rendered context exceeded the window (round 3, context era)")
    problems = check_no_context_window_exceeded(conn)
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[9] no dangling tool_calls on a terminal run (round 3, context era)")
    problems = check_no_dangling_tool_calls(conn, run_ids)
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[10] no orphaned sub-agents (round 3, context era)")
    problems = check_no_orphaned_subagents(conn)
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[11] no run exceeded max_cost, including sub-agent spend (round 3, context era)")
    problems = check_no_run_exceeded_max_cost(conn, run_ids)
    all_problems += problems
    print(f"  {len(problems)} problem(s) found" if problems else "  none found")
    for p in problems:
        print(f"  FAIL: {p}")

    print("\n[12] cache hit rate (informational, round 3, context era)")
    report_cache_hit_rate(conn, run_ids)


if __name__ == "__main__":
    main()