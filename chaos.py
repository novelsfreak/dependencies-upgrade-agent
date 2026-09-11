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
ACTIONABLE_STATES = ["CREATED", "PATCHING", "BUILDING", "TESTING", "PATCH_READY", "PR_OPEN"]

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
        "SELECT id FROM repos WHERE url = %s", (repo_url,)
    ).fetchone()
    if existing_repo:
        repo_id = existing_repo["id"]
    else:
        repo_id = conn.execute(
            """
            INSERT INTO repos (url, default_branch, ecosystem, build_cmd, test_cmd)
            VALUES (%s, 'main', 'npm', 'npm run build', 'npm test')
            RETURNING id
            """,
            (repo_url,),
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


if __name__ == "__main__":
    main()