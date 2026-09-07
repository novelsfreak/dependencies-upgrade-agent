"""
The state machine, expressed as a dict of handler functions.

Every handler has the same shape: (run, conn, worker_id) -> (next_state,
checkpoint_delta). conn and worker_id exist so handlers whose work spans
a long-running subprocess (BUILDING, TESTING) can heartbeat the lease
while they wait. Most handlers ignore both -- the signature is uniform
so the worker loop never special-cases "this handler needs extra args."

Deliberately fewer states than the full design doc for week 1 -- no
REVISING loop, no PLANNING, no AWAITING_REVIEW yet. Those come once
there's a real agent and real GitHub integration to react to.

Two states are NOT in HANDLERS on purpose: AWAITING_CI (and later
AWAITING_REVIEW). They only move on a webhook. A worker's claim() query
filters on ACTIONABLE_STATES, so rows in these states are structurally
invisible to claim() -- not skipped by convention, but never selected
by the WHERE clause in the first place. See core/claim.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import psycopg

from core.heartbeat_guard import HeartbeatGuard, LeaseLostError

Handler = Any  # (run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]

BUILD_TIMEOUT_SECONDS = 600
RUNS_LOG_DIR = Path("runs")


def handle_created(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    Nothing to do yet except move forward. This is where, later, the
    orchestrator would decide PLANNING vs. going straight to a
    deterministic patch (design doc's cost cascade). For week 1, every
    CREATED run goes straight to PATCHING.
    """
    return "PATCHING", {}


def handle_patching(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    THE FAKE AGENT.

    Two code paths, chosen by whether checkpoint gives a repo_url:

    - Real repo (Day 4 onward): clone it, then run
      `npm install {dep}@{target_version}`. This is npm's own job --
      it updates package.json AND package-lock.json together, using
      npm's real dependency resolution. A manual string-replace on
      package.json alone would leave the lockfile stale, and `npm ci`
      (used in BUILDING) refuses to run against a stale lockfile by
      design -- it's meant for reproducible installs, not for guessing
      how to reconcile a manual edit. This bit us on the first real-repo
      run, which is exactly why it's handled properly now instead of
      papered over.

    - No repo_url (stub, Days 1-3 tests still exercise this): fabricate
      a minimal package.json and do a plain string-replace. There's no
      real lockfile in this path, and no `npm ci` will ever run against
      it, so the mismatch problem above doesn't apply here.

    Either way: fresh temp dir per attempt, never mutate anything in
    place (Day 7 chaos fix -- a retry must never see a half-applied
    patch from a previous, interrupted attempt).

    Real repo/dependency/target-version lookups would come from joining
    through candidate_id -> dependency_id -> dependencies/candidates
    tables. For week 1, we accept them directly on the checkpoint so we
    can exercise the state machine without wiring up those joins yet.
    """
    checkpoint = run.get("checkpoint") or {}
    manifest_path = checkpoint.get("manifest_path", "package.json")
    dep_name = checkpoint.get("dep_name", "axios")
    current_version = checkpoint.get("current_version", "1.6.2")
    target_version = checkpoint.get("target_version", "1.7.0")
    repo_url = checkpoint.get("repo_url")

    # Always work in a fresh temp dir per attempt -- never mutate
    # anything in place. This is the fix for the Day 7 chaos failure
    # mode: "a handler that mutates state on disk before committing to
    # the DB, so a retry sees a half-applied patch."
    work_dir = Path(tempfile.mkdtemp(prefix=f"run-{run['id']}-"))

    if repo_url:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(work_dir / "repo")],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(f"git clone failed: {result.stderr}")
        repo_dir = work_dir / "repo"

        manifest_full_path = repo_dir / manifest_path
        before = manifest_full_path.read_text()

        # npm's own resolver updates package.json AND package-lock.json
        # together. package-lock=true is the default but stated
        # explicitly: this must never silently skip the lockfile.
        install = subprocess.run(
            ["npm", "install", f"{dep_name}@{target_version}", "--package-lock=true"],
            cwd=repo_dir, capture_output=True, text=True, timeout=180,
        )
        if install.returncode != 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(
                f"npm install {dep_name}@{target_version} failed: {install.stderr[-800:]}"
            )

        after = manifest_full_path.read_text()
        if after == before:
            # npm exited 0 but changed nothing -- e.g. target_version
            # resolves to what's already installed. Loud failure, not a
            # silently-green no-op run.
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ValueError(
                f"npm install reported success but {manifest_path} did not change "
                f"-- {dep_name} may already be at the requested version"
            )
    else:
        # Week 1 stub content, unchanged from Day 3 -- lets existing
        # tests keep working without a repo_url. Deliberately does NOT
        # derive file content from current_version: doing so would make
        # the stub always "agree" with whatever the caller claims the
        # current version is, defeating the mismatch check below.
        repo_dir = work_dir
        _STUB_VERSION = "1.6.2"
        stub_manifest = repo_dir / manifest_path
        stub_manifest.parent.mkdir(parents=True, exist_ok=True)
        if not stub_manifest.exists():
            stub_manifest.write_text(
                json.dumps(
                    {"name": "fake-repo", "dependencies": {dep_name: _STUB_VERSION}},
                    indent=2,
                )
            )

        manifest_full_path = stub_manifest
        original = manifest_full_path.read_text()
        patched = original.replace(
            f'"{dep_name}": "{current_version}"',
            f'"{dep_name}": "{target_version}"',
        )

        if patched == original:
            # The version string we expected to find wasn't there. Don't
            # silently "succeed" with a no-op patch -- fail loudly so this
            # shows up as a real error, not a mysteriously green run.
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ValueError(
                f'expected "{dep_name}": "{current_version}" in {manifest_path}, not found'
            )

        manifest_full_path.write_text(patched)
        patch_summary = (
            f"--- a/{manifest_path}\n+++ b/{manifest_path}\n"
            f'- "{dep_name}": "{current_version}"\n'
            f'+ "{dep_name}": "{target_version}"\n'
        )

    if repo_url:
        # For the real-repo path, describe what actually changed via git
        # diff rather than reconstructing it by hand -- npm may have
        # touched package-lock.json too, and a hand-written summary
        # would silently omit that.
        diff = subprocess.run(
            ["git", "diff", "--stat"], cwd=repo_dir, capture_output=True, text=True,
        )
        patch_summary = diff.stdout or f"(no diff output) bumped {dep_name} to {target_version}"

    patch_path = work_dir / "changes.patch"
    patch_path.write_text(patch_summary)

    return "BUILDING", {
        "work_dir": str(work_dir),
        "repo_dir": str(repo_dir),
        "patch_path": str(patch_path),
        "manifest_path": manifest_path,
        "dep_name": dep_name,
        "target_version": target_version,
    }


def _run_subprocess_step(
    run: dict,
    conn: psycopg.Connection,
    worker_id: str,
    cmd: list[str],
    log_name: str,
) -> tuple[int, str]:
    """
    Shared machinery for handle_building and handle_testing: run `cmd`
    in the run's repo_dir, heartbeat the lease for the duration, enforce
    a hard timeout, and persist the full log to disk.

    Returns (exit_code, log_path). Does NOT decide next_state -- that's
    the caller's job, since "exit code 0 means what" differs between a
    build and a test run only in name, not in mechanics.
    """
    checkpoint = run.get("checkpoint") or {}
    repo_dir = checkpoint.get("repo_dir") or checkpoint.get("work_dir")
    if not repo_dir:
        raise RuntimeError("no repo_dir/work_dir in checkpoint -- did PATCHING run first?")

    log_dir = RUNS_LOG_DIR / str(run["id"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # The heartbeat guard renews the lease every 30s while we block on
    # proc below. If it ever loses the lease, it kills proc itself --
    # we don't have to poll for that ourselves, we just notice the
    # process died early and lease_lost is set.
    guard = HeartbeatGuard(conn, run["id"], worker_id, proc)
    with guard:
        try:
            stdout, _ = proc.communicate(timeout=BUILD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            log_path.write_text(stdout + "\n\n[TIMED OUT after {}s]".format(BUILD_TIMEOUT_SECONDS))
            raise TimeoutError(f"{' '.join(cmd)} exceeded {BUILD_TIMEOUT_SECONDS}s")

    log_path.write_text(stdout)

    if guard.lease_lost:
        # We were superseded mid-build. The other worker now owns this
        # run -- we must not report success or failure back to the DB
        # at all, since release() would be writing to a run we no
        # longer have any authority over. LeaseLostError (not a plain
        # RuntimeError) so the worker loop can tell this apart from an
        # ordinary failure and skip release()/backoff entirely.
        raise LeaseLostError(f"lease lost during {log_name}, another worker has taken over")

    return proc.returncode, str(log_path)


def handle_building(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    Real subprocess build: npm ci && npm run build, in the repo cloned
    by handle_patching. Full log goes to disk; only exit code + log
    path go in the checkpoint (the design doc's error-shaping work --
    parsing that log into structured errors -- comes later, not week 1).
    """
    repo_dir = (run.get("checkpoint") or {}).get("repo_dir") or (run.get("checkpoint") or {}).get("work_dir")

    ci = subprocess.run(
        ["npm", "ci"], cwd=repo_dir, capture_output=True, text=True, timeout=BUILD_TIMEOUT_SECONDS,
    )
    if ci.returncode != 0:
        log_dir = RUNS_LOG_DIR / str(run["id"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "npm_ci.log"
        log_path.write_text(ci.stdout + ci.stderr)
        raise RuntimeError(f"npm ci failed (exit {ci.returncode}), see {log_path}")

    exit_code, log_path = _run_subprocess_step(
        run, conn, worker_id, ["npm", "run", "build"], "build.log"
    )

    if exit_code != 0:
        raise RuntimeError(f"npm run build failed (exit {exit_code}), see {log_path}")

    return "TESTING", {"build_log": log_path, "build_exit_code": exit_code}


def handle_testing(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    Real subprocess test run: npm test. Same mechanics as
    handle_building -- see _run_subprocess_step.
    """
    exit_code, log_path = _run_subprocess_step(
        run, conn, worker_id, ["npm", "test"], "test.log"
    )

    if exit_code != 0:
        raise RuntimeError(f"npm test failed (exit {exit_code}), see {log_path}")

    return "PATCH_READY", {"test_log": log_path, "test_exit_code": exit_code}


def handle_patch_ready(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    Week 1 stub for what will become the outbox insert (Day 5): open a
    branch, open a PR. For now just advance state so the pipeline can
    be exercised end to end before the outbox exists.
    """
    return "PR_OPEN", {}


def handle_pr_open(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    A PR is open. From here a worker has nothing left to do -- the next
    move depends on GitHub telling us CI finished. So this handler's
    only job is to hand the run off into AWAITING_CI, a state that
    claim() will never select again until a webhook moves it out.
    """
    return "AWAITING_CI", {}


# AWAITING_CI is deliberately absent from this dict. If claim() ever
# somehow returned a run in that state (it shouldn't -- see
# ACTIONABLE_STATES in core/claim.py), HANDLERS[run["state"]] would
# raise a KeyError rather than silently doing nothing. Loud failure
# over silent one.
HANDLERS: dict[str, Handler] = {
    "CREATED": handle_created,
    "PATCHING": handle_patching,
    "BUILDING": handle_building,
    "TESTING": handle_testing,
    "PATCH_READY": handle_patch_ready,
    "PR_OPEN": handle_pr_open,
}
