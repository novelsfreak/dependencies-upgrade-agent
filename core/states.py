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

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
import logging

import psycopg

from adapters import ADAPTERS
from adapters.base import BuildError, StepResult, add_source_context
from core.heartbeat_guard import HeartbeatGuard, LeaseLostError
from core.logs import log_dir_for
from core.repo_lock import LockContention, try_lock_repo, unlock_repo
from sandbox.executor import kill_sandbox, start_sandbox
from sandbox.network import EGRESS_NETWORK, PROXY_URL, ensure_egress_proxy

Handler = Any  # (run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]

BUILD_TIMEOUT_SECONDS = 600


class BuildFailed(RuntimeError):
    """
    Raised by handle_building/handle_testing when the adapter's own
    parse_build/parse_test says the step failed. str(error) is a JSON
    dump of the StepResult, not prose -- handle_failure's
    checkpoint_delta stores str(error) as "last_error" verbatim, so this
    is what makes that field genuinely structured (file/line/code/
    context, capped at 6, true error_count) instead of a raw text tail.
    """
    def __init__(self, step_result: StepResult):
        self.step_result = step_result
        super().__init__(json.dumps(dataclasses.asdict(step_result)))


# 137 = 128 + SIGKILL(9), Linux/Docker's universal convention for "this
# process was killed", the OOM killer included. Checked here, not in
# any one adapter, because the convention is a Docker/OS-level fact,
# not something tsc/jest/mypy/pytest output ever encodes themselves.
_OOM_EXIT_CODE = 137


def _classify_infra_failure(result: StepResult, exit_code: int) -> None:
    """
    Reclassifies a StepResult's status from "failed" to "infra_error"
    in place when the exit code itself says this wasn't the tool
    reporting a real error -- it was the container being killed out
    from under it. Distinguishing the two matters: "failed" means the
    code has a real problem to fix; "infra_error" means retrying with
    more memory (or just retrying) is the right move, not debugging the
    diff.
    """
    if exit_code == _OOM_EXIT_CODE and result.status == "failed":
        result.status = "infra_error"


def _timeout_step_result(cmd: list[str], timeout_seconds: int, log_path: Path) -> StepResult:
    return StepResult(
        status="timeout",
        error_count=1,
        errors=[BuildError(
            file="", line=None, col=None, code=None,
            message=f"{' '.join(cmd)} exceeded {timeout_seconds}s",
        )],
        log_ref=str(log_path),
    )


# Only "install" genuinely needs the registry -- the build and test
# phases execute the third-party code that npm/pip just downloaded,
# which is exactly the code we don't trust with a network. Splitting
# by phase means that untrusted code never has anywhere to reach.
PHASE_NETWORK = {
    "install": EGRESS_NETWORK,
    "build": "none",
    "test": "none",
}

# Explicit allowlist, never the host's environment. Nothing here is
# secret-shaped on purpose -- see test_no_secrets_in_sandbox_env.
BASE_SANDBOX_ENV = {"CI": "true"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("states")



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
    ecosystem = checkpoint.get("ecosystem", "npm")
    adapter = ADAPTERS[ecosystem]

    # Always work in a fresh temp dir per attempt -- never mutate
    # anything in place. This is the fix for the Day 7 chaos failure
    # mode: "a handler that mutates state on disk before committing to
    # the DB, so a retry sees a half-applied patch."
    work_dir = Path(tempfile.mkdtemp(prefix=f"run-{run['id']}-"))
    base_sha = None  # only meaningful on the real-repo path; stub path has no git history to diff

    if repo_url:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(work_dir / "repo")],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(f"git clone failed: {result.stderr}")
        repo_dir = work_dir / "repo"

        # The clone's own HEAD, before the bump commit goes on top --
        # this is what handle_building's patch extraction later diffs
        # against, so the patch is exactly "what the bump changed"
        # regardless of whether it added, deleted, or modified files.
        base_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True,
        ).stdout.strip()

        manifest_full_path = repo_dir / manifest_path
        before = manifest_full_path.read_text()

        # Ecosystem-specific: the adapter knows how to update its
        # manifest AND lockfile together (see NpmAdapter.bump's
        # docstring for why a plain string-replace isn't enough).
        try:
            adapter.bump(repo_dir, dep_name, target_version)
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        after = manifest_full_path.read_text()
        if after == before:
            # bump() exited 0 but changed nothing -- e.g. target_version
            # resolves to what's already installed. Loud failure, not a
            # silently-green no-op run.
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ValueError(
                f"{ecosystem} bump reported success but {manifest_path} did not change "
                f"-- {dep_name} may already be at the requested version"
            )

        # npm only touches the working tree -- without an actual commit,
        # HEAD never moves past the cloned commit, so handle_patch_ready's
        # `git push HEAD:refs/heads/branch` would push a branch identical
        # to main (GitHub then rejects the PR with "No commits between
        # main and <branch>").
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, text=True)
        commit = subprocess.run(
            ["git", "-c", "user.email=agent@example.com", "-c", "user.name=upgrade-agent",
             "commit", "-m", f"Upgrade {dep_name} to {target_version}"],
            cwd=repo_dir, capture_output=True, text=True,
        )
        if commit.returncode != 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(f"git commit failed: {commit.stderr[-800:]}")
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
        "ecosystem": ecosystem,
        "base_sha": base_sha,
    }


def _run_subprocess_step(
    run: dict,
    conn: psycopg.Connection,
    worker_id: str,
    cmd: list[str],
    log_name: str,
    phase: str,  # "install" | "build" | "test" -- feeds the container name
) -> tuple[int, str, str, int]:
    """
    Shared machinery for handle_building and handle_testing: run `cmd`
    inside a disposable container, heartbeat the lease for the duration,
    enforce a hard timeout, and persist the full log to disk.

    Returns (exit_code, stdout, log_path, duration_ms). Does NOT decide
    next_state or shape errors -- callers own that (they're the ones who
    know which adapter method, parse_build vs. parse_test, applies).
    """
    checkpoint = run.get("checkpoint") or {}
    repo_dir = checkpoint.get("repo_dir") or checkpoint.get("work_dir")
    if not repo_dir:
        raise RuntimeError("no repo_dir/work_dir in checkpoint -- did PATCHING run first?")

    adapter = ADAPTERS[checkpoint["ecosystem"]]

    # runs/{run_id}/{attempt}/{phase}.log -- attempt-scoped so a retried
    # run's logs from a prior failed attempt aren't overwritten, and so
    # log_ref (e.g. "370/1/build") unambiguously names one execution.
    log_dir = log_dir_for(run["id"], run["attempt"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    container_name = f"upgrade-{run['id']}-{run['attempt']}-{phase}"
    # sandbox_image lives on repos, not runs -- runs has no such column.
    # Per this file's own week-1 pattern (repo_url, dep_name, etc. are
    # threaded through checkpoint rather than joined live), whatever
    # creates the run is expected to have copied repos.sandbox_image
    # onto checkpoint at creation time. release()'s checkpoint merge
    # (`checkpoint || delta`) preserves it across every later state
    # transition without handle_patching needing to forward it by hand.
    # Falls back to the adapter's own default_image, not a literal.
    image = checkpoint.get("sandbox_image") or adapter.default_image

    network = PHASE_NETWORK[phase]
    env = dict(BASE_SANDBOX_ENV)
    volumes = {}
    if phase == "install":
        # Only the install phase gets a route out, and only to the
        # allowlisted proxy -- never straight to the internet.
        ensure_egress_proxy()
        env["HTTP_PROXY"] = PROXY_URL
        env["HTTPS_PROXY"] = PROXY_URL
        env["NO_PROXY"] = "localhost,127.0.0.1"
        # Shared across every run of this ecosystem -- core.repo_lock is
        # what keeps two concurrent installs of the SAME repo from
        # corrupting it; unrelated repos installing at the same time
        # still share this volume, same as the plan describes.
        volumes[adapter.cache_volume] = adapter.cache_mount_path

    started_at = time.monotonic()
    proc = start_sandbox(
        image=image,
        workdir=Path(repo_dir),
        cmd=cmd,
        name=container_name,
        network=network,
        env=env,
        volumes=volumes,
    )

    # HeartbeatGuard must kill by container name now, not proc.kill() --
    # proc here is the `docker run` wrapper; killing it leaves the
    # container running orphaned. See kill_sandbox().
    guard = HeartbeatGuard(conn, run["id"], worker_id, kill_fn=lambda: kill_sandbox(container_name))
    with guard:
        try:
            stdout, _ = proc.communicate(timeout=BUILD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            kill_sandbox(container_name)
            stdout, _ = proc.communicate()
            log_path.write_text(stdout + "\n\n[TIMED OUT after {}s]".format(BUILD_TIMEOUT_SECONDS))
            # Routed through the same StepResult/BuildFailed shape as a
            # real build/test failure, status="timeout" instead of
            # "failed" -- so checkpoint.last_error is structured JSON
            # here too, not a plain string a future model would have to
            # special-case.
            raise BuildFailed(_timeout_step_result(cmd, BUILD_TIMEOUT_SECONDS, log_path))

    duration_ms = int((time.monotonic() - started_at) * 1000)
    log_path.write_text(stdout)

    if guard.lease_lost:
        raise LeaseLostError(f"lease lost during {log_name}, another worker has taken over")

    return proc.returncode, stdout, str(log_path), duration_ms

def extract_patch(repo_dir: Path, base_sha: str, out_path: Path) -> None:
    """
    Diffs base_sha (the clone's original HEAD, captured in handle_patching
    before the bump commit went on top) against the current HEAD, and
    writes the result to out_path.

    Run directly on the host, not sandboxed: git diff between two
    already-committed shas doesn't execute any repo-controlled code --
    there's nothing here to sandbox against, unlike install/build/test.
    /repo is bind-mounted, so this sees exactly the file state a
    container-side diff would see after the build step wrote to it --
    host and container share the same underlying directory.

    Diffing two real commits, rather than uncommitted worktree changes,
    is what makes new/deleted/modified files all fall out of one plain
    `git diff` -- no --no-index special-casing for new files, and no
    risk of a .gitignore'd file leaking in (it was never `git add`ed
    into either commit to begin with).
    """
    result = subprocess.run(
        ["git", "diff", base_sha, "HEAD"],
        cwd=repo_dir, capture_output=True, text=True, timeout=60,
    )
    out_path.write_text(result.stdout)


def handle_building(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    checkpoint = run.get("checkpoint") or {}
    if "ecosystem" not in checkpoint:
        raise RuntimeError("no ecosystem in checkpoint -- did PATCHING run first?")
    adapter = ADAPTERS[checkpoint["ecosystem"]]
    repo_dir = Path(checkpoint.get("repo_dir") or checkpoint["work_dir"])

    # Per-repo, not per-run: two runs against the SAME repo must not
    # install concurrently into the shared cache volume. Keyed on
    # repo_url (falling back to work_dir for the no-repo stub path,
    # where every run gets its own tempdir anyway and contention is
    # impossible) rather than a numeric repos.id -- avoids needing that
    # join here, and is just as unique.
    #
    # Scoped to install+build only (this call), NOT held across into
    # handle_testing: that's a separately-claimable state that could be
    # picked up by a different worker's connection, and releasing an
    # advisory lock from a connection that never held it is a silent
    # no-op in Postgres -- it would leak this lock on the ORIGINAL
    # worker's connection until that worker process exits (exactly the
    # "advisory lock held by a dead connection" bug the plan's own
    # chaos section warns about). install+build is also where the only
    # actually-shared mutable resource (the cache volume) gets touched;
    # TESTING only reads this run's own already-populated repo_dir.
    repo_key = f"repo:{checkpoint.get('repo_url') or checkpoint['work_dir']}"
    if not try_lock_repo(conn, repo_key):
        raise LockContention(repo_key)

    try:
        # There's no adapter.parse_install in the Protocol -- install
        # failures (missing package, registry unreachable, a
        # package.json/lockfile mismatch) are a different, usually
        # simpler class of problem than a build/test failure. But they
        # still deserve structured shape rather than a bare message, so
        # this reuses parse_build: none of its tool-specific regexes
        # (tsc, mypy) match install-time text, so it safely falls
        # through to the adapter's generic single-error capture -- see
        # fixtures/parsers/npm_ci_lockfile_mismatch.log for a real
        # example this was verified against.
        exit_code, stdout, log_path, duration_ms = _run_subprocess_step(
            run, conn, worker_id, adapter.install_cmd(), "install.log", phase="install"
        )
        if exit_code != 0:
            install_result = adapter.parse_build(exit_code, stdout, "")
            install_result.log_ref = log_path
            install_result.duration_ms = duration_ms
            _classify_infra_failure(install_result, exit_code)
            raise BuildFailed(install_result)

        exit_code, stdout, log_path, duration_ms = _run_subprocess_step(
            run, conn, worker_id, adapter.build_cmd(), "build.log", phase="build"
        )
        # stderr is always "" here: start_sandbox merges it into stdout
        # (stderr=subprocess.STDOUT in executor.py), so the adapter
        # only ever sees combined output as stdout.
        build_result = adapter.parse_build(exit_code, stdout, "")
        build_result.log_ref = log_path
        build_result.duration_ms = duration_ms
        add_source_context(build_result.errors, repo_dir)
        _classify_infra_failure(build_result, exit_code)

        if build_result.status != "ok":
            raise BuildFailed(build_result)
    finally:
        unlock_repo(conn, repo_key)

    checkpoint_delta = {
        "build_log": log_path,
        "build_exit_code": exit_code,
        "build_result": dataclasses.asdict(build_result),
    }

    base_sha = checkpoint.get("base_sha")
    if base_sha:
        patch_path = log_dir_for(run["id"], run["attempt"]) / "patch.diff"
        extract_patch(repo_dir, base_sha, patch_path)
        checkpoint_delta["patch_path"] = str(patch_path)

    return "TESTING", checkpoint_delta


def handle_testing(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    checkpoint = run.get("checkpoint") or {}
    if "ecosystem" not in checkpoint:
        raise RuntimeError("no ecosystem in checkpoint -- did PATCHING run first?")
    adapter = ADAPTERS[checkpoint["ecosystem"]]
    repo_dir = Path(checkpoint.get("repo_dir") or checkpoint["work_dir"])

    exit_code, stdout, log_path, duration_ms = _run_subprocess_step(
        run, conn, worker_id, adapter.test_cmd(), "test.log", phase="test"
    )
    test_result = adapter.parse_test(exit_code, stdout, "")
    test_result.log_ref = log_path
    test_result.duration_ms = duration_ms
    add_source_context(test_result.errors, repo_dir)
    _classify_infra_failure(test_result, exit_code)

    if test_result.status != "ok":
        raise BuildFailed(test_result)

    return "PATCH_READY", {
        "test_log": log_path,
        "test_exit_code": exit_code,
        "test_result": dataclasses.asdict(test_result),
    }

def handle_patch_ready(run: dict, conn: psycopg.Connection, worker_id: str) -> tuple[str, dict]:
    """
    Push the branch, then record "I intend to open a PR" as an outbox
    row IN THE SAME TRANSACTION as the state change to PR_OPEN.

    This does NOT call the GitHub PR-creation API itself -- that's
    publisher.py's job, running as a separate process. Why split it
    this way: GitHub's create-PR call and our DB update can't be one
    atomic operation (they're two different systems), so if this
    handler called GitHub directly and then crashed before writing to
    the DB, a retry would call GitHub again and create a duplicate PR.
    By only ever writing a durable INTENT here -- transactionally, with
    the state change -- the actual GitHub call becomes something a
    separate, retryable process can attempt as many times as it needs
    to, safely, because it's driven off a row that either fully exists
    or (if this transaction never committed) doesn't exist at all.

    git push IS done directly here, not through the outbox. Pushing to
    a fixed, deterministic branch name is naturally idempotent -- git
    just fast-forwards or no-ops if nothing changed -- so it doesn't
    need the same durability treatment as PR creation, which is NOT
    idempotent on GitHub's side.

    IMPORTANT: unlike every other handler, this one commits the state
    change itself, as part of the same transaction as the outbox
    insert. It returns a checkpoint_delta with "_released": True so the
    worker loop knows NOT to also call release() -- doing so would run
    a second, redundant UPDATE outside this transaction and defeat the
    whole point of writing both changes atomically.
    """
    checkpoint = run.get("checkpoint") or {}
    repo_dir = checkpoint.get("repo_dir")
    dep_name = checkpoint.get("dep_name")
    target_version = checkpoint.get("target_version")

    if not repo_dir:
        raise RuntimeError("no repo_dir in checkpoint -- this run's patch used the stub path, not a real repo")

    token = os.environ.get("GITHUB_TOKEN")
    repo_slug = os.environ.get("GITHUB_REPO")

    if not token or not repo_slug:
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPO must be set (see .env)")

    log.info("Pushing to repo %s", repo_slug)

    # Deterministic branch name -- same run, same dep, same target
    # version always produces the same branch. This is what lets the
    # publisher ask "does a PR already exist for this branch?" instead
    # of having to trust its own bookkeeping alone.
    branch = f"agent/upgrade/npm/{dep_name}-{target_version}"

    # Authenticate the remote via token, then push. Done as separate
    # steps so the token never appears in a command that could show up
    # in a process listing or shell history.
    set_url = subprocess.run(
        ["git", "remote", "set-url", "origin", f"https://x-access-token:{token}@github.com/{repo_slug}.git"],
        cwd=repo_dir, capture_output=True, text=True, timeout=30,
    )
    if set_url.returncode != 0:
        raise RuntimeError(f"git remote set-url failed: {set_url.stderr[-800:]}")

    # Plain --force, not --force-with-lease: the lease check needs a
    # known remote tip to compare against, but handle_patching's clone
    # never fetches this branch (only ever `main`), so git has no lease
    # to check -- it would reject this push every time the branch
    # already has any commit on it, on every retry, identically, since
    # retries reuse the same clone that still never fetches it. Chaos
    # testing proved this isn't hypothetical: 9 of 10 seeded runs died
    # this way. This branch is exclusively owned by this automation (no
    # human ever pushes to it directly), so the safety --force-with-lease
    # buys elsewhere doesn't apply here -- plain --force is what actually
    # delivers the "fixed branch name is naturally idempotent" behavior
    # this function's own docstring describes.
    push = subprocess.run(
        ["git", "push", "--force", "origin", f"HEAD:refs/heads/{branch}"],
        cwd=repo_dir, capture_output=True, text=True, timeout=60,
    )
    if push.returncode != 0:
        raise RuntimeError(f"git push failed: {push.stderr[-800:]}")

    idempotency_key = f"{run['id']}:open_pr"
    payload = {
        "repo": repo_slug,
        "head_branch": branch,
        "base_branch": "main",
        "title": f"Upgrade {dep_name} to {target_version}",
        "body": f"Automated upgrade for {dep_name} to {target_version}.\n\nrun_id: {run['id']}",
    }

    new_checkpoint = {**checkpoint, "branch": branch}

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE runs SET state = %s, checkpoint = %s::jsonb, lease_owner = NULL, "
            "lease_expires_at = NULL, updated_at = now() WHERE id = %s",
            ("PR_OPEN", json.dumps(new_checkpoint), run["id"]),
        )
        cur.execute(
            """
            INSERT INTO outbox (run_id, kind, payload, idempotency_key)
            VALUES (%s, 'open_pr', %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            """,
            (run["id"], json.dumps(payload), idempotency_key),
        )
    conn.commit()

    return "PR_OPEN", {"_released": True}


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
