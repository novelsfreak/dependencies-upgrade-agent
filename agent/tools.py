# agent/tools.py
#
# Every tool takes only the arguments the model itself supplies --
# repo_dir/run/conn/worker_id are bound in by build_tools()'s closures,
# not part of any tool's schema, so the model can never influence which
# repo or run it's operating on.
from __future__ import annotations

import dataclasses
import fnmatch
import re
import subprocess
from pathlib import Path
from typing import Callable

from adapters import ADAPTERS
from core.logs import read_log as _read_log

_MAX_SEARCH_MATCHES = 50
_MAX_LIST_FILES = 200
_DEFAULT_READ_LINES = 200
_IGNORED_DIR_NAMES = {".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".pytest_cache"}


class GiveUp(Exception):
    """
    Raised by the give_up tool to break the agent loop cleanly. Not an
    error -- same spirit as LeaseLostError/LockContention elsewhere in
    this codebase: a distinct exception type because the loop needs to
    treat "the model chose to stop" completely differently from "a tool
    call blew up."
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _resolve_safe_path(repo_dir: Path, user_path: str) -> Path:
    """
    Path containment, checked on every tool that takes a path. The
    model will eventually produce something like "../../../etc/passwd"
    -- not maliciously, just because it lost track of where it is in a
    big repo -- and this must fail loudly and specifically enough that
    the model can correct itself next turn.
    """
    resolved = (repo_dir / user_path).resolve()
    repo_root = repo_dir.resolve()
    if repo_root != resolved and repo_root not in resolved.parents:
        raise ValueError(
            f"path {user_path!r} resolves outside the repository -- "
            f"paths must be relative to the repo root, use list_files to see what's there"
        )
    return resolved


def _iter_repo_files(repo_dir: Path):
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _IGNORED_DIR_NAMES for part in path.relative_to(repo_dir).parts):
            continue
        yield path


def list_files(repo_dir: Path, glob: str = "**/*") -> str:
    matches = sorted(
        str(p.relative_to(repo_dir))
        for p in repo_dir.glob(glob)
        if p.is_file() and not any(part in _IGNORED_DIR_NAMES for part in p.relative_to(repo_dir).parts)
    )
    total = len(matches)
    shown = matches[:_MAX_LIST_FILES]
    body = "\n".join(shown)
    if total > len(shown):
        body += f"\n... ({total} total matches, showing first {len(shown)} -- narrow your glob)"
    return body or f"no files matched {glob!r}"


def read_file(repo_dir: Path, path: str, start: int = 1, end: int = _DEFAULT_READ_LINES) -> str:
    try:
        resolved = _resolve_safe_path(repo_dir, path)
    except ValueError as e:
        # Every tool here returns a descriptive string rather than
        # raising -- list_files/search already never raise; relying on
        # the agent loop's own generic exception handler to turn this
        # into a string would work by accident, not by contract.
        return str(e)

    if not resolved.is_file():
        # A guess is more useful than a bare "not found" -- the model
        # is usually one directory or one typo away, not fabricating a
        # path from nothing.
        candidates = [
            str(p.relative_to(repo_dir)) for p in repo_dir.rglob(resolved.name)
            if p.is_file()
        ][:5]
        hint = f" Did you mean: {', '.join(candidates)}?" if candidates else " Use list_files first."
        return f"No such file: {path}.{hint}"

    lines = resolved.read_text(errors="replace").splitlines()
    total = len(lines)
    if end - start > 2000:
        return f"requested range too large ({end - start} lines) -- read at most 2000 lines at a time"

    window = lines[max(0, start - 1):end]
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(window, start=max(1, start)))
    if end < total:
        numbered += f"\n... ({total} lines total, showing {start}-{min(end, total)})"
    return numbered or f"{path} is empty"


def search(repo_dir: Path, pattern: str, glob: str = "**/*") -> str:
    """
    Pure-Python line-scan, not a subprocess call to ripgrep: this
    environment has no real `rg` binary on PATH (only a shell alias
    that bypasses subprocess entirely), and a worker container running
    this later may not either. Same interface/behavior the plan asks
    for -- capped matches, true count reported -- without a host
    dependency.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"invalid regex {pattern!r}: {e}"

    matches: list[str] = []
    total = 0
    for path in repo_dir.glob(glob):
        if not path.is_file() or any(part in _IGNORED_DIR_NAMES for part in path.relative_to(repo_dir).parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(repo_dir)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                total += 1
                if len(matches) < _MAX_SEARCH_MATCHES:
                    matches.append(f"{rel}:{lineno}:{line.strip()}")

    if total == 0:
        return f"no matches for {pattern!r}"
    body = "\n".join(matches)
    if total > len(matches):
        body += f"\n... ({total} total matches, showing first {len(matches)})"
    return body


_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
_BARE_HUNK_RE = re.compile(r"^@@\s*@@|^@@\s*$", re.MULTILINE)


def apply_patch(repo_dir: Path, diff: str) -> str:
    """
    git apply --check first, always -- never half-apply a multi-hunk
    diff. Falls back to -C1 (relaxed context matching) per the plan's
    own mitigation for a model reconstructing a diff from a file it
    read many turns ago and getting context lines slightly wrong, then
    to classic patch(1) if even that fails.

    That third fallback exists for a real, reproduced case: gpt-oss-120b
    tends to write minimal hunks containing ONLY the changed line(s),
    zero lines of surrounding context. That's a well-formed unified
    diff -- header counts match the body, content matches the file
    exactly -- but `git apply` unconditionally refuses to apply a hunk
    with no context at all, even with -C1 (confirmed directly: -C1
    reduces how closely context must match, it doesn't waive the
    requirement that some exist). Classic `patch(1)` has no such
    requirement and applies the identical diff without complaint --
    verified against the exact diff a live run produced before this
    fallback existed.

    On success, commits immediately (matching handle_patching's own
    existing pattern) so extract_patch's base_sha diffing in
    core/states.py keeps working unmodified on this same run later.
    """
    if _BARE_HUNK_RE.search(diff) and not _HUNK_HEADER_RE.search(diff):
        # Observed for real: gpt-oss-120b repeatedly emitted a bare
        # "@@" with no line numbers across FIVE consecutive attempts in
        # one live run, each one burning a full turn (and a chunk of
        # the rate-limited token budget) on git's own unhelpful "No
        # valid patches in input" / "patch with only garbage at line N"
        # -- neither message tells the model what's actually wrong.
        # Catching the specific, common shape of the mistake here means
        # the very first attempt gets a fix-it-now answer instead of
        # five rounds of guessing.
        return (
            "patch did not apply: hunk header is missing line numbers. Every hunk header "
            "must look like '@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@' (e.g. "
            "'@@ -1,7 +1,7 @@'), not a bare '@@'. Count OLD_START from line 1 of the file "
            "content read_file already showed you."
        )

    diff_path = repo_dir / ".agent_patch.diff"
    diff_path.write_text(diff)
    try:
        check = subprocess.run(
            ["git", "apply", "--check", str(diff_path)],
            cwd=repo_dir, capture_output=True, text=True,
        )
        use_patch_binary = False
        extra_flag = None
        if check.returncode != 0:
            check_relaxed = subprocess.run(
                ["git", "apply", "--check", "-C1", str(diff_path)],
                cwd=repo_dir, capture_output=True, text=True,
            )
            if check_relaxed.returncode == 0:
                extra_flag = "-C1"
            else:
                check_patch = subprocess.run(
                    ["patch", "--dry-run", "-p1", "-i", str(diff_path)],
                    cwd=repo_dir, capture_output=True, text=True,
                )
                if check_patch.returncode == 0:
                    use_patch_binary = True
                else:
                    return (
                        f"patch did not apply: {check.stderr.strip()}\n\n"
                        f"Use read_file to see the CURRENT content of the target file(s) "
                        f"before retrying -- your context lines don't match what's actually there."
                    )

        if use_patch_binary:
            result = subprocess.run(
                ["patch", "-p1", "-i", str(diff_path)], cwd=repo_dir, capture_output=True, text=True,
            )
        else:
            apply_cmd = ["git", "apply"] + ([extra_flag] if extra_flag else []) + [str(diff_path)]
            result = subprocess.run(apply_cmd, cwd=repo_dir, capture_output=True, text=True)
        if result.returncode != 0:
            return f"patch passed --check but failed to apply: {result.stderr.strip()}"
    finally:
        diff_path.unlink(missing_ok=True)

    subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, text=True)
    commit = subprocess.run(
        ["git", "-c", "user.email=agent@example.com", "-c", "user.name=upgrade-agent",
         "commit", "-m", "agent patch"],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if commit.returncode != 0:
        if "nothing to commit" in commit.stdout:
            return "patch applied cleanly but changed nothing (a no-op diff)"
        return f"patch applied but commit failed: {commit.stderr.strip()}"

    applied = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD"], cwd=repo_dir, capture_output=True, text=True,
    )
    return f"patch applied and committed. Current diff from before your patch:\n\n{applied.stdout}"


def read_log_tool(run_id: int, attempt: int, kind: str, offset: int = 0, limit: int = 200) -> str:
    try:
        result = _read_log(run_id, attempt, kind, offset, limit)
    except FileNotFoundError as e:
        return str(e)
    lines = "\n".join(result["lines"])
    if result["truncated"]:
        lines += f"\n... ({result['total_lines']} lines total, showing {offset}-{offset + len(result['lines'])})"
    return lines or "(empty)"


def build_tools(run: dict, conn, worker_id: str) -> dict[str, Callable[..., str]]:
    """
    Binds a tool dict to one run/conn/worker_id via closures -- tool
    schemas never expose repo_dir/run_id/conn, only what the model
    itself should be choosing (a path, a pattern, a diff). This is the
    TOOLS registry the loop dispatches into, built fresh per run rather
    than a single global dict, since these need per-run context.
    """
    checkpoint = run.get("checkpoint") or {}
    repo_dir = Path(checkpoint.get("repo_dir") or checkpoint["work_dir"])

    def _run_build() -> str:
        # Reuses handle_building exactly -- same sandbox, same repo
        # lock, same cache volume, same StepResult shaping. This is
        # the model's own iterative feedback; the final BUILDING state
        # after the loop ends re-verifies independently regardless of
        # what the model believes happened here.
        from core.states import BuildFailed, handle_building
        try:
            _next_state, delta = handle_building(run, conn, worker_id)
            run["checkpoint"].update(delta)
            return f"build ok. {delta.get('build_result', {})}"
        except BuildFailed as e:
            return f"build failed: {e}"

    def _run_tests(filter: str | None = None) -> str:
        from core.states import BuildFailed, handle_testing
        try:
            _next_state, delta = handle_testing(run, conn, worker_id)
            run["checkpoint"].update(delta)
            return f"tests ok. {delta.get('test_result', {})}"
        except BuildFailed as e:
            return f"tests failed: {e}"

    def _give_up(reason: str) -> str:
        raise GiveUp(reason)

    return {
        "list_files": lambda glob="**/*": list_files(repo_dir, glob),
        "read_file": lambda path, start=1, end=_DEFAULT_READ_LINES: read_file(repo_dir, path, start, end),
        "search": lambda pattern, glob="**/*": search(repo_dir, pattern, glob),
        "apply_patch": lambda diff: apply_patch(repo_dir, diff),
        "run_build": lambda: _run_build(),
        "run_tests": lambda filter=None: _run_tests(filter),
        "read_log": lambda kind, offset=0, limit=200: read_log_tool(run["id"], run["attempt"], kind, offset, limit),
        "give_up": lambda reason: _give_up(reason),
    }
