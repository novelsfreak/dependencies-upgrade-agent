# adapters/base.py
#
# The plugin boundary between the state machine (core/states.py) and any
# one ecosystem's toolchain. core/states.py should never again contain a
# literal "npm ci" or "pytest" -- it asks an adapter for a command and
# gets back a StepResult, and has no idea which ecosystem it just ran.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol


@dataclass
class BuildError:
    """
    One structured error extracted from a build/test tool's own output.
    Day 3 only defines this shape; Day 4 is what actually populates
    `symbol`/`context` properly instead of leaving them None.
    """
    file: str
    line: int | None
    col: int | None
    code: str | None       # e.g. "TS2554"
    message: str
    symbol: str | None = None   # e.g. "HttpClient.request"
    context: str | None = None  # +/-5 lines of source, Day 4


@dataclass
class StepResult:
    """
    What every adapter's parse_build/parse_test hands back. `errors` is
    capped (Day 4 fixes the cap at 6) but `error_count` is always the
    true total -- the model needs to know it's seeing some of 40, not
    all of 3. `log_ref` and `duration_ms` aren't known to the adapter
    itself (it only sees exit_code/stdout/stderr), so callers fill
    those in after parse_build/parse_test returns.
    """
    status: Literal["ok", "failed", "timeout", "infra_error"]
    error_count: int
    errors: list[BuildError] = field(default_factory=list)
    truncated: bool = False
    log_ref: str = ""
    duration_ms: int = 0


def add_source_context(errors: list[BuildError], repo_dir: Path, context_lines: int = 5) -> None:
    """
    Populate `context` on each error by reading +/-`context_lines` lines
    around it from the repo on disk. Generic across ecosystems -- it
    only needs a file+line, not tool-specific knowledge -- so it lives
    here rather than in any one adapter, and is called by the orchestrator
    (core/states.py) after parse_build/parse_test, which don't have
    repo_dir in their signature.
    """
    for error in errors:
        if not error.file or error.line is None:
            continue
        path = repo_dir / error.file
        if not path.is_file():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        start = max(0, error.line - 1 - context_lines)
        end = min(len(lines), error.line + context_lines)
        error.context = "\n".join(lines[start:end])


class BuildAdapter(Protocol):
    """
    One implementation per ecosystem (npm, pip, ...). `repos.ecosystem`
    selects which adapter a run uses; `repos.sandbox_image` overrides
    `default_image` per repo.
    """
    ecosystem: str
    default_image: str

    # A named docker volume and the path it mounts to, install-phase
    # only (a cold install can take minutes; over a long soak that's
    # most of the wall clock). Shared across every run of this
    # ecosystem, which is exactly why it needs core.repo_lock guarding
    # concurrent access to it, not because the mount itself is unsafe.
    cache_volume: str
    cache_mount_path: str

    def install_cmd(self) -> list[str]: ...
    def build_cmd(self) -> list[str]: ...
    def test_cmd(self, filter: str | None = None) -> list[str]: ...

    def manifest_paths(self) -> list[str]: ...
    def bump(self, repo_dir: Path, dep: str, version: str) -> None: ...

    def parse_build(self, exit_code: int, stdout: str, stderr: str) -> StepResult: ...
    def parse_test(self, exit_code: int, stdout: str, stderr: str) -> StepResult: ...
