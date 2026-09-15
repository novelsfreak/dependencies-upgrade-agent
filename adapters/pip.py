# adapters/pip.py
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from adapters.base import BuildError, StepResult

_MAX_MESSAGE_CHARS = 4000
_MAX_ERRORS = 6

# Our own tooling, not the target repo's -- installed alongside whatever
# the repo's own requirements.txt declares, same idea as a CI runner
# bringing its own linters. A repo isn't expected to carry
# pytest-json-report as a devDependency just so we can parse its output.
_TOOLING_PACKAGES = ["mypy", "pytest", "pytest-json-report"]


class PipAdapter:
    ecosystem = "pip"
    # Official astral image: python + uv preinstalled, nothing to bootstrap.
    default_image = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
    cache_volume = "uv-cache"
    cache_mount_path = "/root/.cache/uv"  # confirmed via `uv cache dir`

    def install_cmd(self) -> list[str]:
        """
        `--system` would install into the CONTAINER's own site-packages,
        which is gone the instant this --rm container exits -- the
        build and test phases are separate containers from a fresh
        image, and only the bind-mounted /repo persists between them.
        So install into a venv INSIDE /repo instead (the pip analogue
        of npm's node_modules landing in repo_dir): build/test then
        reach it by calling /repo/.venv/bin/<tool> directly, since
        there's no shell around to "activate" it non-interactively.
        """
        script = (
            "uv venv /repo/.venv && "
            "uv pip install --python /repo/.venv/bin/python "
            f"-r requirements.txt {' '.join(_TOOLING_PACKAGES)}"
        )
        return ["sh", "-c", script]

    def build_cmd(self) -> list[str]:
        """
        Python has no compile step. Decided deliberately: "build" here
        means a static type-check (mypy), not a no-op -- it's real
        signal that can catch a breaking upgrade before any test runs,
        which a no-op would just throw away. The tradeoff: a repo with
        no mypy config gets a build step that always reports zero
        errors, which is harmless but not informative for that repo.
        """
        return ["/repo/.venv/bin/mypy", ".", "--output", "json", "--no-error-summary"]

    def test_cmd(self, filter: str | None = None) -> list[str]:
        """
        pytest-json-report writes its report to a FILE
        (--json-report-file), unlike jest which can put JSON straight
        on stdout -- there's no stdout-JSON mode for pytest. To keep
        parse_test's signature identical to NpmAdapter's (just
        exit_code/stdout/stderr, no repo_dir), this cats that file to
        stdout as the very last thing -- after pytest's own console
        output is discarded to /dev/null, so stdout ends up being
        exactly the JSON and nothing else. `exit $code` preserves
        pytest's real exit status, which `cat` would otherwise clobber.
        """
        pytest_args = ["/repo/.venv/bin/pytest", "--json-report",
                        "--json-report-file=/repo/.pytest_report.json", "-q"]
        if filter is not None:
            pytest_args += ["-k", shlex.quote(filter)]
        pytest_cmd = " ".join(pytest_args)
        script = (
            f"{pytest_cmd} > /dev/null 2>&1; "
            "code=$?; cat /repo/.pytest_report.json; exit $code"
        )
        return ["sh", "-c", script]

    def manifest_paths(self) -> list[str]:
        return ["requirements.txt"]

    def bump(self, repo_dir: Path, dep: str, version: str) -> None:
        """
        requirements.txt has no separate lockfile to keep in sync (unlike
        npm's package.json/package-lock.json pair) -- a pinned line is
        the whole spec, so a direct rewrite is the real, complete fix
        here, not a shortcut the way it would be for npm.
        """
        req_path = repo_dir / "requirements.txt"
        if not req_path.exists():
            raise RuntimeError(f"no requirements.txt at {req_path}")

        lines = req_path.read_text().splitlines()
        new_lines = []
        found = False
        for line in lines:
            stripped = line.strip()
            name = stripped.split("==")[0].split(">=")[0].split("<")[0].strip()
            if name.lower() == dep.lower():
                new_lines.append(f"{dep}=={version}")
                found = True
            else:
                new_lines.append(line)

        if not found:
            raise ValueError(f"{dep} not found in requirements.txt")

        req_path.write_text("\n".join(new_lines) + "\n")

    def parse_build(self, exit_code: int, stdout: str, stderr: str) -> StepResult:
        """
        mypy --output json emits one JSON object per line (JSON Lines,
        not a single array or an array-of-objects) -- see
        fixtures/parsers/mypy_type_error.jsonl for genuine captured
        output this was built against.
        """
        if exit_code == 0:
            return StepResult(status="ok", error_count=0, errors=[])

        all_errors: list[BuildError] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                diag = json.loads(line)
            except json.JSONDecodeError:
                continue
            if diag.get("severity") != "error":
                continue
            all_errors.append(
                BuildError(
                    file=diag.get("file", ""),
                    line=diag.get("line"),
                    col=diag.get("column"),
                    code=diag.get("code"),
                    message=diag.get("message", ""),
                )
            )

        if not all_errors:
            return self._generic_failure(stdout + "\n" + stderr)

        return StepResult(
            status="failed",
            error_count=len(all_errors),
            errors=all_errors[:_MAX_ERRORS],
            truncated=len(all_errors) > _MAX_ERRORS,
        )

    def parse_test(self, exit_code: int, stdout: str, stderr: str) -> StepResult:
        """
        pytest --json-report writes its report to a file
        (--json-report-file), not stdout -- unlike jest, there's no
        stdout-JSON mode. The caller is expected to have read that file
        and pass its contents here as `stdout`; see
        fixtures/parsers/pytest_failing.json for the genuine schema.
        """
        if exit_code == 0:
            return StepResult(status="ok", error_count=0, errors=[])

        parsed = self._parse_pytest_json(stdout)
        if parsed is not None:
            return parsed

        return self._generic_failure(stdout + "\n" + stderr)

    def _parse_pytest_json(self, report_text: str) -> StepResult | None:
        try:
            data = json.loads(report_text.strip())
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or "tests" not in data:
            return None

        all_errors: list[BuildError] = []
        for test in data["tests"]:
            if test.get("outcome") != "failed":
                continue
            call = test.get("call", {})
            crash = call.get("crash") or {}
            # pytest reports the container's absolute path (-w /repo);
            # relativize for the same reason as NpmAdapter's jest
            # parsing -- add_source_context's repo_dir / file join needs
            # a relative path, not an absolute right-hand side.
            file_path = crash.get("path", "")
            if file_path.startswith("/repo/"):
                file_path = file_path[len("/repo/"):]
            all_errors.append(
                BuildError(
                    file=file_path,
                    line=crash.get("lineno"),
                    col=None,
                    code=None,
                    message=crash.get("message", "")[:_MAX_MESSAGE_CHARS],
                    symbol=test.get("nodeid"),
                )
            )

        error_count = data.get("summary", {}).get("failed", len(all_errors))
        return StepResult(
            status="failed",
            error_count=error_count,
            errors=all_errors[:_MAX_ERRORS],
            truncated=error_count > min(len(all_errors), _MAX_ERRORS),
        )

    def _generic_failure(self, raw_output: str) -> StepResult:
        tail = raw_output.strip()[-_MAX_MESSAGE_CHARS:]
        error = BuildError(file="", line=None, col=None, code=None, message=tail)
        return StepResult(status="failed", error_count=1, errors=[error])
