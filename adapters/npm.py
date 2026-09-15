# adapters/npm.py
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from adapters.base import BuildError, StepResult

_MAX_MESSAGE_CHARS = 4000
_MAX_ERRORS = 6

# tsc's default (non---pretty) diagnostic format, one per line:
#   src/main.tsx(15,7): error TS2322: Type 'string' is not assignable to type 'number'.
# Confirmed empirically: tsc auto-disables ANSI/box "pretty" output when
# stdout isn't a TTY (exactly our case -- subprocess.PIPE), so this is
# what tsc actually emits by default inside the sandbox, no extra flag
# needed.
_TSC_ERROR_RE = re.compile(
    r"^(?P<file>[^\n():]+)\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<message>.+)$",
    re.MULTILINE,
)

# jest/vitest --json stack frames look like:
#   at Object.toBe (/repo/__tests__/sample.test.js:5:17)
# Used to recover a line number jest's JSON doesn't give directly.
_JEST_STACK_LINE_RE = re.compile(r":(?P<line>\d+):(?P<col>\d+)\)?\s*$", re.MULTILINE)


class NpmAdapter:
    ecosystem = "npm"
    default_image = "node:20"
    cache_volume = "npm-cache"
    cache_mount_path = "/root/.npm"

    def install_cmd(self) -> list[str]:
        return ["npm", "ci"]

    def build_cmd(self) -> list[str]:
        return ["npm", "run", "build"]

    def test_cmd(self, filter: str | None = None) -> list[str]:
        if filter is None:
            return ["npm", "test"]
        return ["npm", "test", "--", filter]

    def manifest_paths(self) -> list[str]:
        return ["package.json", "package-lock.json"]

    def bump(self, repo_dir: Path, dep: str, version: str) -> None:
        """
        --package-lock-only, not a string-replace on package.json: `npm
        ci` (install_cmd, run inside the sandbox) refuses to run against
        a package-lock.json that disagrees with package.json, and a
        manual edit only ever touches the former. This updates both,
        via npm's own resolver, without installing into node_modules
        here -- that happens later, inside the sandbox, via install_cmd().
        """
        result = subprocess.run(
            ["npm", "install", f"{dep}@{version}", "--package-lock-only"],
            cwd=repo_dir, capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"npm install {dep}@{version} --package-lock-only failed: "
                f"{result.stderr[-800:]}"
            )

    def parse_build(self, exit_code: int, stdout: str, stderr: str) -> StepResult:
        """
        `npm run build` here means `tsc -b && vite build` -- ask tsc for
        structured diagnostics first (its default, non-TTY output already
        is the parseable form; see _TSC_ERROR_RE). If nothing matches
        that shape (e.g. vite itself failed, not tsc), fall back to a
        single generic error rather than silently returning nothing.
        """
        if exit_code == 0:
            return StepResult(status="ok", error_count=0, errors=[])

        combined = stdout + "\n" + stderr
        matches = list(_TSC_ERROR_RE.finditer(combined))
        if matches:
            all_errors = [
                BuildError(
                    file=m.group("file").strip(),
                    line=int(m.group("line")),
                    col=int(m.group("col")),
                    code=m.group("code"),
                    message=m.group("message").strip(),
                )
                for m in matches
            ]
            return StepResult(
                status="failed",
                error_count=len(all_errors),
                errors=all_errors[:_MAX_ERRORS],
                truncated=len(all_errors) > _MAX_ERRORS,
            )

        return self._generic_failure(combined)

    def parse_test(self, exit_code: int, stdout: str, stderr: str) -> StepResult:
        """
        Tries jest/vitest `--json` output first (a JSON object on its
        own, with a top-level "testResults" list -- see
        fixtures/parsers/jest_failing.json for the real schema this was
        built against). Falls back to a generic error if stdout isn't
        that JSON shape, e.g. a test runner that doesn't support --json,
        or a real crash before any JSON was ever written.
        """
        if exit_code == 0:
            return StepResult(status="ok", error_count=0, errors=[])

        parsed = self._parse_jest_json(stdout)
        if parsed is not None:
            return parsed

        return self._generic_failure(stdout + "\n" + stderr)

    def _parse_jest_json(self, stdout: str) -> StepResult | None:
        try:
            data = json.loads(stdout.strip())
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or "testResults" not in data:
            return None

        all_errors: list[BuildError] = []
        for test_file in data["testResults"]:
            # jest reports the container's absolute path (we run with
            # -w /repo); relativize so add_source_context's
            # repo_dir / error.file join works instead of pathlib
            # discarding repo_dir for an absolute right-hand side.
            file_name = test_file.get("name", "")
            if file_name.startswith("/repo/"):
                file_name = file_name[len("/repo/"):]
            for assertion in test_file.get("assertionResults", []):
                if assertion.get("status") != "failed":
                    continue
                failure_messages = assertion.get("failureMessages") or [""]
                message = failure_messages[0]
                line = col = None
                stack_match = _JEST_STACK_LINE_RE.search(message)
                if stack_match:
                    line = int(stack_match.group("line"))
                    col = int(stack_match.group("col"))
                all_errors.append(
                    BuildError(
                        file=file_name,
                        line=line,
                        col=col,
                        code=None,
                        message=message.split("\n")[0].strip() or message.strip(),
                        symbol=assertion.get("fullName") or assertion.get("title"),
                    )
                )

        error_count = data.get("numFailedTests", len(all_errors)) or len(all_errors)
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
