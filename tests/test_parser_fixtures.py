"""
Parser fixtures: real, saved tool output run through the real parser
with no Docker and no network. These run in milliseconds on every
commit -- the fixtures themselves came from actually breaking a real
repo (tsc_type_error.log) and actually running a failing test suite
(jest_failing.json), not hand-written guesses at the tool's schema.
"""
from __future__ import annotations

from pathlib import Path

from adapters.npm import NpmAdapter
from adapters.pip import PipAdapter

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "parsers"


def test_tsc_type_error_fixture():
    log = (FIXTURES_DIR / "tsc_type_error.log").read_text()
    result = NpmAdapter().parse_build(exit_code=2, stdout=log, stderr="")

    assert result.status == "failed"
    assert result.error_count == 2
    assert not result.truncated
    assert result.errors[0].file == "src/main.tsx"
    assert result.errors[0].line == 15
    assert result.errors[0].col == 7
    assert result.errors[0].code == "TS2322"
    assert "not assignable" in result.errors[0].message
    assert result.errors[1].code == "TS6133"


def test_jest_failing_fixture():
    raw = (FIXTURES_DIR / "jest_failing.json").read_text()
    result = NpmAdapter().parse_test(exit_code=1, stdout=raw, stderr="")

    assert result.status == "failed"
    assert result.error_count == 1
    assert not result.truncated
    assert result.errors[0].file == "__tests__/sample.test.js"
    assert result.errors[0].line == 5
    assert result.errors[0].symbol == "this one fails on purpose"


def test_generic_fallback_for_unparseable_output():
    result = NpmAdapter().parse_build(exit_code=1, stdout="some vite bundling error\n", stderr="")

    assert result.status == "failed"
    assert result.error_count == 1
    assert "vite bundling error" in result.errors[0].message


def test_ok_status_on_zero_exit():
    result = NpmAdapter().parse_build(exit_code=0, stdout="", stderr="")
    assert result.status == "ok"
    assert result.error_count == 0
    assert result.errors == []


def test_mypy_type_error_fixture():
    log = (FIXTURES_DIR / "mypy_type_error.jsonl").read_text()
    result = PipAdapter().parse_build(exit_code=1, stdout=log, stderr="")

    assert result.status == "failed"
    assert result.error_count == 2
    assert result.errors[0].file == "main.py"
    assert result.errors[0].code == "import-untyped"
    assert result.errors[1].line == 4
    assert result.errors[1].code == "return-value"


def test_pytest_failing_fixture():
    raw = (FIXTURES_DIR / "pytest_failing.json").read_text()
    result = PipAdapter().parse_test(exit_code=1, stdout=raw, stderr="")

    assert result.status == "failed"
    assert result.error_count == 1
    assert result.errors[0].file == "test_main.py"
    assert result.errors[0].line == 7
    assert result.errors[0].symbol == "test_main.py::test_add_wrong_on_purpose"


def test_npm_ci_lockfile_mismatch_fixture():
    # Real npm ci output, captured by actually bumping package.json's
    # declared react range without updating package-lock.json and
    # running npm ci against a real clone -- npm's own "package.json
    # and lockfile disagree" check, not fabricated text.
    log = (FIXTURES_DIR / "npm_ci_lockfile_mismatch.log").read_text()
    result = NpmAdapter().parse_build(exit_code=1, stdout=log, stderr="")

    assert result.status == "failed"
    assert result.error_count == 1
    assert "Missing" in result.errors[0].message
    assert "lock file" in result.errors[0].message


def test_caps_at_six_and_reports_true_total():
    # 8 real-shaped tsc lines from one breaking change (the plan's own
    # example: the same error 40 times) -- must return 6, but still
    # report 8 as the true count so the model knows it's seeing a slice.
    lines = "\n".join(
        f'src/file{i}.ts({i},1): error TS2554: Expected 2 arguments, but got 3.'
        for i in range(8)
    )
    result = NpmAdapter().parse_build(exit_code=2, stdout=lines, stderr="")

    assert result.status == "failed"
    assert result.error_count == 8
    assert len(result.errors) == 6
    assert result.truncated is True
