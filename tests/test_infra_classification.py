"""
Distinguishing "the code is broken" (status="failed") from "the sandbox
died out from under it" (status="timeout" / "infra_error") -- these two
scenarios are deliberately synthetic (nobody wants to actually wait out
a real timeout or crash a machine's memory to get a fixture), unlike
the tsc/jest/mypy/pytest/lockfile fixtures which are all genuine
captured output. What's under test here is the classification logic
in core/states.py, not any adapter's parsing.
"""
from __future__ import annotations

from pathlib import Path

from adapters.base import BuildError, StepResult
from core.states import BuildFailed, _classify_infra_failure, _timeout_step_result


def test_oom_exit_code_reclassified_as_infra_error():
    result = StepResult(
        status="failed",
        error_count=1,
        errors=[BuildError(file="", line=None, col=None, code=None, message="Killed")],
    )
    _classify_infra_failure(result, exit_code=137)
    assert result.status == "infra_error"


def test_ordinary_failure_exit_code_not_reclassified():
    result = StepResult(status="failed", error_count=1, errors=[])
    _classify_infra_failure(result, exit_code=1)
    assert result.status == "failed"


def test_ok_status_never_reclassified_even_at_oom_exit_code():
    # Shouldn't happen in practice (a killed process doesn't exit 0),
    # but the classifier must not touch a status that isn't "failed".
    result = StepResult(status="ok", error_count=0, errors=[])
    _classify_infra_failure(result, exit_code=137)
    assert result.status == "ok"


def test_timeout_produces_structured_step_result():
    result = _timeout_step_result(["npm", "test"], 600, Path("runs/1/0/test.log"))
    assert result.status == "timeout"
    assert result.error_count == 1
    assert "600s" in result.errors[0].message
    assert result.log_ref == "runs/1/0/test.log"


def test_timeout_step_result_carries_through_buildfailed_json():
    import dataclasses
    import json

    result = _timeout_step_result(["npm", "run", "build"], 600, Path("runs/2/0/build.log"))
    err = BuildFailed(result)
    parsed = json.loads(str(err))
    assert parsed["status"] == "timeout"
    assert parsed == dataclasses.asdict(result)
