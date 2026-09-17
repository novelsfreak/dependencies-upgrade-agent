# core/logs.py
#
# The on-disk layout for a run's full, unabridged logs, and pull-based
# access to them. StepResult only ever carries up to 6 errors -- this is
# how a caller (a human today, a model's tool call in week 3) gets the
# rest, keyed off the log_ref a StepResult points at.
from __future__ import annotations

from pathlib import Path

RUNS_LOG_DIR = Path("runs")


def log_dir_for(run_id: int, attempt: int) -> Path:
    return RUNS_LOG_DIR / str(run_id) / str(attempt)


def log_path_for(run_id: int, attempt: int, kind: str) -> Path:
    return log_dir_for(run_id, attempt) / f"{kind}.log"


def read_log(
    run_id: int,
    attempt: int,
    kind: str,
    offset: int = 0,
    limit: int = 200,
) -> dict:
    """
    Pull-based access to a run's full log. Nothing calls this yet --
    it becomes a tool the model can call in week 3, when a StepResult's
    six errors aren't enough and it needs to see more of the same log
    itself pointed at via `log_ref`.
    """
    path = log_path_for(run_id, attempt, kind)
    if not path.exists():
        raise FileNotFoundError(f"no log at {path}")

    lines = path.read_text(errors="replace").splitlines()
    total = len(lines)
    window = lines[offset: offset + limit]
    return {
        "lines": window,
        "offset": offset,
        "total_lines": total,
        "truncated": offset + limit < total,
    }
