# adapters/__init__.py
#
# Registry, same pattern as core/states.py's HANDLERS dict:
# repos.ecosystem selects the adapter a run uses.
from adapters.base import BuildAdapter, BuildError, StepResult
from adapters.npm import NpmAdapter
from adapters.pip import PipAdapter

ADAPTERS: dict[str, BuildAdapter] = {
    "npm": NpmAdapter(),
    "pip": PipAdapter(),
}

__all__ = ["BuildAdapter", "BuildError", "StepResult", "ADAPTERS"]
