# core/triage.py
#
# Week 4 Day 5: most upgrades don't need a model at all. A patch bump
# with no breaking-change markers in its changelog is, empirically,
# never a call-site problem -- semver's own contract is that patch
# releases don't change the public API. Spending real Groq turns on
# those is pure waste; the deterministic PATCHING->BUILDING->TESTING->
# PATCH_READY path (use_agent=False) already handles them correctly,
# for zero tokens.
from __future__ import annotations

import os
import re

_BREAKING_MARKER_RE = re.compile(r"\bBREAKING\b", re.IGNORECASE)


def _deny_list() -> set[str]:
    raw = os.environ.get("UPGRADE_DENY_LIST", "")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def has_breaking_markers(changelog_text: str) -> bool:
    return bool(_BREAKING_MARKER_RE.search(changelog_text))


def classify_upgrade(dep_name: str, semver_jump: str, changelog_text: str) -> tuple[str, str, bool]:
    """
    Returns (classification, reason, fallback_to_agent).

    classification is "AUTO" | "AGENT" | "SKIP".

    fallback_to_agent only ever applies to AUTO, and only for a minor
    bump: patch releases are, per the plan's own rule, confident enough
    to just retry deterministically on failure like any other
    infrastructure hiccup. A minor bump is less certain -- semver's
    contract is looser there -- so a REAL build failure on one should
    hand off to the agent instead of retrying a deterministic path
    that would fail identically every time.
    """
    if dep_name.lower() in _deny_list():
        return "SKIP", f"{dep_name!r} is on the deny list (UPGRADE_DENY_LIST)", False

    breaking = has_breaking_markers(changelog_text)

    if semver_jump == "patch" and not breaking:
        return "AUTO", "patch bump, no breaking-change markers in changelog", False
    if semver_jump == "minor" and not breaking:
        return "AUTO", "minor bump, no breaking-change markers -- falls back to AGENT on a real build failure", True

    return "AGENT", f"{semver_jump} bump or breaking-change markers present in changelog", False
