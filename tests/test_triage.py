"""
Week 4 Day 5: deterministic classification of an upgrade candidate,
before any Groq call happens. Pure function, no DB/network/Docker.
"""
from __future__ import annotations

from core.triage import classify_upgrade, has_breaking_markers


def test_patch_bump_no_breaking_markers_is_auto_no_fallback():
    cls, reason, fallback = classify_upgrade("lodash", "patch", "Fixed a typo in the docs.")
    assert cls == "AUTO"
    assert fallback is False


def test_minor_bump_no_breaking_markers_is_auto_with_fallback():
    cls, reason, fallback = classify_upgrade("axios", "minor", "Added a new optional config field.")
    assert cls == "AUTO"
    assert fallback is True


def test_patch_bump_with_breaking_marker_is_agent():
    cls, reason, fallback = classify_upgrade("weird-pkg", "patch", "BREAKING: dropped Node 12 support.")
    assert cls == "AGENT"
    assert fallback is False


def test_major_bump_is_always_agent_even_with_no_markers():
    cls, reason, fallback = classify_upgrade("uuid", "major", "Just some internal refactoring, nothing notable.")
    assert cls == "AGENT"


def test_unknown_semver_jump_is_agent():
    cls, reason, fallback = classify_upgrade("mystery-pkg", "unknown", "")
    assert cls == "AGENT"


def test_deny_list_wins_over_everything_else(monkeypatch):
    monkeypatch.setenv("UPGRADE_DENY_LIST", "react, react-dom")
    cls, reason, fallback = classify_upgrade("react", "patch", "totally harmless fix")
    assert cls == "SKIP"
    assert "deny list" in reason


def test_deny_list_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("UPGRADE_DENY_LIST", "React")
    cls, _, _ = classify_upgrade("react", "patch", "")
    assert cls == "SKIP"


def test_has_breaking_markers_is_case_insensitive():
    assert has_breaking_markers("this has a Breaking change") is True
    assert has_breaking_markers("nothing notable here") is False


def test_has_breaking_markers_does_not_match_substring_of_other_word():
    # "breakingpoint" or similar should not spuriously trigger.
    assert has_breaking_markers("we hit a breakingpoint in testing") is False
