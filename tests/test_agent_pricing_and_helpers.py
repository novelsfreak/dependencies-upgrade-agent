"""
Pure-function tests for cost accounting and loop control-flow helpers --
no Groq calls, no Docker.
"""
from __future__ import annotations

from agent.loop import _call_hash, _classify_semver_jump
from agent.pricing import compute_cost_cents


def test_compute_cost_cents_matches_known_rate():
    # gpt-oss-120b: $0.15/M input, $0.60/M output (verified against
    # multiple trackers as of 2026-09-16, see agent/pricing.py).
    cost = compute_cost_cents("openai/gpt-oss-120b", tokens_in=1_000_000, tokens_out=1_000_000)
    assert abs(cost - 75.0) < 0.001  # ($0.15 + $0.60) * 100 cents


def test_compute_cost_cents_unknown_model_raises():
    try:
        compute_cost_cents("not-a-real-model", 100, 100)
        assert False, "should have raised"
    except ValueError as e:
        assert "not-a-real-model" in str(e)


def test_call_hash_is_stable_regardless_of_arg_order():
    h1 = _call_hash("read_file", {"path": "a.js", "start": 1})
    h2 = _call_hash("read_file", {"start": 1, "path": "a.js"})
    assert h1 == h2


def test_call_hash_differs_for_different_args():
    h1 = _call_hash("read_file", {"path": "a.js"})
    h2 = _call_hash("read_file", {"path": "b.js"})
    assert h1 != h2


def test_classify_semver_jump_major():
    assert _classify_semver_jump("3.4.0", "14.0.2") == "major"


def test_classify_semver_jump_minor():
    assert _classify_semver_jump("1.6.2", "1.7.0") == "minor"


def test_classify_semver_jump_patch():
    assert _classify_semver_jump("1.6.2", "1.6.3") == "patch"


def test_classify_semver_jump_unknown_current():
    assert _classify_semver_jump("unknown", "1.7.0") == "unknown"
