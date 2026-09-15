"""
Day 2's two permanent controls: the build/test phase must have no
network, and it must never see anything secret-shaped. Neither of
these is something you'd deliberately break -- they exist to catch a
future "just add --env-file .env to fix this" or "just drop
--network=none, it's blocking something" done tired, months from now.

Requires Docker. Run with: uv run pytest tests/test_sandbox_network.py -v
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from sandbox.executor import start_sandbox

pytestmark = pytest.mark.skipif(
    subprocess.run(["docker", "info"], capture_output=True).returncode != 0,
    reason="Docker daemon not available",
)

# Token-shaped patterns worth failing the build over. Not exhaustive --
# just the shapes common enough that seeing one crossing into a
# --network=none container is a near-certain sign something leaked in
# that shouldn't have.
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"[A-Za-z0-9+/]{32,}={0,2}"),  # generic base64-ish blob
]


def test_no_network_in_build_phase():
    """Build/test containers run with --network=none. A request to a
    raw IP (no DNS involved) must fail to even connect."""
    proc = start_sandbox(
        image="curlimages/curl:8.10.1",
        workdir=Path("/tmp"),
        cmd=["curl", "-s", "--max-time", "5", "http://1.1.1.1"],
        name="test-no-network-build-phase",
        network="none",
    )
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode != 0, (
        f"container reached the network despite --network=none: {stdout!r}"
    )


def test_no_secrets_in_sandbox_env():
    """
    Build/test containers get an explicit env allowlist, never a copy
    of the host's environment. Prove it by planting a real-looking
    token on the host process and confirming it never appears inside
    the container.
    """
    fake_token = "ghp_" + "a" * 36
    os.environ["_TEST_FAKE_GITHUB_TOKEN"] = fake_token
    try:
        proc = start_sandbox(
            image="alpine:3.20",
            workdir=Path("/tmp"),
            cmd=["env"],
            name="test-no-secrets-sandbox-env",
            network="none",
        )
        stdout, _ = proc.communicate(timeout=15)
    finally:
        del os.environ["_TEST_FAKE_GITHUB_TOKEN"]

    assert fake_token not in stdout, "planted host token leaked into sandbox env"
    for line in stdout.splitlines():
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(line), (
                f"token-shaped value in sandbox env: {line!r}"
            )
