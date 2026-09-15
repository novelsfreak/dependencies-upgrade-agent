# sandbox/network.py
#
# Manages the Phase A (install) egress path: a user-defined bridge
# network plus an allowlist proxy container attached to it. Phase B
# (build/test) never touches any of this -- it runs with
# --network=none, enforced directly in executor.start_sandbox.
from __future__ import annotations

import subprocess
from pathlib import Path

EGRESS_NETWORK = "agent-egress"
EGRESS_PROXY_NAME = "agent-egress-proxy"
EGRESS_PROXY_IMAGE = "agent-egress-proxy:latest"
PROXY_URL = f"http://{EGRESS_PROXY_NAME}:3128"

PROXY_BUILD_CONTEXT = Path(__file__).resolve().parent / "proxy"


def ensure_egress_network() -> None:
    """Idempotently create the bridge network install-phase containers
    and the allowlist proxy both attach to."""
    result = subprocess.run(
        ["docker", "network", "inspect", EGRESS_NETWORK],
        capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["docker", "network", "create", EGRESS_NETWORK],
            check=True, capture_output=True,
        )


def _ensure_proxy_image() -> None:
    """Build the proxy image from sandbox/proxy/ if it isn't already
    present. Built locally rather than pulled -- the allowlist lives in
    this repo's squid.conf, not in a third-party tag we'd have to trust."""
    result = subprocess.run(
        ["docker", "image", "inspect", EGRESS_PROXY_IMAGE],
        capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["docker", "build", "-t", EGRESS_PROXY_IMAGE, str(PROXY_BUILD_CONTEXT)],
            check=True, capture_output=True,
        )


def ensure_egress_proxy() -> None:
    """
    Idempotently get the allowlist proxy running. Safe to call before
    every install phase, and on worker boot -- a no-op if it's already up.
    """
    ensure_egress_network()
    _ensure_proxy_image()

    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}",
         "--filter", f"name=^{EGRESS_PROXY_NAME}$"],
        capture_output=True, text=True,
    )
    if EGRESS_PROXY_NAME in result.stdout.splitlines():
        return

    subprocess.run(
        ["docker", "run", "-d", "--rm",
         "--name", EGRESS_PROXY_NAME,
         "--network", EGRESS_NETWORK,
         EGRESS_PROXY_IMAGE],
        check=True, capture_output=True,
    )
