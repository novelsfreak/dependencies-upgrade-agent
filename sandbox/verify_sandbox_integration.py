# sandbox/verify_sandbox_integration.py
#
# End-to-end verification that handle_building actually launches the
# SANDBOXED container with the image from the run's checkpoint (not
# runs.sandbox_image, which doesn't exist -- see core/states.py's
# comment on this). Runs a real worker.main process against a real
# seeded run and inspects the live container.
#
# This does NOT reliably test the lease-loss/kill_fn mechanism end to
# end -- a real `npm ci` can finish or fail in well under
# HEARTBEAT_INTERVAL_SECONDS (30s), so there's no guarantee the
# container survives long enough for a heartbeat check to ever fire.
# For that mechanism specifically, use verify_lease_kill.py instead,
# which uses a deterministic `sleep 120` so timing isn't a confound.
#
# Known caveat, unrelated to this code: on macOS + Docker Desktop,
# `npm ci` inside the bind-mounted container can occasionally fail with
# ETXTBSY while executing a package's freshly-installed postinstall
# binary (seen with esbuild) -- a host<->VM filesystem sync race on
# bind mounts, not a bug in this repo. If you see that, it's usually
# not reproducible on every run.
#
# Run from the project root:
#   uv run python -m sandbox.verify_sandbox_integration
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")
GITHUB_REPO = os.environ["GITHUB_REPO"]


def docker_container_names() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout
    return [n for n in out.splitlines() if n]


def seed_one_run(conn: psycopg.Connection) -> int:
    repo_url = f"https://github.com/{GITHUB_REPO}.git"
    repo = conn.execute(
        "SELECT id, ecosystem, sandbox_image FROM repos WHERE url = %s", (repo_url,)
    ).fetchone()
    if repo:
        repo_id, ecosystem, sandbox_image = repo["id"], repo["ecosystem"], repo["sandbox_image"] or "node:20"
    else:
        ecosystem = "npm"
        sandbox_image = "node:20"
        repo_id = conn.execute(
            """
            INSERT INTO repos (url, default_branch, ecosystem, build_cmd, test_cmd, sandbox_image)
            VALUES (%s, 'main', 'npm', 'npm run build', 'npm test', %s) RETURNING id
            """,
            (repo_url, sandbox_image),
        ).fetchone()["id"]

    dep = conn.execute(
        "SELECT id FROM dependencies WHERE repo_id = %s AND name = 'debug'", (repo_id,)
    ).fetchone()
    dep_id = dep["id"] if dep else conn.execute(
        """
        INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path)
        VALUES (%s, 'debug', 'npm', 'unknown', 'package.json') RETURNING id
        """,
        (repo_id,),
    ).fetchone()["id"]

    candidate_id = conn.execute(
        """
        INSERT INTO candidates (dependency_id, target_version, semver_jump, status)
        VALUES (%s, 'latest', 'unknown', 'new') RETURNING id
        """,
        (dep_id,),
    ).fetchone()["id"]

    checkpoint = {
        "repo_url": repo_url,
        "manifest_path": "package.json",
        "dep_name": "debug",
        "current_version": "unknown",
        "target_version": "latest",
        "sandbox_image": sandbox_image,
        "ecosystem": ecosystem,
    }
    return conn.execute(
        """
        INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at)
        VALUES (%s, 'CREATED', %s::jsonb, now()) RETURNING id
        """,
        (candidate_id, json.dumps(checkpoint)),
    ).fetchone()["id"]


def main() -> None:
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    run_id = seed_one_run(conn)
    print(f"[1] seeded run {run_id}")

    worker = subprocess.Popen(
        [sys.executable, "-m", "worker.main", "verify-sandbox-worker"],
        cwd=ROOT_DIR, stdout=open("/tmp/verify_sandbox_worker.log", "w"), stderr=subprocess.STDOUT,
    )
    print(f"[2] started worker pid={worker.pid}, waiting for its sandbox container...")

    container_name = f"upgrade-{run_id}-0-install"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if container_name in docker_container_names():
            break
        time.sleep(0.5)
    else:
        print("FAIL: container never appeared -- check /tmp/verify_sandbox_worker.log")
        worker.kill()
        conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
        sys.exit(1)

    image = subprocess.run(
        ["docker", "inspect", container_name, "--format", "{{.Config.Image}}"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"[3] container {container_name} is running, image={image}")
    ok = image == "node:20"
    print(f"=== {'PASS' if ok else 'FAIL'}: sandbox_image was {'correctly' if ok else 'NOT'} read from checkpoint ===")

    print("\n=== cleanup ===")
    worker.kill()
    subprocess.run(["docker", "kill", container_name], capture_output=True)
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    print(f"deleted test run {run_id}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
