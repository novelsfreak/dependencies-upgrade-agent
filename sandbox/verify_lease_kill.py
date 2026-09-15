# sandbox/verify_lease_kill.py
#
# Deterministic verification that HeartbeatGuard's kill_fn actually
# tears down a sandboxed container (not just the `docker run` CLI
# wrapper) the instant a run's lease is lost to another worker.
#
# Uses `sleep 120` instead of a real build/test command so the result
# depends only on the mechanism under test, not on real npm/container
# timing or flakiness. Run from the project root:
#
#   uv run python -m sandbox.verify_lease_kill
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from core.heartbeat_guard import HeartbeatGuard
from sandbox.executor import kill_sandbox, start_sandbox

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


def docker_container_names() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout
    return [n for n in out.splitlines() if n]


def main() -> None:
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)

    candidate = conn.execute("SELECT id FROM candidates LIMIT 1").fetchone()
    if not candidate:
        print("no candidates row exists to attach a test run to -- seed one first "
              "(e.g. run chaos.py once, or insert one by hand)")
        return

    run_id = conn.execute(
        """
        INSERT INTO runs (candidate_id, state, lease_owner, lease_expires_at, checkpoint)
        VALUES (%s, 'BUILDING', 'verify-worker-a', now() + interval '2 minutes', '{}'::jsonb)
        RETURNING id
        """,
        (candidate["id"],),
    ).fetchone()["id"]
    print(f"[1] seeded fake in-progress run {run_id}, owned by verify-worker-a")

    container_name = f"verify-lease-kill-{run_id}"
    proc = start_sandbox(image="node:20", workdir=Path("/tmp"), cmd=["sleep", "120"], name=container_name, network="none")
    print(f"[2] started long-running container {container_name}")
    time.sleep(1)
    print(f"    confirmed in docker ps: {container_name in docker_container_names()}")

    guard = HeartbeatGuard(conn, run_id, "verify-worker-a", kill_fn=lambda: kill_sandbox(container_name))
    with guard:
        time.sleep(2)
        print("[3] simulating another worker stealing the lease...")
        conn.execute("UPDATE runs SET lease_owner = 'someone-else' WHERE id = %s", (run_id,))

        print("[4] waiting up to 40s for HeartbeatGuard's next heartbeat() check to notice "
              "(checks happen every HEARTBEAT_INTERVAL_SECONDS=30s)...")
        try:
            proc.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            pass

    print(f"[5] guard.lease_lost = {guard.lease_lost}")
    still_running = container_name in docker_container_names()
    print(f"[6] container still in docker ps: {still_running}")

    ok = guard.lease_lost and not still_running
    print(
        f"\n=== {'PASS' if ok else 'FAIL'}: kill_fn "
        f"{'correctly killed the container on lease loss' if ok else 'did NOT tear the container down as expected'} ==="
    )

    subprocess.run(["docker", "kill", container_name], capture_output=True)
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    print(f"cleaned up test run {run_id}")


if __name__ == "__main__":
    main()
