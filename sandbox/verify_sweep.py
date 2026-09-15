# sandbox/verify_sweep.py
#
# Deterministic verification that sweep_orphaned_containers() actually
# kills containers whose run is no longer leased, and leaves alone
# containers whose run still holds a live lease.
#
# Run from the project root:
#   uv run python -m sandbox.verify_sweep
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

from sandbox.executor import start_sandbox
from worker.main import sweep_orphaned_containers

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
        print("no candidates row exists to attach test runs to -- seed one first "
              "(e.g. run chaos.py once, or insert one by hand)")
        return

    # Run A: lease already expired -- should be swept.
    orphan_run_id = conn.execute(
        """
        INSERT INTO runs (candidate_id, state, lease_owner, lease_expires_at, checkpoint)
        VALUES (%s, 'BUILDING', 'dead-worker', now() - interval '1 minute', '{}'::jsonb)
        RETURNING id
        """,
        (candidate["id"],),
    ).fetchone()["id"]

    # Run B: lease still live -- should survive.
    live_run_id = conn.execute(
        """
        INSERT INTO runs (candidate_id, state, lease_owner, lease_expires_at, checkpoint)
        VALUES (%s, 'BUILDING', 'live-worker', now() + interval '5 minutes', '{}'::jsonb)
        RETURNING id
        """,
        (candidate["id"],),
    ).fetchone()["id"]

    orphan_name = f"upgrade-{orphan_run_id}-0-install"
    live_name = f"upgrade-{live_run_id}-0-install"

    start_sandbox(image="node:20", workdir=Path("/tmp"), cmd=["sleep", "60"], name=orphan_name, network="none")
    start_sandbox(image="node:20", workdir=Path("/tmp"), cmd=["sleep", "60"], name=live_name, network="none")
    time.sleep(1)
    print(f"[1] started {orphan_name} (orphan) and {live_name} (live lease)")
    print(f"    both running: {orphan_name in docker_container_names()}, {live_name in docker_container_names()}")

    print("[2] running sweep_orphaned_containers()...")
    sweep_orphaned_containers(conn)
    time.sleep(1)

    names_after = docker_container_names()
    orphan_killed = orphan_name not in names_after
    live_survived = live_name in names_after

    ok = orphan_killed and live_survived
    print(f"[3] orphan killed: {orphan_killed}, live-leased survived: {live_survived}")
    print(f"\n=== {'PASS' if ok else 'FAIL'}: sweep {'correctly' if ok else 'incorrectly'} discriminated leased vs. orphaned ===")

    subprocess.run(["docker", "kill", orphan_name], capture_output=True)
    subprocess.run(["docker", "kill", live_name], capture_output=True)
    conn.execute("DELETE FROM runs WHERE id = ANY(%s)", ([orphan_run_id, live_run_id],))
    print(f"cleaned up test runs {orphan_run_id}, {live_run_id}")


if __name__ == "__main__":
    main()
