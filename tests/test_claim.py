"""
Day 2 concurrency proof: 5 threads hammering claim() against 100 seeded
runs must never let two threads claim the same run.

Requires a live Postgres with migrations applied (see docker-compose.yml).
Run with: uv run pytest tests/test_claim.py -v
"""
import os
import threading

import psycopg
import pytest

from core.claim import claim

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")

N_RUNS = 100
N_WORKERS = 5


@pytest.fixture
def seeded_conn():
    """
    A fresh connection with 100 CREATED runs seeded, wired to a throwaway
    repo/dependency/candidate chain (foreign keys must be satisfied).
    Cleans up after itself so the test is repeatable.
    """
    conn = psycopg.connect(DSN, autocommit=True)

    repo_id = conn.execute(
        """
        INSERT INTO repos (url, ecosystem, build_cmd, test_cmd)
        VALUES ('test://claim-test-repo', 'npm', 'npm run build', 'npm test')
        RETURNING id
        """
    ).fetchone()[0]

    dep_id = conn.execute(
        """
        INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path)
        VALUES (%s, 'axios', 'npm', '1.6.2', 'package.json')
        RETURNING id
        """,
        (repo_id,),
    ).fetchone()[0]

    cand_id = conn.execute(
        """
        INSERT INTO candidates (dependency_id, target_version)
        VALUES (%s, '1.7.0')
        RETURNING id
        """,
        (dep_id,),
    ).fetchone()[0]

    run_ids = []
    with conn.cursor() as cur:
        for _ in range(N_RUNS):
            cur.execute(
                """
                INSERT INTO runs (candidate_id, state, next_attempt_at)
                VALUES (%s, 'CREATED', now())
                RETURNING id
                """,
                (cand_id,),
            )
            run_ids.append(cur.fetchone()[0])

    yield conn, run_ids

    # cleanup — cascade manually since we didn't add ON DELETE CASCADE
    conn.execute("DELETE FROM outbox WHERE run_id = ANY(%s)", (run_ids,))
    conn.execute("DELETE FROM runs WHERE id = ANY(%s)", (run_ids,))
    conn.execute("DELETE FROM candidates WHERE id = %s", (cand_id,))
    conn.execute("DELETE FROM dependencies WHERE id = %s", (dep_id,))
    conn.execute("DELETE FROM repos WHERE id = %s", (repo_id,))
    conn.close()


def test_no_double_claims(seeded_conn):
    _, run_ids = seeded_conn
    claimed_by: dict[int, list[str]] = {rid: [] for rid in run_ids}
    lock = threading.Lock()

    def worker_loop(worker_id: str):
        # own connection per thread — psycopg connections aren't thread-safe
        with psycopg.connect(DSN, autocommit=False) as conn:
            while True:
                row = claim(conn, worker_id)
                if row is None:
                    break
                with lock:
                    claimed_by[row["id"]].append(worker_id)

    threads = [
        threading.Thread(target=worker_loop, args=(f"worker-{i}",))
        for i in range(N_WORKERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # every run claimed, and claimed by exactly one worker
    never_claimed = [rid for rid, owners in claimed_by.items() if len(owners) == 0]
    double_claimed = {rid: owners for rid, owners in claimed_by.items() if len(owners) > 1}

    assert not never_claimed, f"{len(never_claimed)} runs were never claimed: {never_claimed}"
    assert not double_claimed, f"double claims detected: {double_claimed}"
