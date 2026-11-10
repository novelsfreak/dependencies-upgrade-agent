# bench.py
#
# Week 5 Day 6: the frozen benchmark. "Frozen" here means pinned repo
# state (local git fixtures, single commit each, never touched after
# creation -- see bench/fixtures.json for their exact SHAs) and pinned
# dependency versions (exact current->target strings), with the
# changelog itself vendored (bench/vendored_changelogs/*.json, fetched
# once for real and replayed from disk from then on -- see agent/loop.py's
# vendored_changelog_text checkpoint key) so a benchmark run's numbers
# reflect this project's own code, not a live changelog host's latency
# or content drifting between runs.
#
# What this is NOT, and says so rather than pretending otherwise: the
# plan's own "20 runs across many repos." Same real infrastructure
# constraint Week 4 Day 6 hit (a handful of real test repos, real Groq
# rate limits, real daily token caps) -- three real fixtures, run for
# real against the real worker, real Docker sandbox, real Groq API.
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from core.claim import claim_specific, release
from core.heartbeat_guard import LeaseLostError
from core.repo_lock import LockContention
from core.states import HANDLERS
from agent.context.report import cache_hit_rate_for_run
from agent.context.retrieval import search_before_read_compliance
from core.taxonomy import classify_run
from agent.messages import total_cost_cents
from worker.main import handle_failure

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")
FIXTURES_PATH = ROOT / "bench" / "fixtures.json"
VENDORED_DIR = ROOT / "bench" / "vendored_changelogs"
RESULTS_PATH = ROOT / "bench" / "results.jsonl"

TERMINAL_STATES = {"PATCH_READY", "PR_OPEN", "AWAITING_CI", "MERGED_READY", "ESCALATED", "FAILED", "SKIPPED"}


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip() or "uncommitted"


def _load_fixtures() -> list[dict]:
    return json.loads(FIXTURES_PATH.read_text())


def _vendored_changelog(dep_name: str) -> dict | None:
    path = VENDORED_DIR / f"{dep_name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def seed_run(conn: psycopg.Connection, fixture: dict) -> tuple[int, int, int, int]:
    repo = conn.execute("SELECT id FROM repos WHERE url = %s", (fixture["repo_url"],)).fetchone()
    if repo is None:
        repo = conn.execute(
            "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
            "(%s, 'npm', 'npm run build', 'npm test') RETURNING id",
            (fixture["repo_url"],),
        ).fetchone()
    repo_id = repo["id"]

    dep = conn.execute(
        "SELECT id FROM dependencies WHERE repo_id = %s AND name = %s", (repo_id, fixture["dep_name"])
    ).fetchone()
    if dep is None:
        dep = conn.execute(
            "INSERT INTO dependencies (repo_id, name, ecosystem, current_version, manifest_path) "
            "VALUES (%s, %s, 'npm', %s, 'package.json') RETURNING id",
            (repo_id, fixture["dep_name"], fixture["current_version"]),
        ).fetchone()
    dep_id = dep["id"]

    candidate_id = conn.execute(
        "INSERT INTO candidates (dependency_id, target_version, semver_jump, status) "
        "VALUES (%s, %s, %s, 'new') RETURNING id",
        (dep_id, fixture["target_version"], fixture["semver_jump"]),
    ).fetchone()["id"]

    checkpoint = {
        "repo_url": fixture["repo_url"], "dep_name": fixture["dep_name"],
        "current_version": fixture["current_version"], "target_version": fixture["target_version"],
        "manifest_path": "package.json", "ecosystem": "npm", "semver_jump": fixture["semver_jump"],
        "bench_label": fixture["label"],
    }
    vendored = _vendored_changelog(fixture["dep_name"])
    if vendored:
        checkpoint["vendored_changelog_text"] = vendored["text"]
        checkpoint["vendored_changelog_source"] = vendored["source"]

    run_id = conn.execute(
        "INSERT INTO runs (candidate_id, state, checkpoint, next_attempt_at) "
        "VALUES (%s, 'CREATED', %s::jsonb, now()) RETURNING id",
        (candidate_id, json.dumps(checkpoint)),
    ).fetchone()["id"]
    conn.commit()
    return run_id, repo_id, dep_id, candidate_id


def advance_until_terminal(conn: psycopg.Connection, run_id: int, worker_id: str, timeout_seconds: int = 900) -> str:
    """
    Reuses the exact same claim/dispatch/release/failure machinery
    worker/main.py runs in production -- a benchmark that exercised
    different code than what actually ships would be measuring the
    wrong thing.
    """
    start = time.monotonic()
    while True:
        row = conn.execute("SELECT state FROM runs WHERE id = %s", (run_id,)).fetchone()
        conn.commit()
        if row["state"] in TERMINAL_STATES:
            return row["state"]
        if time.monotonic() - start > timeout_seconds:
            return "TIMEOUT"

        run = claim_specific(conn, run_id, worker_id)
        if run is None:
            time.sleep(1)
            continue
        try:
            handler = HANDLERS[run["state"]]
            next_state, checkpoint_delta = handler(run, conn, worker_id)
            if not checkpoint_delta.get("_released"):
                release(conn, run["id"], next_state, checkpoint_delta)
        except LeaseLostError:
            pass
        except LockContention:
            release(conn, run["id"], run["state"], next_attempt_at_sql="now() + interval '5 seconds'")
        except Exception as e:
            handle_failure(conn, run, e)


def collect_metrics(conn: psycopg.Connection, run_id: int) -> dict:
    row = conn.execute("SELECT state, checkpoint FROM runs WHERE id = %s", (run_id,)).fetchone()
    turns = conn.execute(
        "SELECT count(*) c FROM run_messages WHERE run_id = %s AND role = 'assistant'", (run_id,)
    ).fetchone()["c"]
    tokens_in = conn.execute(
        "SELECT COALESCE(SUM(tokens_in), 0) s FROM run_messages WHERE run_id = %s", (run_id,)
    ).fetchone()["s"]
    tokens_out = conn.execute(
        "SELECT COALESCE(SUM(tokens_out), 0) s FROM run_messages WHERE run_id = %s", (run_id,)
    ).fetchone()["s"]
    n_compactions = conn.execute(
        "SELECT count(*) c FROM compaction_summaries WHERE run_id = %s", (run_id,)
    ).fetchone()["c"]
    category, evidence = classify_run(conn, run_id)
    cache_hit_rate = cache_hit_rate_for_run(conn, run_id)
    retrieval = search_before_read_compliance(conn, run_id)
    return {
        "final_state": row["state"],
        "turns": turns,
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_cents": round(total_cost_cents(conn, run_id), 4),
        "compactions": n_compactions,
        "cache_hit_rate": round(cache_hit_rate, 4) if cache_hit_rate is not None else None,
        "search_before_read_compliance": retrieval["compliance_rate"],
        "taxonomy": category,
        "taxonomy_evidence": evidence,
    }


def _cleanup(conn: psycopg.Connection, run_id: int, dep_id: int, candidate_id: int) -> None:
    conn.execute("DELETE FROM compaction_summaries WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM run_messages WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM outbox WHERE run_id = %s", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = %s", (run_id,))
    conn.execute("DELETE FROM candidates WHERE id = %s", (candidate_id,))
    conn.execute("DELETE FROM dependencies WHERE id = %s", (dep_id,))
    conn.commit()


def run_benchmark(code_version: str) -> list[dict]:
    fixtures = _load_fixtures()
    sha = git_sha()
    results = []
    with psycopg.connect(DSN, autocommit=False, row_factory=dict_row) as conn:
        for fixture in fixtures:
            t0 = time.monotonic()
            run_id, repo_id, dep_id, candidate_id = seed_run(conn, fixture)
            worker_id = f"bench-{code_version}-{fixture['label']}"
            final_state = advance_until_terminal(conn, run_id, worker_id)
            wall_clock = time.monotonic() - t0

            metrics = collect_metrics(conn, run_id)
            metrics.update({
                "label": fixture["label"],
                "run_id": run_id,
                "wall_clock_seconds": round(wall_clock, 1),
                "code_version": code_version,
                "git_sha": sha,
                "timestamp": time.time(),
            })
            results.append(metrics)
            print(json.dumps(metrics, indent=2))
            _cleanup(conn, run_id, dep_id, candidate_id)

    with open(RESULTS_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def summarize(results: list[dict]) -> dict:
    turns = [r["turns"] for r in results]
    costs = [r["cost_cents"] for r in results]
    wall = [r["wall_clock_seconds"] for r in results]
    succeeded = [r for r in results if r["final_state"] in ("PATCH_READY", "PR_OPEN", "AWAITING_CI", "MERGED_READY")]
    return {
        "success_rate": len(succeeded) / len(results) if results else 0,
        "median_turns": statistics.median(turns) if turns else None,
        "median_cost_cents": statistics.median(costs) if costs else None,
        "median_wall_clock_seconds": statistics.median(wall) if wall else None,
        "total_cost_cents": round(sum(costs), 4),
    }


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "week5"
    results = run_benchmark(label)
    print("\n=== summary ===")
    print(json.dumps(summarize(results), indent=2))
