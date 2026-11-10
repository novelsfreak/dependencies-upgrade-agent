"""
Thin wrapper around the GitHub REST API for the two things the
publisher needs: pushing a branch (via git, not this API) and creating
a PR, plus checking whether a PR already exists for a branch.

Deliberately minimal -- no retry logic here, no auth flows beyond a
static PAT. The publisher owns retry/backoff; this module just knows
how to make one HTTP call at a time and raise if it fails.
"""
from __future__ import annotations

import os

import requests

GITHUB_API = "https://api.github.com"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def find_open_pr(token: str, repo: str, head_branch: str) -> dict | None:
    """
    repo is "owner/name". head_branch is just the branch name (no
    owner prefix needed when checking your own repo's branches).

    This is the idempotency check on the PUBLISHER side, distinct from
    the idempotency_key on the outbox row. The outbox key stops the
    same INTENT from being recorded twice; this stops the same intent
    from producing two PRs if the publisher retries after a call whose
    result it never got to observe (e.g. it crashed between GitHub
    creating the PR and the publisher reading the response).
    """
    owner = repo.split("/")[0]
    resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/pulls",
        headers=_headers(token),
        params={"head": f"{owner}:{head_branch}", "state": "open"},
        timeout=30,
    )
    resp.raise_for_status()
    prs = resp.json()
    return prs[0] if prs else None


def create_pr(token: str, repo: str, head_branch: str, base_branch: str, title: str, body: str) -> dict:
    resp = requests.post(
        f"{GITHUB_API}/repos/{repo}/pulls",
        headers=_headers(token),
        json={"title": title, "head": head_branch, "base": base_branch, "body": body},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def list_check_runs_for_ref(token: str, repo: str, ref: str) -> list[dict]:
    """
    Week 4 Day 2. Every check run GitHub has recorded for one commit --
    a check SUITE (what the webhook payload carries) can bundle several
    check RUNS (one per CI job), and only the runs themselves carry the
    actual failure output. Deliberately NOT called from api/webhooks.py
    -- that handler has a hard 10-second budget and this is a network
    call; it's called later from core/states.py's handle_revising,
    which runs on the normal worker poll loop with no such constraint.
    """
    resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/commits/{ref}/check-runs",
        headers=_headers(token),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("check_runs", [])


def summarize_failed_check_runs(check_runs: list[dict]) -> str:
    """
    Each check run's `output.summary`/`output.text` is exactly what a
    CI job chose to report -- usually the actual error output truncated
    to something readable, which is exactly what the agent needs to
    diagnose a real failure rather than a bare "CI failed."
    """
    failed = [cr for cr in check_runs if cr.get("conclusion") not in ("success", "neutral", "skipped", None)]
    if not failed:
        return "CI reported the check suite as failed, but no individual check run's own conclusion says why."

    parts = []
    for cr in failed:
        name = cr.get("name", "unknown check")
        conclusion = cr.get("conclusion", "unknown")
        output = cr.get("output") or {}
        summary = (output.get("summary") or "").strip()
        text = (output.get("text") or "").strip()
        body = "\n".join(p for p in (summary, text) if p) or "(no output text provided by this check run)"
        parts.append(f"### {name} ({conclusion})\n{body}")
    return "\n\n".join(parts)


def post_comment(token: str, repo: str, issue_number: int, body: str) -> dict:
    """
    Week 4 Day 3. PR review threads and plain PR discussion both use
    the Issues comment endpoint on GitHub's API -- a pull request IS an
    issue, API-wise. issue_number is the PR number.
    """
    resp = requests.post(
        f"{GITHUB_API}/repos/{repo}/issues/{issue_number}/comments",
        headers=_headers(token),
        json={"body": body},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
