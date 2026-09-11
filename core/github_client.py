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
