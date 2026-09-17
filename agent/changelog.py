# agent/changelog.py
#
# Cascade, first hit wins: GitHub releases API -> CHANGELOG.md -> npm
# registry metadata. Real HTTP calls to public, unauthenticated
# endpoints (npm registry, GitHub's public API, raw.githubusercontent)
# -- no token needed, nothing to sandbox against.
from __future__ import annotations

import re

import requests

_MAX_CHANGELOG_CHARS = 8000
_REQUEST_TIMEOUT = 15


def _parse_version(v: str) -> tuple[int, ...]:
    """Best-effort semver-ish parse: strips a leading 'v' and any
    pre-release/build suffix, returns (major, minor, patch, ...)."""
    v = v.lstrip("vV").split("-")[0].split("+")[0]
    parts = []
    for p in v.split("."):
        m = re.match(r"\d+", p)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts) or (0,)


def _in_range(version: str, current: str, target: str) -> bool:
    try:
        v, c, t = _parse_version(version), _parse_version(current), _parse_version(target)
    except (ValueError, AttributeError):
        return False
    return c < v <= t


def _github_repo_for_npm_package(dep_name: str) -> tuple[str, str] | None:
    try:
        resp = requests.get(f"https://registry.npmjs.org/{dep_name}", timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    repo_url = (resp.json().get("repository") or {}).get("url", "")
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", repo_url)
    return (match.group(1), match.group(2)) if match else None


def _from_github_releases(
    owner: str, repo: str, current_version: str, target_version: str
) -> tuple[str, tuple[int, ...]] | None:
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/releases",
            params={"per_page": 100}, timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    releases = resp.json()
    if not isinstance(releases, list):
        return None

    in_range = [r for r in releases if _in_range(r.get("tag_name", ""), current_version, target_version)]
    if not in_range:
        return None

    in_range.sort(key=lambda r: _parse_version(r["tag_name"]))
    sections = [f"## {r['tag_name']}\n\n{r.get('body') or '(no notes)'}" for r in in_range]
    earliest = _parse_version(in_range[0]["tag_name"])
    return "\n\n".join(sections), earliest


def _from_changelog_md(
    owner: str, repo: str, current_version: str, target_version: str
) -> tuple[str, tuple[int, ...]] | None:
    for branch in ("main", "master"):
        try:
            resp = requests.get(
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/CHANGELOG.md",
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            return _extract_changelog_range(resp.text, current_version, target_version)
    return None


def _extract_changelog_range(
    text: str, current_version: str, target_version: str
) -> tuple[str, tuple[int, ...]] | None:
    """
    Keep-a-changelog / standard-version style headers: "## [X.Y.Z](...)"
    or plain "## X.Y.Z". Keeps any section whose version falls in
    (current, target].
    """
    header_re = re.compile(r"^##\s+\[?v?(\d+\.\d+\.\d+[^\]\s(]*)", re.MULTILINE)
    headers = list(header_re.finditer(text))
    sections = []  # (version_tuple, text)
    for i, m in enumerate(headers):
        version = m.group(1)
        if not _in_range(version, current_version, target_version):
            continue
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections.append((_parse_version(version), text[m.start():end].strip()))
    if not sections:
        return None

    # Oldest first, not document order (changelogs are newest-first):
    # the earliest post-current-version entries are the ones most
    # likely to contain the actual breaking change, and are what should
    # survive _MAX_CHANGELOG_CHARS truncation -- not the newest patch
    # notes. Verified against uuid: without this, the real breaking
    # change at v7.0.0 got truncated out in favor of v14.x bug fixes.
    sections.sort(key=lambda s: s[0])
    return "\n\n".join(s[1] for s in sections), sections[0][0]


def _from_npm_description(dep_name: str) -> str:
    try:
        resp = requests.get(f"https://registry.npmjs.org/{dep_name}", timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return "(no changelog available -- registry lookup failed)"
    description = data.get("description", "")
    readme = (data.get("readme") or "")[:2000]
    return f"{description}\n\n{readme}".strip() or "(no changelog available)"


def fetch_changelog(dep_name: str, current_version: str, target_version: str) -> tuple[str, str]:
    """
    Returns (text, source) where source is one of "github-releases",
    "changelog-md", or "npm-description" -- callers attach this as the
    trust-delimiter's source attribute.

    NOT simple "first hit wins" by source order: GitHub's Releases API
    "hitting" (non-empty) doesn't mean it's complete -- a repo can have
    Releases only for its recent history while CHANGELOG.md goes back
    further. Verified empirically against uuid: Releases only covers
    v11+, silently missing the actual breaking change at v7.0.0 that
    CHANGELOG.md documents. So both sources are tried, and whichever
    one's earliest covered version is CLOSEST to current_version wins
    -- that's the one with the most complete view of the actual gap.
    """
    repo = _github_repo_for_npm_package(dep_name)
    candidates: list[tuple[tuple[int, ...], str, str]] = []

    if repo:
        owner, name = repo
        releases = _from_github_releases(owner, name, current_version, target_version)
        if releases:
            text, earliest = releases
            candidates.append((earliest, text, "github-releases"))

        changelog = _from_changelog_md(owner, name, current_version, target_version)
        if changelog:
            text, earliest = changelog
            candidates.append((earliest, text, "changelog-md"))

    if candidates:
        candidates.sort(key=lambda c: c[0])
        _earliest, text, source = candidates[0]
        return text[:_MAX_CHANGELOG_CHARS], source

    return _from_npm_description(dep_name)[:_MAX_CHANGELOG_CHARS], "npm-description"
