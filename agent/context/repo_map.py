# agent/context/repo_map.py
#
# Week 6 Day 5: a cheap, cached repo map -- directory structure, entry
# points, where tests live, framework versions, the manifest. Fills the
# "repo_map" segment declared (but unused) since Week 5 Day 2. Generated
# once per repo and reused as long as the repo's structure hasn't
# changed, keyed by a structural hash rather than regenerated every run
# -- both to avoid redoing the work and because a stable, byte-identical
# repo_map matters for Week 6 Day 4's prompt-cache prefix.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

_IGNORED_DIR_NAMES = {".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".pytest_cache"}
_MAX_TREE_ENTRIES = 150
_TEST_DIR_MARKERS = ("test", "tests", "__tests__", "spec")


def _list_tree(repo_dir: Path) -> list[str]:
    paths = sorted(
        str(p.relative_to(repo_dir)) for p in repo_dir.rglob("*")
        if p.is_file() and not any(part in _IGNORED_DIR_NAMES for part in p.relative_to(repo_dir).parts)
    )
    return paths


def _find_test_locations(paths: list[str]) -> list[str]:
    hits = set()
    for p in paths:
        parts = Path(p).parts
        if any(marker in parts for marker in _TEST_DIR_MARKERS):
            hits.add(str(Path(*parts[:1])) if len(parts) > 1 else p)
        elif ".test." in p or ".spec." in p:
            hits.add(str(Path(p).parent))
    return sorted(hits)[:20]


def _manifest_summary(repo_dir: Path) -> str:
    pkg = repo_dir / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text())
        except (json.JSONDecodeError, OSError):
            return "package.json present but unparseable"
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        lines = [f"name: {data.get('name', '?')}"]
        if data.get("scripts"):
            lines.append("scripts: " + ", ".join(sorted(data["scripts"])))
        if data.get("main"):
            lines.append(f"entry point: {data['main']}")
        if deps:
            lines.append(f"{len(deps)} dependencies, including: " + ", ".join(sorted(deps)[:10]))
        return "\n".join(lines)
    return "no package.json found"


def _structural_hash(paths: list[str], manifest_text: str) -> str:
    payload = "\n".join(paths) + "\n---\n" + manifest_text
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def generate_repo_map(repo_dir: Path) -> tuple[str, str]:
    """Returns (map_text, structural_hash). Pure and deterministic --
    same repo tree + manifest always produces the same text, which is
    what makes caching it (get_or_build_repo_map below) safe."""
    paths = _list_tree(repo_dir)
    manifest = _manifest_summary(repo_dir)
    test_locations = _find_test_locations(paths)

    shown = paths[:_MAX_TREE_ENTRIES]
    tree_text = "\n".join(shown)
    if len(paths) > len(shown):
        tree_text += f"\n... ({len(paths)} files total, showing first {len(shown)})"

    map_text = f"""Repository structure ({len(paths)} files):
{tree_text}

Manifest:
{manifest}

Test files found in: {', '.join(test_locations) if test_locations else '(none detected)'}"""

    return map_text, _structural_hash(paths, manifest)


def get_or_build_repo_map(conn: psycopg.Connection, repo_id: int, repo_dir: Path) -> str:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT repo_map_cache, repo_map_structural_hash FROM repos WHERE id = %s", (repo_id,))
        row = cur.fetchone()

    map_text, structural_hash = generate_repo_map(repo_dir)
    if row and row["repo_map_structural_hash"] == structural_hash and row["repo_map_cache"]:
        return row["repo_map_cache"]

    conn.execute(
        "UPDATE repos SET repo_map_cache = %s, repo_map_structural_hash = %s WHERE id = %s",
        (map_text, structural_hash, repo_id),
    )
    conn.commit()
    return map_text
