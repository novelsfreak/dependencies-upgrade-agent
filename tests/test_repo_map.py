"""Week 6 Day 5: the cached repo map, real filesystem, real Postgres."""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from agent.context.repo_map import generate_repo_map, get_or_build_repo_map

DSN = os.environ.get("DATABASE_URL", "postgresql://agent:agent@localhost:5431/agent")


@pytest.fixture
def real_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.js").write_text("module.exports = {};\n")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "index.test.js").write_text("assert(true);\n")
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "fixture", "main": "src/index.js",
        "scripts": {"build": "true", "test": "true"},
        "dependencies": {"chalk": "5.0.0"},
    }))
    return tmp_path


def test_generate_repo_map_finds_structure_manifest_and_tests(real_repo):
    map_text, structural_hash = generate_repo_map(real_repo)
    assert "src/index.js" in map_text
    assert "fixture" in map_text
    assert "test" in map_text.lower()
    assert isinstance(structural_hash, str) and len(structural_hash) == 16


def test_generate_repo_map_is_deterministic(real_repo):
    text1, hash1 = generate_repo_map(real_repo)
    text2, hash2 = generate_repo_map(real_repo)
    assert text1 == text2
    assert hash1 == hash2


def test_generate_repo_map_hash_changes_when_structure_changes(real_repo):
    _text1, hash1 = generate_repo_map(real_repo)
    (real_repo / "src" / "new_file.js").write_text("// new\n")
    _text2, hash2 = generate_repo_map(real_repo)
    assert hash1 != hash2


@pytest.fixture
def seeded_repo():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
    repo_id = conn.execute(
        "INSERT INTO repos (url, ecosystem, build_cmd, test_cmd) VALUES "
        "('test://repo-map-fixture', 'npm', 'npm run build', 'npm test') RETURNING id"
    ).fetchone()["id"]
    yield conn, repo_id
    conn.execute("DELETE FROM repos WHERE id = %s", (repo_id,))
    conn.close()


def test_get_or_build_repo_map_returns_identical_text_and_persists_it(seeded_repo, real_repo):
    """
    generate_repo_map itself (a local filesystem walk + one manifest
    read) is cheap enough that get_or_build_repo_map always recomputes
    it to check the structural hash -- what the cache actually buys is
    NOT skipping that recomputation, but guaranteeing the returned text
    is the persisted, byte-identical value rather than a fresh object
    that happens to look the same (the property Week 6 Day 4's prompt
    caching depends on), and avoiding a needless DB write when nothing
    changed.
    """
    conn, repo_id = seeded_repo

    first = get_or_build_repo_map(conn, repo_id, real_repo)
    row_after_first = conn.execute(
        "SELECT repo_map_cache, repo_map_structural_hash FROM repos WHERE id = %s", (repo_id,)
    ).fetchone()
    assert row_after_first["repo_map_cache"] == first

    second = get_or_build_repo_map(conn, repo_id, real_repo)
    row_after_second = conn.execute(
        "SELECT repo_map_cache, repo_map_structural_hash FROM repos WHERE id = %s", (repo_id,)
    ).fetchone()
    assert second == first
    assert row_after_second["repo_map_structural_hash"] == row_after_first["repo_map_structural_hash"]


def test_get_or_build_repo_map_regenerates_after_structural_change(seeded_repo, real_repo):
    conn, repo_id = seeded_repo
    first = get_or_build_repo_map(conn, repo_id, real_repo)
    (real_repo / "src" / "another.js").write_text("// x\n")
    second = get_or_build_repo_map(conn, repo_id, real_repo)
    assert first != second
