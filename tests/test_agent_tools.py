"""
Unit tests for the Week 3 agent tools -- pure mechanics, no Groq calls,
no Docker. These are the tools the model calls; if these are wrong, no
amount of prompt engineering fixes it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.tools import apply_patch, list_files, read_file, search


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.js").write_text("line1\nline2\nline3\n")
    (tmp_path / "src" / "b.js").write_text("const x = require('uuid/v4');\n")
    (tmp_path / "README.md").write_text("hello\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("should be ignored\n")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path)
    subprocess.run(
        ["git", "-c", "user.email=a@a.com", "-c", "user.name=a", "commit", "-q", "-m", "init"],
        cwd=tmp_path,
    )
    return tmp_path


def test_list_files_finds_real_files_and_skips_node_modules(repo_dir):
    result = list_files(repo_dir, "**/*.js")
    assert "src/a.js" in result
    assert "src/b.js" in result
    assert "junk.js" not in result


def test_read_file_returns_numbered_lines(repo_dir):
    result = read_file(repo_dir, "src/a.js")
    assert "1: line1" in result
    assert "2: line2" in result


def test_read_file_respects_range(repo_dir):
    result = read_file(repo_dir, "src/a.js", start=2, end=2)
    assert "line2" in result
    assert "line1" not in result
    assert "line3" not in result


def test_read_file_missing_file_suggests_alternative(repo_dir):
    result = read_file(repo_dir, "a.js")
    assert "No such file" in result
    assert "src/a.js" in result  # the suggestion


def test_read_file_path_containment_blocks_escape(repo_dir):
    result = read_file(repo_dir, "../../../etc/passwd")
    assert "outside the repository" in result


def test_search_finds_the_deep_import_pattern(repo_dir):
    result = search(repo_dir, r"require\('uuid/v4'\)")
    assert "src/b.js:1:" in result


def test_search_reports_true_count_when_no_matches(repo_dir):
    result = search(repo_dir, "nonexistent_pattern_xyz")
    assert "no matches" in result


def test_search_invalid_regex_reports_clearly(repo_dir):
    result = search(repo_dir, "[unclosed")
    assert "invalid regex" in result


def test_apply_patch_add_delete_modify(repo_dir):
    diff = """--- a/src/a.js
+++ b/src/a.js
@@ -1,3 +1,3 @@
 line1
-line2
+line2 modified
 line3
diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+new content
"""
    result = apply_patch(repo_dir, diff)
    assert "applied and committed" in result
    assert (repo_dir / "src" / "a.js").read_text() == "line1\nline2 modified\nline3\n"
    assert (repo_dir / "new.txt").read_text() == "new content\n"


def test_apply_patch_rejects_mismatched_context(repo_dir):
    diff = """--- a/src/a.js
+++ b/src/a.js
@@ -1,3 +1,3 @@
 this context does not match
-anything real
+so this should be rejected
 at all
"""
    result = apply_patch(repo_dir, diff)
    assert "did not apply" in result
    assert "read_file" in result  # points the model at recovery


def test_apply_patch_zero_context_hunk_falls_back_to_patch_binary(repo_dir):
    # Observed for real against gpt-oss-120b: it wrote a well-formed
    # unified diff (correct header, content matches the file exactly)
    # but with ZERO lines of surrounding context -- just the changed
    # line(s). `git apply` unconditionally rejects that, even with -C1
    # (confirmed: -C1 only relaxes how closely context must match, it
    # doesn't waive the requirement that some exists). Classic
    # patch(1) has no such requirement, which is exactly why it's the
    # third fallback here.
    diff = """--- a/src/b.js
+++ b/src/b.js
@@ -1,1 +1,1 @@
-const x = require('uuid/v4');
+const x = require('uuid');
"""
    result = apply_patch(repo_dir, diff)
    assert "applied and committed" in result
    assert (repo_dir / "src" / "b.js").read_text() == "const x = require('uuid');\n"


def test_apply_patch_bare_hunk_header_gets_specific_correction(repo_dir):
    # Observed for real against gpt-oss-120b: it repeatedly emitted a
    # bare "@@" with no line numbers, and git's own error ("No valid
    # patches in input") never told it what was actually wrong -- five
    # consecutive turns were burned on this exact mistake in one live
    # run before the loop's token budget ran out. This is the specific,
    # named error that's supposed to let it self-correct in one turn.
    diff = """--- a/src/b.js
+++ b/src/b.js
@@
-const x = require('uuid/v4');
+const { v4: x } = require('uuid');
"""
    result = apply_patch(repo_dir, diff)
    assert "hunk header is missing line numbers" in result
    assert "@@ -1,7 +1,7 @@" in result  # the example format, not the model's broken one
