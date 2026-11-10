"""
Unit tests for the Week 3 agent tools -- pure mechanics, no Groq calls,
no Docker. These are the tools the model calls; if these are wrong, no
amount of prompt engineering fixes it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from adapters.base import BuildError, StepResult
from agent.tools import _correlate_changelog, apply_patch, list_files, read_file, search, write_findings
from core.states import BuildFailed


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


def test_search_suggests_a_read_range_around_each_match(repo_dir):
    result = search(repo_dir, r"require\('uuid/v4'\)")
    assert "suggested read_file range:" in result


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


def test_apply_patch_is_idempotent_on_replay(repo_dir):
    # Week 4 Day 1: a resumed run may re-execute an apply_patch call
    # whose commit already landed before a crash cut off the tool
    # result (agent/loop.py's _pending_tool_calls repair path). The
    # second call must recognize that and no-op, not error out on a
    # diff that no longer matches the (already-changed) file.
    diff = """--- a/src/a.js
+++ b/src/a.js
@@ -1,3 +1,3 @@
 line1
-line2
+line2 modified
 line3
"""
    first = apply_patch(repo_dir, diff)
    assert "applied and committed" in first

    second = apply_patch(repo_dir, diff)
    assert "already applied" in second
    # Nothing changed the second time -- one commit, same file content.
    assert (repo_dir / "src" / "a.js").read_text() == "line1\nline2 modified\nline3\n"


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


# --- Week 4 Day 4: changelog correlation ------------------------------------

# Real shape (source and symbol), synthetic text -- kept network-
# independent for CI. Verified live against the ACTUAL uuid v3->v9
# changelog during development: this exact symbol ("v4") and exact
# error message shape correctly found and attached the real "v4()
# method...removed" breaking-change section, which is what this
# fixture text is modeled on.
_REAL_SHAPE_CHANGELOG = """## [7.0.0](https://github.com/uuidjs/uuid/compare/v3.4.0...v7.0.0)

### BREAKING CHANGES

- The default export, which used to be the v4() method, has been removed.
- Deep imports of the different uuid version functions are deprecated.
"""


def test_correlate_changelog_attaches_matching_section_for_real_error_shape():
    step_result = StepResult(
        status="failed", error_count=1,
        errors=[BuildError(
            file="src/order.js", line=1, col=None, code=None,
            message="Package subpath './v4' is not defined by exports", symbol="v4",
        )],
    )
    result = _correlate_changelog(BuildFailed(step_result), _REAL_SHAPE_CHANGELOG)
    assert "v4() method" in result
    assert "removed" in result


def test_correlate_changelog_dedupes_repeated_symbols():
    step_result = StepResult(
        status="failed", error_count=2,
        errors=[
            BuildError(file="a.js", line=1, col=None, code=None, message="m1", symbol="v4"),
            BuildError(file="b.js", line=2, col=None, code=None, message="m2", symbol="v4"),
        ],
    )
    result = _correlate_changelog(BuildFailed(step_result), _REAL_SHAPE_CHANGELOG)
    assert result.count("v4() method") == 1  # attached once, not once per error


def test_correlate_changelog_returns_empty_when_no_symbol_matches():
    step_result = StepResult(
        status="failed", error_count=1,
        errors=[BuildError(file="a.js", line=1, col=None, code=None, message="m", symbol=None)],
    )
    assert _correlate_changelog(BuildFailed(step_result), _REAL_SHAPE_CHANGELOG) == ""


def test_correlate_changelog_returns_empty_with_no_changelog_text():
    step_result = StepResult(
        status="failed", error_count=1,
        errors=[BuildError(file="a.js", line=1, col=None, code=None, message="m", symbol="v4")],
    )
    assert _correlate_changelog(BuildFailed(step_result), "") == ""


def test_write_findings_accepts_a_note_under_the_cap():
    result = write_findings("Changed: a.js. Tried and failed: nothing yet. Open: run build.")
    assert "findings saved" in result


def test_write_findings_rejects_an_oversized_note_without_truncating():
    oversized = "x " * 5000  # well over the 1200-token cap
    result = write_findings(oversized)
    assert "nothing was saved" in result
    assert "condense" in result.lower()
