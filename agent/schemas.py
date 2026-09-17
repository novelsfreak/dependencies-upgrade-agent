# agent/schemas.py
#
# Descriptions are prompt engineering, not documentation -- each one
# says WHEN to reach for the tool, not just what it does. That's the
# sentence that actually changes model behavior.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in the repository matching a glob pattern (default: everything). "
                "Use this FIRST when you don't know the repo's layout yet -- cheaper than reading "
                "files speculatively. Capped at 200 results; narrow the glob if you hit the cap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "e.g. 'src/**/*.ts', 'package.json'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search file contents by regex across the repository. Use this to find every call "
                "site of a symbol BEFORE you change its signature or usage -- missing one is the "
                "most common way an upgrade patch is incomplete. Returns file:line:text, capped at "
                "50 matches with the true total reported. Prefer this over reading files one by one "
                "to hunt for usages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "a Python regex, e.g. \"require\\('uuid/v4'\\)\""},
                    "glob": {"type": "string", "description": "restrict to matching files, default everything"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a line range from one file (default: first 200 lines). Paths are relative to "
                "the repo root. Ranged on purpose -- do not assume you need the whole file; read the "
                "region search pointed you at."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "description": "1-indexed, default 1"},
                    "end": {"type": "integer", "description": "default 200, max 2000-line window"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff to the repository. The `diff` argument MUST be a standard "
                "unified diff exactly as `git diff` would print it (---/+++ file headers, @@ hunk "
                "headers, context lines). Do NOT use the OpenAI Codex \"*** Begin Patch\" format or "
                "any other patch convention -- this tool only accepts unified diff text. "
                "Every hunk header MUST include line numbers in the form "
                "'@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@' (e.g. '@@ -1,7 +1,7 @@') -- "
                "a bare '@@' with no numbers will always be rejected. Count OLD_START from 1 in "
                "the file content read_file already showed you; OLD_COUNT/NEW_COUNT are the number "
                "of context+changed lines on each side of the hunk. Validated "
                "with `git apply --check` before anything is touched -- either every hunk lands or "
                "none do. If it's rejected, the error tells you why; use read_file to see the "
                "file's ACTUAL current content before retrying rather than guessing at line numbers "
                "again. On success this commits immediately and returns the applied diff so you "
                "know exactly what the file looks like now."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diff": {
                        "type": "string",
                        "description": (
                            "a unified diff, i.e. the exact text output of `git diff` -- "
                            "starts with '--- a/<path>' and '+++ b/<path>' lines"
                        ),
                    },
                },
                "required": ["diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_build",
            "description": (
                "Install dependencies and build the repository, sandboxed. Returns a structured "
                "result: ok, or up to 6 errors with file/line/code/message/source-context and the "
                "TRUE total error count if there were more than 6. Call this after apply_patch to "
                "check your work -- don't assume a patch compiles."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the test suite, sandboxed. Same structured result shape as run_build. Call this "
                "only after run_build succeeds -- a broken build makes test failures meaningless."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "optional: run only tests matching this"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log",
            "description": (
                "Pull the full raw log for a build/test/install step, beyond the 6 errors run_build/"
                "run_tests already showed you. Use this ONLY when 6 errors genuinely aren't enough to "
                "see the pattern -- most of the time they are."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["install", "build", "test"]},
                    "offset": {"type": "integer", "description": "line offset, default 0"},
                    "limit": {"type": "integer", "description": "lines to return, default 200"},
                },
                "required": ["kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": (
                "Stop and hand this run back for a human to look at. Call this when you've made a "
                "genuine attempt and are stuck -- a peer dependency conflict, a migration that isn't "
                "documented, or the same fix failing repeatedly. This is not a failure on your part; "
                "continuing to retry a truly stuck run wastes money for no better outcome."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "specific -- what you tried, what's blocking you"},
                },
                "required": ["reason"],
            },
        },
    },
]
