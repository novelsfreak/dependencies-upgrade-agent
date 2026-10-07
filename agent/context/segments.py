# agent/context/segments.py
#
# Week 5's own segment vocabulary (weeks-5-6-plan.md, "Add a segment
# label to everything the assembler will eventually emit"): system,
# tools, repo_map, brief, working_set, tool_results, scratchpad -- plus
# "assistant", which is not one of the ContextAssembler's budgeted
# segments (the model's own tool-call turns travel paired with their
# tool result for compaction purposes -- you can never compact one side
# of a tool_call/tool_result pair without orphaning the other), but is
# tracked as its own line here because Day 1's own worked example
# expects to see it: "the slope will be nearly all tool results...
# everything else is flat." Without a separate bucket for the model's
# own text, there's nothing to compare that flatness against.
from __future__ import annotations

SEGMENTS = [
    "system",
    "tools",
    "repo_map",
    "brief",
    "working_set",
    "tool_results",
    "scratchpad",
    "assistant",
]

# Segments the ContextAssembler (Week 5 Day 2) actually budgets and
# applies an eviction/compaction policy to. "assistant" and "tools" are
# real token cost but aren't independently managed: assistant turns are
# compacted as part of their tool_results pair, and tool schemas are a
# fixed argument to every API call, not a message in the conversation.
BUDGETED_SEGMENTS = ["system", "repo_map", "brief", "working_set", "tool_results", "scratchpad"]
