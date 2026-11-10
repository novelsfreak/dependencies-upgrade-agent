# agent/context/tokenizer.py
#
# Week 5 Day 1: real tokenizer counts, not len(text)/4 -- the plan's
# own explicit instruction, since the char-based heuristic this project
# used through Week 4 (agent/loop.py's chars_per_token, recalibrated
# every turn against Groq's own usage.prompt_tokens) drifts badly on
# JSON and code, which is most of what a tool result actually is.
#
# gpt-oss-120b's real tokenizer IS registered in tiktoken as
# "o200k_harmony" -- confirmed directly, not assumed:
# tiktoken.encoding_for_model("gpt-oss-120b") resolves to it with no
# guessing on our part (OpenAI's own harmony chat-format encoding for
# the gpt-oss family, which is what Groq hosts unmodified). This counts
# raw text/JSON, not the exact wire format with role wrapping and
# special tokens Groq's backend uses internally, so it's still an
# estimate of the true billed count -- just a vastly better-grounded
# one than a fixed chars-per-token guess.
from __future__ import annotations

import functools
import json

import tiktoken

_MODEL_NAME = "gpt-oss-120b"


@functools.lru_cache(maxsize=1)
def _encoder():
    return tiktoken.encoding_for_model(_MODEL_NAME)


def count_tokens(text: str) -> int:
    """
    disallowed_special=() is required, not decorative: tiktoken raises
    by default if the text contains anything that LOOKS like one of its
    special tokens (e.g. a literal "<|endoftext|>"-shaped substring) --
    a real risk here, since this tokenizes arbitrary changelog text and
    third-party build/test output, not text this project authored.
    """
    if not text:
        return 0
    return len(_encoder().encode(text, disallowed_special=()))


def count_message_tokens(content: object) -> int:
    return count_tokens(json.dumps(content, default=str))
