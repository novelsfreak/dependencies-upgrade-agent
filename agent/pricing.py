# agent/pricing.py
#
# Verified against multiple third-party trackers (console.groq.com's own
# pricing page is JS-rendered and didn't yield a fetchable number) as of
# 2026-09-16. Re-check console.groq.com/docs/models if cost figures in
# checkpoint.cost_cents start looking implausible -- Groq's rates do move.
MODEL_PRICING_PER_MILLION = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}

DEFAULT_MODEL = "openai/gpt-oss-120b"


def compute_cost_cents(model: str, tokens_in: int, tokens_out: int) -> float:
    """
    tokens_out includes reasoning tokens -- gpt-oss-120b is a reasoning
    model and Groq bills completion_tokens as a whole, reasoning and
    visible content together (confirmed empirically: a 3-word answer
    consumed 142 of 156 completion tokens on internal reasoning).
    """
    pricing = MODEL_PRICING_PER_MILLION.get(model)
    if pricing is None:
        raise ValueError(f"no pricing configured for model {model!r} -- add it to MODEL_PRICING_PER_MILLION")
    dollars = (tokens_in / 1_000_000) * pricing["input"] + (tokens_out / 1_000_000) * pricing["output"]
    return dollars * 100
