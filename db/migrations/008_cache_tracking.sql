-- Week 6 Day 4: prompt caching on Groq is fully automatic (no
-- cache_control param the way Anthropic's API needs -- confirmed
-- directly against console.groq.com/docs/prompt-caching, not
-- assumed), and reports its own effect via
-- usage.prompt_tokens_details.cached_tokens on every response. Storing
-- it per turn is what makes a real hit rate computable across a run
-- (or a whole benchmark) rather than eyeballed from log lines.
alter table run_messages add column cached_tokens int;
