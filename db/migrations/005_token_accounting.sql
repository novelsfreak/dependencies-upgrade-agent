-- Week 5 Day 1: per-segment, real-tokenizer token accounting. Every
-- persisted turn already knows its role; this adds which of the plan's
-- context segments it belongs to and how big it really is (tiktoken's
-- o200k_harmony encoding -- gpt-oss's actual tokenizer -- not
-- len(text)/4), so "which tokens carried weight" is queryable instead
-- of narrated.
alter table run_messages add column tokens_by_segment jsonb;
