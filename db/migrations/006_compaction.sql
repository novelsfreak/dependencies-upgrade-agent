-- Week 5 Day 3: a persisted, human-readable record of every compaction
-- event. "A bad compaction is a brutal failure to diagnose -- the
-- model starts behaving strangely six turns later and nothing in the
-- transcript explains why" (the plan's own words) -- so every
-- compaction is logged here, never only implied by what disappeared
-- from the rendered view. run_messages itself is NEVER modified by
-- compaction; this table is purely additive.
create table compaction_summaries (
    id               bigint generated always as identity primary key,
    run_id           bigint not null references runs (id),
    revision         int not null default 0,
    covers_seq_start int not null,
    covers_seq_end   int not null,
    summary_text     text not null,
    tokens_before    int not null,
    tokens_after     int not null,
    cost_cents       numeric(10,4) not null default 0,
    created_at       timestamptz not null default now()
);

create index compaction_summaries_run_id_idx on compaction_summaries (run_id, revision);
