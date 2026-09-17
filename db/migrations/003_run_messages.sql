-- Week 3 Day 1: persistence for the agent conversation loop.
--
-- One row per turn (both the model's own message and each tool result
-- sent back to it), so a killed worker can rebuild `messages` from here
-- on resume (Week 4) instead of losing everything past the last
-- checkpoint. unique(run_id, seq) is what makes replaying idempotent --
-- writing the same turn twice is a no-op, not a duplicate.
create table run_messages (
    id          bigint generated always as identity primary key,
    run_id      bigint not null references runs (id),
    seq         int not null,
    role        text not null,          -- 'system' | 'user' | 'assistant' | 'tool'
    content     jsonb not null,         -- raw SDK message dict, as returned/sent
    tokens_in   int,
    tokens_out  int,
    cost_cents  numeric(10,4),
    created_at  timestamptz not null default now(),
    unique (run_id, seq)
);

create index run_messages_run_id_idx on run_messages (run_id, seq);
