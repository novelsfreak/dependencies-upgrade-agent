-- 002_inbound_events.sql
-- Deferred from 001_init.sql on purpose: not needed until Day 6, when
-- webhooks start arriving. Splitting migrations by when a table is
-- actually needed, rather than front-loading everything on Day 1, also
-- doubles as an early proof that migrate.py correctly applies more than
-- one file in order.

create table inbound_events (
    id           bigint generated always as identity primary key,
    source       text not null,            -- 'github', future: others
    external_id  text not null,            -- GitHub's X-GitHub-Delivery header
    run_id       bigint references runs (id),
    payload      jsonb not null,
    processed_at timestamptz,
    created_at   timestamptz not null default now(),

    -- THE entire dedupe strategy. GitHub redelivers on failure/timeout
    -- and can send the same event more than once even without a prior
    -- failure. Without this constraint, processing an event twice could
    -- double-apply a state transition (e.g. move a run backward to
    -- PATCHING twice, wasting a retry attempt on nothing). With it, a
    -- duplicate INSERT is simply rejected by the database -- the
    -- webhook handler doesn't need its own logic to detect "have I seen
    -- this delivery before," Postgres enforces it unconditionally.
    unique (source, external_id)
);

create index inbound_events_unprocessed_idx on inbound_events (created_at) where processed_at is null;
