-- 001_init.sql
-- Core tables for the dependency upgrade agent skeleton.
-- No LLM, no sandbox yet — just enough to prove durability.

create table repos (
    id               bigint generated always as identity primary key,
    url              text not null unique,
    default_branch   text not null default 'main',
    ecosystem        text not null,
    build_cmd        text not null,
    test_cmd         text not null,
    sandbox_image    text,
    egress_allowlist jsonb not null default '[]',
    created_at       timestamptz not null default now()
);

create table dependencies (
    id             bigint generated always as identity primary key,
    repo_id        bigint not null references repos (id),
    name           text not null,
    ecosystem      text not null,
    current_version text not null,
    manifest_path  text not null,
    created_at     timestamptz not null default now(),
    unique (repo_id, name)
);

create table candidates (
    id             bigint generated always as identity primary key,
    dependency_id  bigint not null references dependencies (id),
    target_version text not null,
    released_at    timestamptz,
    changelog_url  text,
    semver_jump    text,          -- 'patch' | 'minor' | 'major'
    status         text not null default 'new',
    created_at     timestamptz not null default now()
);

create table runs (
    id               bigint generated always as identity primary key,
    candidate_id     bigint not null references candidates (id),

    state            text not null,
    attempt          int not null default 0,

    lease_owner      text,
    lease_expires_at timestamptz,
    next_attempt_at  timestamptz not null default now(),

    checkpoint       jsonb not null default '{}',
    last_error       text,

    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

-- The claim query hits this every second. Without it you get a seq scan
-- under load and the whole durability story falls over on throughput.
create index runs_state_next_attempt_idx on runs (state, next_attempt_at);

create table outbox (
    id               bigint generated always as identity primary key,
    run_id           bigint not null references runs (id),
    kind             text not null,
    payload          jsonb not null,
    idempotency_key  text not null unique,
    published_at     timestamptz,
    attempts         int not null default 0,
    created_at       timestamptz not null default now()
);

-- Publisher polls unpublished rows; this index keeps that cheap too.
create index outbox_unpublished_idx on outbox (created_at) where published_at is null;
