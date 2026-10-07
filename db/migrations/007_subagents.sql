-- Week 6 Day 1: a sub-agent IS a run -- same state machine, same
-- claim/lease/crash-resume machinery worker/main.py already has,
-- rather than a parallel system built from scratch. parent_run_id
-- links it to the run that spawned it; (parent_run_id, task_type,
-- task_target) is the stable idempotency key the plan asks for ("a
-- sub-agent task must be identified by a stable key... same
-- idempotency reasoning as your outbox from week 1") -- re-spawning
-- after a crash looks this up before creating a new row.
alter table runs add column parent_run_id bigint references runs (id);
alter table runs add column task_type text;
alter table runs add column task_target text;

create unique index runs_subagent_key_idx on runs (parent_run_id, task_type, task_target)
    where parent_run_id is not null;
create index runs_parent_run_id_idx on runs (parent_run_id) where parent_run_id is not null;
