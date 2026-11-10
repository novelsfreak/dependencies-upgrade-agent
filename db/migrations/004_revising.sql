-- Week 4 Day 2: the REVISING path starts a fresh, compacted
-- conversation per revision round rather than replaying the raw N-turn
-- history forever -- cheaper, and per the plan's own guidance usually
-- gets to green faster than continuing a long stale conversation.
--
-- `revision` tags which round a message belongs to. unique(run_id, seq)
-- alone can't survive a fresh round restarting at seq 0, so it becomes
-- unique(run_id, revision, seq). Older revisions' rows are kept, not
-- deleted, for audit/debugging -- load_messages only ever looks at the
-- CURRENT revision (checkpoint->>'revision_count', kept in checkpoint
-- rather than a new runs column since checkpoint is already this
-- project's convention for per-run scratch state).
alter table run_messages add column revision int not null default 0;

alter table run_messages drop constraint run_messages_run_id_seq_key;
alter table run_messages add constraint run_messages_run_id_revision_seq_key unique (run_id, revision, seq);
