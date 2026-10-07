-- Week 6 Day 5: a cached repo map per repo, invalidated on structural
-- change (not regenerated every run against the same repo). Keyed by a
-- structural hash (file listing + manifest content) rather than a
-- timestamp -- a repo whose file tree and manifest haven't changed
-- gets the SAME map text back, which matters for prompt-cache byte
-- stability (Week 6 Day 4) just as much as for not wasting the work.
alter table repos add column repo_map_cache text;
alter table repos add column repo_map_structural_hash text;
