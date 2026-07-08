# Transformation stage: parent owns all Solr I/O; workers are pure compute

## Status

accepted

## Context

The transformation stage regrids source **Granules** onto ECCO **Grids** using
process-based parallelism (`--multiprocesses`), because the regridding is CPU-bound.
Historically each worker was also its own Solr client: it pre-populated a
**Transformation** doc (write + hard commit), read that doc back to get its id, and
wrote per-field status. So Solr write/commit load scaled with worker count and could
not be batched — a full-archive reprocess with many workers took Solr down via a
commit / searcher-warming / connection storm.

## Decision

Make transformation workers **pure compute**: a worker loads a Granule, regrids it,
writes the output netCDF, and returns `TxResult` records — it makes **zero** Solr
calls. The **parent** (`TxJobFactory`) owns all Solr I/O and does it batched: one bulk
pre-populate write marking the batch in-progress (with **client-generated uuid** ids,
which removes the per-field read-back), then one bulk write recording results, flushed
with a single hard commit per dataset. Solr therefore sees exactly one writer
regardless of `--multiprocesses`. The harvest-quality (unprocessable-granule) check
also moves to the parent's job-planning, so workers only ever receive processable
Granules.

## Considered options

- **Keep per-worker writes and tune them** (connection pooling, `commitWithin`,
  retry/backoff). This is the simpler, conventional design and was in fact the
  quick-wins pass. Rejected as the *primary* mechanism because it only turns the storm
  down — its intensity still scales with worker count. This refactor supersedes it
  structurally for the transform path; the tuning remains as defense-in-depth.
- **Client-generated uuids vs Solr-generated ids.** Chosen uuids so the parent knows
  every doc id before dispatch and never has to read a doc back to find it. Valid
  because the collection's `uniqueKey` is the default schemaless string `id`.

## Consequences

- Solr load is decoupled from CPU parallelism; the commit storm is impossible by
  construction (one writer), not merely tuned down.
- Workers become unit-testable (assert they make no Solr calls).
- Status metadata is now batched rather than persisted per-granule, so a hard kill
  (OOM/segfault) can leave a batch's docs at `success_b=false` / in-progress. This is
  recovered on the next run by the existing filesystem fallback
  (`need_to_transform`: output file exists + newer than source + current version), and
  no output is lost because the expensive artifact — the output netCDF — is still
  written per-granule in the worker. No new reconciliation logic was added.
- Scope is transformation-only. The aggregation stage has the same worker-owns-Solr
  shape and is left as future work.
