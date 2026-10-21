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
  (OOM/segfault) can leave a batch's docs at `success_b=false` / in-progress. No output
  is lost, because the expensive artifact — the output netCDF — is still written
  per-granule in the worker.
- Scope is transformation-only. The aggregation stage has the same worker-owns-Solr
  shape and is left as future work.

## Amendment: recovery after a hard kill

This ADR originally claimed the leftover `success_b=false` docs were "recovered on the
next run by the existing filesystem fallback (`need_to_transform`)" with "no new
reconciliation logic". **That was wrong**, and it cost a full re-run of a large job batch
before it was noticed.

`need_to_transform` is reached only via the `for/else` in `get_tx_jobs` — i.e. only when
**no** transformation doc exists. But `prepopulate_jobs` writes a doc for every
dispatched (grid, field) with `commitWithin=15s`, so on any run long enough to matter
those docs are committed and searchable before the kill. The next run finds them, routes
to `need_to_update`, sees `success_b=false`, and re-transforms — with every valid output
netCDF sitting on disk, untouched. A 25k-job run dying at the end redid all 25k.
`transformation_in_progress_b` was written in five places and read in none.

Two changes close the gap:

1. **`recoverable_from_disk` / `record_recovered`** — an unsuccessful doc whose output on
   disk is trustworthy is adopted via one bulk **atomic update against the existing doc
   id** (not `reconstruct_tx_solr_doc`, which mints a fresh uuid and would duplicate the
   doc). Guarded on `origin_checksum_s` still matching the granule: if it was
   re-harvested, the output came from the old source and must be rebuilt.
2. **Incremental flushing** — `execute_jobs` consumes `imap_unordered` through
   `drain_results`, recording results every `SOLR_FLUSH_INTERVAL`. `starmap` returned
   only after every job finished, so there was nothing to flush and a late crash
   discarded all earned status. The parent is still the sole writer and cadence is keyed
   to result count, not worker count, so this ADR's core property holds: Solr load stays
   decoupled from `--multiprocesses`.

### The trap: a failed transformation still writes a plausible output file

Trusting the filesystem requires more than "exists + newer than source + current
version". When `perform_mapping` throws, `Transformation.transform` substitutes
`make_empty_record(...)` and that all-fill-value record flows through the **same**
`save_netcdf` call, stamped with the **current** `transformation_version` and a fresh
mtime. Those three checks therefore pass for a file representing a genuine failure.

Accepting it would record `success_b=true` and permanently launder real failures into
successes — worse than the re-run bug it fixes, and invisible. So `usable_output_exists`
also rejects any output whose variable attrs carry an `empty_record_note` in
`FAILURE_RECORD_NOTES`. Note the asymmetry: the *field missing from source data* empty
record deliberately sets `mapping_success=True` and must stay **accepted**, or those
granules re-transform forever. The discriminator is the note text, not the emptiness.
`transformation_version` is a global attr while `empty_record_note` is a variable attr;
`_inspect_output_file` reads both in one open, since this runs once per candidate file.
