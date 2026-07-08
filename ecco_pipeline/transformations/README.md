# Transformations

## Transformation job factory
Determines which target grid/field combinations a single granule for a single dataset need to be transformed, either because the transformation has not yet happened, the harvested granule has been updated, or the transformation configuration has been modified. The harvest-quality (unprocessable-granule) check happens here during job planning, so workers only ever receive processable granules.

The factory (`TxJobFactory`) owns **all** Solr I/O and does it batched: one bulk write marks the batch in-progress (assigning each doc a client-generated `uuid`), a single bulk write records the results returned by the workers, and one hard commit flushes them per dataset. Solr therefore sees exactly one writer regardless of the worker count. See `docs/adr/0001-transformation-parent-owns-solr-io.md`.

Supports Python's multiprocessing to execute transformations in parallel (`--multiprocesses`, honored up to the machine's CPU count). Workers are **pure compute** — they make zero Solr calls (see below). Each worker caches opened grids and unpickled mapping factors per-process, so a given grid/factors set is loaded once and reused across the granules that worker handles, rather than re-read from disk per granule.

## Transformation

A grid transformation occurs for a single data granule to a single target grid for a single field. 

1. Make mapping factors (ie: mappings from source to target grid) via `utils.processing_utils.transformation_utils.generalized_grid_product()` -> `utils.processing_utils.transformation_utils.find_mappings_from_source_to_target()`. These are cached on disk and get reused for future pipeline runs for a given dataset. 

2. An arbitrary number of preprocessing functions can be applied to the data prior to transformation to the target grid. ex: masking flagged data

2. Make array of target shape with transformed (or reprojected) data values via `utils.processing_utils.transformation_utils.transform_to_target_grid()`

3. Apply arbitrary number of postprocessing functions to the data. ex: converting units

4. Metadata is set, the transformed netCDF is saved, and its checksum is computed. The worker returns a `TxResult` record for the parent to persist to Solr — the worker itself writes nothing to Solr.

If an error occurs during the transformation process, an "empty record" is saved instead. Worker exceptions are captured and returned as a per-granule status marker rather than being silently dropped, so one bad granule no longer aborts the rest of the batch and failures are surfaced in the run's `N of M granules failed` summary.