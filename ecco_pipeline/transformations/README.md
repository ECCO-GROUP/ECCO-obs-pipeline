# Transformations

## Transformation job factory
Determines which target grid/field combinations a single granule for a single dataset need to be transformed, either because the transformation has not yet happened, the harvested granule has been updated, or the transformation configuration has been modified. The harvest-quality (unprocessable-granule) check happens here during job planning, so workers only ever receive processable granules.

The factory (`TxJobFactory`) owns **all** Solr I/O and does it batched: one bulk write marks the batch in-progress (assigning each doc a client-generated `uuid`), a single bulk write records the results returned by the workers, and one hard commit flushes them per dataset. Solr therefore sees exactly one writer regardless of the worker count. See `docs/adr/0001-transformation-parent-owns-solr-io.md`.

Supports Python's multiprocessing to execute transformations in parallel (`--multiprocesses`, honored up to the machine's CPU count). Workers are **pure compute** — they make zero Solr calls (see below). Each worker caches opened grids and unpickled mapping factors per-process, so a given grid/factors set is loaded once and reused across the granules that worker handles, rather than re-read from disk per granule.

## Transformation

A grid transformation occurs for a single data granule to a single target grid for a single field. 

1. Make mapping factors (ie: mappings from source to target grid). The factors are the same 3-tuple shape regardless of source geometry, so steps 2–4 below are identical for both source types.

2. An arbitrary number of preprocessing functions can be applied to the data prior to transformation to the target grid. ex: masking flagged data

2. Make array of target shape with transformed (or reprojected) data values via `utils.processing_utils.transformation_utils.transform_to_target_grid()`

3. Apply arbitrary number of postprocessing functions to the data. ex: converting units

4. Metadata is set, the transformed netCDF is saved, and its checksum is computed. The worker returns a `TxResult` record for the parent to persist to Solr — the worker itself writes nothing to Solr.

If an error occurs during the transformation process, an "empty record" is saved instead. Worker exceptions are captured and returned as a per-granule status marker rather than being silently dropped, so one bad granule no longer aborts the rest of the batch and failures are surfaced in the run's `N of M granules failed` summary.

## Source geometry (`source_type`)

A dataset config declares its source geometry via `source_type`. The two paths differ **only** in how mapping factors are built (step 1); everything downstream is shared.

### `grid` (default)

The source is a regular lon/lat grid that is static from granule to granule. Factors are built via `generalized_grid_product()` → `find_mappings_from_source_to_target()` (a target→source radius search on pyresample's kd-tree). Because the synthetic source grid is identical for every granule of the collection, factors are **cached** — both in a per-worker in-memory cache and an on-disk pickle keyed by `grid + hemi + t_version` — and reused across granules and runs. `pregenerate_factors()` warms this cache once before dispatch so the workers don't stampede building the same factors.

### `along_track`

The source is a 1-D satellite ground track: per-sample `latitude`/`longitude` (configurable via `lat_var`/`lon_var`) whose geometry differs every granule. Factors are built via `along_track_factors()` using **nearest-cell binning** — each observation is assigned to its single nearest grid cell on the sphere and the cell value is the `nanmean` of the observations that landed in it. See [`docs/adr/0002-alongtrack-nearest-cell-binning.md`](../../docs/adr/0002-alongtrack-nearest-cell-binning.md) for why this replaces the pyresample radius core rather than reusing it, and for the two load-bearing details (spherical/ECEF nearest-neighbour, per-cell radius cap).

Because the track geometry differs every granule, along-track factors are **not cacheable**: `make_factors` bypasses both the in-memory cache and the on-disk pickle for `source_type == along_track` (the shared `grid + hemi + t_version` key would otherwise make every granule silently reuse the first granule's factors), and `pregenerate_factors()` is a no-op for it. Factors depend only on the coordinates, which are shared across a granule's fields, so one factor set still serves every field of the granule (like the grid path); NaN field *values* are handled downstream by `nanmean`, so per-field QC masking (a `pre_transformation`) drops out automatically. Along-track output is expected to be **sparse** (most cells empty on any given granule) and aggregates like any other gridded output, since the output grid shape is identical regardless of `source_type`.

Along-track config fields (see `conf/ds_configs/NASA_SSH_REF_ALONGTRACK_V11.yaml` for a worked example):

| field | meaning |
| --- | --- |
| `source_type: along_track` | selects the binning path |
| `lat_var` / `lon_var` | per-sample coordinate variable names (default `latitude`/`longitude`) |
| `allow_nearest_neighbor: false` | keep output honestly sparse — never fill point-free cells (defaults off for track, on for grid) |
| `mapping_operation: nanmean` | per-cell averaging op (nanmean when the source has NaN values) |