# Processing utils

Shared, mostly-stateless helpers used by the transformation stage (and, for a few,
aggregation). These modules hold the numerical and I/O primitives; the orchestration
that calls them lives in `transformations/` and `aggregations/`.

## `transformation_utils.py`

The mapping primitives — how source observations become values on a target grid.

- **`transform_to_target_grid(...)`** — the shared averaging core. Given the factor
  3-tuple, a source field, and the target shape, it produces the target-shaped array by
  applying the per-cell `operation` (`mean`/`nanmean`/`median`/`nanmedian`/`nearest`).
  Both source types feed this unchanged. `allow_nearest_neighbor` controls whether
  point-free cells may borrow a nearest source value (on for grid, off for along-track).
- **Grid path** — `generalized_grid_product()` synthesizes a regular lon/lat source grid
  from the config (`data_res`, `area_extent`, `dims`, `proj_info`), and
  `find_mappings_from_source_to_target()` builds the target→source radius factors on
  pyresample's kd-tree.
- **Along-track path** — `along_track_factors(lons, lats, grid_ds)` bins 1-D track
  observations to their nearest grid cell (ADR 0002). It converts both observations and
  cell centers to 3-D ECEF unit vectors before a `k=1` kd-tree query (correct at the
  antimeridian and high latitude, unlike a planar lon/lat tree), then discards points
  beyond the matched cell's own radius (`AlongTrackTarget` precomputes the cell tree and
  the meter→unit-sphere-chord radius cap). Returns the same 3-tuple shape as the grid
  path, with `nearest_source_index_to_target_index_i = {}`.

The **factor 3-tuple** every mapping function returns:
`(source_indices_within_target_radius_i, num_source_indices_within_target_radius_i,
nearest_source_index_to_target_index_i)` — respectively a `{cell -> source point indices}`
dict, a per-cell count vector, and a `{cell -> single nearest source index}` dict.
Indices reference the **full** (unraveled) source field array, since
`transform_to_target_grid` ravels `ds[field].values` and indexes it with them.

## `ds_functions.py`

Named, config-referenced functions applied to a dataset at three points in the pipeline.
Each is looked up by name from the relevant `pre_transformations` / `post_transformations`
/ `preprocessing` config list, so adding a per-dataset behavior means adding a method here
and naming it in the config — no changes to the transformation code.

- **`PreprocessingFuncs`** — run while opening irregular source files (e.g. extracting a
  netCDF group, merging variables) before anything else.
- **`PretransformationFuncs`** — run on the source `xr.Dataset` before mapping, typically
  to mask flagged/invalid data to NaN (e.g. `NASA_SSH_mask_nasa_flag`, the sea-ice QA
  maskers).
- **`PosttransformationFuncs`** — run on the transformed `xr.DataArray` after mapping,
  typically unit conversions or time fixes.

## `records.py`

Output-record construction and serialization: `make_empty_record()` builds a target-shaped
DataArray for a date/grid (the template every transformation fills and the fallback on
failure), `TimeBound` centers a record's time within its averaging period, and
`save_netcdf()` / `save_binary()` write the results.

## `llc_array_conversion.py`

Layout conversions for ECCO LLC (lat-lon-cap) grids — tiles ↔ faces ↔ compact
representations — used where the LLC tile layout must be reshaped.
