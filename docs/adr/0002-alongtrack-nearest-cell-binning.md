# Along-track transformation uses nearest-cell binning, not the pyresample radius core

## Status

accepted

## Context

Most **Datasets** arrive on a regular lon/lat grid that is static from **Granule** to
Granule, and the transformation stage maps them onto an ECCO **Grid** via
`find_mappings_from_source_to_target` — a target→source radius search built on pyresample's
kd-tree. Along-track altimetry (e.g. NASA-SSH) instead delivers 1-D observations along a
satellite ground track (per-sample `latitude`/`longitude`) whose geometry differs every
Granule. We needed to map that onto the same Grids.

## Decision

Along-track uses **source-centric nearest-cell binning** in a dedicated function
(`along_track_factors` in `transformation_utils.py`): assign each valid observation to its
single nearest Grid cell (`k=1`), then average the observations that land in each cell
(`nanmean`). It does **not** reuse `find_mappings_from_source_to_target`. It does reuse the
downstream averaging (`transform_to_target_grid`) by returning the same 3-tuple factor
shape, with `nearest_source_index_to_target_index_i = {}`.

Two properties of the binning are load-bearing and non-obvious:

- **Spherical (ECEF) nearest-neighbour, not planar lon/lat.** Both observations and cell
  centers are converted to 3-D unit vectors before the kd-tree query. A kd-tree on raw
  lon/lat degrees is wrong at the antimeridian (a 0.2°-apart pair reads as 359.8° apart),
  makes a degree-valued distance cap latitude-dependent, and distorts "nearest" by cos(lat)
  at high latitude. Pyresample does ECEF internally — which is why the grid path never had
  this bug and why hand-rolling a lon/lat kd-tree (as an early prototype did) is a trap.
- **Per-cell radius cap.** After an unbounded `k=1` query, observations farther than the
  matched cell's own radius (`effective_grid_radius` / `RAD` / `0.5*sqrt(rA)`, converted
  from meters to a unit-sphere chord) are discarded, so out-of-domain / over-land strays
  are dropped rather than smeared into a distant coastal cell.

## Considered options

- **Reuse the pyresample core (target→source radius-gather).** The original plan's premise
  ("the core is source-agnostic; a track is just a flat swath"). Rejected: it lets one
  observation land in multiple overlapping cells or none; it needs `source_grid_min_L` /
  `source_grid_max_L` spacing estimates feeding an inverse-square neighbour bound that
  assumes areal (2-D) source density and misfits a 1-D track; and it needs an
  `allow_nearest_neighbor=False` fallback plus a degenerate-granule guard. Nearest-cell
  binning is a clean partition that needs none of that and leaves point-free cells NaN by
  construction.

## Consequences

- Along-track factors are **not cacheable** — track geometry differs per Granule, so
  `make_factors` bypasses both the in-memory `_factors_cache` and the on-disk pickle for
  `source_type == along_track`, and `pregenerate_factors` is a no-op for it. (The
  per-worker in-memory cache is keyed by grid+hemi+t_version, which is identical across a
  collection's Granules; leaving it active would silently reuse the first Granule's factors
  for all subsequent Granules a worker handles.)
- The grid path is byte-for-byte unchanged; along-track is a `source_type` branch only.
- Switching binning algorithms later means reprocessing along-track datasets and bumping
  their `t_version`, so this is expensive to reverse.
