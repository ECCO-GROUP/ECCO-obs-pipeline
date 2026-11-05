# Grids

Contains netCDF files for supported target grids. Each grid must adhere to the following requirements:

- Must contain a `name` global attribute. Typically this is the same as the grid filename, but it does not have to be: the `name` attribute is the grid's identity everywhere in the pipeline, and the filename is only used to find the file. The attribute must be unique across all files here.
- Must contain a `type` global attribute. Must be one of `llc` or `latlon`
- Must contain a variable containing the effective grid radius of the grid, ideally named `effective_grid_radius`

These are enforced at startup by `utils/pipeline_utils/grid_registry.py`, which scans this directory and owns the name-to-path mapping. A grid file missing either attribute, declaring an unrecognized `type`, or claiming a `name` another file already uses aborts the run rather than being skipped.

## Adding a grid

1. Drop the netCDF file in this directory.
2. Add a block for it in `ecco_pipeline/conf/grid_coverage.yaml` naming the datasets it should receive. A grid with no block aborts the run when enabled, which is deliberate: without it a new grid would silently inherit every dataset, which is wrong for a grid meant for a handful.
3. Add its `name` to `GRIDS` in `global_settings.py`, and run the pipeline once with `--grids_to_solr` to register it.