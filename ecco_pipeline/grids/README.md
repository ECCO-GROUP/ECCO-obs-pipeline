# Grids

Contains netCDF files for supported target grids. Each grid must adhere to the following requirements:

- Must contain a `name` global attribute. Typically this is the same as the grid filename
- Must contain a `type` global attribute. Must be one of `llc` or `latlon`
- Must contain a variable containing the effective grid radius of the grid, ideally named `effective_grid_radius`