# Configuration files

## Dataset configs

Config file per dataset contains all information needed for a dataset to be run through the pipeline from harvesting through aggregation. See README in `ecco_pipeline/conf/ds_configs` for more information. `ecco_pipeline/conf/ds_configs/deprecated` contains configs for datasets no longer supported, typically because they have been supplanted by a newer version.

## grid_coverage.yaml

Declares which datasets are transformed to which target grids. Keyed by grid, because grids are the axis that changes: adding a grid is one new block here, rather than an edit to every dataset config that should not receive it.

```yaml
grids:
  ECCO_llc90:
    datasets: all
  ECCO_llc540:
    datasets:
      - AVHRR_OI-NCEI-L4-GLOB-v2.0
  TPOSE:
    datasets: all
    exclude_ecco_variables: [sea_ice]
```

Per grid:

- `datasets` is either the string `all` or an explicit list of dataset config names. An empty list is valid, for a grid declared but not yet populated.
- `exclude_ecco_variables` is optional, and drops datasets whose `ecco_variable` appears in it. This is where the sea ice / TPOSE rule lives, since TPOSE is a tropical Pacific grid.

Grid keys are a grid file's `name` global attribute, which is not necessarily its filename. See `ecco_pipeline/grids/README.md`.

### How it interacts with GRIDS and --grids_to_use

Three knobs compose, narrowing in order:

1. `GRIDS` in `global_settings.py` is which grids this deployment runs at all.
2. `--grids_to_use` filters that list for one run.
3. `grid_coverage.yaml` decides, per dataset, which of those remain.

A dataset is processed to the intersection. Coverage can never widen the set beyond what `GRIDS` and `--grids_to_use` enabled.

### Validation

The run aborts at startup, before any stage, if:

- a grid enabled for the run has no block here;
- a block names a grid with no file in `ecco_pipeline/grids/`;
- a block names a dataset with no config in `conf/ds_configs/`;
- `exclude_ecco_variables` names a value no dataset config uses.

That third check is what keeps a version bump honest. Bumping a dataset version moves the old config into `conf/ds_configs/deprecated/`, which makes any coverage entry naming it dangle in the same commit that adds the replacement, so a bump cannot quietly drop a grid.

### Adding and removing

A newly added dataset joins the `datasets: all` grids automatically, and the explicitly-listed grids not at all. Removing a dataset from a grid stops future processing but does **not** delete output already written or its Solr documents; the pipeline simply stops updating them. Cleaning those up is manual.

## global_settings.py

Script that contains some settings that are used globally throughout the pipeline, such as the location of the output directory. This file must be manually set up after cloning the repo.

`GRIDS` lists the grids this deployment runs. Every name in it needs a block in `grid_coverage.yaml`.