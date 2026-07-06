# Changelog

All notable changes to ECCO-obs-pipeline are documented here.

Each release is named after a marine animal, in the spirit of the ocean data this pipeline serves.
Version numbers follow [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **MAJOR** — breaking changes (incompatible config or schema changes)
- **MINOR** — new features, backwards-compatible
- **PATCH** — bug fixes and small improvements

---

## [Unreleased]

---

## [v2.2.1] — 2026-07-06

### Bug Fixes

- **Dependencies**: pinned `netcdf4>=1.6.4` (was `>=1.6`), which the lock had resolved to 1.6.3. That wheel bundles libnetcdf 4.9.0, which noisily probes every variable for quantization attributes and makes HDF5 dump `can't locate attribute: '_Quantize…'` diagnostics on files not written with quantization. netcdf4 ≥ 1.6.4 ships libnetcdf ≥ 4.9.2 (and HDF5 1.14), silencing the diagnostics. The messages were harmless stderr noise — no output was affected — but cluttered pipeline logs.

---

## [v2.2.0] — 2026-07-06

### Bug Fixes

- **Transformation (radius bin averaging)**: fixed a bug in `find_mappings_from_source_to_target` where the within-radius mask was compared with `is True` against a NumPy array, which always evaluated `False`. As a result no source points were ever recorded as falling within a target cell's radius, and every transformation silently fell back to single nearest-neighbor resampling instead of averaging all source points within the cell radius. Regridded outputs will now differ (true radius bin averaging), so all non-deprecated dataset `t_version`s were bumped to force re-transformation (see Dataset Updates).

### Improvements

- **Tests**: added a golden-output characterization test for the transformation step (`test_golden_transformation.py`) that runs the real (un-mocked) NASA-SSH → ECCO_llc90 regrid and compares against a committed baseline. Guards against numerical drift when bumping numpy/xarray/pyresample/pyproj/netcdf4.
- **Tests**: regenerated the `golden_transformation` fixtures (both the input granule and the golden baseline) from the same post-fix pipeline run so they reflect the corrected radius bin-averaging output (see Bug Fixes). The input and golden must come from the same run to stay consistent.
- **Dependency management**: migrated from Conda (`environment.yml`) to `uv` + `pyproject.toml`. Runtime dependency pins are unchanged; `pytest` and `jupyter` moved to a `dev` dependency group. Install with `uv sync`; run with `uv run ...`.
- **Dependencies (Stage A)**: modernized the core scientific pins to their latest within the numpy 1.x era — `numpy` `>=1.26,<2`, `xarray` `>=2024.6`, `pyresample` `>=1.28,<2`, `pyproj` `>=3.6,<4`, `netcdf4` `>=1.6,<2`. Hard `==` pins are now abstract ranges in `pyproject.toml`, with exact versions recorded in `uv.lock`. Python stays 3.10; the numpy 2.x / Python 3.12 jump is deferred to Stage B.
- **CI**: added a `Tests` GitHub Actions workflow that runs `ruff` and `pytest` on pull requests to `main` (Python 3.10, via `uv`). Added `ruff` and `pre-commit` to the `dev` group and scoped `ruff` to Python sources (notebooks excluded, matching the pre-commit hook). The test suite now bootstraps a `conf/global_settings.py` from the committed template when absent, so tests run on a fresh checkout without manual setup.

### Dataset Updates

- **All non-deprecated datasets**: bumped `t_version` (one step per config) to force re-transformation following the radius bin-averaging bug fix (see Bug Fixes). This also invalidates the versioned factors cache so mappings are regenerated rather than reloaded from stale pickles.
- **AMSR-2_OSI-408**: Removed the `confidence_level` field, which is no longer included in the product.

### New Features

- **NASA_SSH_REF_SIMPLE_GRID_V11**: added processing support with a quicklook validation notebook.

---

## [v2.1.1] — 2026-05-22

### Bug Fixes

- **Dashboard**: All-time totals now exclude deprecated datasets.

### Improvements

- **Dashboard**: deprecated datasets excluded from counts, Solr outage banner, data-freshness caption, Recent Activity → Dataset Inspector cross-link, Compact view for the inspector, hidden Streamlit Deploy button, ECCO favicon added.

---

## [v2.1.0] — 2026-05-22

### Bug Fixes

- **RDEFT4**: updated CMR concept ID to `C3205181648-NSIDC_CPRD` and provider host to `data.nsidc.earthdatacloud.nasa.gov`; the previous `NSIDC_ECS` collection no longer returns granules.

- **ATL20_V004**: bug fix in validation notebooks for improper opening of sample granules

### Dataset Updates

- **ATL21_V004_daily/monthly**: updated CMR concept ID to `C3826284331-NSIDC_CPRD` and provider host to `data.nsidc.earthdatacloud.nasa.gov`; the previous `NSIDC_ECS` collection no longer returns granules. Also created validation notebooks.

- **ATL21_V003**: deprecated in favor of `ATL21_V004`

---

## [v2.0.0] — 2026-05-21 — _Architeuthis dux_

### Breaking Changes

- **Solr schema migration required**: `date_dt` field migration, removal of `descendants` fields, and addition of observability fields. Existing Solr cores must be updated before running — see Migration Notes below. ([#106](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/106))
- **Globally shared runtime config**: pipeline configuration is now managed via a shared runtime config object rather than passed through individual function calls. Any code or scripts that directly call internal pipeline functions will need to be updated. ([#69](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/69))

### New Features

- **Pipeline Status Dashboard**: new dashboard for monitoring pipeline run status ([#109](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/109))
- **Parallelized harvesting**: enumeration and downloads are now parallelized across all web-scraped harvesters, significantly reducing harvest time ([#103](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/103))
- **Unit test suites**: full test coverage added for the harvester, transformation, utility, and aggregation modules ([#94](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/94), [#95](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/95), [#96](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/96))
- **ATL21 support**: added harvesting and processing support for the ATL21 dataset
- **AVHRR ice removal**: added support for AVHRR-based sea ice removal during transformation
- **Dataset config templates**: added reusable templates to simplify adding new dataset configurations
- **Improved CLI**: better argument parsing configuration to allow bypassing the interactive menu more reliably ([#68](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/68))

### Dataset Updates

- **G10016**: updated to v3 ([#86](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/86))
- **G02202**: updated to v5 ([#80](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/80))
- **LOCEAN**: deprecated v8, implemented support for v10 ([#85](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/85))
- **Sea ice dataset**: updated to v6.3, deprecated v6.2 ([#77](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/77))
- **MASCON**: deprecated in favor of updated version ([#74](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/74))

### Improvements

- Improved handling of loss of Solr data to reduce re-processing on recovery ([#101](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/101))
- Sea ice processing updates ([#99](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/99))
- Full logging overhaul for cleaner, more consistent output ([#58](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/58))
- Reduced redundant work in ATL daily and GRACE mascon harvesters
- Pinned versions for compute-related packages for reproducibility
- Applied `ruff` code formatting across the codebase ([#75](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/75))
- Tweaked default pipeline run order when not using the interactive menu
- Added `CITATION.cff` for academic citation support ([#76](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/76))

### Bug Fixes

- Fixed stale `grid_file_path` variable in `grids_to_solr` ([#102](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/102))
- Fixed incorrect date window logic ([#97](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/97))
- Fixed incorrect netCDF path being written to Solr aggregation docs
- Fixed incorrect start and end dates in dataset records
- Fixed ATL daily harvesting bug
- Fixed projection error in grid handling ([#73](https://github.com/ECCO-GROUP/ECCO-obs-pipeline/pull/73))
- Fixed missing endianness in binary output
- Fixed extraction of daily files from monthly granules
- Fixed hemispherical descendants handling
- Fixed units bug for GRACE datasets (incorrect conversion to cm)
- Fixed single-processor aggregation crash
- Fixed issue where a large number of log files would slow down git commands
- Added filter to skip `icdrft` files during harvesting

### Migration Notes

Upgrading from v1.0.0 requires a **Solr schema change**. The easiest path forwrad is:

- Wipe Solr
- Rerun pipeline - the pipeline will pick up the local files and repopulate Solr

---

## [v1.0.0] — 2024-03-26 — _Aurelia aurita_

First official release of the ECCO obs pipeline.

Named after the moon jellyfish (_Aurelia aurita_) — like the pipeline, it gracefully glides through the ocean.

### Features

- Framework for harvesting and transforming datasets used as ECCO model inputs
- Data sources: PO.DAAC, NSIDC, OSISAF, iFremer
- Output formats: binary and netCDF, daily and monthly average
- Solr database integration for maintaining dataset state between pipeline runs
- CLI with `--dataset`, `--step`, `--grids_to_use`, `--log_level`, and `--menu` flags

---

[Unreleased]: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/compare/v2.2.1...HEAD
[v2.2.1]: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/compare/v2.2.0...v2.2.1
[v2.2.0]: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/compare/v2.1.1...v2.2.0
[v2.1.1]: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/compare/v2.1.0...v2.1.1
[v2.1.0]: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/compare/v2.0.0...v2.1.0
[v2.0.0]: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/compare/v1.0.0...v2.0.0
[v1.0.0]: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/releases/tag/v1.0.0
