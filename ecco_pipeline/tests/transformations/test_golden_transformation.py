"""
Golden-output characterization test for the transformation (regridding) step.

Unlike the other transformation tests, this one does NOT mock pyresample, pyproj,
or netCDF I/O. It runs the *real* regrid of a NASA-SSH granule onto the ECCO_llc90
grid and compares the interpolated field against a committed golden reference that
was produced by the pre-upgrade dependency stack.

Purpose: detect numerical / qualitative drift when bumping numpy, xarray,
pyresample, pyproj, or netcdf4. The rest of the suite mocks these libraries out, so
it cannot see a change in resampled values — this test is the oracle that can.

- PASS  -> the regridded values match the baseline within tolerance; the bump is
           numerically safe for this dataset/grid.
- FAIL  -> a dependency bump changed pipeline output. That is signal, not a flake:
           investigate before merging (see DEPENDENCY_MODERNIZATION_PLAN.md).

Regenerating the baseline (only when an output change is *intended* and reviewed):
run the pipeline transform for NASA_SSH_REF_SIMPLE_GRID_V11 on the input fixture and
replace golden_ECCO_llc90_ssha_*.nc.

Note: slower than the other unit tests (computes real resampling factors).
"""

import os

import numpy as np
import pytest
import xarray as xr
import yaml

from transformations import grid_transformation
from transformations.grid_transformation import Transformation

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "..", "fixtures", "golden_transformation")

INPUT_FILE = os.path.join(_FIXTURES, "input_NASA-SSH_alt_ref_simple_grid_v1_1_20260601.nc")
GOLDEN_FILE = os.path.join(_FIXTURES, "golden_ECCO_llc90_ssha_NASA-SSH_alt_ref_simple_grid_v1_1_20260601.nc")
CONFIG_FILE = os.path.join(_HERE, "..", "..", "conf", "ds_configs", "NASA_SSH_REF_SIMPLE_GRID_V11.yaml")
GRID_FILE = os.path.join(_HERE, "..", "..", "grids", "ECCO_llc90.nc")

# Matches the Solr `date_dt` string the pipeline feeds a real transform job (incl.
# the trailing "Z"), so we reproduce the baseline exactly.
GRANULE_DATE = "2026-06-01T00:00:00Z"
OUTPUT_VAR = "ssha_interpolated_to_ECCO_llc90"

# Large sentinel written by the transform in place of NaN (nc4 default fill for f4).
# Normalize it back to NaN so the comparison is independent of how the on-disk
# golden encoded empty cells.
_FILL_THRESHOLD = 1e30

pytestmark = pytest.mark.skipif(
    not (os.path.exists(INPUT_FILE) and os.path.exists(GOLDEN_FILE)),
    reason="golden_transformation fixtures not present",
)


def _normalize(da: xr.DataArray) -> np.ndarray:
    """Empty/fill/inf cells -> NaN, so both sides use one representation for 'no data'."""
    return da.where(np.isfinite(da) & (da < _FILL_THRESHOLD)).values


@pytest.fixture
def config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def test_nasa_ssh_regrid_matches_golden(config, tmp_path, monkeypatch):
    # Send the pyresample factor cache to an isolated temp dir: forces a fresh
    # computation and keeps the real OUTPUT_DIR untouched.
    monkeypatch.setattr(grid_transformation, "OUTPUT_DIR", str(tmp_path))

    grid_ds = xr.open_dataset(GRID_FILE)
    golden = xr.open_dataset(GOLDEN_FILE)

    T = Transformation(config, INPUT_FILE, GRANULE_DATE)
    ds = T.load_file(INPUT_FILE)
    factors = T.make_factors(grid_ds)
    results = T.transform(grid_ds, factors, ds, fields=T.fields)

    # Config defines one field (ssha) -> one (dataset, success, error) tuple.
    assert len(results) == 1
    field_ds, success, error_message = results[0]
    assert success, f"transform reported failure: {error_message}"
    assert OUTPUT_VAR in field_ds.data_vars

    result_da = field_ds[OUTPUT_VAR]
    golden_da = golden[OUTPUT_VAR]

    # Structure must line up (shape/dim-names of the ECCO llc90 output).
    assert result_da.dims == golden_da.dims
    assert result_da.shape == golden_da.shape

    # Averaging-period center time (a daily granule is centered at noon). Asserted
    # separately so a time-coord regression is not mistaken for a value regression.
    np.testing.assert_array_equal(result_da["time"].values, golden_da["time"].values)

    # The scientific assertion: the regridded values themselves. Compared on raw
    # arrays (attrs like interpolation_date are non-deterministic and ignored).
    # NaNs in matching cells are treated as equal.
    np.testing.assert_allclose(
        _normalize(result_da),
        _normalize(golden_da),
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    )
