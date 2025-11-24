"""
Unit tests for ds_functions module (PreprocessingFuncs, PretransformationFuncs, PosttransformationFuncs).
All file I/O is mocked where appropriate.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr

from utils.processing_utils.ds_functions import (
    FuncNotFound,
    PreprocessingFuncs,
    PretransformationFuncs,
    PosttransformationFuncs,
)


class PreprocessingFuncsTestCase(unittest.TestCase):
    """Tests for the PreprocessingFuncs class."""

    def setUp(self):
        self.funcs = PreprocessingFuncs()

    def test_call_function_not_found(self):
        """Test that FuncNotFound is raised for unknown function."""
        with self.assertRaises(FuncNotFound):
            self.funcs.call_function("nonexistent_func", "/fake/path", [])

    @patch("xarray.open_dataset")
    def test_ATL20_V004_monthly(self, mock_open):
        """Test ATL20_V004_monthly preprocessing."""
        # Create mock datasets
        base_ds = xr.Dataset(
            {
                "grid_x": xr.DataArray([1, 2, 3]),
                "grid_y": xr.DataArray([4, 5, 6]),
                "crs": xr.DataArray(0),
                "other_var": xr.DataArray([7, 8, 9]),
            }
        )

        group_ds = xr.Dataset({"test_field": xr.DataArray([10, 11, 12])})

        def open_side_effect(path, **kwargs):
            if "group" in kwargs:
                return group_ds
            return base_ds

        mock_open.side_effect = open_side_effect

        # Create mock field
        mock_field = MagicMock()
        mock_field.name = "test_field"

        result = self.funcs.call_function("ATL20_V004_monthly", "/fake/path.nc", [mock_field])

        self.assertIn("grid_x", result)
        self.assertIn("grid_y", result)
        self.assertIn("crs", result)
        self.assertIn("test_field", result)
        self.assertNotIn("other_var", result)

    @patch("xarray.open_dataset")
    def test_nsidc_seaice_nt_extraction(self, mock_open):
        """Test nsidc_seaice_nt_extraction preprocessing."""
        base_ds = xr.Dataset({"base_var": xr.DataArray([1, 2, 3])})

        group_ds = xr.Dataset({"raw_nt_seaice_conc": xr.DataArray([0.5, 0.6, 0.7])})

        def open_side_effect(path, **kwargs):
            if "group" in kwargs:
                return group_ds
            return base_ds

        mock_open.side_effect = open_side_effect

        result = self.funcs.call_function("nsidc_seaice_nt_extraction", "/fake/path.nc", [])

        self.assertIn("base_var", result)
        self.assertIn("raw_nt_seaice_conc", result)


class PretransformationFuncsTestCase(unittest.TestCase):
    """Tests for the PretransformationFuncs class."""

    def setUp(self):
        self.funcs = PretransformationFuncs()

    def test_call_function_not_found(self):
        """Test that FuncNotFound is raised for unknown function."""
        ds = xr.Dataset()
        with self.assertRaises(FuncNotFound):
            self.funcs.call_function("nonexistent_func", ds)

    def test_call_functions_multiple(self):
        """Test calling multiple functions in sequence."""
        ds = xr.Dataset({"analysed_sst": xr.DataArray(np.array([280.0, 270.0, 290.0]), dims=["x"])})

        result = self.funcs.call_functions(["AVHRR_remove_ice_or_near_ice"], ds)

        self.assertTrue(np.isnan(result["analysed_sst"].values[1]))

    def test_AVHRR_remove_ice_or_near_ice_cold_temps(self):
        """Test AVHRR removes values colder than -0.5C."""
        ds = xr.Dataset(
            {
                "analysed_sst": xr.DataArray(
                    np.array([280.0, 272.0, 290.0]),  # 272K = -1.15C
                    dims=["x"],
                )
            }
        )

        result = self.funcs.AVHRR_remove_ice_or_near_ice(ds)

        self.assertFalse(np.isnan(result["analysed_sst"].values[0]))
        self.assertTrue(np.isnan(result["analysed_sst"].values[1]))
        self.assertFalse(np.isnan(result["analysed_sst"].values[2]))

    def test_AVHRR_remove_ice_or_near_ice_with_ice_fraction(self):
        """Test AVHRR removes values where sea ice is present."""
        ds = xr.Dataset(
            {
                "analysed_sst": xr.DataArray(np.array([280.0, 285.0, 290.0]), dims=["x"]),
                "sea_ice_fraction": xr.DataArray(np.array([0.0, 0.5, 0.0]), dims=["x"]),
            }
        )

        result = self.funcs.AVHRR_remove_ice_or_near_ice(ds)

        self.assertFalse(np.isnan(result["analysed_sst"].values[0]))
        self.assertTrue(np.isnan(result["analysed_sst"].values[1]))
        self.assertFalse(np.isnan(result["analysed_sst"].values[2]))

    def test_RDEFT4_remove_negative_values(self):
        """Test RDEFT4 removes negative values."""
        ds = xr.Dataset(
            {
                "sea_ice_thickness": xr.DataArray(np.array([1.5, -0.5, 2.0]), dims=["x"]),
                "lat": xr.DataArray([45.0, 50.0, 55.0], dims=["x"]),
                "lon": xr.DataArray([-120.0, -110.0, -100.0], dims=["x"]),
            }
        )

        result = self.funcs.RDEFT4_remove_negative_values(ds)

        self.assertFalse(np.isnan(result["sea_ice_thickness"].values[0]))
        self.assertTrue(np.isnan(result["sea_ice_thickness"].values[1]))
        self.assertFalse(np.isnan(result["sea_ice_thickness"].values[2]))
        # lat/lon should be unchanged
        self.assertEqual(result["lat"].values[1], 50.0)

    def test_mask_nsidc_seaice(self):
        """Test mask_nsidc_seaice masks based on QA flags."""
        ds = xr.Dataset(
            {
                "cdr_seaice_conc": xr.DataArray(
                    np.array([[0.5, 0.6, 1.5]]),  # 1.5 is out of range
                    dims=["y", "x"],
                ),
                "cdr_seaice_conc_stdev": xr.DataArray(np.array([[0.1, 0.1, 0.1]]), dims=["y", "x"]),
                "cdr_seaice_conc_qa_flag": xr.DataArray(
                    np.array([[0, 8, 0]]),  # 8 = no input data flag
                    dims=["y", "x"],
                ),
            }
        )

        result = self.funcs.mask_nsidc_seaice(ds)

        # First value should be preserved (no flags)
        self.assertFalse(np.isnan(result["cdr_seaice_conc"].values[0, 0]))
        # Second value should be NaN (flag 8 set)
        self.assertTrue(np.isnan(result["cdr_seaice_conc"].values[0, 1]))
        # Third value should be NaN (out of range)
        self.assertTrue(np.isnan(result["cdr_seaice_conc"].values[0, 2]))

    def test_GRACE_MASCON(self):
        """Test GRACE_MASCON masks land points."""
        ds = xr.Dataset(
            {
                "lwe_thickness": xr.DataArray(np.array([10.0, 20.0, 30.0]), dims=["x"]),
                "uncertainty": xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["x"]),
                "land_mask": xr.DataArray(
                    np.array([0.0, 1.0, 0.0]),  # Middle point is land
                    dims=["x"],
                ),
            }
        )

        result = self.funcs.GRACE_MASCON(ds)

        self.assertFalse(np.isnan(result["lwe_thickness"].values[0]))
        self.assertTrue(np.isnan(result["lwe_thickness"].values[1]))
        self.assertFalse(np.isnan(result["lwe_thickness"].values[2]))
        self.assertTrue(np.isnan(result["uncertainty"].values[1]))


class PosttransformationFuncsTestCase(unittest.TestCase):
    """Tests for the PosttransformationFuncs class."""

    def setUp(self):
        self.funcs = PosttransformationFuncs()

    def test_call_function_not_found(self):
        """Test that FuncNotFound is raised for unknown function."""
        da = xr.DataArray([1, 2, 3])
        with self.assertRaises(FuncNotFound):
            self.funcs.call_function("nonexistent_func", da)

    def test_call_functions_multiple(self):
        """Test calling multiple functions in sequence."""
        da = xr.DataArray(np.array([1.0, 2.0, 3.0]), attrs={"units": "m"})

        result = self.funcs.call_functions(["meters_to_cm"], da)

        self.assertEqual(result.attrs["units"], "cm")
        np.testing.assert_array_equal(result.values, [100.0, 200.0, 300.0])

    def test_meters_to_cm(self):
        """Test meters_to_cm conversion."""
        da = xr.DataArray(np.array([1.0, 2.5, 0.01]), attrs={"units": "m"})

        result = self.funcs.meters_to_cm(da)

        self.assertEqual(result.attrs["units"], "cm")
        np.testing.assert_array_almost_equal(result.values, [100.0, 250.0, 1.0])

    def test_kelvin_to_celsius(self):
        """Test kelvin_to_celsius conversion."""
        da = xr.DataArray(np.array([273.15, 283.15, 373.15]), attrs={"units": "K"})

        result = self.funcs.kelvin_to_celsius(da)

        self.assertEqual(result.attrs["units"], "Celsius")
        np.testing.assert_array_almost_equal(result.values, [0.0, 10.0, 100.0])

    def test_seaice_concentration_to_fraction(self):
        """Test seaice_concentration_to_fraction conversion."""
        da = xr.DataArray(np.array([0.0, 50.0, 100.0]), attrs={"units": "%"})

        result = self.funcs.seaice_concentration_to_fraction(da)

        self.assertEqual(result.attrs["units"], "1")
        np.testing.assert_array_almost_equal(result.values, [0.0, 0.5, 1.0])

    def test_MEaSUREs_fix_time(self):
        """Test MEaSUREs_fix_time corrects time bounds."""
        time_val = np.datetime64("2020-01-15T12:30:45.123456789", "ns")
        da = xr.DataArray(
            np.array([[1.0]]),
            dims=["time", "x"],
            coords={"time": [time_val], "time_start": ("time", [str(time_val)]), "time_end": ("time", [str(time_val)])},
        )

        result = self.funcs.MEaSUREs_fix_time(da)

        # time_start should be start of day
        expected_start = str(np.datetime64("2020-01-15", "ns"))
        self.assertEqual(str(result.time_start.values[0]), expected_start)

        # time_end should be start of next day
        expected_end = str(np.datetime64("2020-01-16", "ns"))
        self.assertEqual(str(result.time_end.values[0]), expected_end)


if __name__ == "__main__":
    unittest.main()
