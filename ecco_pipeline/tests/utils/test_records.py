"""
Unit tests for records module (TimeBound, make_empty_record, save_netcdf).
All file I/O is mocked where appropriate.
"""

import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from utils.processing_utils.records import TimeBound, make_empty_record, save_netcdf, NETCDF_FILL_VALUE, DTYPE


class TimeBoundTestCase(unittest.TestCase):
    """Tests for the TimeBound class."""

    def test_timebound_daily_from_start(self):
        """Test daily time bounds computed from start date."""
        start = np.datetime64("2020-01-15", "ns")
        tb = TimeBound(rec_avg_start=start, period="AVG_DAY")

        expected_end = np.datetime64("2020-01-16", "ns")
        self.assertEqual(tb._end, expected_end)

        expected_center = start + (expected_end - start) / 2
        self.assertEqual(tb.center, expected_center)

        self.assertEqual(len(tb.bounds), 2)
        self.assertEqual(tb.bounds[0], start)
        self.assertEqual(tb.bounds[1], expected_end)

    def test_timebound_daily_from_end(self):
        """Test daily time bounds computed from end date."""
        end = np.datetime64("2020-01-16", "ns")
        tb = TimeBound(rec_avg_end=end, period="AVG_DAY")

        expected_start = np.datetime64("2020-01-15", "ns")
        self.assertEqual(tb._start, expected_start)

    def test_timebound_monthly_from_start(self):
        """Test monthly time bounds computed from start date."""
        start = np.datetime64("2020-01-01", "ns")
        tb = TimeBound(rec_avg_start=start, period="AVG_MON")

        expected_end = np.datetime64("2020-02-01", "ns")
        self.assertEqual(tb._end, expected_end)

    def test_timebound_monthly_from_end(self):
        """Test monthly time bounds computed from end date."""
        end = np.datetime64("2020-02-01", "ns")
        tb = TimeBound(rec_avg_end=end, period="AVG_MON")

        expected_start = np.datetime64("2020-01-01", "ns")
        self.assertEqual(tb._start, expected_start)

    def test_timebound_weekly(self):
        """Test weekly time bounds."""
        start = np.datetime64("2020-01-01", "ns")
        tb = TimeBound(rec_avg_start=start, period="AVG_WEEK")

        expected_end = np.datetime64("2020-01-08", "ns")
        self.assertEqual(tb._end, expected_end)

    def test_timebound_yearly(self):
        """Test yearly time bounds."""
        start = np.datetime64("2020-01-01", "ns")
        tb = TimeBound(rec_avg_start=start, period="AVG_YEAR")

        expected_end = np.datetime64("2021-01-01", "ns")
        self.assertEqual(tb._end, expected_end)

    def test_timebound_requires_exactly_one_date(self):
        """Test that exactly one of start or end must be provided."""
        # When both are provided, should raise ValueError
        start = np.datetime64("2020-01-01", "ns")
        end = np.datetime64("2020-01-02", "ns")
        with self.assertRaises(ValueError):
            TimeBound(rec_avg_start=start, rec_avg_end=end, period="AVG_DAY")

    def test_timebound_invalid_period(self):
        """Test that invalid period raises error."""
        start = np.datetime64("2020-01-01", "ns")
        with self.assertRaises(ValueError):
            TimeBound(rec_avg_start=start, period="INVALID")

    def test_timebound_center_calculation(self):
        """Test that center is correctly calculated as midpoint."""
        start = np.datetime64("2020-01-01T00:00:00", "ns")
        tb = TimeBound(rec_avg_start=start, period="AVG_DAY")

        # Center should be at noon (12 hours from start)
        center_hour = (tb.center - tb._start) / np.timedelta64(1, "h")
        self.assertEqual(center_hour, 12)


class MakeEmptyRecordTestCase(unittest.TestCase):
    """Tests for the make_empty_record function."""

    def get_mock_model_grid(self, shape=(10, 20)):
        """Create a mock model grid dataset."""
        y_dim, x_dim = shape
        XC = xr.DataArray(
            np.random.rand(y_dim, x_dim).astype(np.float32),
            dims=["j", "i"],
            attrs={"long_name": "longitude", "units": "degrees_east"},
        )
        YC = xr.DataArray(
            np.random.rand(y_dim, x_dim).astype(np.float32),
            dims=["j", "i"],
            attrs={"long_name": "latitude", "units": "degrees_north"},
        )

        return xr.Dataset({"XC": XC, "YC": YC, "j": np.arange(y_dim), "i": np.arange(x_dim)})

    def test_make_empty_record_shape(self):
        """Test that empty record has correct shape."""
        model_grid = self.get_mock_model_grid((10, 20))
        record_date = "2020-01-15"

        result = make_empty_record(record_date, model_grid)

        self.assertEqual(result.shape, (1, 10, 20))

    def test_make_empty_record_filled_with_nan(self):
        """Test that empty record is filled with NaN."""
        model_grid = self.get_mock_model_grid((5, 5))
        record_date = "2020-01-15"

        result = make_empty_record(record_date, model_grid)

        self.assertTrue(np.all(np.isnan(result.values)))

    def test_make_empty_record_has_time_coordinate(self):
        """Test that empty record has time coordinate."""
        model_grid = self.get_mock_model_grid()
        record_date = "2020-01-15"

        result = make_empty_record(record_date, model_grid)

        self.assertIn("time", result.coords)
        self.assertEqual(result.time.values[0], np.datetime64("2020-01-15", "ns"))

    def test_make_empty_record_has_time_bounds(self):
        """Test that empty record has time_start and time_end."""
        model_grid = self.get_mock_model_grid()
        record_date = "2020-01-15"

        result = make_empty_record(record_date, model_grid)

        self.assertIn("time_start", result.coords)
        self.assertIn("time_end", result.coords)

    def test_make_empty_record_has_spatial_coords(self):
        """Test that empty record has XC and YC coordinates."""
        model_grid = self.get_mock_model_grid()
        record_date = "2020-01-15"

        result = make_empty_record(record_date, model_grid)

        self.assertIn("XC", result.coords)
        self.assertIn("YC", result.coords)

    def test_make_empty_record_preserves_coord_attrs(self):
        """Test that XC/YC attributes are preserved."""
        model_grid = self.get_mock_model_grid()
        record_date = "2020-01-15"

        result = make_empty_record(record_date, model_grid)

        self.assertEqual(result.XC.attrs["long_name"], "longitude")
        self.assertEqual(result.YC.attrs["long_name"], "latitude")

    def test_make_empty_record_dtype(self):
        """Test that empty record uses correct dtype."""
        model_grid = self.get_mock_model_grid()
        record_date = "2020-01-15"

        result = make_empty_record(record_date, model_grid)

        self.assertEqual(result.dtype, DTYPE)


class SaveNetcdfTestCase(unittest.TestCase):
    """Tests for the save_netcdf function."""

    def get_mock_dataset(self):
        """Create a mock dataset for saving."""
        data = xr.DataArray(np.random.rand(1, 5, 5).astype(np.float32), dims=["time", "y", "x"], name="test_var")
        data = data.assign_coords(time=[np.datetime64("2020-01-15", "ns")], y=np.arange(5), x=np.arange(5))
        return data.to_dataset()

    def test_save_netcdf_creates_file(self):
        """Test that save_netcdf creates a file."""
        ds = self.get_mock_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_netcdf(ds, "test_output.nc", tmpdir)

            output_path = os.path.join(tmpdir, "test_output.nc")
            self.assertTrue(os.path.exists(output_path))

    def test_save_netcdf_creates_directory(self):
        """Test that save_netcdf creates output directory if needed."""
        ds = self.get_mock_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "path")
            save_netcdf(ds, "test_output.nc", nested_dir)

            output_path = os.path.join(nested_dir, "test_output.nc")
            self.assertTrue(os.path.exists(output_path))

    def test_save_netcdf_fills_nan(self):
        """Test that NaN values are replaced with fill value."""
        ds = self.get_mock_dataset()
        ds["test_var"].values[0, 0, 0] = np.nan

        with tempfile.TemporaryDirectory() as tmpdir:
            save_netcdf(ds, "test_output.nc", tmpdir)

            output_path = os.path.join(tmpdir, "test_output.nc")
            # Open with mask_and_scale=False to see raw fill values
            loaded = xr.open_dataset(output_path, mask_and_scale=False)

            self.assertAlmostEqual(loaded["test_var"].values[0, 0, 0], NETCDF_FILL_VALUE, places=5)
            loaded.close()

    def test_save_netcdf_with_dataarray(self):
        """Test that save_netcdf handles DataArray input."""
        da = xr.DataArray(np.random.rand(1, 5, 5).astype(np.float32), dims=["time", "y", "x"], name="test_var")
        da = da.assign_coords(time=[np.datetime64("2020-01-15", "ns")], y=np.arange(5), x=np.arange(5))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_netcdf(da, "test_output.nc", tmpdir)

            output_path = os.path.join(tmpdir, "test_output.nc")
            self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
