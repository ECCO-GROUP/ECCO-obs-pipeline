"""
Unit tests for aggregation module (Aggregation class).
All Solr calls and file I/O are mocked.
"""

import json
import unittest
from collections import defaultdict
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import xarray as xr
from baseclasses import Field

from aggregations.aggregation import Aggregation


class AggregationInitTestCase(unittest.TestCase):
    """Tests for Aggregation class initialization."""

    def get_base_config(self):
        """Return a base configuration for testing."""
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "2.0",
            "do_monthly_aggregation": True,
            "remove_nan_days_from_data": True,
            "skipna_in_mean": False,
            "data_time_scale": "daily",
            "original_dataset_title": "Test Dataset",
            "original_dataset_short_name": "TEST",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "Test Ref",
            "original_dataset_doi": "10.1234/test",
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "Sea Surface Height Anomaly",
                    "standard_name": "sea_surface_height_above_sea_level",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
        }

    def get_mock_grid(self):
        """Return a mock grid dictionary."""
        return {
            "grid_name_s": "TEST_GRID",
            "grid_path_s": "/path/to/grid.nc",
            "grid_type_s": "latlon",
        }

    def get_mock_field(self):
        """Return a mock Field object."""
        field = MagicMock(spec=Field)
        field.name = "ssha"
        field.long_name = "Sea Surface Height Anomaly"
        return field

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_basic_initialization(self, mock_query):
        """Test basic Aggregation initialization."""
        # Mock dataset metadata query
        mock_query.return_value = [
            {
                "start_date_dt": "2020-01-01T00:00:00Z",
                "end_date_dt": "2020-12-31T00:00:00Z",
                "data_time_scale_s": "daily",
            }
        ]

        config = self.get_base_config()
        grid = self.get_mock_grid()
        field = self.get_mock_field()
        year = "2020"

        agg = Aggregation(config, grid, year, field)

        self.assertEqual(agg.ds_name, "TEST_DATASET")
        self.assertEqual(agg.version, "2.0")
        self.assertEqual(agg.year, "2020")
        self.assertEqual(agg.field, field)
        self.assertEqual(agg.grid, grid)
        self.assertTrue(agg.do_monthly_aggregation)
        self.assertTrue(agg.remove_nan_days_from_data)
        self.assertFalse(agg.skipna_in_mean)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_initialization_sets_ds_meta(self, mock_query):
        """Test that dataset metadata is loaded."""
        mock_query.return_value = [
            {
                "start_date_dt": "2020-01-01T00:00:00Z",
                "data_time_scale_s": "daily",
            }
        ]

        config = self.get_base_config()
        agg = Aggregation(config, self.get_mock_grid(), "2020", self.get_mock_field())

        self.assertIsNotNone(agg.ds_meta)
        self.assertEqual(agg.ds_meta["data_time_scale_s"], "daily")

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_initialization_error_no_transformed_granules(self, mock_query):
        """Test that error is raised when no transformed granules exist."""
        # Return dataset metadata without start_date_dt
        mock_query.return_value = [{"data_time_scale_s": "daily"}]

        config = self.get_base_config()

        with self.assertRaises(Exception) as context:
            Aggregation(config, self.get_mock_grid(), "2020", self.get_mock_field())

        self.assertIn("No transformed granules to aggregate", str(context.exception))

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_str_method(self, mock_query):
        """Test string representation of Aggregation."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]

        config = self.get_base_config()
        agg = Aggregation(config, self.get_mock_grid(), "2020", self.get_mock_field())

        result = str(agg)
        self.assertIn("TEST_GRID", result)
        self.assertIn("ssha", result)
        self.assertIn("2020", result)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_transformations_dict_initialized(self, mock_query):
        """Test that transformations defaultdict is initialized."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]

        config = self.get_base_config()
        agg = Aggregation(config, self.get_mock_grid(), "2020", self.get_mock_field())

        self.assertIsInstance(agg.transformations, defaultdict)


class AggregationMakeEmptyDateTestCase(unittest.TestCase):
    """Tests for Aggregation.make_empty_date method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "1.0",
            "data_time_scale": "daily",
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "Sea Surface Height Anomaly",
                    "standard_name": "sea_surface_height_above_sea_level",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
        }

    def get_mock_grid(self):
        return {
            "grid_name_s": "TEST_GRID",
            "grid_path_s": "/path/to/grid.nc",
            "grid_type_s": "latlon",
        }

    def create_mock_model_grid(self):
        """Create a mock model grid dataset."""
        XC = np.linspace(-180, 180, 360)
        YC = np.linspace(-90, 90, 180)
        lon_grid, lat_grid = np.meshgrid(XC, YC)

        grid_ds = xr.Dataset(
            {
                "XC": (["j", "i"], lon_grid),
                "YC": (["j", "i"], lat_grid),
            }
        )
        return grid_ds

    @patch("aggregations.aggregation.records.make_empty_record")
    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_make_empty_date_daily(self, mock_query, mock_make_empty):
        """Test making empty date for daily data."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z", "data_time_scale_s": "daily"}]

        # Mock make_empty_record to return a DataArray with time_start and time_end
        mock_da = xr.DataArray(
            np.full((1, 180, 360), np.nan),
            dims=["time", "j", "i"],
            coords={
                "time": [np.datetime64("2020-01-15", "ns")],
                "time_start": (["time"], [np.datetime64("2020-01-15", "ns")]),
                "time_end": (["time"], [np.datetime64("2020-01-15", "ns")]),
            },
        )
        mock_make_empty.return_value = mock_da

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        model_grid_ds = self.create_mock_model_grid()
        result = agg.make_empty_date("2020-01-15", model_grid_ds, self.get_mock_grid())

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("time_bnds", result.coords)
        # time_start and time_end are dropped at the end of make_empty_date
        self.assertNotIn("time_start", result.coords)
        self.assertNotIn("time_end", result.coords)
        mock_make_empty.assert_called_once_with("2020-01-15", model_grid_ds)

    @patch("aggregations.aggregation.records.make_empty_record")
    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_make_empty_date_monthly(self, mock_query, mock_make_empty):
        """Test making empty date for monthly data."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z", "data_time_scale_s": "monthly"}]

        mock_da = xr.DataArray(
            np.full((1, 180, 360), np.nan),
            dims=["time", "j", "i"],
            coords={
                "time": [np.datetime64("2020-01-01", "ns")],
                "time_start": (["time"], [np.datetime64("2020-01-01", "ns")]),
                "time_end": (["time"], [np.datetime64("2020-01-01", "ns")]),
            },
        )
        mock_make_empty.return_value = mock_da

        config = self.get_base_config()
        config["data_time_scale"] = "monthly"
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        model_grid_ds = self.create_mock_model_grid()
        result = agg.make_empty_date("2020-01-01", model_grid_ds, self.get_mock_grid())

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("time_bnds", result.coords)
        # time_start and time_end are dropped at the end of make_empty_date
        self.assertNotIn("time_start", result.coords)
        self.assertNotIn("time_end", result.coords)


class AggregationGetFilepathsTestCase(unittest.TestCase):
    """Tests for Aggregation.get_filepaths method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "1.0",
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "Sea Surface Height Anomaly",
                    "standard_name": "sea_surface_height_above_sea_level",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
        }

    def get_mock_grid(self):
        return {
            "grid_name_s": "TEST_GRID",
            "grid_path_s": "/path/to/grid.nc",
        }

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_get_filepaths_single_file_per_date(self, mock_query):
        """Test getting filepaths when there's one file per date."""
        # First call for ds_meta, second for transformations, third for granule
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],
            [
                {
                    "date_s": "2020-01-15T00:00:00Z",
                    "transformation_file_path_s": "/path/to/file1.nc",
                    "pre_transformation_file_path_s": "/path/to/pre1.nc",
                }
            ],
            [{"id": "granule1"}],  # harvested metadata
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        filepaths = agg.get_filepaths()

        self.assertIn("2020-01-15T00:00:00Z", filepaths)
        self.assertEqual(len(filepaths["2020-01-15T00:00:00Z"]), 1)
        self.assertEqual(filepaths["2020-01-15T00:00:00Z"][0], "/path/to/file1.nc")

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_get_filepaths_multiple_files_per_date(self, mock_query):
        """Test getting filepaths when there are multiple files per date (ascending/descending)."""
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],
            [
                {
                    "date_s": "2020-01-15T00:00:00Z",
                    "transformation_file_path_s": "/path/to/asc.nc",
                    "pre_transformation_file_path_s": "/path/to/pre_asc.nc",
                },
                {
                    "date_s": "2020-01-15T00:00:00Z",
                    "transformation_file_path_s": "/path/to/desc.nc",
                    "pre_transformation_file_path_s": "/path/to/pre_desc.nc",
                },
            ],
            [{"id": "granule1"}],
            [{"id": "granule2"}],
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        filepaths = agg.get_filepaths()

        self.assertEqual(len(filepaths["2020-01-15T00:00:00Z"]), 2)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_get_filepaths_updates_transformations_list(self, mock_query):
        """Test that transformations list is updated."""
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],
            [
                {
                    "date_s": "2020-01-15T00:00:00Z",
                    "transformation_file_path_s": "/path/to/file1.nc",
                    "pre_transformation_file_path_s": "/path/to/pre1.nc",
                }
            ],
            [{"id": "granule1", "filename_s": "test.nc"}],
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        agg.get_filepaths()

        self.assertIn("ssha", agg.transformations)
        self.assertGreater(len(agg.transformations["ssha"]), 0)


class AggregationOpenAndConcatTestCase(unittest.TestCase):
    """Tests for Aggregation.open_and_concat method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "1.0",
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "Sea Surface Height Anomaly",
                    "standard_name": "sea_surface_height_above_sea_level",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
        }

    def get_mock_grid(self):
        return {
            "grid_name_s": "TEST_GRID",
        }

    def create_mock_dataset(self, has_nan=False):
        """Create a mock dataset."""
        data = np.random.rand(1, 10, 10) if not has_nan else np.full((1, 10, 10), np.nan)
        return xr.Dataset(
            {
                "ssha": (["time", "j", "i"], data),
                "time": [np.datetime64("2020-01-15", "ns")],
            }
        )

    @patch("xarray.open_dataset")
    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_open_and_concat_single_file_per_date(self, mock_query, mock_open_ds):
        """Test opening and concatenating with single file per date."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]
        mock_open_ds.return_value = self.create_mock_dataset()

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        filepaths = {
            "2020-01-15": ["/path/to/file1.nc"],
            "2020-01-16": ["/path/to/file2.nc"],
        }

        result = agg.open_and_concat(filepaths)

        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual(len(result.time), 2)

    @patch("xarray.open_dataset")
    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_open_and_concat_multiple_files_both_nan(self, mock_query, mock_open_ds):
        """Test handling multiple files where both have all NaN."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]

        # Both files have NaN data
        mock_open_ds.return_value = self.create_mock_dataset(has_nan=True)

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        filepaths = {
            "2020-01-15": ["/path/to/file1.nc", "/path/to/file2.nc"],
        }

        result = agg.open_and_concat(filepaths)

        self.assertIsInstance(result, xr.Dataset)

    @patch("xarray.open_dataset")
    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_open_and_concat_multiple_files_one_valid(self, mock_query, mock_open_ds):
        """Test handling multiple files where one has valid data."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]

        # First file has NaN, second has valid data
        mock_open_ds.side_effect = [
            self.create_mock_dataset(has_nan=True),
            self.create_mock_dataset(has_nan=False),
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        filepaths = {
            "2020-01-15": ["/path/to/file1.nc", "/path/to/file2.nc"],
        }

        result = agg.open_and_concat(filepaths)

        self.assertIsInstance(result, xr.Dataset)


class AggregationGetMissingDatesTestCase(unittest.TestCase):
    """Tests for Aggregation.get_missing_dates method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "1.0",
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "Sea Surface Height Anomaly",
                    "standard_name": "sea_surface_height_above_sea_level",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
        }

    def get_mock_grid(self):
        return {
            "grid_name_s": "TEST_GRID",
        }

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_get_missing_dates_daily_complete_year(self, mock_query):
        """Test getting missing dates for daily data with complete year."""
        # First query for ds_meta, second for transformations
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z", "data_time_scale_s": "daily"}],
            [
                {"date_s": f"2020-{m:02d}-{d:02d}T00:00:00Z"}
                for m in range(1, 13)
                for d in range(1, 32)
                if m in [1, 3, 5, 7, 8, 10, 12] or d <= 30
            ],
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        missing = agg.get_missing_dates()

        # 2020 is a leap year with 366 days, should have some missing since we didn't include all dates
        self.assertIsInstance(missing, list)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_get_missing_dates_daily_missing_some(self, mock_query):
        """Test getting missing dates for daily data with some missing."""
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z", "data_time_scale_s": "daily"}],
            [
                {"date_s": "2020-01-01T00:00:00Z"},
                {"date_s": "2020-01-02T00:00:00Z"},
                # 2020-01-03 is missing
                {"date_s": "2020-01-04T00:00:00Z"},
            ],
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        missing = agg.get_missing_dates()

        self.assertIn("2020-01-03", missing)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_get_missing_dates_monthly(self, mock_query):
        """Test getting missing dates for monthly data."""
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z", "data_time_scale_s": "monthly"}],
            [
                {"date_s": "2020-01-01T00:00:00Z"},
                {"date_s": "2020-02-01T00:00:00Z"},
                # March missing
                {"date_s": "2020-04-01T00:00:00Z"},
            ],
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        missing = agg.get_missing_dates()

        self.assertIn("2020-03-01", missing)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_get_missing_dates_monthly_with_tolerance(self, mock_query):
        """Test that monthly dates within tolerance are not marked missing."""
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z", "data_time_scale_s": "monthly"}],
            [
                {"date_s": "2020-01-01T00:00:00Z"},
                {"date_s": "2020-02-03T00:00:00Z"},  # Within tolerance of 2020-02-01
            ],
        ]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        missing = agg.get_missing_dates()

        # February should not be marked as missing due to tolerance
        self.assertNotIn("2020-02-01", missing)


class AggregationMonthlyAggregationTestCase(unittest.TestCase):
    """Tests for Aggregation.monthly_aggregation method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "1.0",
            "remove_nan_days_from_data": True,
            "skipna_in_mean": False,
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "Sea Surface Height Anomaly",
                    "standard_name": "sea_surface_height_above_sea_level",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
        }

    def get_mock_grid(self):
        return {
            "grid_name_s": "TEST_GRID",
        }

    def create_daily_annual_dataset(self):
        """Create a mock daily dataset for a year."""
        # Create 365 days of data
        times = [np.datetime64(f"2020-01-01", "ns") + np.timedelta64(i, "D") for i in range(365)]
        time_bnds = np.array([[t, t + np.timedelta64(1, "D")] for t in times])
        data = np.random.rand(365, 10, 10).astype(np.float32)

        ds = xr.Dataset(
            {
                "ssha": (["time", "j", "i"], data),
            },
            coords={
                "time": times,
                "time_bnds": (["time", "nv"], time_bnds),
            },
        )
        ds.attrs["test_attr"] = "test_value"
        return ds

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_monthly_aggregation_basic(self, mock_query):
        """Test basic monthly aggregation."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        ds = self.create_daily_annual_dataset()
        uuid_str = "test-uuid-123"

        result = agg.monthly_aggregation(ds, "ssha", uuid_str)

        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual(len(result.time), 12)  # 12 months
        self.assertIn("time_bnds", result.coords)
        self.assertEqual(result.attrs["uuid"], uuid_str)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_monthly_aggregation_skipna_false(self, mock_query):
        """Test monthly aggregation with skipna=False."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]

        config = self.get_base_config()
        config["skipna_in_mean"] = False
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        ds = self.create_daily_annual_dataset()
        # Add some NaN values
        ds["ssha"].values[0, 0, 0] = np.nan

        result = agg.monthly_aggregation(ds, "ssha", "uuid")

        self.assertIsInstance(result, xr.Dataset)

    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_monthly_aggregation_remove_nan_days(self, mock_query):
        """Test monthly aggregation with remove_nan_days_from_data=True."""
        mock_query.return_value = [{"start_date_dt": "2020-01-01T00:00:00Z"}]

        config = self.get_base_config()
        config["remove_nan_days_from_data"] = True
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        ds = self.create_daily_annual_dataset()
        # Set entire day to NaN
        ds["ssha"].values[0, :, :] = np.nan

        result = agg.monthly_aggregation(ds, "ssha", "uuid")

        self.assertIsInstance(result, xr.Dataset)


class AggregationGenerateProvenanceTestCase(unittest.TestCase):
    """Tests for Aggregation.generate_provenance method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "1.0",
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "Sea Surface Height Anomaly",
                    "standard_name": "sea_surface_height_above_sea_level",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
        }

    def get_mock_grid(self):
        return {
            "grid_name_s": "TEST_GRID",
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("aggregations.aggregation.solr_utils.solr_update")
    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_generate_provenance_updates_existing_descendants(self, mock_query, mock_update, mock_file):
        """Test updating existing descendants documents."""
        # First call for ds_meta, second for descendants, third for aggregation docs
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],
            [{"id": "descendant1", "date_s": "2020-01-15T00:00:00Z"}],  # Existing descendants
            [{"id": "agg1"}],  # Aggregation docs
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        solr_paths = {
            "daily_bin": "/path/to/daily.bin",
            "daily_netCDF": "/path/to/daily.nc",
        }

        agg.generate_provenance("TEST_GRID", solr_paths, True)

        # Check that update was called
        mock_update.assert_called_once()
        update_body = mock_update.call_args[0][0]
        self.assertEqual(update_body[0]["id"], "descendant1")

    @patch("builtins.open", new_callable=mock_open)
    @patch("aggregations.aggregation.solr_utils.solr_update")
    @patch("aggregations.aggregation.solr_utils.solr_query")
    def test_generate_provenance_exports_json(self, mock_query, mock_update, mock_file):
        """Test that JSON file is exported."""
        mock_query.side_effect = [
            [{"start_date_dt": "2020-01-01T00:00:00Z", "dataset_s": "TEST_DATASET"}],
            [],  # No existing descendants
            [{"id": "agg1"}],  # Aggregation docs
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        field = MagicMock()
        field.name = "ssha"
        agg = Aggregation(config, self.get_mock_grid(), "2020", field)

        solr_paths = {
            "daily_bin": "/path/to/daily.bin",
            "daily_netCDF": "/path/to/daily.nc",
        }

        agg.generate_provenance("TEST_GRID", solr_paths, True)

        # Check that file was opened for writing
        self.assertTrue(mock_file.called)


if __name__ == "__main__":
    unittest.main()
