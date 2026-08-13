"""
Unit tests for grid_transformation module (Transformation class).
All file I/O and Solr calls are mocked.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr

import transformations.grid_transformation as grid_transformation
from transformations.grid_transformation import Transformation

_GRID_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "grids", "ECCO_llc90.nc"
)


class TransformationInitTestCase(unittest.TestCase):
    """Tests for Transformation class initialization."""

    def get_base_config(self):
        """Return a base configuration for testing."""
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "data_time_scale": "daily",
            "fields": [
                {
                    "name": "test_field",
                    "long_name": "Test Field",
                    "standard_name": "test",
                    "units": "1",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
            "original_dataset_title": "Test Dataset",
            "original_dataset_short_name": "TEST",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "Test Ref",
            "original_dataset_doi": "10.1234/test",
            "t_version": 1.0,
            "a_version": 1.0,
            "data_res": 0.25,
            "area_extent": [-180, -90, 180, 90],
            "dims": [720, 360],
            "proj_info": {
                "area_id": "test",
                "area_name": "Test",
                "proj_id": "test",
                "proj4_args": "+proj=latlong",
            },
            "notes": "",
        }

    def test_basic_initialization(self):
        """Test basic Transformation initialization."""
        config = self.get_base_config()
        source_path = "/data/test_file_20200115.nc"

        T = Transformation(config, source_path, "2020-01-15")

        self.assertEqual(T.ds_name, "TEST_DATASET")
        self.assertEqual(T.file_name, "test_file_20200115")
        self.assertEqual(T.date, "2020-01-15")
        self.assertEqual(T.transformation_version, 1.0)

    def test_compute_data_res_float(self):
        """Test data resolution as float."""
        config = self.get_base_config()
        config["data_res"] = 0.5

        T = Transformation(config, "/data/test.nc", "2020-01-01")

        self.assertEqual(T.data_res, 0.5)

    def test_compute_data_res_fraction_string(self):
        """Test data resolution as fraction string."""
        config = self.get_base_config()
        config["data_res"] = "1/4"

        T = Transformation(config, "/data/test.nc", "2020-01-01")

        self.assertEqual(T.data_res, 0.25)

    def test_compute_data_res_number_string(self):
        """Test data resolution as number string."""
        config = self.get_base_config()
        config["data_res"] = "0.125"

        T = Transformation(config, "/data/test.nc", "2020-01-01")

        self.assertEqual(T.data_res, 0.125)

    def test_get_hemi_north(self):
        """Test hemisphere detection for north."""
        config = self.get_base_config()
        config["hemi_pattern"] = {"north": "_nh_", "south": "_sh_"}
        config["area_extent_nh"] = [-180, 0, 180, 90]
        config["dims_nh"] = [360, 180]
        config["proj_info_nh"] = config["proj_info"]

        T = Transformation(config, "/data/test_nh_20200115.nc", "2020-01-15")

        self.assertEqual(T.hemi, "_nh")

    def test_get_hemi_south(self):
        """Test hemisphere detection for south."""
        config = self.get_base_config()
        config["hemi_pattern"] = {"north": "_nh_", "south": "_sh_"}
        config["area_extent_sh"] = [-180, -90, 180, 0]
        config["dims_sh"] = [360, 180]
        config["proj_info_sh"] = config["proj_info"]

        T = Transformation(config, "/data/test_sh_20200115.nc", "2020-01-15")

        self.assertEqual(T.hemi, "_sh")

    def test_get_hemi_no_pattern(self):
        """Test no hemisphere when pattern not in config."""
        config = self.get_base_config()

        T = Transformation(config, "/data/test_20200115.nc", "2020-01-15")

        self.assertEqual(T.hemi, "")

    def test_source_type_defaults_to_grid(self):
        """Absent source_type defaults to the grid path with nearest-neighbor on."""
        config = self.get_base_config()

        T = Transformation(config, "/data/test_20200115.nc", "2020-01-15")

        self.assertEqual(T.source_type, "grid")
        self.assertEqual(T.lat_var, "latitude")
        self.assertEqual(T.lon_var, "longitude")
        self.assertTrue(T.allow_nearest_neighbor)

    def test_along_track_config_fields(self):
        """along_track defaults nearest-neighbor off; lat/lon vars are configurable."""
        config = self.get_base_config()
        config["source_type"] = "along_track"
        config["lat_var"] = "lat"
        config["lon_var"] = "lon"

        T = Transformation(config, "/data/test_20200115.nc", "2020-01-15")

        self.assertEqual(T.source_type, "along_track")
        self.assertEqual(T.lat_var, "lat")
        self.assertEqual(T.lon_var, "lon")
        self.assertFalse(T.allow_nearest_neighbor)

    def test_allow_nearest_neighbor_explicit_override(self):
        """An explicit allow_nearest_neighbor overrides the source_type default."""
        config = self.get_base_config()
        config["source_type"] = "along_track"
        config["allow_nearest_neighbor"] = True

        T = Transformation(config, "/data/test_20200115.nc", "2020-01-15")

        self.assertTrue(T.allow_nearest_neighbor)


class TransformationMakeFactorsTestCase(unittest.TestCase):
    """Tests for Transformation.make_factors method."""

    def setUp(self):
        # make_factors memoizes loaded factors in a module-level per-process cache
        # keyed by factors_path. Clear it between tests so one test's cached entry
        # doesn't hide another test's disk read.
        grid_transformation._factors_cache.clear()
        grid_transformation._grid_ds_cache.clear()

    def get_base_config(self):
        """Return a base configuration for testing."""
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "data_time_scale": "daily",
            "fields": [],
            "original_dataset_title": "Test",
            "original_dataset_short_name": "TEST",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "Ref",
            "original_dataset_doi": "10.1234/test",
            "t_version": 1.0,
            "a_version": 1.0,
            "data_res": 0.25,
            "area_extent": [-180, -90, 180, 90],
            "dims": [720, 360],
            "proj_info": {
                "area_id": "test",
                "area_name": "Test",
                "proj_id": "test",
                "proj4_args": "+proj=latlong",
            },
            "notes": "",
        }

    @patch("transformations.grid_transformation.pickle.dump")
    @patch("transformations.grid_transformation.pickle.load")
    @patch("transformations.grid_transformation.os.path.exists")
    @patch("transformations.grid_transformation.os.makedirs")
    def test_make_factors_loads_existing(
        self, mock_makedirs, mock_exists, mock_load, mock_dump
    ):
        """Test that existing factors are loaded from file."""
        mock_exists.return_value = True
        expected_factors = ({"0": [0, 1]}, [2], {"3": 4})
        mock_load.return_value = expected_factors

        config = self.get_base_config()
        T = Transformation(config, "/data/test.nc", "2020-01-01")

        # Create mock grid
        grid_ds = MagicMock()
        grid_ds.name = "test_grid"

        with patch("builtins.open", MagicMock()):
            result = T.make_factors(grid_ds)

        self.assertEqual(result, expected_factors)
        mock_load.assert_called_once()

    @patch("transformations.grid_transformation.pickle.dump")
    @patch("transformations.grid_transformation.pickle.load")
    @patch("transformations.grid_transformation.os.path.exists")
    @patch("transformations.grid_transformation.os.makedirs")
    def test_make_factors_reads_pickle_once_across_calls(
        self, mock_makedirs, mock_exists, mock_load, mock_dump
    ):
        """Two make_factors calls in one process unpickle the factors file only once."""
        mock_exists.return_value = True
        expected_factors = ({"0": [0, 1]}, [2], {"3": 4})
        mock_load.return_value = expected_factors

        config = self.get_base_config()
        T = Transformation(config, "/data/test.nc", "2020-01-01")

        grid_ds = MagicMock()
        grid_ds.name = "test_grid"

        with patch("builtins.open", MagicMock()):
            first = T.make_factors(grid_ds)
            second = T.make_factors(grid_ds)

        self.assertEqual(first, expected_factors)
        self.assertEqual(second, expected_factors)
        mock_load.assert_called_once()

    @patch(
        "transformations.grid_transformation.transformation_utils.find_mappings_from_source_to_target"
    )
    @patch(
        "transformations.grid_transformation.transformation_utils.generalized_grid_product"
    )
    @patch("transformations.grid_transformation.pr.geometry.SwathDefinition")
    @patch("transformations.grid_transformation.pickle.dump")
    @patch("transformations.grid_transformation.os.path.exists")
    @patch("transformations.grid_transformation.os.makedirs")
    def test_make_factors_creates_new(
        self,
        mock_makedirs,
        mock_exists,
        mock_dump,
        mock_swath,
        mock_grid_product,
        mock_find_mappings,
    ):
        """Test that new factors are created when file doesn't exist."""
        mock_exists.return_value = False

        # Mock grid product return
        mock_grid_product.return_value = (1000, 5000, MagicMock())

        # Mock find_mappings return
        expected_factors = ({"0": [0, 1]}, np.array([2]), {"3": 4})
        mock_find_mappings.return_value = expected_factors

        config = self.get_base_config()
        T = Transformation(config, "/data/test.nc", "2020-01-01")

        # Create mock grid with required attributes
        grid_ds = MagicMock()
        grid_ds.name = "test_grid"
        grid_ds.XC.values.ravel.return_value = np.zeros(25)
        grid_ds.YC.values.ravel.return_value = np.zeros(25)
        grid_ds.RAD.values.ravel.return_value = np.ones(25) * 1000
        # Configure __contains__ to return True for "RAD"
        grid_ds.__contains__.side_effect = lambda x: x == "RAD"

        with patch("builtins.open", MagicMock()):
            result = T.make_factors(grid_ds)

        self.assertEqual(result, expected_factors)
        mock_find_mappings.assert_called_once()

    @patch(
        "transformations.grid_transformation.transformation_utils.along_track_factors"
    )
    @patch("transformations.grid_transformation.pickle.load")
    @patch("transformations.grid_transformation.os.path.exists")
    @patch("transformations.grid_transformation.os.makedirs")
    def test_make_factors_along_track_bypasses_caches(
        self, mock_makedirs, mock_exists, mock_load, mock_along_track
    ):
        """
        along_track make_factors computes fresh per granule: it must call
        along_track_factors with the granule coords and touch neither cache (even if a
        pickle exists on disk for the same grid+hemi+t_version key).
        """
        mock_exists.return_value = True  # a stale pickle exists for this key
        expected = ({0: np.array([1])}, np.array([0, 1]), {})
        mock_along_track.return_value = expected

        config = self.get_base_config()
        config["source_type"] = "along_track"
        config["fields"] = [
            {
                "name": "ssha",
                "long_name": "x",
                "standard_name": "x",
                "units": "m",
                "pre_transformations": [],
                "post_transformations": [],
            }
        ]
        T = Transformation(config, "/data/test.nc", "2020-01-01")

        grid_ds = MagicMock()
        grid_ds.name = "test_grid"

        ds = MagicMock()
        lon_vals = np.array([0.0, 1.0])
        lat_vals = np.array([0.0, 0.0])
        ds.__getitem__.side_effect = lambda key: {
            "longitude": MagicMock(values=lon_vals),
            "latitude": MagicMock(values=lat_vals),
        }[key]

        result = T.make_factors(grid_ds, ds)

        self.assertEqual(result, expected)
        mock_along_track.assert_called_once()
        # Never read the on-disk pickle despite os.path.exists being True.
        mock_load.assert_not_called()
        # Nothing written to the in-memory cache for this grid.
        self.assertEqual(len(grid_transformation._factors_cache), 0)

        # A second granule with different coords recomputes (no silent cache reuse).
        T.make_factors(grid_ds, ds)
        self.assertEqual(mock_along_track.call_count, 2)


class TransformationEmptyAlongTrackTestCase(unittest.TestCase):
    """
    Regression: an along-track granule with zero samples must not crash.

    time/lat/lon/value are per-sample coordinates, so an empty granule has a size-0
    time array. perform_mapping used to do ds["time"].values.ravel()[0] unconditionally
    and raised IndexError; it now falls back to the nominal record date.
    """

    def setUp(self):
        grid_transformation._factors_cache.clear()
        grid_transformation._grid_ds_cache.clear()

    def get_config(self):
        return {
            "ds_name": "TEST_ALONGTRACK",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "data_time_scale": "daily",
            "source_type": "along_track",
            "mapping_operation": "nanmean",
            "allow_nearest_neighbor": False,
            "fields": [
                {
                    "name": "ssha",
                    "long_name": "ssha",
                    "standard_name": "ssha",
                    "units": "m",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
            "original_dataset_title": "T",
            "original_dataset_short_name": "T",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "R",
            "original_dataset_doi": "10.1234/t",
            "t_version": 1.0,
            "a_version": 1.0,
            "notes": "",
        }

    @unittest.skipUnless(os.path.exists(_GRID_FILE), "ECCO_llc90 grid not present")
    def test_empty_granule_does_not_crash(self):
        empty = xr.Dataset(
            {
                "ssha": ("time", np.array([], dtype="float64")),
                "latitude": ("time", np.array([], dtype="float32")),
                "longitude": ("time", np.array([], dtype="float32")),
            },
            coords={"time": ("time", np.array([], dtype="datetime64[ns]"))},
        )
        grid_ds = xr.open_dataset(_GRID_FILE).reset_coords()

        T = Transformation(
            self.get_config(),
            "/data/NASA-SSH_alt_ref_at_v1_1_20200617.nc",
            "2020-06-17T00:00:00Z",
        )
        factors = T.make_factors(grid_ds, empty)
        # No points binned anywhere.
        self.assertEqual(len(factors[0]), 0)

        field_DA = T.perform_mapping(empty, factors, T.fields[0], grid_ds)
        # Time falls back to the nominal record date (no per-sample time to read).
        self.assertEqual(str(field_DA.time.values[0])[:10], "2020-06-17")
        # Every cell is empty (all-NaN) — a valid empty record, not a crash.
        self.assertTrue(np.all(np.isnan(field_DA.values)))


class TransformationLoadFileTestCase(unittest.TestCase):
    """Tests for Transformation.load_file method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "data_time_scale": "daily",
            "fields": [],
            "original_dataset_title": "Test",
            "original_dataset_short_name": "TEST",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "Ref",
            "original_dataset_doi": "10.1234/test",
            "t_version": 1.0,
            "a_version": 1.0,
            "data_res": 0.25,
            "area_extent": [-180, -90, 180, 90],
            "dims": [720, 360],
            "proj_info": {
                "area_id": "test",
                "area_name": "Test",
                "proj_id": "test",
                "proj4_args": "+proj=latlong",
            },
            "notes": "",
            "preprocessing": None,
        }

    @patch("xarray.open_dataset")
    def test_load_file_no_preprocessing(self, mock_open):
        """Test loading file without preprocessing."""
        mock_ds = xr.Dataset({"var": xr.DataArray([1, 2, 3])})
        mock_open.return_value = mock_ds

        config = self.get_base_config()
        T = Transformation(config, "/data/test_file.nc", "2020-01-01")

        result = T.load_file("/data/test_file.nc")

        self.assertEqual(result.attrs["original_file_name"], "test_file")
        mock_open.assert_called_once_with("/data/test_file.nc", decode_times=True)

    @patch("transformations.grid_transformation.PreprocessingFuncs")
    def test_load_file_with_preprocessing(self, mock_funcs_class):
        """Test loading file with preprocessing function."""
        mock_ds = xr.Dataset({"var": xr.DataArray([1, 2, 3])})
        mock_func_instance = MagicMock()
        mock_func_instance.call_function.return_value = mock_ds
        mock_funcs_class.return_value = mock_func_instance

        config = self.get_base_config()
        config["preprocessing"] = "custom_preprocess"

        T = Transformation(config, "/data/test_file.nc", "2020-01-01")
        result = T.load_file("/data/test_file.nc")

        mock_func_instance.call_function.assert_called_once()
        self.assertEqual(result.attrs["original_file_name"], "test_file")


class TransformWorkerPurityTestCase(unittest.TestCase):
    """
    The module-level transform() is pure compute (ADR 0001): it saves output netCDFs
    and returns a TxResult per (grid, field), and makes no Solr calls.
    """

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "data_time_scale": "daily",
            "fields": [
                {
                    "name": "test_field",
                    "long_name": "Test Field",
                    "standard_name": "test",
                    "units": "1",
                    "pre_transformations": [],
                    "post_transformations": [],
                }
            ],
            "original_dataset_title": "Test",
            "original_dataset_short_name": "TEST",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "Ref",
            "original_dataset_doi": "10.1234/test",
            "t_version": 1.0,
            "a_version": 1.0,
            "data_res": 0.25,
            "area_extent": [-180, -90, 180, 90],
            "dims": [720, 360],
            "proj_info": {
                "area_id": "test",
                "area_name": "Test",
                "proj_id": "test",
                "proj4_args": "+proj=latlong",
            },
            "notes": "",
        }

    def test_module_has_no_solr_dependency(self):
        """The worker module must not carry a Solr client at all."""
        self.assertFalse(hasattr(grid_transformation, "solr_utils"))

    @patch("transformations.grid_transformation.file_utils.md5")
    @patch("transformations.grid_transformation.records.save_netcdf")
    @patch("transformations.grid_transformation.os.makedirs")
    @patch.object(Transformation, "transform")
    @patch.object(Transformation, "make_factors")
    @patch("transformations.grid_transformation.load_grid")
    @patch.object(Transformation, "load_file")
    def test_transform_returns_txresults(
        self,
        mock_load_file,
        mock_load_grid,
        mock_make_factors,
        mock_method_transform,
        mock_makedirs,
        mock_save,
        mock_md5,
    ):
        """A successful field yields a TxResult carrying the preassigned doc id,
        the worker-computed checksum, and success=True — with no Solr access."""
        mock_load_file.return_value = MagicMock()
        mock_load_grid.return_value = MagicMock()
        mock_make_factors.return_value = (MagicMock(),)
        mock_method_transform.return_value = [(MagicMock(), True, "")]
        mock_md5.return_value = "cksum"

        field = MagicMock()
        field.name = "test_field"
        tx_jobs = {"grid1": [field]}
        doc_id_map = {("grid1", "test_field"): "doc-1"}

        results = grid_transformation.transform(
            "/data/test.nc", tx_jobs, self.get_base_config(), "2020-01-01", doc_id_map
        )

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.doc_id, "doc-1")
        self.assertEqual(res.grid, "grid1")
        self.assertEqual(res.field, "test_field")
        self.assertTrue(res.success)
        self.assertEqual(res.checksum, "cksum")
        mock_save.assert_called_once()

    @patch("transformations.grid_transformation.os.makedirs")
    @patch.object(Transformation, "make_factors")
    @patch("transformations.grid_transformation.load_grid")
    @patch.object(Transformation, "load_file")
    def test_transform_grid_failure_emits_failure_results(
        self, mock_load_file, mock_load_grid, mock_make_factors, mock_makedirs
    ):
        """A grid-level failure fails every field of that grid, each as a
        TxResult(success=False) carrying its doc id — the batch is never aborted."""
        mock_load_file.return_value = MagicMock()
        mock_load_grid.side_effect = RuntimeError("bad grid")

        field = MagicMock()
        field.name = "test_field"
        tx_jobs = {"grid1": [field]}
        doc_id_map = {("grid1", "test_field"): "doc-1"}

        results = grid_transformation.transform(
            "/data/test.nc", tx_jobs, self.get_base_config(), "2020-01-01", doc_id_map
        )

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertFalse(res.success)
        self.assertEqual(res.doc_id, "doc-1")
        self.assertIn("bad grid", res.error_message)


if __name__ == "__main__":
    unittest.main()
