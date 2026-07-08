"""
Unit tests for grid_transformation module (Transformation class).
All file I/O and Solr calls are mocked.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr

import transformations.grid_transformation as grid_transformation
from transformations.grid_transformation import Transformation


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
            "proj_info": {"area_id": "test", "area_name": "Test", "proj_id": "test", "proj4_args": "+proj=latlong"},
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
            "proj_info": {"area_id": "test", "area_name": "Test", "proj_id": "test", "proj4_args": "+proj=latlong"},
            "notes": "",
        }

    @patch("transformations.grid_transformation.pickle.dump")
    @patch("transformations.grid_transformation.pickle.load")
    @patch("transformations.grid_transformation.os.path.exists")
    @patch("transformations.grid_transformation.os.makedirs")
    def test_make_factors_loads_existing(self, mock_makedirs, mock_exists, mock_load, mock_dump):
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

    @patch("transformations.grid_transformation.transformation_utils.find_mappings_from_source_to_target")
    @patch("transformations.grid_transformation.transformation_utils.generalized_grid_product")
    @patch("transformations.grid_transformation.pr.geometry.SwathDefinition")
    @patch("transformations.grid_transformation.pickle.dump")
    @patch("transformations.grid_transformation.os.path.exists")
    @patch("transformations.grid_transformation.os.makedirs")
    def test_make_factors_creates_new(
        self, mock_makedirs, mock_exists, mock_dump, mock_swath, mock_grid_product, mock_find_mappings
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
            "proj_info": {"area_id": "test", "area_name": "Test", "proj_id": "test", "proj4_args": "+proj=latlong"},
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
            "proj_info": {"area_id": "test", "area_name": "Test", "proj_id": "test", "proj4_args": "+proj=latlong"},
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
        self, mock_load_file, mock_load_grid, mock_make_factors,
        mock_method_transform, mock_makedirs, mock_save, mock_md5,
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
