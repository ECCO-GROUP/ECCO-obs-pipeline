"""
Unit tests for transformation_factory module (TxJobFactory class).
All Solr and file I/O calls are mocked.
"""

import unittest
from unittest.mock import patch, MagicMock

from transformations.transformation_factory import TxJobFactory, multiprocess_transformation


class TxJobFactoryInitTestCase(unittest.TestCase):
    """Tests for TxJobFactory initialization."""

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
            "original_dataset_title": "Test",
            "original_dataset_short_name": "TEST",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "Ref",
            "original_dataset_doi": "10.1234/test",
            "t_version": 1.0,
            "a_version": 1.0,
            "notes": "",
        }

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_initialization(self, mock_config, mock_query):
        """Test TxJobFactory initialization."""
        mock_config.user_cpus = 4
        mock_config.grids_to_use = ["grid1", "grid2"]

        mock_query.return_value = [{"filename_s": "test1.nc", "pre_transformation_file_path_s": "/data/test1.nc"}]

        config = self.get_base_config()
        factory = TxJobFactory(config)

        self.assertEqual(factory.ds_name, "TEST_DATASET")
        self.assertEqual(factory.grids, ["grid1", "grid2"])
        self.assertEqual(len(factory.harvested_granules), 1)

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_initialization_loads_grids_from_solr(self, mock_config, mock_query):
        """Test that grids are loaded from Solr when not specified."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = []  # Empty, should query Solr

        # First call for granules, second for grids
        mock_query.side_effect = [
            [{"filename_s": "test1.nc"}],  # Granules
            [{"grid_name_s": "grid_A"}, {"grid_name_s": "grid_B"}],  # Grids
        ]

        config = self.get_base_config()
        factory = TxJobFactory(config)

        self.assertEqual(factory.grids, ["grid_A", "grid_B"])

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_initialization_excludes_tpose_for_hemi(self, mock_config, mock_query):
        """Test that TPOSE grid is excluded for hemispheric data."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["ECCO_grid", "TPOSE_grid"]

        mock_query.return_value = [{"filename_s": "test1.nc"}]

        config = self.get_base_config()
        config["hemi_pattern"] = {"north": "_nh_", "south": "_sh_"}

        factory = TxJobFactory(config)

        self.assertIn("ECCO_grid", factory.grids)
        self.assertNotIn("TPOSE_grid", factory.grids)


class TxJobFactoryNeedToUpdateTestCase(unittest.TestCase):
    """Tests for TxJobFactory.need_to_update method."""

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
            "t_version": 2.0,
            "a_version": 1.0,
            "notes": "",
        }

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_need_to_update_false_when_current(self, mock_config, mock_query):
        """Test that update is not needed when transformation is current."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []

        config = self.get_base_config()
        factory = TxJobFactory(config)

        granule = {"checksum_s": "abc123"}
        tx = {"success_b": True, "transformation_version_f": 2.0, "origin_checksum_s": "abc123"}

        result = factory.need_to_update(granule, tx)
        self.assertFalse(result)

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_need_to_update_true_when_version_changed(self, mock_config, mock_query):
        """Test that update is needed when version changed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []

        config = self.get_base_config()
        factory = TxJobFactory(config)

        granule = {"checksum_s": "abc123"}
        tx = {
            "success_b": True,
            "transformation_version_f": 1.0,  # Old version
            "origin_checksum_s": "abc123",
        }

        result = factory.need_to_update(granule, tx)
        self.assertTrue(result)

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_need_to_update_true_when_checksum_changed(self, mock_config, mock_query):
        """Test that update is needed when source checksum changed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []

        config = self.get_base_config()
        factory = TxJobFactory(config)

        granule = {"checksum_s": "new_checksum"}
        tx = {"success_b": True, "transformation_version_f": 2.0, "origin_checksum_s": "old_checksum"}

        result = factory.need_to_update(granule, tx)
        self.assertTrue(result)

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_need_to_update_true_when_failed(self, mock_config, mock_query):
        """Test that update is needed when previous transformation failed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []

        config = self.get_base_config()
        factory = TxJobFactory(config)

        granule = {"checksum_s": "abc123"}
        tx = {
            "success_b": False,  # Failed
            "transformation_version_f": 2.0,
            "origin_checksum_s": "abc123",
        }

        result = factory.need_to_update(granule, tx)
        self.assertTrue(result)


class TxJobFactoryFindDataForFactorsTestCase(unittest.TestCase):
    """Tests for TxJobFactory.find_data_for_factors method."""

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
            "notes": "",
        }

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_find_data_for_factors_non_hemi(self, mock_config, mock_query):
        """Test finding data for factors with non-hemispheric data."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = [
            {"pre_transformation_file_path_s": "/data/test1.nc"},
            {"pre_transformation_file_path_s": "/data/test2.nc"},
        ]

        config = self.get_base_config()
        factory = TxJobFactory(config)

        result = factory.find_data_for_factors()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pre_transformation_file_path_s"], "/data/test1.nc")

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_find_data_for_factors_hemi(self, mock_config, mock_query):
        """Test finding data for factors with hemispheric data."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = [
            {"pre_transformation_file_path_s": "/data/test_nh_1.nc"},
            {"pre_transformation_file_path_s": "/data/test_sh_1.nc"},
            {"pre_transformation_file_path_s": "/data/test_nh_2.nc"},
        ]

        config = self.get_base_config()
        config["hemi_pattern"] = {"north": "_nh_", "south": "_sh_"}

        factory = TxJobFactory(config)

        result = factory.find_data_for_factors()

        self.assertEqual(len(result), 2)
        paths = [r["pre_transformation_file_path_s"] for r in result]
        self.assertTrue(any("_nh_" in p for p in paths))
        self.assertTrue(any("_sh_" in p for p in paths))


class TxJobFactoryPipelineCleanupTestCase(unittest.TestCase):
    """Tests for TxJobFactory.pipeline_cleanup method."""

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
            "notes": "",
        }

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pipeline_cleanup_all_successful(self, mock_config, mock_query, mock_update):
        """Test cleanup when all transformations successful."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]

        # Return values for each query
        mock_query.side_effect = [
            [],  # harvested granules
            [{"id": "dataset_id"}],  # dataset metadata
            [{"id": "tx1"}],  # successful transformations
            [],  # failed transformations
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = TxJobFactory(config)

        result = factory.pipeline_cleanup()

        self.assertEqual(result, "All transformations successful")

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pipeline_cleanup_some_failed(self, mock_config, mock_query, mock_update):
        """Test cleanup when some transformations failed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]

        mock_query.side_effect = [
            [],  # harvested granules
            [{"id": "dataset_id"}],  # dataset metadata
            [{"id": "tx1"}],  # successful transformations
            [{"id": "tx2"}, {"id": "tx3"}],  # failed transformations
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = TxJobFactory(config)

        result = factory.pipeline_cleanup()

        self.assertEqual(result, "2 transformations failed")

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pipeline_cleanup_none_performed(self, mock_config, mock_query, mock_update):
        """Test cleanup when no transformations performed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]

        mock_query.side_effect = [
            [],  # harvested granules
            [{"id": "dataset_id"}],  # dataset metadata
            [],  # successful transformations
            [],  # failed transformations
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = TxJobFactory(config)

        result = factory.pipeline_cleanup()

        self.assertEqual(result, "No transformations performed")


class MultiprocessTransformationTestCase(unittest.TestCase):
    """Tests for the multiprocess_transformation function."""

    @patch("transformations.transformation_factory.transform")
    @patch("transformations.transformation_factory.log_config.mp_logging")
    def test_multiprocess_skips_invalid_granule(self, mock_logging, mock_transform):
        """Test that invalid granules are skipped."""
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        config = {"ds_name": "TEST"}
        granule = {"pre_transformation_file_path_s": None, "file_size_l": 0}

        multiprocess_transformation(config, granule, {}, "INFO", "/logs")

        mock_transform.assert_not_called()

    @patch("transformations.transformation_factory.transform")
    @patch("transformations.transformation_factory.log_config.mp_logging")
    def test_multiprocess_calls_transform(self, mock_logging, mock_transform):
        """Test that transform is called for valid granule."""
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        config = {"ds_name": "TEST"}
        granule = {"pre_transformation_file_path_s": "/data/test.nc", "file_size_l": 1000, "date_s": "2020-01-01"}
        tx_jobs = {"grid": ["field"]}

        multiprocess_transformation(config, granule, tx_jobs, "INFO", "/logs")

        mock_transform.assert_called_once_with("/data/test.nc", tx_jobs, config, "2020-01-01")


if __name__ == "__main__":
    unittest.main()
