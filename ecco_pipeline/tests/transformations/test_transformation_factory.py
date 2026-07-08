"""
Unit tests for transformation_factory module (TxJobFactory class).
All Solr and file I/O calls are mocked.
"""

import unittest
from unittest.mock import patch, MagicMock

from transformations.grid_transformation import TxResult
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

    @patch("transformations.transformation_factory.solr_utils.solr_count")
    @patch("transformations.transformation_factory.solr_utils.solr_update")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pipeline_cleanup_all_successful(self, mock_config, mock_query, mock_update, mock_count):
        """Test cleanup when all transformations successful."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]

        # Return values for each query
        mock_query.side_effect = [
            [],  # harvested granules
            [{"id": "dataset_id"}],  # dataset metadata
        ]
        # pipeline_cleanup counts successful then failed transformations.
        mock_count.side_effect = [1, 0]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = TxJobFactory(config)

        result = factory.pipeline_cleanup()

        self.assertEqual(result, "All transformations successful")

    @patch("transformations.transformation_factory.solr_utils.solr_count")
    @patch("transformations.transformation_factory.solr_utils.solr_update")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pipeline_cleanup_some_failed(self, mock_config, mock_query, mock_update, mock_count):
        """Test cleanup when some transformations failed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]

        mock_query.side_effect = [
            [],  # harvested granules
            [{"id": "dataset_id"}],  # dataset metadata
        ]
        mock_count.side_effect = [1, 2]  # successful, failed

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = TxJobFactory(config)

        result = factory.pipeline_cleanup()

        self.assertEqual(result, "2 transformations failed")

    @patch("transformations.transformation_factory.solr_utils.solr_count")
    @patch("transformations.transformation_factory.solr_utils.solr_update")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pipeline_cleanup_none_performed(self, mock_config, mock_query, mock_update, mock_count):
        """Test cleanup when no transformations performed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]

        mock_query.side_effect = [
            [],  # harvested granules
            [{"id": "dataset_id"}],  # dataset metadata
        ]
        mock_count.side_effect = [0, 0]  # successful, failed

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = TxJobFactory(config)

        result = factory.pipeline_cleanup()

        self.assertEqual(result, "No transformations performed")


class MultiprocessTransformationTestCase(unittest.TestCase):
    """Tests for the multiprocess_transformation function (pure-compute wrapper)."""

    @patch("transformations.transformation_factory.solr_utils")
    @patch("transformations.transformation_factory.transform")
    @patch("transformations.transformation_factory.log_config.mp_logging")
    def test_multiprocess_calls_transform_and_makes_no_solr_calls(
        self, mock_logging, mock_transform, mock_solr
    ):
        """transform() is called with the doc_id_map, its results are returned in a
        4-tuple, and the worker path touches Solr zero times (ADR 0001)."""
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger
        worker_results = [TxResult(doc_id="id1", grid="grid", field="field", success=True)]
        mock_transform.return_value = worker_results

        config = {"ds_name": "TEST"}
        granule = {"pre_transformation_file_path_s": "/data/test.nc", "file_size_l": 1000, "date_dt": "2020-01-01"}
        field = MagicMock()
        field.name = "field"
        tx_jobs = {"grid": [field]}
        doc_id_map = {("grid", "field"): "id1"}

        result = multiprocess_transformation(config, granule, tx_jobs, doc_id_map, "INFO", "/logs")

        mock_transform.assert_called_once_with("/data/test.nc", tx_jobs, config, "2020-01-01", doc_id_map)
        # No filename_s on the granule, so the marker falls back to the file path.
        self.assertEqual(result, ("/data/test.nc", "ok", "", worker_results))
        mock_solr.solr_update.assert_not_called()
        mock_solr.solr_query.assert_not_called()

    @patch("transformations.transformation_factory.transform")
    @patch("transformations.transformation_factory.log_config.mp_logging")
    def test_multiprocess_returns_failure_results_on_exception(self, mock_logging, mock_transform):
        """A granule that raises (e.g. in load_file) returns an error marker plus a
        failure TxResult per (grid, field), not an exception — the batch must not abort."""
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger
        mock_transform.side_effect = RuntimeError("load_file blew up")

        config = {"ds_name": "TEST"}
        granule = {
            "filename_s": "bad.nc",
            "pre_transformation_file_path_s": "/data/bad.nc",
            "file_size_l": 1000,
            "date_dt": "2020-01-01",
        }
        field = MagicMock()
        field.name = "field"
        tx_jobs = {"grid": [field]}
        doc_id_map = {("grid", "field"): "id1"}

        # Must not raise — the whole batch would abort otherwise.
        result = multiprocess_transformation(config, granule, tx_jobs, doc_id_map, "INFO", "/logs")

        self.assertEqual(result[0], "bad.nc")
        self.assertEqual(result[1], "error")
        self.assertIn("load_file blew up", result[2])
        failure_results = result[3]
        self.assertEqual(len(failure_results), 1)
        self.assertEqual(failure_results[0].doc_id, "id1")
        self.assertFalse(failure_results[0].success)
        self.assertIn("load_file blew up", failure_results[0].error_message)


class TxJobFactoryParentSolrIOTestCase(unittest.TestCase):
    """Tests for the parent-owned, batched Solr I/O (prepopulate_jobs, record_results,
    record_unprocessable) — the core of ADR 0001."""

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

    def _field(self, name):
        field = MagicMock()
        field.name = name
        return field

    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def _factory(self, mock_config, mock_query):
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []
        return TxJobFactory(self.get_base_config())

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    def test_prepopulate_jobs_batches_and_assigns_ids(self, mock_update):
        """Two (grid, field) pairs across grids collapse into ONE bulk write; each
        gets a doc id in the returned map. A new field mints a uuid; an existing one
        reuses its id via an atomic update."""
        factory = self._factory()
        factory.existing_tx_ids = {("test1.nc", "gridA", "f_existing"): "existing-id"}

        granule = {
            "filename_s": "test1.nc",
            "pre_transformation_file_path_s": "/data/test1.nc",
            "date_dt": "2020-01-01",
            "checksum_s": "abc",
        }
        grid_fields = {"gridA": [self._field("f_existing"), self._field("f_new")]}
        processable = [(granule, grid_fields)]

        doc_id_maps = factory.prepopulate_jobs(processable)

        # One bulk write for the whole batch.
        self.assertEqual(mock_update.call_count, 1)
        _, kwargs = mock_update.call_args
        self.assertFalse(kwargs.get("commit", True))  # commitWithin, not a hard commit
        update_body = mock_update.call_args[0][0]
        self.assertEqual(len(update_body), 2)

        field_map = doc_id_maps["test1.nc"]
        # Existing field reuses its id (atomic update).
        self.assertEqual(field_map[("gridA", "f_existing")], "existing-id")
        existing_doc = next(d for d in update_body if d["id"] == "existing-id")
        self.assertEqual(existing_doc["success_b"], {"set": False})
        self.assertEqual(existing_doc["origin_checksum_s"], {"set": "abc"})
        # New field gets a fresh (non-existing) id and a full doc.
        new_id = field_map[("gridA", "f_new")]
        self.assertNotEqual(new_id, "existing-id")
        new_doc = next(d for d in update_body if d["id"] == new_id)
        self.assertEqual(new_doc["type_s"], "transformation")
        self.assertEqual(new_doc["origin_checksum_s"], "abc")

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    def test_record_results_batches_success_and_failure(self, mock_update):
        """N results collapse into ONE bulk atomic update; success writes file fields,
        failure writes only status; a result with no doc id is skipped, not fatal."""
        factory = self._factory()

        results = [
            TxResult(
                doc_id="id-ok", grid="g", field="f1", success=True,
                output_filename="out.nc", output_path="/o/out.nc",
                checksum="cs", error_message="",
            ),
            TxResult(doc_id="id-fail", grid="g", field="f2", success=False, error_message="boom"),
            TxResult(doc_id=None, grid="g", field="f3", success=False, error_message="no id"),
        ]

        factory.record_results(results)

        self.assertEqual(mock_update.call_count, 1)
        _, kwargs = mock_update.call_args
        self.assertFalse(kwargs.get("commit", True))
        update_body = mock_update.call_args[0][0]
        # The doc_id=None result is skipped.
        self.assertEqual(len(update_body), 2)

        ok = next(d for d in update_body if d["id"] == "id-ok")
        self.assertEqual(ok["success_b"], {"set": True})
        self.assertEqual(ok["transformation_file_path_s"], {"set": "/o/out.nc"})
        self.assertEqual(ok["transformation_checksum_s"], {"set": "cs"})
        self.assertEqual(ok["transformation_version_f"], {"set": 1.0})

        fail = next(d for d in update_body if d["id"] == "id-fail")
        self.assertEqual(fail["success_b"], {"set": False})
        self.assertEqual(fail["error_message_s"], {"set": "boom"})
        self.assertNotIn("transformation_file_path_s", fail)

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    def test_record_unprocessable_records_failures(self, mock_update):
        """Harvest-quality failures are recorded by the parent in one bulk write and
        the count is tracked so cleanup still runs."""
        factory = self._factory()
        factory.existing_tx_ids = {}

        granule = {
            "filename_s": "bad.nc",
            "pre_transformation_file_path_s": None,
            "file_size_l": 0,
            "date_dt": "2020-01-01",
        }
        unprocessable = [(granule, {"gridA": [self._field("f1")]})]

        factory.record_unprocessable(unprocessable)

        self.assertEqual(factory.unprocessable_count, 1)
        self.assertEqual(mock_update.call_count, 1)
        _, kwargs = mock_update.call_args
        self.assertFalse(kwargs.get("commit", True))
        doc = mock_update.call_args[0][0][0]
        self.assertEqual(doc["success_b"], False)
        self.assertIn("not harvested properly", doc["error_message_s"])

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    def test_record_unprocessable_noop_when_empty(self, mock_update):
        """No unprocessable granules → no write, count stays 0."""
        factory = self._factory()
        factory.record_unprocessable([])
        mock_update.assert_not_called()
        self.assertEqual(factory.unprocessable_count, 0)


class TxJobFactoryReconstructTestCase(unittest.TestCase):
    """Tests for TxJobFactory.reconstruct_tx_solr_doc."""

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
            "notes": "",
        }

    @patch("transformations.transformation_factory.file_utils.md5")
    @patch("transformations.transformation_factory.os.path.getmtime")
    @patch("transformations.transformation_factory.solr_utils.solr_update")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_reconstruct_uses_commitwithin(
        self, mock_config, mock_query, mock_update, mock_getmtime, mock_md5
    ):
        """Reconstruct writes must use commitWithin (commit=False), not a per-doc hard
        commit — the latter was a commit storm that hung job generation for minutes."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []
        mock_getmtime.return_value = 0
        mock_md5.return_value = "checksum"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        factory = TxJobFactory(self.get_base_config())

        field = MagicMock()
        field.name = "test_field"
        granule = {
            "filename_s": "test1.nc",
            "date_dt": "2020-01-01T00:00:00Z",
            "pre_transformation_file_path_s": "/data/test1.nc",
            "checksum_s": "abc",
        }

        factory.reconstruct_tx_solr_doc(granule, "grid", field)

        self.assertEqual(mock_update.call_count, 1)
        _, kwargs = mock_update.call_args
        self.assertFalse(kwargs.get("commit", True))


if __name__ == "__main__":
    unittest.main()
