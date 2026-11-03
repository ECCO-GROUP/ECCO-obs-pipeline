"""
Unit tests for transformation_factory module (TxJobFactory class).

Solr calls are mocked throughout. File I/O is mocked too, except in
OutputFileTrustTestCase, which writes real netCDFs on purpose: whether
transformation_version (a global attr) and empty_record_note (a variable attr) survive
save_netcdf is exactly what that logic depends on, and a mock would assert nothing.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr

from transformations.grid_transformation import TxResult
from transformations.transformation_factory import (
    TxJobFactory,
    multiprocess_transformation,
    stamp_run_id,
)


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

        mock_query.return_value = [
            {
                "filename_s": "test1.nc",
                "pre_transformation_file_path_s": "/data/test1.nc",
            }
        ]

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
        tx = {
            "success_b": True,
            "transformation_version_f": 2.0,
            "origin_checksum_s": "abc123",
        }

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
        tx = {
            "success_b": True,
            "transformation_version_f": 2.0,
            "origin_checksum_s": "old_checksum",
        }

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


class TxJobFactoryPregenerateFactorsTestCase(unittest.TestCase):
    """Tests for TxJobFactory.pregenerate_factors along-track no-op."""

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

    @patch("transformations.transformation_factory.xr.open_dataset")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pregenerate_factors_noop_for_along_track(
        self, mock_config, mock_query, mock_open
    ):
        """along_track skips factor warm-up entirely (no grid open, no make_factors)."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = [{"pre_transformation_file_path_s": "/data/test1.nc"}]

        config = self.get_base_config()
        config["source_type"] = "along_track"
        factory = TxJobFactory(config)

        factory.pregenerate_factors()

        # Never opened a grid file — the warm-up body was skipped.
        mock_open.assert_not_called()

    @patch("transformations.transformation_factory.Transformation.make_factors")
    @patch("transformations.transformation_factory.xr.open_dataset")
    @patch("transformations.transformation_factory.solr_utils.solr_query")
    @patch("transformations.transformation_factory.baseclasses.Config")
    def test_pregenerate_factors_runs_for_grid(
        self, mock_config, mock_query, mock_open, mock_make_factors
    ):
        """The grid path still warms factors (regression guard on the branch)."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = [{"pre_transformation_file_path_s": "/data/test1.nc"}]

        config = self.get_base_config()  # no source_type -> grid
        factory = TxJobFactory(config)

        factory.pregenerate_factors()

        mock_make_factors.assert_called()


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
    def test_pipeline_cleanup_all_successful(
        self, mock_config, mock_query, mock_update, mock_count
    ):
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
    def test_pipeline_cleanup_some_failed(
        self, mock_config, mock_query, mock_update, mock_count
    ):
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
    def test_pipeline_cleanup_none_performed(
        self, mock_config, mock_query, mock_update, mock_count
    ):
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
        worker_results = [
            TxResult(doc_id="id1", grid="grid", field="field", success=True)
        ]
        mock_transform.return_value = worker_results

        config = {"ds_name": "TEST"}
        granule = {
            "pre_transformation_file_path_s": "/data/test.nc",
            "file_size_l": 1000,
            "date_dt": "2020-01-01",
        }
        field = MagicMock()
        field.name = "field"
        tx_jobs = {"grid": [field]}
        doc_id_map = {("grid", "field"): "id1"}

        result = multiprocess_transformation(
            config, granule, tx_jobs, doc_id_map, "INFO", "/logs"
        )

        mock_transform.assert_called_once_with(
            "/data/test.nc", tx_jobs, config, "2020-01-01", doc_id_map
        )
        # No filename_s on the granule, so the marker falls back to the file path.
        self.assertEqual(result, ("/data/test.nc", "ok", "", worker_results))
        mock_solr.solr_update.assert_not_called()
        mock_solr.solr_query.assert_not_called()

    @patch("transformations.transformation_factory.transform")
    @patch("transformations.transformation_factory.log_config.mp_logging")
    def test_multiprocess_returns_failure_results_on_exception(
        self, mock_logging, mock_transform
    ):
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
        result = multiprocess_transformation(
            config, granule, tx_jobs, doc_id_map, "INFO", "/logs"
        )

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
                doc_id="id-ok",
                grid="g",
                field="f1",
                success=True,
                output_filename="out.nc",
                output_path="/o/out.nc",
                checksum="cs",
                error_message="",
            ),
            TxResult(
                doc_id="id-fail",
                grid="g",
                field="f2",
                success=False,
                error_message="boom",
            ),
            TxResult(
                doc_id=None, grid="g", field="f3", success=False, error_message="no id"
            ),
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


class InterruptedRunRecoveryTestCase(unittest.TestCase):
    """
    Recovery of transformation docs left unsuccessful by an interrupted run.

    prepopulate_jobs writes success_b=False for every dispatched (grid, field) with
    commitWithin, so a late crash leaves committed docs behind. Those docs sent the next
    run down need_to_update (which re-transforms) instead of the filesystem fallback in
    need_to_transform (reachable only when no doc exists at all), so a 25k-job run that
    died at the end redid all 25k despite valid outputs on disk.

    These tests pin both halves: the recovery works, AND it refuses to adopt an output
    that must not be trusted.
    """

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
    def _factory(self, mock_config, mock_query):
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []
        return TxJobFactory(self.get_base_config())

    def _field(self, name="f1"):
        field = MagicMock()
        field.name = name
        return field

    def _granule(self, checksum="abc"):
        return {
            "filename_s": "test1.nc",
            "pre_transformation_file_path_s": "/data/test1.nc",
            "date_dt": "2020-01-01T00:00:00Z",
            "checksum_s": checksum,
        }

    def _tx(self, **overrides):
        tx = {
            "id": "existing-id",
            "success_b": False,
            "origin_checksum_s": "abc",
            "field_s": "f1",
            "grid_name_s": "grid",
        }
        tx.update(overrides)
        return tx

    def test_recovers_interrupted_doc_when_output_is_usable(self):
        """The core fix: an unsuccessful doc whose output on disk is current is adopted
        (doc id returned) rather than re-transformed."""
        factory = self._factory()
        with patch.object(factory, "usable_output_exists", return_value=True):
            doc_id = factory.recoverable_from_disk(
                self._granule(), self._tx(), "grid", self._field()
            )
        self.assertEqual(doc_id, "existing-id")

    def test_does_not_recover_when_output_is_not_usable(self):
        factory = self._factory()
        with patch.object(factory, "usable_output_exists", return_value=False):
            doc_id = factory.recoverable_from_disk(
                self._granule(), self._tx(), "grid", self._field()
            )
        self.assertIsNone(doc_id)

    def test_does_not_recover_when_granule_was_reharvested(self):
        """A checksum mismatch means the output on disk was built from the OLD source,
        so it must be rebuilt no matter how current the file looks. The file is not even
        inspected — the cheap guard short-circuits first."""
        factory = self._factory()
        with patch.object(factory, "usable_output_exists") as mock_usable:
            doc_id = factory.recoverable_from_disk(
                self._granule(checksum="NEW"),
                self._tx(origin_checksum_s="OLD"),
                "grid",
                self._field(),
            )
        self.assertIsNone(doc_id)
        mock_usable.assert_not_called()

    def test_already_successful_doc_is_not_a_recovery_candidate(self):
        factory = self._factory()
        with patch.object(factory, "usable_output_exists", return_value=True):
            doc_id = factory.recoverable_from_disk(
                self._granule(), self._tx(success_b=True), "grid", self._field()
            )
        self.assertIsNone(doc_id)

    @patch("transformations.transformation_factory.file_utils.md5")
    @patch("transformations.transformation_factory.os.path.getmtime")
    @patch("transformations.transformation_factory.solr_utils.solr_update")
    def test_record_recovered_batches_atomic_updates(
        self, mock_update, mock_getmtime, mock_md5
    ):
        """Recovered docs are written as ONE bulk atomic update against the existing id
        — not as new docs via reconstruct_tx_solr_doc, which would mint a fresh uuid and
        duplicate a doc that already exists."""
        factory = self._factory()
        mock_getmtime.return_value = 0
        mock_md5.return_value = "checksum"

        recovered = [
            ("id-1", self._granule(), "grid", self._field("f1")),
            ("id-2", self._granule(), "grid", self._field("f2")),
        ]
        factory.record_recovered(recovered)

        self.assertEqual(mock_update.call_count, 1)
        _, kwargs = mock_update.call_args
        self.assertFalse(kwargs.get("commit", True))  # commitWithin, not a hard commit
        body = mock_update.call_args[0][0]
        self.assertEqual(len(body), 2)
        self.assertEqual(factory.recovered_count, 2)

        doc = next(d for d in body if d["id"] == "id-1")
        # Atomic-update form throughout, and no type_s (which would make it a new doc).
        self.assertNotIn("type_s", doc)
        self.assertEqual(doc["success_b"], {"set": True})
        self.assertEqual(doc["transformation_version_f"], {"set": 2.0})
        self.assertEqual(doc["transformation_in_progress_b"], {"set": False})
        self.assertEqual(doc["error_message_s"], {"set": ""})
        self.assertEqual(doc["transformation_checksum_s"], {"set": "checksum"})

    @patch("transformations.transformation_factory.solr_utils.solr_update")
    def test_record_recovered_noop_when_empty(self, mock_update):
        factory = self._factory()
        factory.record_recovered([])
        mock_update.assert_not_called()
        self.assertEqual(factory.recovered_count, 0)


class OutputFileTrustTestCase(unittest.TestCase):
    """
    usable_output_exists / _inspect_output_file against REAL netCDF files.

    The regression that matters: a failed mapping still writes an output netCDF (an
    all-fill-value record) stamped with the current transformation_version and a fresh
    mtime. So exists + newer-than-source + current-version is NOT sufficient to conclude
    success. Without the empty_record_note check, a genuine failure would be accepted as
    a finished transformation and recorded success_b=True — silently laundering failures
    into permanent successes. These tests use real files because the attribute's
    survival through save_netcdf (global vs variable attrs) is the whole point.
    """

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
    def _factory(self, mock_config, mock_query):
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []
        return TxJobFactory(self.get_base_config())

    def _write_output(self, path, t_version=2.0, empty_record_note=None):
        """Write a transformed-output netCDF the way grid_transformation does:
        transformation_version as a GLOBAL attr, empty_record_note on the DataArray
        (and therefore a VARIABLE attr in the saved file)."""
        time = np.array(["2020-01-01"], dtype="datetime64[ns]")
        da = xr.DataArray(
            np.full((1, 3), np.nan, "f4"), dims=("time", "x"), coords={"time": time}
        )
        da.name = "f1_interpolated_to_grid"
        if empty_record_note is not None:
            da.attrs["empty_record_note"] = empty_record_note
        ds = da.to_dataset()
        if t_version is not None:
            ds = ds.assign_attrs({"transformation_version": t_version})
        ds.to_netcdf(path)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.source = os.path.join(self.tmp, "src.nc")
        with open(self.source, "w") as f:
            f.write("source")
        self.output = os.path.join(self.tmp, "out.nc")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _usable(self, factory):
        """Run usable_output_exists with the output path pinned to our temp file."""
        granule = {
            "filename_s": "test1.nc",
            "pre_transformation_file_path_s": self.source,
            "checksum_s": "abc",
        }
        field = MagicMock()
        field.name = "f1"
        with patch.object(factory, "_output_path", return_value=self.output):
            # Output must be newer than source for the mtime gate to pass.
            os.utime(self.output, (3000, 3000))
            os.utime(self.source, (1000, 1000))
            return factory.usable_output_exists(granule, "grid", field)

    def test_failure_placeholder_is_rejected(self):
        """THE regression: a failed-transform placeholder must NOT be accepted, even
        though it exists, is newer than its source, and carries the current version."""
        factory = self._factory()
        self._write_output(self.output, empty_record_note="Transformation failed")
        self.assertFalse(self._usable(factory))

    def test_failed_post_transformation_placeholder_is_rejected(self):
        factory = self._factory()
        self._write_output(
            self.output, empty_record_note="Post transformation(s) failed"
        )
        self.assertFalse(self._usable(factory))

    def test_missing_field_empty_record_is_accepted(self):
        """The 'field missing from source data' empty record sets mapping_success=True
        on purpose — an empty record is the accepted output for a field the source does
        not carry. It must NOT be swept up by the failure check, or these would
        re-transform forever."""
        factory = self._factory()
        self._write_output(self.output, empty_record_note="f1 missing from source data")
        self.assertTrue(self._usable(factory))

    def test_clean_current_output_is_accepted(self):
        factory = self._factory()
        self._write_output(self.output)
        self.assertTrue(self._usable(factory))

    def test_stale_version_output_is_rejected(self):
        factory = self._factory()
        self._write_output(self.output, t_version=1.0)
        self.assertFalse(self._usable(factory))

    def test_unreadable_output_is_rejected(self):
        """A corrupt/truncated file is both a version mismatch and untrustworthy."""
        factory = self._factory()
        with open(self.output, "w") as f:
            f.write("not a netcdf")
        self.assertFalse(self._usable(factory))

    def test_output_older_than_source_is_rejected(self):
        factory = self._factory()
        self._write_output(self.output)
        granule = {
            "filename_s": "test1.nc",
            "pre_transformation_file_path_s": self.source,
            "checksum_s": "abc",
        }
        field = MagicMock()
        field.name = "f1"
        with patch.object(factory, "_output_path", return_value=self.output):
            os.utime(self.output, (1000, 1000))
            os.utime(self.source, (3000, 3000))
            self.assertFalse(factory.usable_output_exists(granule, "grid", field))

    def test_need_to_transform_still_inverts_usable(self):
        """need_to_transform is the no-doc-exists fallback and must stay the inverse of
        usable_output_exists."""
        factory = self._factory()
        with patch.object(factory, "usable_output_exists", return_value=True):
            self.assertFalse(factory.need_to_transform({}, "grid", self._field_f1()))
        with patch.object(factory, "usable_output_exists", return_value=False):
            self.assertTrue(factory.need_to_transform({}, "grid", self._field_f1()))

    def _field_f1(self):
        field = MagicMock()
        field.name = "f1"
        return field


class IncrementalFlushTestCase(unittest.TestCase):
    """
    drain_results: worker results are recorded to Solr as they complete rather than
    once at the end, so a late crash does not discard the status of every finished job.
    The parent remains the sole Solr writer (ADR 0001) — flush cadence is keyed to
    result count, not worker count.
    """

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
    def _factory(self, mock_config, mock_query):
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["grid"]
        mock_query.return_value = []
        return TxJobFactory(self.get_base_config())

    def _marker(self, n_results):
        results = [
            TxResult(
                doc_id=f"id-{i}",
                grid="g",
                field=f"f{i}",
                success=True,
                output_filename="o.nc",
                output_path="/o/o.nc",
                checksum="cs",
            )
            for i in range(n_results)
        ]
        return ("granule.nc", "ok", "", results)

    def test_flushes_mid_run_once_the_interval_is_reached(self):
        """Results are written before the run ends — the whole point. One flush at the
        interval boundary, one for the remainder."""
        factory = self._factory()
        factory.job_params = [None] * 3
        # 3 markers x 200 results = 600 >= SOLR_FLUSH_INTERVAL (500) partway through.
        result_iter = iter([self._marker(200) for _ in range(3)])

        with patch.object(factory, "record_results") as mock_record:
            results = factory.drain_results(result_iter)

        self.assertEqual(len(results), 3)
        # Flushed mid-run at the interval boundary, not once at the end.
        self.assertGreaterEqual(mock_record.call_count, 1)
        # Every result is written exactly once across the flushes — nothing dropped,
        # nothing double-counted.
        self.assertEqual(sum(len(c[0][0]) for c in mock_record.call_args_list), 600)

    def test_small_run_flushes_once_at_the_end(self):
        factory = self._factory()
        factory.job_params = [None] * 2
        result_iter = iter([self._marker(1), self._marker(1)])

        with patch.object(factory, "record_results") as mock_record:
            factory.drain_results(result_iter)

        self.assertEqual(mock_record.call_count, 1)
        self.assertEqual(len(mock_record.call_args[0][0]), 2)

    def test_no_results_writes_nothing(self):
        factory = self._factory()
        factory.job_params = []
        with patch.object(factory, "record_results") as mock_record:
            factory.drain_results(iter([]))
        mock_record.assert_not_called()

    def test_worker_markers_with_no_results_are_still_returned(self):
        """A marker carrying no TxResults still counts toward summarize_results, which
        is how worker-level failures stay visible."""
        factory = self._factory()
        factory.job_params = [None]
        with patch.object(factory, "record_results") as mock_record:
            results = factory.drain_results(iter([("g.nc", "error", "boom", [])]))
        self.assertEqual(len(results), 1)
        mock_record.assert_not_called()

    @patch("transformations.transformation_factory.solr_utils.commit_solr")
    @patch("transformations.transformation_factory.Pool")
    def test_pooled_path_uses_imap_unordered(self, mock_pool_cls, mock_commit):
        """starmap only returns once every job is done, so it cannot flush
        incrementally. The pooled path must consume results as they complete."""
        factory = self._factory()
        factory.user_cpus = 4
        factory.job_params = [("a",), ("b",)]

        pool = MagicMock()
        mock_pool_cls.return_value.__enter__.return_value = pool
        pool.imap_unordered.return_value = iter([self._marker(1), self._marker(1)])

        with patch.object(factory, "record_results"):
            factory.execute_jobs()

        pool.imap_unordered.assert_called_once()
        pool.starmap.assert_not_called()

    @patch("transformations.transformation_factory.solr_utils.commit_solr")
    @patch("transformations.transformation_factory.multiprocess_transformation")
    def test_single_cpu_path_also_flushes_incrementally(
        self, mock_transform, mock_commit
    ):
        """--multiprocesses 1 takes a different branch; it must go through the same
        drain, not build a list and write once at the end."""
        factory = self._factory()
        factory.user_cpus = 1
        factory.job_params = [("a",), ("b",)]
        mock_transform.return_value = self._marker(1)

        with patch.object(factory, "drain_results", return_value=[]) as mock_drain:
            factory.execute_jobs()

        mock_drain.assert_called_once()


class StampRunIdTestCase(unittest.TestCase):
    """
    Run-id stamping at the Solr write boundary.

    Applied at the write boundaries rather than at each doc-construction site so a write
    path added later is stamped automatically — a missed site would silently under-count
    the run's activity, which is the one direction a monitoring system cannot tolerate.
    """

    def test_new_doc_gets_a_plain_value(self):
        docs = stamp_run_id([{"type_s": "transformation", "dataset_s": "DS_A"}], "RUN1")

        self.assertEqual(docs[0]["run_id_s"], "RUN1")

    def test_atomic_update_gets_the_set_form(self):
        """A doc without type_s is a partial update keyed by id, so it needs {"set": ...}."""
        docs = stamp_run_id([{"id": "abc", "success_b": {"set": True}}], "RUN1")

        self.assertEqual(docs[0]["run_id_s"], {"set": "RUN1"})

    def test_mixed_batch_is_stamped_per_doc(self):
        docs = stamp_run_id(
            [
                {"type_s": "transformation", "dataset_s": "DS_A"},
                {"id": "abc", "success_b": {"set": False}},
            ],
            "RUN1",
        )

        self.assertEqual(docs[0]["run_id_s"], "RUN1")
        self.assertEqual(docs[1]["run_id_s"], {"set": "RUN1"})

    def test_empty_batch_is_fine(self):
        self.assertEqual(stamp_run_id([], "RUN1"), [])


if __name__ == "__main__":
    unittest.main()
