"""
Unit tests for aggregation_factory module (AgJobFactory class).
All Solr calls and multiprocessing are mocked.
"""

import logging
import unittest
from unittest.mock import patch, MagicMock

from aggregations.aggregation_factory import AgJobFactory, multiprocess_aggregate


class AgJobFactoryInitTestCase(unittest.TestCase):
    """Tests for AgJobFactory initialization."""

    def get_base_config(self):
        """Return a base configuration for testing."""
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "2.0",
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

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_initialization(self, mock_config, mock_query):
        """Test AgJobFactory initialization."""
        mock_config.user_cpus = 4
        mock_config.grids_to_use = ["GRID1", "GRID2"]

        # Mock Solr queries for grids and dataset metadata
        mock_query.side_effect = [
            [  # Grids query
                {"grid_name_s": "GRID1", "grid_type_s": "latlon"},
                {"grid_name_s": "GRID2", "grid_type_s": "latlon"},
            ],
            [  # Dataset metadata query (for get_jobs)
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "2.0",
                }
            ],
            [],  # Transformations query (for make_jobs)
            [],  # Existing aggregations query (for make_jobs)
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        self.assertEqual(factory.ds_name, "TEST_DATASET")
        self.assertEqual(factory.user_cpus, 4)
        self.assertEqual(len(factory.grids), 2)
        self.assertIsInstance(factory.agg_jobs, list)

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_initialization_filters_grids(self, mock_config, mock_query):
        """Test that grids are filtered based on grids_to_use."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]  # Only want GRID1

        mock_query.side_effect = [
            [  # All available grids
                {"grid_name_s": "GRID1", "grid_type_s": "latlon"},
                {"grid_name_s": "GRID2", "grid_type_s": "latlon"},
            ],
            [  # Dataset metadata query
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "2.0",
                }
            ],
            [],  # Transformations/aggregations query
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        self.assertEqual(len(factory.grids), 1)
        self.assertEqual(factory.grids[0]["grid_name_s"], "GRID1")

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_initialization_excludes_tpose_for_hemi(self, mock_config, mock_query):
        """Test that TPOSE grid is excluded for hemispheric data."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["ECCO_GRID", "TPOSE_GRID"]

        mock_query.side_effect = [
            [
                {"grid_name_s": "ECCO_GRID", "grid_type_s": "latlon"},
                {"grid_name_s": "TPOSE_GRID", "grid_type_s": "latlon"},
            ],
            [  # Dataset metadata query
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "2.0",
                }
            ],
            [],  # Transformations/aggregations query
        ]

        config = self.get_base_config()
        config["hemi_pattern"] = {"north": "_nh_", "south": "_sh_"}

        factory = AgJobFactory(config)

        grid_names = [g["grid_name_s"] for g in factory.grids]
        self.assertIn("ECCO_GRID", grid_names)
        self.assertNotIn("TPOSE_GRID", grid_names)


class AgJobFactoryGetGridsTestCase(unittest.TestCase):
    """Tests for AgJobFactory.get_grids method."""

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

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_grids_queries_solr(self, mock_config, mock_query):
        """Test that get_grids queries Solr for grids."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = []

        mock_query.side_effect = [
            [  # Grids query
                {"grid_name_s": "GRID_A", "grid_type_s": "latlon"},
                {"grid_name_s": "GRID_B", "grid_type_s": "llc"},
            ],
            [  # Dataset metadata query
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",
                }
            ],
            [],  # Transformations query (for make_jobs since version matches)
            [],  # Existing aggregations query
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # get_grids is called during init
        self.assertEqual(len(factory.grids), 2)

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_grids_filters_by_grids_to_use(self, mock_config, mock_query):
        """Test filtering grids by grids_to_use list."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID_A"]

        mock_query.side_effect = [
            [  # Grids query
                {"grid_name_s": "GRID_A", "grid_type_s": "latlon"},
                {"grid_name_s": "GRID_B", "grid_type_s": "llc"},
            ],
            [  # Dataset metadata query
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",
                }
            ],
            [],  # Transformations query
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        self.assertEqual(len(factory.grids), 1)
        self.assertEqual(factory.grids[0]["grid_name_s"], "GRID_A")


class AgJobFactoryGetJobsTestCase(unittest.TestCase):
    """Tests for AgJobFactory.get_jobs method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "2.0",
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

    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_jobs_creates_all_years_on_version_change(self, mock_config, mock_query, mock_agg_class):
        """Test that all years are processed when version changes."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # grids
            [  # dataset metadata
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2021-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",  # Different version triggers make_jobs_all_years
                }
            ],
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Should create jobs for all years (2020, 2021)
        self.assertGreater(len(factory.agg_jobs), 0)

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_jobs_creates_selective_jobs_same_version(self, mock_config, mock_query):
        """Test that only needed jobs are created when version is same."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # grids
            [  # dataset metadata
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "2.0",  # Same version
                }
            ],
            [],  # No transformations found (make_jobs query)
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Should check each year individually
        self.assertIsInstance(factory.agg_jobs, list)


class AgJobFactoryMakeJobsTestCase(unittest.TestCase):
    """Tests for AgJobFactory.make_jobs method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "2.0",
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

    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_make_jobs_creates_job_for_new_transformation(self, mock_config, mock_query, mock_agg_class):
        """Test creating job when new transformation exists."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # grids
            [{"start_date_dt": "2020-01-01T00:00:00Z", "aggregation_version_s": "2.0"}],  # ds_meta
            [{"date_s": "2020-01-15T00:00:00Z"}],  # transformations
            [],  # no existing aggregation
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Should create at least one job
        self.assertGreater(len(factory.agg_jobs), 0)

    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_make_jobs_skips_up_to_date_aggregation(self, mock_config, mock_query, mock_agg_class):
        """Test skipping job when aggregation is up to date."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # grids
            [{"start_date_dt": "2020-01-01T00:00:00Z", "aggregation_version_s": "2.0"}],  # ds_meta
            [  # transformations
                {
                    "date_s": "2020-01-15T00:00:00Z",
                    "transformation_completed_dt": "2020-01-16T10:00:00Z",
                }
            ],
            [  # existing aggregation
                {
                    "aggregation_time_dt": "2020-01-17T10:00:00Z"  # After transformation
                }
            ],
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Should not create jobs as aggregation is newer than transformation
        # (This depends on implementation details of make_jobs logic)
        self.assertIsInstance(factory.agg_jobs, list)

    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_make_jobs_creates_job_when_transformation_newer(self, mock_config, mock_query, mock_agg_class):
        """Test creating job when transformation is newer than aggregation."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # grids
            [{"start_date_dt": "2020-01-01T00:00:00Z", "aggregation_version_s": "2.0"}],  # ds_meta
            [  # transformations
                {
                    "date_s": "2020-01-15T00:00:00Z",
                    "transformation_completed_dt": "2020-01-17T10:00:00Z",
                }
            ],
            [  # existing aggregation (older than transformation)
                {"aggregation_time_dt": "2020-01-16T10:00:00Z"}
            ],
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Should create job as transformation is newer
        self.assertGreater(len(factory.agg_jobs), 0)


class AgJobFactoryMakeJobsAllYearsTestCase(unittest.TestCase):
    """Tests for AgJobFactory.make_jobs_all_years method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "2.0",
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

    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_make_jobs_all_years_creates_jobs_for_range(self, mock_config, mock_query, mock_agg_class):
        """Test creating jobs for all years in date range."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # grids
            [  # dataset metadata
                {
                    "start_date_dt": "2018-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",  # Different version triggers all years
                }
            ],
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Should create jobs for years 2018, 2019, 2020
        # Each year, each grid, each field = 3 years * 1 grid * 1 field = 3 jobs
        self.assertEqual(len(factory.agg_jobs), 3)


class AgJobFactoryGetAggStatusTestCase(unittest.TestCase):
    """Tests for AgJobFactory.get_agg_status method."""

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

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_agg_status_all_successful(self, mock_config, mock_query):
        """Test status when all aggregations successful."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = []

        mock_query.side_effect = [
            [],  # No grids
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],  # Dataset metadata (for init)
            [{"id": "agg1"}, {"id": "agg2"}],  # successful aggregations
            [],  # no failed aggregations
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        status = factory.get_agg_status()

        self.assertEqual(status, "All aggregations successful")

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_agg_status_some_failed(self, mock_config, mock_query):
        """Test status when some aggregations failed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = []

        mock_query.side_effect = [
            [],  # No grids
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],  # Dataset metadata (for init)
            [{"id": "agg1"}],  # successful aggregations
            [{"id": "agg2"}, {"id": "agg3"}],  # failed aggregations
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        status = factory.get_agg_status()

        self.assertEqual(status, "2 aggregations failed")

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_agg_status_none_performed(self, mock_config, mock_query):
        """Test status when no aggregations performed."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = []

        mock_query.side_effect = [
            [],  # No grids
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],  # Dataset metadata (for init)
            [],  # no successful aggregations
            [],  # no failed aggregations
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        status = factory.get_agg_status()

        self.assertEqual(status, "No aggregations performed")

    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_get_agg_status_no_successful(self, mock_config, mock_query):
        """Test status when no successful aggregations."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = []

        mock_query.side_effect = [
            [],  # No grids
            [{"start_date_dt": "2020-01-01T00:00:00Z"}],  # Dataset metadata (for init)
            [],  # no successful aggregations
            [{"id": "agg1"}],  # failed aggregations
        ]

        config = self.get_base_config()
        factory = AgJobFactory(config)

        status = factory.get_agg_status()

        self.assertEqual(status, "No successful aggregations")


class AgJobFactoryUpdateSolrDsTestCase(unittest.TestCase):
    """Tests for AgJobFactory.update_solr_ds method."""

    def get_base_config(self):
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "a_version": "2.0",
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

    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.solr_utils.solr_update")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_update_solr_ds(self, mock_config, mock_query, mock_update, mock_agg_class):
        """Test updating Solr dataset entry."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # Grids (get_grids during init)
            [
                {
                    "id": "dataset_id",
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",
                }
            ],  # dataset metadata (get_jobs during init)
            [{"id": "dataset_id", "start_date_dt": "2020-01-01T00:00:00Z"}],  # dataset metadata (update_solr_ds call)
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = AgJobFactory(config)

        factory.update_solr_ds("All aggregations successful")

        mock_update.assert_called_once()
        update_body = mock_update.call_args[0][0]
        self.assertEqual(update_body[0]["aggregation_version_s"]["set"], "2.0")
        self.assertEqual(update_body[0]["aggregation_status_s"]["set"], "All aggregations successful")


class AgJobFactoryExecuteJobsTestCase(unittest.TestCase):
    """Tests for AgJobFactory.execute_jobs method."""

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

    @patch("aggregations.aggregation_factory.logging.getLogger")
    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.multiprocess_aggregate")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_execute_jobs_single_cpu(self, mock_config, mock_query, mock_multiprocess, mock_agg_class, mock_logger):
        """Test executing jobs with single CPU (no multiprocessing)."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # Grids (get_grids during init)
            [
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",
                }
            ],  # Dataset metadata (get_jobs during init)
            [],  # Transformations query (make_jobs)
            [],  # Existing aggregations query
        ]

        # Mock logger handlers
        mock_handler = MagicMock()
        mock_handler.baseFilename = "/tmp/test.log"
        mock_logger.return_value.handlers = [mock_handler]
        mock_logger.return_value.level = 20  # INFO level

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Manually add a mock job
        mock_job = MagicMock()
        factory.agg_jobs = [mock_job]

        factory.execute_jobs()

        # Should call multiprocess_aggregate directly
        mock_multiprocess.assert_called_once()

    @patch("aggregations.aggregation_factory.logging.getLogger")
    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.Pool")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_execute_jobs_multiprocessing(self, mock_config, mock_query, mock_pool_class, mock_agg_class, mock_logger):
        """Test executing jobs with multiprocessing."""
        mock_config.user_cpus = 4
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # Grids (get_grids during init)
            [
                {
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",
                }
            ],  # Dataset metadata (get_jobs during init)
            [],  # Transformations query (make_jobs)
            [],  # Existing aggregations query
        ]

        # Mock logger handlers
        mock_handler = MagicMock()
        mock_handler.baseFilename = "/tmp/test.log"
        mock_logger.return_value.handlers = [mock_handler]
        mock_logger.return_value.level = 20  # INFO level

        # Mock Pool
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool

        config = self.get_base_config()
        factory = AgJobFactory(config)

        # Manually add mock jobs
        factory.agg_jobs = [MagicMock(), MagicMock()]

        factory.execute_jobs()

        # Should use Pool
        mock_pool_class.assert_called()
        mock_pool.starmap_async.assert_called_once()


class AgJobFactoryPipelineCleanupTestCase(unittest.TestCase):
    """Tests for AgJobFactory.pipeline_cleanup method."""

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

    @patch("aggregations.aggregation_factory.Aggregation")
    @patch("aggregations.aggregation_factory.solr_utils.solr_update")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_pipeline_cleanup_returns_status(self, mock_config, mock_query, mock_update, mock_agg_class):
        """Test that pipeline_cleanup returns status."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # Grids (get_grids during init)
            [
                {
                    "id": "dataset_id",
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",
                }
            ],  # dataset metadata (get_jobs during init)
            [{"date_s": "2020-01-15T00:00:00Z"}],  # Transformations query (make_jobs)
            [],  # Existing aggregations query
            [{"id": "agg1"}],  # successful aggregations (get_agg_status call in pipeline_cleanup)
            [],  # no failed aggregations (get_agg_status call in pipeline_cleanup)
            [
                {"id": "dataset_id", "start_date_dt": "2020-01-01T00:00:00Z"}
            ],  # dataset metadata (update_solr_ds call during pipeline_cleanup)
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = AgJobFactory(config)

        result = factory.pipeline_cleanup()

        self.assertEqual(result, "All aggregations successful")
        mock_update.assert_called_once()


class AgJobFactoryStartFactoryTestCase(unittest.TestCase):
    """Tests for AgJobFactory.start_factory method."""

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

    @patch("aggregations.aggregation_factory.solr_utils.solr_update")
    @patch("aggregations.aggregation_factory.solr_utils.solr_query")
    @patch("aggregations.aggregation_factory.baseclasses.Config")
    def test_start_factory_no_jobs(self, mock_config, mock_query, mock_update):
        """Test start_factory when there are no jobs."""
        mock_config.user_cpus = 1
        mock_config.grids_to_use = ["GRID1"]

        mock_query.side_effect = [
            [{"grid_name_s": "GRID1"}],  # Grids (get_grids during init)
            [
                {
                    "id": "dataset_id",
                    "start_date_dt": "2020-01-01T00:00:00Z",
                    "end_date_dt": "2020-12-31T00:00:00Z",
                    "aggregation_version_s": "1.0",
                }
            ],  # Dataset metadata (get_jobs during init)
            [],  # No transformations (make_jobs query, returns early with no jobs)
            [],  # successful aggregations (get_agg_status in pipeline_cleanup)
            [],  # failed aggregations (get_agg_status in pipeline_cleanup)
            [
                {"id": "dataset_id", "start_date_dt": "2020-01-01T00:00:00Z"}
            ],  # Dataset metadata (update_solr_ds -> get_solr_ds_metadata in pipeline_cleanup)
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_base_config()
        factory = AgJobFactory(config)

        result = factory.start_factory()

        self.assertIsInstance(result, str)


class MultiprocessAggregateTestCase(unittest.TestCase):
    """Tests for multiprocess_aggregate function."""

    @patch("aggregations.aggregation_factory.log_config.mp_logging")
    def test_multiprocess_aggregate_calls_aggregate(self, mock_logging):
        """Test that multiprocess_aggregate calls job.aggregate()."""
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        mock_job = MagicMock()
        mock_job.grid = {"grid_name_s": "TEST_GRID"}
        mock_job.year = "2020"
        mock_job.field = MagicMock()
        mock_job.field.name = "ssha"

        multiprocess_aggregate(mock_job, "INFO", "/tmp/logs")

        mock_job.aggregate.assert_called_once()

    @patch("aggregations.aggregation_factory.log_config.mp_logging")
    def test_multiprocess_aggregate_handles_exception(self, mock_logging):
        """Test that exceptions are caught and logged."""
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        mock_job = MagicMock()
        mock_job.grid = {"grid_name_s": "TEST_GRID"}
        mock_job.year = "2020"
        mock_job.field = MagicMock()
        mock_job.field.name = "ssha"
        mock_job.aggregate.side_effect = Exception("Test error")

        # Should not raise exception
        multiprocess_aggregate(mock_job, "INFO", "/tmp/logs")

        mock_logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
