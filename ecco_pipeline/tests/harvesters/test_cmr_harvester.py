"""
Unit tests for CMR harvester.
All Solr and HTTP calls are mocked - no external dependencies required.
"""
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

from harvesters.cmr_harvester import CMR_Harvester, harvester
from harvesters.enumeration.cmr_enumerator import CMRGranule


def get_mock_config():
    """Return a mock configuration dictionary for CMR datasets."""
    return {
        "ds_name": "SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205",
        "start": "20200101T00:00:00Z",
        "end": "20200131T00:00:00Z",
        "harvester_type": "cmr",
        "cmr_concept_id": "C2036882072-POCLOUD",
        "provider": "POCLOUD",
        "filename_date_fmt": "%Y%m%d%H",
        "filename_date_regex": r"\d{10}",
        "data_time_scale": "daily",
        "hemi_pattern": None,
        "fields": [
            {
                "name": "SLA",
                "long_name": "Sea Level Anomaly",
                "standard_name": "sea_surface_height_above_sea_level",
                "units": "m",
                "pre_transformations": [],
                "post_transformations": []
            }
        ],
        "original_dataset_title": "MEaSUREs Gridded Sea Surface Height Anomalies",
        "original_dataset_short_name": "SEA_SURFACE_HEIGHT_ALT_GRIDS",
        "original_dataset_url": "https://podaac.jpl.nasa.gov",
        "original_dataset_reference": "JPL Reference",
        "original_dataset_doi": "10.5067/SLREF-CDRV3",
        "ddir": None,
        "preprocessing": None,
        "t_version": 1.0,
        "a_version": 1.0,
        "notes": ""
    }


def create_mock_cmr_granule(filename, date, provider="POCLOUD"):
    """Helper to create a mock CMRGranule."""
    granule = MagicMock(spec=CMRGranule)
    granule.url = f"https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/TEST/{filename}"
    granule.id = f"G12345-{provider}"
    granule.mod_time = date
    granule.collection_id = "C2036882072-POCLOUD"
    return granule


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.cmr_harvester.CMRQuery")
class CMRHarvesterTestCase(unittest.TestCase):
    """Tests for the CMR_Harvester class."""

    def test_harvester_initialization(self, mock_cmr_query, mock_clean, mock_solr_query):
        """Test CMR_Harvester initializes correctly."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)

                self.assertEqual(h.ds_name, "SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205")
                self.assertEqual(h.cmr_concept_id, "C2036882072-POCLOUD")
                self.assertEqual(h.provider, "POCLOUD")
                mock_cmr_query.assert_called_once()

    def test_fetch_no_granules(self, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch when no granules are available."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)
                h.fetch()

                self.assertEqual(len(h.updated_solr_docs), 0)

    @patch("harvesters.cmr_harvester.requests.get")
    def test_fetch_downloads_file(self, mock_requests, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch downloads files correctly."""
        mock_solr_query.return_value = []

        # Create mock granule
        mock_granule = create_mock_cmr_granule(
            "ssh_grids_v2205_2020011512.nc",
            datetime(2020, 1, 16, 10, 30)
        )
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = [mock_granule]
        mock_cmr_query.return_value = mock_query_instance

        # Mock successful download
        mock_response = MagicMock()
        mock_response.content = b"mock netcdf content"
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)
                h.fetch()

                # Verify download occurred
                mock_requests.assert_called_once()
                self.assertEqual(len(h.updated_solr_docs), 2)  # granule + descendant

    @patch("harvesters.cmr_harvester.requests.get")
    def test_fetch_skips_nrt_files(self, mock_requests, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch skips NRT (Near Real Time) files."""
        mock_solr_query.return_value = []

        # Create mock NRT granule
        mock_granule = create_mock_cmr_granule(
            "ssh_grids_v2205_NRT_2020011512.nc",
            datetime(2020, 1, 16, 10, 30)
        )
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = [mock_granule]
        mock_cmr_query.return_value = mock_query_instance

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)
                h.fetch()

                # Should not download NRT files
                mock_requests.assert_not_called()

    @patch("harvesters.cmr_harvester.requests.get")
    def test_fetch_skips_out_of_range_dates(self, mock_requests, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch skips granules outside date range."""
        mock_solr_query.return_value = []

        # Create mock granule with date outside range (2019)
        mock_granule = create_mock_cmr_granule(
            "ssh_grids_v2205_2019011512.nc",
            datetime(2019, 1, 16, 10, 30)
        )
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = [mock_granule]
        mock_cmr_query.return_value = mock_query_instance

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)
                h.fetch()

                # Should not download
                mock_requests.assert_not_called()

    @patch("harvesters.cmr_harvester.requests.get")
    def test_fetch_handles_download_failure(self, mock_requests, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch handles download failures gracefully."""
        mock_solr_query.return_value = []

        mock_granule = create_mock_cmr_granule(
            "ssh_grids_v2205_2020011512.nc",
            datetime(2020, 1, 16, 10, 30)
        )
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = [mock_granule]
        mock_cmr_query.return_value = mock_query_instance

        # Mock failed download
        mock_requests.side_effect = Exception("Download failed")

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)
                h.fetch()

                # Should still create Solr docs with failure status
                self.assertGreater(len(h.updated_solr_docs), 0)
                granule_doc = [d for d in h.updated_solr_docs if d.get("type_s") == "granule"][0]
                self.assertFalse(granule_doc["harvest_success_b"])

    @patch("harvesters.cmr_harvester.requests.get")
    def test_fetch_skips_existing_up_to_date_files(self, mock_requests, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch skips files that are already up to date."""
        mock_solr_query.return_value = []

        mock_granule = create_mock_cmr_granule(
            "ssh_grids_v2205_2020011512.nc",
            datetime(2020, 1, 16, 10, 30)
        )
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = [mock_granule]
        mock_cmr_query.return_value = mock_query_instance

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)

                # Mark file as already downloaded and up to date
                h.solr_docs["ssh_grids_v2205_2020011512.nc"] = {
                    "harvest_success_b": True,
                    "download_time_dt": "2020-12-01T00:00:00Z"
                }

                h.fetch()

                # Should not download
                mock_requests.assert_not_called()

    @patch("harvesters.cmr_harvester.requests.get")
    def test_fetch_multiple_granules(self, mock_requests, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch handles multiple granules."""
        mock_solr_query.return_value = []

        mock_granules = [
            create_mock_cmr_granule(
                f"ssh_grids_v2205_202001{i:02d}12.nc",
                datetime(2020, 1, i+1, 10, 30)
            )
            for i in range(1, 4)
        ]
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = mock_granules
        mock_cmr_query.return_value = mock_query_instance

        mock_response = MagicMock()
        mock_response.content = b"mock netcdf content"
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)
                h.fetch()

                self.assertEqual(mock_requests.call_count, 3)
                # 3 granules * 2 docs = 6
                self.assertEqual(len(h.updated_solr_docs), 6)

    @patch("harvesters.cmr_harvester.time.sleep")
    @patch("harvesters.cmr_harvester.requests.get")
    def test_dl_file_retries_on_failure(self, mock_requests, mock_sleep, mock_cmr_query, mock_clean, mock_solr_query):
        """Test dl_file retries on initial failure."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance

        # First call fails, second succeeds
        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = Exception("First attempt failed")

        mock_success = MagicMock()
        mock_success.content = b"content"
        mock_success.raise_for_status = MagicMock()

        mock_requests.side_effect = [mock_fail, mock_success]

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)

                # Create temp file path
                test_file = os.path.join(tmpdir, "test.nc")

                # Should succeed after retry
                h.dl_file("https://example.com/file.nc", test_file)

                self.assertEqual(mock_requests.call_count, 2)
                mock_sleep.assert_called_once_with(5)


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.harvesterclasses.solr_utils.solr_update")
@patch("harvesters.cmr_harvester.CMRQuery")
class CMRHarvesterFunctionTestCase(unittest.TestCase):
    """Tests for the harvester() module function."""

    def test_harvester_function_standard(self, mock_cmr_query, mock_update, mock_clean, mock_solr_query):
        """Test the harvester() function runs standard fetch workflow."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                status = harvester(config)

                self.assertIsInstance(status, str)
                mock_update.assert_called()

    def test_harvester_function_atl20(self, mock_cmr_query, mock_update, mock_clean, mock_solr_query):
        """Test the harvester() function calls fetch_atl_daily for ATL20."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = get_mock_config()
        config["ds_name"] = "ATL20_V004_daily"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                with patch.object(CMR_Harvester, 'fetch_atl_daily') as mock_fetch:
                    status = harvester(config)
                    mock_fetch.assert_called_once()

    def test_harvester_function_tellus_grac_grfo(self, mock_cmr_query, mock_update, mock_clean, mock_solr_query):
        """Test the harvester() function calls fetch_tellus_grac_grfo for TELLUS dataset."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = get_mock_config()
        config["ds_name"] = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                with patch.object(CMR_Harvester, 'fetch_tellus_grac_grfo') as mock_fetch:
                    status = harvester(config)
                    mock_fetch.assert_called_once()

    def test_harvester_function_rdeft4(self, mock_cmr_query, mock_update, mock_clean, mock_solr_query):
        """Test the harvester() function calls fetch_rdeft4 for RDEFT4."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = get_mock_config()
        config["ds_name"] = "RDEFT4"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                with patch.object(CMR_Harvester, 'fetch_rdeft4') as mock_fetch:
                    status = harvester(config)
                    mock_fetch.assert_called_once()

    def test_harvester_function_tellus_tolerance(self, mock_cmr_query, mock_update, mock_clean, mock_solr_query):
        """Test the harvester() function calls fetch_tolerance_filter for other TELLUS datasets."""
        mock_solr_query.return_value = []
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = []
        mock_cmr_query.return_value = mock_query_instance
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = get_mock_config()
        config["ds_name"] = "TELLUS_GRFO_L3_OTHER"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                with patch.object(CMR_Harvester, 'fetch_tolerance_filter') as mock_fetch:
                    status = harvester(config)
                    mock_fetch.assert_called_once()


class CMRHarvesterSpecialFetchTestCase(unittest.TestCase):
    """Tests for special fetch methods (fetch_rdeft4, fetch_tolerance_filter, etc.)."""

    @patch("harvesters.harvesterclasses.solr_utils.solr_query")
    @patch("harvesters.harvesterclasses.solr_utils.clean_solr")
    @patch("harvesters.cmr_harvester.CMRQuery")
    @patch("harvesters.cmr_harvester.requests.get")
    def test_fetch_rdeft4_filters_end_of_month(self, mock_requests, mock_cmr_query, mock_clean, mock_solr_query):
        """Test fetch_rdeft4 filters to end-of-month granules."""
        mock_solr_query.return_value = []

        # Create granules for different days
        mock_granules = [
            create_mock_cmr_granule(f"RDEFT4_20200115.nc", datetime(2020, 1, 16)),
            create_mock_cmr_granule(f"RDEFT4_20200131.nc", datetime(2020, 2, 1)),  # End of month
            create_mock_cmr_granule(f"RDEFT4_20200220.nc", datetime(2020, 2, 21)),
        ]
        mock_query_instance = MagicMock()
        mock_query_instance.query.return_value = mock_granules
        mock_cmr_query.return_value = mock_query_instance

        mock_response = MagicMock()
        mock_response.content = b"mock content"
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        config = get_mock_config()
        config["ds_name"] = "RDEFT4"
        config["filename_date_fmt"] = "%Y%m%d"
        config["filename_date_regex"] = r"\d{8}"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CMR_Harvester(config)
                h.fetch_rdeft4()

                # Should only download the end-of-month file
                self.assertEqual(mock_requests.call_count, 1)


if __name__ == "__main__":
    unittest.main()
