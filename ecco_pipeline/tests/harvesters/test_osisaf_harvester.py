"""
Unit tests for OSISAF harvester.
All Solr and HTTP calls are mocked - no external dependencies required.
"""
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

from harvesters.osisaf_harvester import OSISAF_Harvester, harvester
from harvesters.enumeration.osisaf_enumerator import OSISAFGranule


def get_mock_config():
    """Return a mock configuration dictionary for OSISAF datasets."""
    return {
        "ds_name": "SSMIS_OSI-450-a_nh",
        "start": "20200101T00:00:00Z",
        "end": "20200131T00:00:00Z",
        "harvester_type": "osisaf",
        "filename_date_fmt": "%Y%m%d",
        "filename_date_regex": r"\d{8}",
        "data_time_scale": "daily",
        "hemi_pattern": {"north": "_nh_", "south": "_sh_"},
        "fields": [
            {
                "name": "ice_conc",
                "long_name": "Sea Ice Concentration",
                "standard_name": "sea_ice_area_fraction",
                "units": "1",
                "pre_transformations": [],
                "post_transformations": []
            }
        ],
        "original_dataset_title": "OSISAF Sea Ice Concentration",
        "original_dataset_short_name": "OSI-450-a",
        "original_dataset_url": "https://osi-saf.eumetsat.int",
        "original_dataset_reference": "OSISAF Reference",
        "original_dataset_doi": "10.15770/EUM_SAF_OSI",
        "ddir": "ice/conc_cdr_cont",
        "preprocessing": None,
        "t_version": 1.0,
        "a_version": 1.0,
        "notes": ""
    }


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.osisaf_harvester.search_osisaf")
class OSISAFHarvesterTestCase(unittest.TestCase):
    """Tests for the OSISAF_Harvester class."""

    def test_harvester_initialization(self, mock_search, mock_clean, mock_query):
        """Test OSISAF_Harvester initializes correctly."""
        mock_query.return_value = []
        mock_search.return_value = []

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = OSISAF_Harvester(config)

                self.assertEqual(h.ds_name, "SSMIS_OSI-450-a_nh")
                self.assertEqual(h.ddir, "ice/conc_cdr_cont")
                mock_search.assert_called_once()

    def test_fetch_no_granules(self, mock_search, mock_clean, mock_query):
        """Test fetch when no granules are available."""
        mock_query.return_value = []
        mock_search.return_value = []

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = OSISAF_Harvester(config)
                h.fetch()

                self.assertEqual(len(h.updated_solr_docs), 0)

    @patch("harvesters.osisaf_harvester.requests.get")
    def test_fetch_downloads_file(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch downloads files correctly."""
        mock_query.return_value = []

        mock_granule = OSISAFGranule(
            url="https://thredds.met.no/thredds/fileServer/ice_conc_nh_20200115.nc",
            mod_time=datetime(2020, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        mock_response = MagicMock()
        mock_response.content = b"mock netcdf content"
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = OSISAF_Harvester(config)
                h.fetch()

                mock_requests.assert_called_once()
                self.assertEqual(len(h.updated_solr_docs), 2)

    @patch("harvesters.osisaf_harvester.requests.get")
    def test_fetch_skips_icdrft_files(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch skips fast track (icdrft) files."""
        mock_query.return_value = []

        # Create mock granule with icdrft in filename
        mock_granule = OSISAFGranule(
            url="https://thredds.met.no/thredds/fileServer/ice_conc_nh_icdrft_20200115.nc",
            mod_time=datetime(2020, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = OSISAF_Harvester(config)
                h.fetch()

                # Should not download icdrft files
                mock_requests.assert_not_called()

    @patch("harvesters.osisaf_harvester.requests.get")
    def test_fetch_handles_download_failure(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch handles download failures gracefully."""
        mock_query.return_value = []

        mock_granule = OSISAFGranule(
            url="https://thredds.met.no/thredds/fileServer/ice_conc_nh_20200115.nc",
            mod_time=datetime(2020, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        mock_requests.side_effect = Exception("Download failed")

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = OSISAF_Harvester(config)
                h.fetch()

                self.assertGreater(len(h.updated_solr_docs), 0)
                granule_doc = [d for d in h.updated_solr_docs if d.get("type_s") == "granule"][0]
                self.assertFalse(granule_doc["harvest_success_b"])


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.harvesterclasses.solr_utils.solr_update")
@patch("harvesters.osisaf_harvester.search_osisaf")
class OSISAFHarvesterFunctionTestCase(unittest.TestCase):
    """Tests for the harvester() module function."""

    def test_harvester_function(self, mock_search, mock_update, mock_clean, mock_query):
        """Test the harvester() function runs complete workflow."""
        mock_query.return_value = []
        mock_search.return_value = []
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                status = harvester(config)

                self.assertIsInstance(status, str)
                mock_update.assert_called()


if __name__ == "__main__":
    unittest.main()
