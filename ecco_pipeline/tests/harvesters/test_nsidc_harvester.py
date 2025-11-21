"""
Unit tests for NSIDC harvester.
All Solr and HTTP calls are mocked - no external dependencies required.
"""
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

from harvesters.nsidc_harvester import NSIDC_Harvester, harvester
from harvesters.enumeration.nsidc_enumerator import NSIDCGranule


def get_mock_config():
    """Return a mock configuration dictionary for NSIDC datasets."""
    return {
        "ds_name": "G10016_V3",
        "start": "20200101T00:00:00Z",
        "end": "20200131T00:00:00Z",
        "harvester_type": "nsidc",
        "filename_date_fmt": "%Y%m%d",
        "filename_date_regex": r"\d{8}",
        "data_time_scale": "daily",
        "hemi_pattern": {"north": "_nh_", "south": "_sh_"},
        "fields": [
            {
                "name": "cdr_seaice_conc",
                "long_name": "Sea Ice Concentration",
                "standard_name": "sea_ice_area_fraction",
                "units": "1",
                "pre_transformations": [],
                "post_transformations": []
            }
        ],
        "original_dataset_title": "NOAA/NSIDC Climate Data Record",
        "original_dataset_short_name": "G10016",
        "original_dataset_url": "https://nsidc.org/data/g10016",
        "original_dataset_reference": "NSIDC Reference",
        "original_dataset_doi": "10.7265/N59P2ZTG",
        "ddir": "CDR",
        "preprocessing": None,
        "t_version": 1.0,
        "a_version": 1.0,
        "notes": ""
    }


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.nsidc_harvester.search_nsidc")
class NSIDCHarvesterTestCase(unittest.TestCase):
    """Tests for the NSIDC_Harvester class."""

    def test_harvester_initialization(self, mock_search, mock_clean, mock_query):
        """Test NSIDC_Harvester initializes correctly."""
        mock_query.return_value = []
        mock_search.return_value = []

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = NSIDC_Harvester(config)

                self.assertEqual(harvester.ds_name, "G10016_V3")
                self.assertEqual(harvester.ddir, "CDR")
                mock_search.assert_called_once()

    def test_fetch_no_granules(self, mock_search, mock_clean, mock_query):
        """Test fetch when no granules are available."""
        mock_query.return_value = []
        mock_search.return_value = []

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = NSIDC_Harvester(config)
                h.fetch()

                self.assertEqual(len(h.updated_solr_docs), 0)

    @patch("harvesters.nsidc_harvester.requests.get")
    def test_fetch_downloads_file(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch downloads files correctly."""
        mock_query.return_value = []

        # Create mock granule
        mock_granule = NSIDCGranule(
            url="https://example.com/seaice_conc_daily_nh_20200115.nc",
            mod_time=datetime(2020, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        # Mock successful download
        mock_response = MagicMock()
        mock_response.content = b"mock netcdf content"
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = NSIDC_Harvester(config)
                h.fetch()

                # Verify download occurred
                mock_requests.assert_called_once()
                self.assertEqual(len(h.updated_solr_docs), 2)  # granule + descendant

    @patch("harvesters.nsidc_harvester.requests.get")
    def test_fetch_skips_out_of_range_dates(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch skips granules outside date range."""
        mock_query.return_value = []

        # Create mock granule with date outside range
        mock_granule = NSIDCGranule(
            url="https://example.com/seaice_conc_daily_nh_20190115.nc",  # 2019
            mod_time=datetime(2019, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = NSIDC_Harvester(config)
                h.fetch()

                # Should not download
                mock_requests.assert_not_called()

    @patch("harvesters.nsidc_harvester.requests.get")
    def test_fetch_handles_download_failure(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch handles download failures gracefully."""
        mock_query.return_value = []

        mock_granule = NSIDCGranule(
            url="https://example.com/seaice_conc_daily_nh_20200115.nc",
            mod_time=datetime(2020, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        # Mock failed download
        mock_requests.side_effect = Exception("Download failed")

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = NSIDC_Harvester(config)
                h.fetch()

                # Should still create Solr docs with failure status
                self.assertGreater(len(h.updated_solr_docs), 0)
                # Check that harvest_success_b is False
                granule_doc = [d for d in h.updated_solr_docs if d.get("type_s") == "granule"][0]
                self.assertFalse(granule_doc["harvest_success_b"])

    @patch("harvesters.nsidc_harvester.requests.get")
    def test_fetch_skips_existing_up_to_date_files(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch skips files that are already up to date."""
        # Setup existing Solr doc
        mock_query.return_value = []

        mock_granule = NSIDCGranule(
            url="https://example.com/seaice_conc_daily_nh_20200115.nc",
            mod_time=datetime(2020, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = NSIDC_Harvester(config)

                # Mark file as already downloaded and up to date
                h.solr_docs["seaice_conc_daily_nh_20200115.nc"] = {
                    "harvest_success_b": True,
                    "download_time_dt": "2020-12-01T00:00:00Z"  # Downloaded later than mod_time
                }

                h.fetch()

                # Should not download
                mock_requests.assert_not_called()


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.harvesterclasses.solr_utils.solr_update")
@patch("harvesters.nsidc_harvester.search_nsidc")
class NSIDCHarvesterFunctionTestCase(unittest.TestCase):
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
