"""
Unit tests for CATDS harvester.
All Solr and HTTP calls are mocked - no external dependencies required.
"""
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

from harvesters.catds_harvester import CATDS_Harvester, harvester
from harvesters.enumeration.catds_enumerator import CATDSGranule


def get_mock_config():
    """Return a mock configuration dictionary for CATDS datasets."""
    return {
        "ds_name": "L3_DEBIAS_LOCEAN_v10_q09",
        "start": "20200101T00:00:00Z",
        "end": "20200131T00:00:00Z",
        "harvester_type": "catds",
        "filename_date_fmt": "%Y%m%d",
        "filename_date_regex": r"\d{8}",
        "data_time_scale": "daily",
        "hemi_pattern": None,
        "fields": [
            {
                "name": "sss",
                "long_name": "Sea Surface Salinity",
                "standard_name": "sea_surface_salinity",
                "units": "psu",
                "pre_transformations": [],
                "post_transformations": []
            }
        ],
        "original_dataset_title": "SMOS L3 Sea Surface Salinity",
        "original_dataset_short_name": "LOCEAN_v10",
        "original_dataset_url": "https://www.catds.fr",
        "original_dataset_reference": "CATDS Reference",
        "original_dataset_doi": "10.12770/xxxx",
        "ddir": "L3_DEBIAS_LOCEAN_v10",
        "preprocessing": None,
        "t_version": 1.0,
        "a_version": 1.0,
        "notes": ""
    }


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.catds_harvester.search_catds")
class CATDSHarvesterTestCase(unittest.TestCase):
    """Tests for the CATDS_Harvester class."""

    def test_harvester_initialization(self, mock_search, mock_clean, mock_query):
        """Test CATDS_Harvester initializes correctly."""
        mock_query.return_value = []
        mock_search.return_value = []

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CATDS_Harvester(config)

                self.assertEqual(h.ds_name, "L3_DEBIAS_LOCEAN_v10_q09")
                self.assertEqual(h.ddir, "L3_DEBIAS_LOCEAN_v10")
                mock_search.assert_called_once()

    def test_fetch_no_granules(self, mock_search, mock_clean, mock_query):
        """Test fetch when no granules are available."""
        mock_query.return_value = []
        mock_search.return_value = []

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CATDS_Harvester(config)
                h.fetch()

                self.assertEqual(len(h.updated_solr_docs), 0)

    @patch("harvesters.catds_harvester.requests.get")
    def test_fetch_downloads_file(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch downloads files correctly."""
        mock_query.return_value = []

        mock_granule = CATDSGranule(
            url="https://data.catds.fr/cecos-locean/Ocean_products/L3_DEBIAS_LOCEAN_v10/SM_OPER_20200115.nc",
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
                h = CATDS_Harvester(config)
                h.fetch()

                mock_requests.assert_called_once()
                self.assertEqual(len(h.updated_solr_docs), 2)

    @patch("harvesters.catds_harvester.requests.get")
    def test_fetch_skips_out_of_range_dates(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch skips granules outside date range."""
        mock_query.return_value = []

        mock_granule = CATDSGranule(
            url="https://data.catds.fr/cecos-locean/Ocean_products/L3_DEBIAS_LOCEAN_v10/SM_OPER_20190115.nc",
            mod_time=datetime(2019, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CATDS_Harvester(config)
                h.fetch()

                mock_requests.assert_not_called()

    @patch("harvesters.catds_harvester.requests.get")
    def test_fetch_handles_download_failure(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch handles download failures gracefully."""
        mock_query.return_value = []

        mock_granule = CATDSGranule(
            url="https://data.catds.fr/cecos-locean/Ocean_products/L3_DEBIAS_LOCEAN_v10/SM_OPER_20200115.nc",
            mod_time=datetime(2020, 1, 16, 10, 30)
        )
        mock_search.return_value = [mock_granule]

        mock_requests.side_effect = Exception("Download failed")

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CATDS_Harvester(config)
                h.fetch()

                self.assertGreater(len(h.updated_solr_docs), 0)
                granule_doc = [d for d in h.updated_solr_docs if d.get("type_s") == "granule"][0]
                self.assertFalse(granule_doc["harvest_success_b"])

    @patch("harvesters.catds_harvester.requests.get")
    def test_fetch_multiple_granules(self, mock_requests, mock_search, mock_clean, mock_query):
        """Test fetch handles multiple granules."""
        mock_query.return_value = []

        mock_granules = [
            CATDSGranule(
                url=f"https://data.catds.fr/cecos-locean/Ocean_products/L3_DEBIAS_LOCEAN_v10/SM_OPER_202001{i:02d}.nc",
                mod_time=datetime(2020, 1, i+1, 10, 30)
            )
            for i in range(15, 18)
        ]
        mock_search.return_value = mock_granules

        mock_response = MagicMock()
        mock_response.content = b"mock netcdf content"
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        config = get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                h = CATDS_Harvester(config)
                h.fetch()

                self.assertEqual(mock_requests.call_count, 3)
                # 3 granules * 2 docs (granule + descendant) = 6
                self.assertEqual(len(h.updated_solr_docs), 6)


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
@patch("harvesters.harvesterclasses.solr_utils.solr_update")
@patch("harvesters.catds_harvester.search_catds")
class CATDSHarvesterFunctionTestCase(unittest.TestCase):
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
