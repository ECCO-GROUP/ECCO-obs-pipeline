"""
Unit tests for CMR enumerator.
All external calls (CMR API) are mocked - no external dependencies required.
"""
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

from harvesters.enumeration.cmr_enumerator import CMRGranule, CMRQuery, URLNotFound


class CMRGranuleTestCase(unittest.TestCase):
    """Tests for the CMRGranule class."""

    def get_mock_query_result(self, provider="POCLOUD"):
        """Return a mock CMR query result."""
        return {
            "id": "G12345-TEST",
            "updated": "2020-01-15T12:30:45.123Z",
            "collection_concept_id": "C12345-TEST",
            "links": [
                {
                    "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                    "href": f"https://archive.earthdata.nasa.gov/{provider}/data/file.nc"
                },
                {
                    "rel": "http://esipfed.org/ns/fedsearch/1.1/metadata#",
                    "href": "https://example.com/metadata.xml"
                }
            ]
        }

    def test_cmr_granule_initialization(self):
        """Test CMRGranule initializes correctly."""
        query_result = self.get_mock_query_result("POCLOUD")
        granule = CMRGranule(query_result, "POCLOUD")

        self.assertEqual(granule.id, "G12345-TEST")
        self.assertEqual(granule.collection_id, "C12345-TEST")
        self.assertIsInstance(granule.mod_time, datetime)
        self.assertEqual(granule.mod_time.year, 2020)
        self.assertEqual(granule.mod_time.month, 1)
        self.assertEqual(granule.mod_time.day, 15)

    def test_cmr_granule_url_extraction(self):
        """Test URL extraction from links."""
        query_result = self.get_mock_query_result("POCLOUD")
        granule = CMRGranule(query_result, "POCLOUD")

        self.assertIn("POCLOUD", granule.url)
        self.assertTrue(granule.url.endswith(".nc"))

    def test_cmr_granule_multiple_providers(self):
        """Test URL extraction with different providers."""
        providers = ["POCLOUD", "NSIDC_ECS", "GES_DISC", "LPDAAC_ECS"]

        for provider in providers:
            query_result = self.get_mock_query_result(provider)
            granule = CMRGranule(query_result, provider)
            self.assertIn(provider, granule.url)

    def test_cmr_granule_url_not_found(self):
        """Test URLNotFound exception when provider URL missing."""
        query_result = {
            "id": "G12345-TEST",
            "updated": "2020-01-15T12:30:45.123Z",
            "collection_concept_id": "C12345-TEST",
            "links": [
                {
                    "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                    "href": "https://example.com/other_provider/file.nc"
                }
            ]
        }

        with self.assertRaises(URLNotFound):
            CMRGranule(query_result, "POCLOUD")

    def test_cmr_granule_wrong_provider(self):
        """Test URLNotFound when provider doesn't match available links."""
        query_result = self.get_mock_query_result("POCLOUD")

        with self.assertRaises(URLNotFound):
            CMRGranule(query_result, "WRONG_PROVIDER")


class CMRQueryTestCase(unittest.TestCase):
    """Tests for the CMRQuery class."""

    def get_mock_harvester(self):
        """Return a mock Harvester object."""
        harvester = MagicMock()
        harvester.cmr_concept_id = "C12345-TEST"
        harvester.start = datetime(2020, 1, 1)
        harvester.end = datetime(2020, 1, 31)
        harvester.provider = "POCLOUD"
        return harvester

    def test_cmr_query_initialization(self):
        """Test CMRQuery initializes correctly."""
        harvester = self.get_mock_harvester()
        query = CMRQuery(harvester)

        self.assertEqual(query.concept_id, "C12345-TEST")
        self.assertEqual(query.start_date, datetime(2020, 1, 1))
        self.assertEqual(query.end_date, datetime(2020, 1, 31))
        self.assertEqual(query.provider, "POCLOUD")

    @patch("harvesters.enumeration.cmr_enumerator.GranuleQuery")
    def test_cmr_query_success(self, mock_granule_query):
        """Test successful CMR query."""
        # Setup mock
        mock_api = MagicMock()
        mock_granule_query.return_value = mock_api

        mock_result = {
            "id": "G12345-TEST",
            "updated": "2020-01-15T12:30:45.123Z",
            "collection_concept_id": "C12345-TEST",
            "links": [
                {
                    "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                    "href": "https://archive.earthdata.nasa.gov/POCLOUD/data/file.nc"
                }
            ]
        }

        mock_api.concept_id.return_value = mock_api
        mock_api.temporal.return_value = mock_api
        mock_api.get_all.return_value = [mock_result]

        # Execute
        harvester = self.get_mock_harvester()
        query = CMRQuery(harvester)
        granules = query.query()

        # Verify
        self.assertEqual(len(granules), 1)
        self.assertIsInstance(granules[0], CMRGranule)
        mock_api.concept_id.assert_called_with("C12345-TEST")
        mock_api.temporal.assert_called_once()

    @patch("harvesters.enumeration.cmr_enumerator.GranuleQuery")
    def test_cmr_query_empty_results(self, mock_granule_query):
        """Test CMR query with no results."""
        mock_api = MagicMock()
        mock_granule_query.return_value = mock_api
        mock_api.concept_id.return_value = mock_api
        mock_api.temporal.return_value = mock_api
        mock_api.get_all.return_value = []

        harvester = self.get_mock_harvester()
        query = CMRQuery(harvester)
        granules = query.query()

        self.assertEqual(len(granules), 0)

    @patch("harvesters.enumeration.cmr_enumerator.GranuleQuery")
    def test_cmr_query_multiple_granules(self, mock_granule_query):
        """Test CMR query returning multiple granules."""
        mock_api = MagicMock()
        mock_granule_query.return_value = mock_api

        mock_results = [
            {
                "id": f"G{i}-TEST",
                "updated": f"2020-01-{i:02d}T12:30:45.123Z",
                "collection_concept_id": "C12345-TEST",
                "links": [
                    {
                        "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                        "href": f"https://archive.earthdata.nasa.gov/POCLOUD/data/file{i}.nc"
                    }
                ]
            }
            for i in range(1, 6)
        ]

        mock_api.concept_id.return_value = mock_api
        mock_api.temporal.return_value = mock_api
        mock_api.get_all.return_value = mock_results

        harvester = self.get_mock_harvester()
        query = CMRQuery(harvester)
        granules = query.query()

        self.assertEqual(len(granules), 5)

    @patch("harvesters.enumeration.cmr_enumerator.time.sleep")
    @patch("harvesters.enumeration.cmr_enumerator.GranuleQuery")
    def test_cmr_query_retry_on_failure(self, mock_granule_query, mock_sleep):
        """Test CMR query retries on RuntimeError."""
        mock_api = MagicMock()
        mock_granule_query.return_value = mock_api

        # First call raises RuntimeError, retry succeeds
        mock_result = {
            "id": "G12345-TEST",
            "updated": "2020-01-15T12:30:45.123Z",
            "collection_concept_id": "C12345-TEST",
            "links": [
                {
                    "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                    "href": "https://archive.earthdata.nasa.gov/POCLOUD/data/file.nc"
                }
            ]
        }

        mock_api.concept_id.return_value = mock_api
        mock_api.temporal.return_value = mock_api
        mock_api.get_all.side_effect = [RuntimeError, [mock_result]]

        harvester = self.get_mock_harvester()
        query = CMRQuery(harvester)
        granules = query.query()

        self.assertEqual(len(granules), 1)
        # Verify sleep was called for retry
        mock_sleep.assert_called()

    @patch("harvesters.enumeration.cmr_enumerator.time.sleep")
    @patch("harvesters.enumeration.cmr_enumerator.GranuleQuery")
    def test_cmr_query_max_retries_exceeded(self, mock_granule_query, mock_sleep):
        """Test CMR query fails after max retries."""
        mock_api = MagicMock()
        mock_granule_query.return_value = mock_api
        mock_api.concept_id.return_value = mock_api
        mock_api.temporal.return_value = mock_api
        mock_api.get_all.side_effect = RuntimeError("CMR unavailable")

        harvester = self.get_mock_harvester()
        query = CMRQuery(harvester)

        with self.assertRaises(RuntimeError):
            query.query()


if __name__ == "__main__":
    unittest.main()
