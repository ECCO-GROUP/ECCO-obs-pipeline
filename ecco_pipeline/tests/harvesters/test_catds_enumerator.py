"""
Unit tests for CATDS enumerator.
All external HTTP calls are mocked - no external dependencies required.
"""
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from harvesters.enumeration.catds_enumerator import CATDSGranule, search_catds


class CATDSGranuleTestCase(unittest.TestCase):
    """Tests for the CATDSGranule dataclass."""

    def test_catds_granule_creation(self):
        """Test CATDSGranule creation."""
        url = "https://data.catds.fr/cecos-locean/Ocean_products/L3_DEBIAS_LOCEAN_v10/SM_OPER_MIR_OSUDP2_20200101.nc"
        mod_time = datetime(2020, 1, 15, 12, 30)

        granule = CATDSGranule(url, mod_time)

        self.assertEqual(granule.url, url)
        self.assertEqual(granule.mod_time, mod_time)


class SearchCATDSTestCase(unittest.TestCase):
    """Tests for the search_catds function."""

    def get_mock_harvester(self):
        """Return a mock Harvester object."""
        harvester = MagicMock()
        harvester.ds_name = "L3_DEBIAS_LOCEAN_v10_q09"
        harvester.start = datetime(2020, 1, 1)
        harvester.end = datetime(2020, 12, 31)
        harvester.ddir = "L3_DEBIAS_LOCEAN_v10"
        return harvester

    def get_file_listing_html(self):
        """Return mock HTML for CATDS file listing."""
        # The CATDS parser expects: rows[3:-1] with tokens[1].find("a")["href"] and tokens[2].text
        # So we need at least 4 header/footer rows plus data rows
        return """
        <html>
        <body>
        <table>
            <tr><th>Name</th><th>Last modified</th><th>Size</th></tr>
            <tr><td></td><td><a href="/">Parent Directory</a></td><td></td></tr>
            <tr><td></td><td><a href="..">../</a></td><td></td></tr>
            <tr>
                <td></td>
                <td><a href="SM_OPER_MIR_OSUDP2_20200101.nc">SM_OPER_MIR_OSUDP2_20200101.nc</a></td>
                <td>2020-01-15 10:30  </td>
            </tr>
            <tr>
                <td></td>
                <td><a href="SM_OPER_MIR_OSUDP2_20200102.nc">SM_OPER_MIR_OSUDP2_20200102.nc</a></td>
                <td>2020-01-16 11:45  </td>
            </tr>
            <tr>
                <td></td>
                <td><a href="SM_OPER_MIR_OSUDP2_20200103.nc">SM_OPER_MIR_OSUDP2_20200103.nc</a></td>
                <td>2020-01-17 09:15  </td>
            </tr>
            <tr><td colspan="3">Total: 3 files</td></tr>
        </table>
        </body>
        </html>
        """

    @patch("harvesters.enumeration.catds_enumerator.requests.get")
    def test_search_catds_basic(self, mock_get):
        """Test basic CATDS search functionality."""
        harvester = self.get_mock_harvester()

        mock_response = MagicMock()
        mock_response.text = self.get_file_listing_html()
        mock_get.return_value = mock_response

        granules = search_catds(harvester)

        # Should have 3 granules (rows 3-5, skipping first 3 and last 1)
        self.assertEqual(len(granules), 3)
        self.assertIsInstance(granules[0], CATDSGranule)

    @patch("harvesters.enumeration.catds_enumerator.requests.get")
    def test_search_catds_url_construction(self, mock_get):
        """Test that correct URL is constructed."""
        harvester = self.get_mock_harvester()

        mock_response = MagicMock()
        mock_response.text = self.get_file_listing_html()
        mock_get.return_value = mock_response

        search_catds(harvester)

        # Verify the correct URL was called
        call_args = mock_get.call_args
        url = call_args[0][0]
        self.assertIn("data.catds.fr", url)
        self.assertIn("L3_DEBIAS_LOCEAN_v10", url)
        self.assertIn("Ocean_products", url)

    @patch("harvesters.enumeration.catds_enumerator.requests.get")
    def test_search_catds_granule_urls(self, mock_get):
        """Test that granule URLs are correctly constructed."""
        harvester = self.get_mock_harvester()

        mock_response = MagicMock()
        mock_response.text = self.get_file_listing_html()
        mock_get.return_value = mock_response

        granules = search_catds(harvester)

        for granule in granules:
            self.assertTrue(granule.url.startswith("https://data.catds.fr"))
            self.assertIn("L3_DEBIAS_LOCEAN_v10", granule.url)
            self.assertTrue(granule.url.endswith(".nc"))

    @patch("harvesters.enumeration.catds_enumerator.requests.get")
    def test_search_catds_modification_times(self, mock_get):
        """Test that modification times are correctly parsed."""
        harvester = self.get_mock_harvester()

        mock_response = MagicMock()
        mock_response.text = self.get_file_listing_html()
        mock_get.return_value = mock_response

        granules = search_catds(harvester)

        # Check first granule's modification time
        self.assertEqual(granules[0].mod_time.year, 2020)
        self.assertEqual(granules[0].mod_time.month, 1)
        self.assertEqual(granules[0].mod_time.day, 15)
        self.assertEqual(granules[0].mod_time.hour, 10)
        self.assertEqual(granules[0].mod_time.minute, 30)

    @patch("harvesters.enumeration.catds_enumerator.requests.get")
    def test_search_catds_empty_listing(self, mock_get):
        """Test handling of empty directory listing."""
        harvester = self.get_mock_harvester()

        # Parser uses rows[3:-1], so with only 4 rows total, we get empty slice
        empty_html = """
        <html>
        <body>
        <table>
            <tr><th>Name</th><th>Last modified</th><th>Size</th></tr>
            <tr><td></td><td><a href="/">Parent Directory</a></td><td></td></tr>
            <tr><td></td><td><a href="..">../</a></td><td></td></tr>
            <tr><td colspan="3">Total: 0 files</td></tr>
        </table>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.text = empty_html
        mock_get.return_value = mock_response

        granules = search_catds(harvester)

        self.assertEqual(len(granules), 0)

    @patch("harvesters.enumeration.catds_enumerator.requests.get")
    def test_search_catds_different_ddir(self, mock_get):
        """Test with different ddir values."""
        harvester = self.get_mock_harvester()
        harvester.ddir = "custom_directory"

        mock_response = MagicMock()
        mock_response.text = self.get_file_listing_html()
        mock_get.return_value = mock_response

        search_catds(harvester)

        call_args = mock_get.call_args
        url = call_args[0][0]
        self.assertIn("custom_directory", url)


if __name__ == "__main__":
    unittest.main()
