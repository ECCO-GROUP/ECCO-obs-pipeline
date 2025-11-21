"""
Unit tests for NSIDC enumerator.
All external HTTP calls are mocked - no external dependencies required.
"""
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from harvesters.enumeration.nsidc_enumerator import NSIDCGranule, search_nsidc


class NSIDCGranuleTestCase(unittest.TestCase):
    """Tests for the NSIDCGranule dataclass."""

    def test_nsidc_granule_creation(self):
        """Test NSIDCGranule creation."""
        url = "https://noaadata.apps.nsidc.org/NOAA/G10016_V3/CDR/north/daily/2020/file.nc"
        mod_time = datetime(2020, 1, 15, 12, 30)

        granule = NSIDCGranule(url, mod_time)

        self.assertEqual(granule.url, url)
        self.assertEqual(granule.mod_time, mod_time)


class SearchNSIDCTestCase(unittest.TestCase):
    """Tests for the search_nsidc function."""

    def get_mock_harvester(self):
        """Return a mock Harvester object."""
        harvester = MagicMock()
        harvester.ds_name = "G10016_V3"
        harvester.start = datetime(2020, 1, 1)
        harvester.end = datetime(2020, 12, 31)
        harvester.ddir = "CDR"
        return harvester

    def get_year_listing_html(self):
        """Return mock HTML for year directory listing."""
        return """
        <html>
        <body>
        <a href="/">Parent</a>
        <a href="2019/">2019/</a>
        <a href="2020/">2020/</a>
        <a href="2021/">2021/</a>
        </body>
        </html>
        """

    def get_file_listing_html(self):
        """Return mock HTML for file listing with modification times."""
        # The parser uses link.next_sibling.split() to get date/time tokens
        return """<html>
<body>
<a href="/">Parent</a>
<a href="seaice_conc_daily_nh_20200101.nc">seaice_conc_daily_nh_20200101.nc</a> 15-Jan-2020 10:30   12345
<a href="seaice_conc_daily_nh_20200102.nc">seaice_conc_daily_nh_20200102.nc</a> 16-Jan-2020 11:45   12346
</body>
</html>"""

    @patch("harvesters.enumeration.nsidc_enumerator.requests.get")
    def test_search_nsidc_basic(self, mock_get):
        """Test basic NSIDC search functionality."""
        harvester = self.get_mock_harvester()

        # Setup mock responses
        mock_responses = []

        # Response for north hemisphere year listing
        north_year_response = MagicMock()
        north_year_response.text = self.get_year_listing_html()
        mock_responses.append(north_year_response)

        # Response for 2020 file listing (north)
        north_file_response = MagicMock()
        north_file_response.text = self.get_file_listing_html()
        mock_responses.append(north_file_response)

        # Response for south hemisphere year listing
        south_year_response = MagicMock()
        south_year_response.text = self.get_year_listing_html()
        mock_responses.append(south_year_response)

        # Response for 2020 file listing (south)
        south_file_response = MagicMock()
        south_file_response.text = self.get_file_listing_html()
        mock_responses.append(south_file_response)

        mock_get.side_effect = mock_responses

        granules = search_nsidc(harvester)

        # Should have granules from both hemispheres
        self.assertGreater(len(granules), 0)
        self.assertIsInstance(granules[0], NSIDCGranule)

    @patch("harvesters.enumeration.nsidc_enumerator.requests.get")
    def test_search_nsidc_url_construction(self, mock_get):
        """Test that correct URLs are constructed."""
        harvester = self.get_mock_harvester()

        # Track calls
        calls = []

        def track_calls(url):
            calls.append(url)
            response = MagicMock()
            if "daily" in url and "/20" not in url:
                response.text = self.get_year_listing_html()
            else:
                response.text = self.get_file_listing_html()
            return response

        mock_get.side_effect = track_calls

        search_nsidc(harvester)

        # Verify base URLs contain expected components
        self.assertTrue(any("G10016_V3" in call for call in calls))
        self.assertTrue(any("CDR" in call for call in calls))
        self.assertTrue(any("north" in call for call in calls))
        self.assertTrue(any("south" in call for call in calls))

    @patch("harvesters.enumeration.nsidc_enumerator.requests.get")
    def test_search_nsidc_date_filtering(self, mock_get):
        """Test that years outside date range are skipped."""
        harvester = self.get_mock_harvester()
        harvester.start = datetime(2020, 1, 1)
        harvester.end = datetime(2020, 12, 31)

        call_urls = []

        def track_calls(url):
            call_urls.append(url)
            response = MagicMock()
            if "daily" in url and "/20" not in url:
                response.text = self.get_year_listing_html()
            else:
                response.text = self.get_file_listing_html()
            return response

        mock_get.side_effect = track_calls

        search_nsidc(harvester)

        # Should only fetch 2020, not 2019 or 2021
        year_fetches = [url for url in call_urls if "/2019/" in url or "/2021/" in url]
        self.assertEqual(len(year_fetches), 0)

    @patch("harvesters.enumeration.nsidc_enumerator.requests.get")
    def test_search_nsidc_no_ddir(self, mock_get):
        """Test search when ddir is not specified."""
        harvester = self.get_mock_harvester()
        harvester.ddir = None

        def respond(url):
            response = MagicMock()
            if "daily" in url and "/20" not in url:
                response.text = self.get_year_listing_html()
            else:
                response.text = self.get_file_listing_html()
            return response

        mock_get.side_effect = respond

        granules = search_nsidc(harvester)

        # Should still work without ddir
        self.assertIsInstance(granules, list)

    @patch("harvesters.enumeration.nsidc_enumerator.requests.get")
    def test_search_nsidc_granule_attributes(self, mock_get):
        """Test that granules have correct attributes."""
        harvester = self.get_mock_harvester()

        def respond(url):
            response = MagicMock()
            if "daily" in url and "/20" not in url:
                response.text = self.get_year_listing_html()
            else:
                response.text = self.get_file_listing_html()
            return response

        mock_get.side_effect = respond

        granules = search_nsidc(harvester)

        for granule in granules:
            self.assertTrue(granule.url.startswith("http"))
            self.assertIsInstance(granule.mod_time, datetime)


if __name__ == "__main__":
    unittest.main()
