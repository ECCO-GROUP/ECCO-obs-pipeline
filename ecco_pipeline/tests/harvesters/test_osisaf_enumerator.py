"""
Unit tests for OSISAF enumerator.
All external HTTP calls are mocked - no external dependencies required.
"""
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from harvesters.enumeration.osisaf_enumerator import OSISAFGranule, search_osisaf


class OSISAFGranuleTestCase(unittest.TestCase):
    """Tests for the OSISAFGranule dataclass."""

    def test_osisaf_granule_creation(self):
        """Test OSISAFGranule creation."""
        url = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc/2020/01/ice_conc_20200101.nc"
        mod_time = datetime(2020, 1, 15, 12, 30)

        granule = OSISAFGranule(url, mod_time)

        self.assertEqual(granule.url, url)
        self.assertEqual(granule.mod_time, mod_time)


class SearchOSISAFTestCase(unittest.TestCase):
    """Tests for the search_osisaf function."""

    def get_mock_harvester(self, time_scale="daily"):
        """Return a mock Harvester object."""
        harvester = MagicMock()
        harvester.ds_name = "OSI-450-a"
        harvester.start = datetime(2020, 1, 1)
        harvester.end = datetime(2020, 12, 31)
        harvester.ddir = "ice/conc_cdr_cont"
        harvester.data_time_scale = time_scale
        return harvester

    def get_year_catalog_xml(self):
        """Return mock XML for year catalog."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
                 xmlns:xlink="http://www.w3.org/1999/xlink">
            <catalogRef xlink:title="2020" xlink:href="2020/catalog.xml"/>
            <catalogRef xlink:title="monthly" xlink:href="monthly/catalog.xml"/>
        </catalog>
        """

    def get_month_catalog_xml(self):
        """Return mock XML for month catalog."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
                 xmlns:xlink="http://www.w3.org/1999/xlink">
            <catalogRef xlink:title="01" xlink:href="01/catalog.xml"/>
            <catalogRef xlink:title="02" xlink:href="02/catalog.xml"/>
        </catalog>
        """

    def get_daily_catalog_xml(self):
        """Return mock XML for daily file catalog."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
            <dataset name="files">
                <dataset name="ice_conc_nh_20200101.nc" urlPath="osisaf/met.no/ice/conc/2020/01/ice_conc_nh_20200101.nc">
                    <date type="modified">2020-01-15T10:30:00Z</date>
                </dataset>
                <dataset name="ice_conc_nh_20200102.nc" urlPath="osisaf/met.no/ice/conc/2020/01/ice_conc_nh_20200102.nc">
                    <date type="modified">2020-01-16T11:45:00Z</date>
                </dataset>
            </dataset>
        </catalog>
        """

    def get_monthly_file_catalog_xml(self):
        """Return mock XML for monthly file catalog."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
            <dataset name="files">
                <dataset name="ice_conc_nh_202001.nc" urlPath="osisaf/met.no/ice/conc/2020/ice_conc_nh_202001.nc">
                    <date type="modified">2020-02-01T10:30:00Z</date>
                </dataset>
            </dataset>
        </catalog>
        """

    @patch("harvesters.enumeration.osisaf_enumerator.requests.session")
    def test_search_osisaf_daily_basic(self, mock_session_class):
        """Test basic OSISAF daily search functionality."""
        harvester = self.get_mock_harvester("daily")

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        def mock_get(url):
            response = MagicMock()
            if url.endswith("catalog.xml") and "/2020/" not in url and "/01/" not in url and "/02/" not in url:
                response.text = self.get_year_catalog_xml()
            elif "/2020/catalog.xml" in url:
                response.text = self.get_month_catalog_xml()
            else:
                response.text = self.get_daily_catalog_xml()
            return response

        mock_session.get.side_effect = mock_get

        granules = search_osisaf(harvester)

        self.assertGreater(len(granules), 0)
        self.assertIsInstance(granules[0], OSISAFGranule)

    @patch("harvesters.enumeration.osisaf_enumerator.requests.session")
    def test_search_osisaf_monthly(self, mock_session_class):
        """Test OSISAF monthly search functionality."""
        harvester = self.get_mock_harvester("monthly")

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        def mock_get(url):
            response = MagicMock()
            if "monthly/catalog.xml" in url and "/2020/" not in url:
                response.text = self.get_year_catalog_xml()
            elif "/2020/catalog.xml" in url:
                response.text = self.get_monthly_file_catalog_xml()
            else:
                response.text = self.get_monthly_file_catalog_xml()
            return response

        mock_session.get.side_effect = mock_get

        granules = search_osisaf(harvester)

        self.assertIsInstance(granules, list)
        for granule in granules:
            self.assertIsInstance(granule, OSISAFGranule)

    @patch("harvesters.enumeration.osisaf_enumerator.requests.session")
    def test_search_osisaf_url_construction(self, mock_session_class):
        """Test that correct URLs are constructed."""
        harvester = self.get_mock_harvester("daily")

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        calls = []

        def mock_get(url):
            calls.append(url)
            response = MagicMock()
            if url.endswith("catalog.xml") and "/2020/" not in url and "/01/" not in url and "/02/" not in url:
                response.text = self.get_year_catalog_xml()
            elif "/2020/catalog.xml" in url:
                response.text = self.get_month_catalog_xml()
            else:
                response.text = self.get_daily_catalog_xml()
            return response

        mock_session.get.side_effect = mock_get

        search_osisaf(harvester)

        # Verify URL contains expected components
        self.assertTrue(any("thredds.met.no" in call for call in calls))
        self.assertTrue(any("osisaf" in call for call in calls))

    @patch("harvesters.enumeration.osisaf_enumerator.requests.session")
    def test_search_osisaf_granule_attributes(self, mock_session_class):
        """Test that granules have correct attributes."""
        harvester = self.get_mock_harvester("daily")

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        def mock_get(url):
            response = MagicMock()
            if url.endswith("catalog.xml") and "/2020/" not in url and "/01/" not in url and "/02/" not in url:
                response.text = self.get_year_catalog_xml()
            elif "/2020/catalog.xml" in url:
                response.text = self.get_month_catalog_xml()
            else:
                response.text = self.get_daily_catalog_xml()
            return response

        mock_session.get.side_effect = mock_get

        granules = search_osisaf(harvester)

        for granule in granules:
            self.assertTrue(granule.url.startswith("http"))
            self.assertIn("fileServer", granule.url)
            self.assertIsInstance(granule.mod_time, datetime)

    @patch("harvesters.enumeration.osisaf_enumerator.requests.session")
    def test_search_osisaf_skips_monthly_dir(self, mock_session_class):
        """Test that monthly directory is skipped in daily search."""
        harvester = self.get_mock_harvester("daily")

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        calls = []

        def mock_get(url):
            calls.append(url)
            response = MagicMock()
            if url.endswith("catalog.xml") and "/2020/" not in url and "/01/" not in url and "/02/" not in url:
                response.text = self.get_year_catalog_xml()
            elif "/2020/catalog.xml" in url:
                response.text = self.get_month_catalog_xml()
            else:
                response.text = self.get_daily_catalog_xml()
            return response

        mock_session.get.side_effect = mock_get

        search_osisaf(harvester)

        # Should not fetch monthly catalog in daily mode
        monthly_calls = [c for c in calls if "/monthly/" in c and "catalog.xml" in c]
        self.assertEqual(len(monthly_calls), 0)


if __name__ == "__main__":
    unittest.main()
