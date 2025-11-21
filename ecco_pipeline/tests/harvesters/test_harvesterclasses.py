"""
Unit tests for harvester base classes (Granule and Harvester).
All Solr calls are mocked - no Solr deployment required.
"""
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

from harvesters.harvesterclasses import Granule, Harvester


class GranuleTestCase(unittest.TestCase):
    """Tests for the Granule class."""

    def setUp(self):
        self.ds_name = "TEST_DATASET"
        self.local_fp = "/tmp/test/2020/test_file_20200101.nc"
        self.date = datetime(2020, 1, 1)
        self.modified_time = datetime(2020, 1, 15, 12, 0, 0)
        self.url = "https://example.com/data/test_file_20200101.nc"

    def test_granule_initialization(self):
        """Test that Granule initializes correctly."""
        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        self.assertEqual(granule.ds_name, self.ds_name)
        self.assertEqual(granule.local_fp, self.local_fp)
        self.assertEqual(granule.filename, "test_file_20200101.nc")
        self.assertEqual(granule.datetime, self.date)
        self.assertEqual(granule.modified_time, self.modified_time)
        self.assertEqual(granule.url, self.url)

    def test_gen_granule_doc(self):
        """Test that granule Solr document is generated correctly."""
        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        self.assertEqual(granule.solr_item["type_s"], "granule")
        self.assertEqual(granule.solr_item["dataset_s"], self.ds_name)
        self.assertEqual(granule.solr_item["filename_s"], "test_file_20200101.nc")
        self.assertEqual(granule.solr_item["source_s"], self.url)
        self.assertEqual(granule.solr_item["date_s"], "2020-01-01T00:00:00Z")

    def test_gen_descendant_doc(self):
        """Test that descendant Solr document is generated correctly."""
        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        self.assertEqual(granule.descendant_item["type_s"], "descendants")
        self.assertEqual(granule.descendant_item["dataset_s"], self.ds_name)
        self.assertEqual(granule.descendant_item["filename_s"], "test_file_20200101.nc")
        self.assertEqual(granule.descendant_item["source_s"], self.url)

    @patch("harvesters.harvesterclasses.file_utils.md5")
    def test_update_item_success(self, mock_md5):
        """Test update_item with successful download."""
        mock_md5.return_value = "abc123checksum"

        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        # Create a temporary file for the test
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            granule.local_fp = temp_path
            solr_docs = {}
            granule.update_item(solr_docs, success=True)

            self.assertTrue(granule.solr_item["harvest_success_b"])
            self.assertEqual(granule.solr_item["checksum_s"], "abc123checksum")
            self.assertEqual(granule.solr_item["pre_transformation_file_path_s"], temp_path)
            self.assertGreater(granule.solr_item["file_size_l"], 0)
        finally:
            os.unlink(temp_path)

    def test_update_item_failure(self):
        """Test update_item with failed download."""
        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        solr_docs = {}
        granule.update_item(solr_docs, success=False)

        self.assertFalse(granule.solr_item["harvest_success_b"])
        self.assertEqual(granule.solr_item["pre_transformation_file_path_s"], "")
        self.assertEqual(granule.solr_item["file_size_l"], 0)

    def test_update_item_with_existing_id(self):
        """Test update_item preserves existing Solr ID."""
        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        solr_docs = {
            "test_file_20200101.nc": {"id": "existing-solr-id-123"}
        }
        granule.update_item(solr_docs, success=False)

        self.assertEqual(granule.solr_item["id"], "existing-solr-id-123")

    def test_update_descendant_success(self):
        """Test update_descendant with successful status."""
        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        # Pre-populate solr_item with path
        granule.solr_item["pre_transformation_file_path_s"] = self.local_fp

        descendants_docs = {}
        granule.update_descendant(descendants_docs, success=True)

        self.assertTrue(granule.descendant_item["harvest_success_b"])
        self.assertEqual(
            granule.descendant_item["pre_transformation_file_path_s"],
            self.local_fp
        )

    def test_get_solr_docs(self):
        """Test get_solr_docs returns both documents."""
        granule = Granule(
            self.ds_name,
            self.local_fp,
            self.date,
            self.modified_time,
            self.url
        )

        docs = granule.get_solr_docs()

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["type_s"], "granule")
        self.assertEqual(docs[1]["type_s"], "descendants")


@patch("harvesters.harvesterclasses.solr_utils.solr_query")
@patch("harvesters.harvesterclasses.solr_utils.clean_solr")
class HarvesterTestCase(unittest.TestCase):
    """Tests for the Harvester base class. All Solr calls are mocked."""

    def get_mock_config(self):
        """Return a mock configuration dictionary."""
        return {
            "ds_name": "TEST_DATASET",
            "start": "20200101T00:00:00Z",
            "end": "20201231T00:00:00Z",
            "harvester_type": "nsidc",
            "filename_date_fmt": "%Y%m%d",
            "filename_date_regex": r"\d{8}",
            "data_time_scale": "daily",
            "hemi_pattern": {"north": "_nh_", "south": "_sh_"},
            "fields": [
                {
                    "name": "test_field",
                    "long_name": "Test Field",
                    "standard_name": "test",
                    "units": "1",
                    "pre_transformations": [],
                    "post_transformations": []
                }
            ],
            "original_dataset_title": "Test Dataset",
            "original_dataset_short_name": "TEST",
            "original_dataset_url": "https://example.com",
            "original_dataset_reference": "Test Ref",
            "original_dataset_doi": "10.1234/test",
            "ddir": "test_dir",
            "preprocessing": None,
            "t_version": 1.0,
            "a_version": 1.0,
            "notes": ""
        }

    def test_harvester_initialization(self, mock_clean, mock_query):
        """Test Harvester initializes correctly."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                self.assertEqual(harvester.ds_name, "TEST_DATASET")
                self.assertEqual(harvester.start, datetime(2020, 1, 1))
                self.assertEqual(harvester.end, datetime(2020, 12, 31))
                self.assertEqual(harvester.ddir, "test_dir")
                self.assertTrue(os.path.exists(harvester.target_dir))

                # Verify Solr was called
                self.assertTrue(mock_clean.called)
                self.assertTrue(mock_query.called)

    def test_harvester_cmr_parsing(self, mock_clean, mock_query):
        """Test Harvester parses CMR-specific config."""
        mock_query.return_value = []

        config = self.get_mock_config()
        config["harvester_type"] = "cmr"
        config["cmr_concept_id"] = "C12345-TEST"
        config["provider"] = "TEST_PROVIDER"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                self.assertEqual(harvester.cmr_concept_id, "C12345-TEST")
                self.assertEqual(harvester.provider, "TEST_PROVIDER")

    def test_check_update_new_file(self, mock_clean, mock_query):
        """Test check_update returns True for new files."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                # File not in solr_docs
                result = harvester.check_update(
                    "new_file.nc",
                    datetime(2020, 1, 1)
                )
                self.assertTrue(result)

    def test_check_update_failed_previous(self, mock_clean, mock_query):
        """Test check_update returns True for previously failed files."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                # Add failed file to solr_docs
                harvester.solr_docs["failed_file.nc"] = {
                    "harvest_success_b": False,
                    "download_time_dt": "2020-01-01T00:00:00Z"
                }

                result = harvester.check_update(
                    "failed_file.nc",
                    datetime(2020, 1, 1)
                )
                self.assertTrue(result)

    def test_check_update_outdated_file(self, mock_clean, mock_query):
        """Test check_update returns True for outdated files."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                # Add outdated file to solr_docs
                harvester.solr_docs["outdated_file.nc"] = {
                    "harvest_success_b": True,
                    "download_time_dt": "2020-01-01T00:00:00Z"
                }

                # New modification time is later
                result = harvester.check_update(
                    "outdated_file.nc",
                    datetime(2020, 6, 1)
                )
                self.assertTrue(result)

    def test_check_update_up_to_date(self, mock_clean, mock_query):
        """Test check_update returns False for up-to-date files."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                # Add up-to-date file to solr_docs
                harvester.solr_docs["current_file.nc"] = {
                    "harvest_success_b": True,
                    "download_time_dt": "2020-06-01T00:00:00Z"
                }

                # Modification time is earlier than download time
                result = harvester.check_update(
                    "current_file.nc",
                    datetime(2020, 1, 1)
                )
                self.assertFalse(result)

    def test_need_to_download_missing_file(self, mock_clean, mock_query):
        """Test need_to_download returns True for missing files."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                granule = Granule(
                    "TEST",
                    "/nonexistent/path/file.nc",
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 1),
                    "https://example.com/file.nc"
                )

                self.assertTrue(harvester.need_to_download(granule))

    def test_need_to_download_outdated_file(self, mock_clean, mock_query):
        """Test need_to_download returns True for outdated local files."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                # Create a temporary file with old modification time
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(b"old content")
                    temp_path = f.name

                try:
                    # Set file mtime to old date
                    old_time = datetime(2019, 1, 1).timestamp()
                    os.utime(temp_path, (old_time, old_time))

                    granule = Granule(
                        "TEST",
                        temp_path,
                        datetime(2020, 1, 1),
                        datetime(2020, 6, 1),  # Newer modification time
                        "https://example.com/file.nc"
                    )

                    self.assertTrue(harvester.need_to_download(granule))
                finally:
                    os.unlink(temp_path)

    def test_make_ds_doc(self, mock_clean, mock_query):
        """Test make_ds_doc creates correct dataset document."""
        mock_query.return_value = []

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                ds_doc = harvester.make_ds_doc(
                    "https://example.com/source",
                    "2020-01-01T00:00:00Z"
                )

                self.assertEqual(ds_doc["type_s"], "dataset")
                self.assertEqual(ds_doc["dataset_s"], "TEST_DATASET")
                self.assertEqual(ds_doc["short_name_s"], "TEST")
                self.assertEqual(ds_doc["data_time_scale_s"], "daily")

    @patch("harvesters.harvesterclasses.solr_utils.solr_update")
    def test_post_fetch_no_updates(self, mock_update, mock_clean, mock_query):
        """Test post_fetch when no downloads occurred."""
        mock_query.return_value = []
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)
                harvester.updated_solr_docs = []

                status = harvester.post_fetch("https://example.com/source")

                self.assertIn("harvested", status.lower())

    @patch("harvesters.harvesterclasses.solr_utils.solr_update")
    def test_post_fetch_with_updates(self, mock_update, mock_clean, mock_query):
        """Test post_fetch when downloads occurred."""
        mock_query.return_value = []
        mock_response = Mock()
        mock_response.status_code = 200
        mock_update.return_value = mock_response

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)
                harvester.updated_solr_docs = [
                    {
                        "type_s": "granule",
                        "harvest_success_b": True,
                        "download_time_dt": "2020-01-01T00:00:00Z",
                        "date_s": "2020-01-01T00:00:00Z"
                    }
                ]

                status = harvester.post_fetch("https://example.com/source")

                # Verify solr_update was called
                self.assertTrue(mock_update.called)

    def test_harvester_status_all_success(self, mock_clean, mock_query):
        """Test harvester_status when all granules succeeded."""
        # First call for init, then for status checks
        mock_query.side_effect = [
            [],  # Initial granule query
            [],  # Initial descendant query
            [],  # Failed granules query
            [{"id": "1"}]  # Successful granules query
        ]

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)
                status = harvester.harvester_status()

                self.assertEqual(status, "All granules successfully harvested")

    def test_harvester_status_with_failures(self, mock_clean, mock_query):
        """Test harvester_status when some granules failed."""
        # First call for init, then for status checks
        mock_query.side_effect = [
            [],  # Initial granule query
            [],  # Initial descendant query
            [{"id": "1"}, {"id": "2"}],  # Failed granules query
            [{"id": "3"}]  # Successful granules query
        ]

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)
                status = harvester.harvester_status()

                self.assertIn("2", status)
                self.assertIn("failed", status)

    def test_get_solr_docs_with_existing(self, mock_clean, mock_query):
        """Test get_solr_docs retrieves existing documents."""
        mock_query.side_effect = [
            [
                {"filename_s": "file1.nc", "id": "id1"},
                {"filename_s": "file2.nc", "id": "id2"}
            ],
            [
                {"filename_s": "file1.nc", "id": "desc1"}
            ]
        ]

        config = self.get_mock_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
                harvester = Harvester(config)

                self.assertEqual(len(harvester.solr_docs), 2)
                self.assertEqual(harvester.solr_docs["file1.nc"]["id"], "id1")
                self.assertEqual(len(harvester.descendant_docs), 1)


if __name__ == "__main__":
    unittest.main()
