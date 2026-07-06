"""
Pytest configuration and shared fixtures for harvester tests.
"""
import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Add ecco_pipeline to path for imports
_ECCO_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ECCO_PIPELINE_DIR)

# conf/global_settings.py is gitignored (it holds machine-specific paths like
# OUTPUT_DIR), so on a fresh checkout — CI or a new clone — it does not exist. The
# pipeline modules import it at module load, so collection would fail before any
# test runs. Seed it from the committed template if absent; tests patch OUTPUT_DIR
# wherever they touch the filesystem, so the placeholder paths are sufficient.
_SETTINGS = os.path.join(_ECCO_PIPELINE_DIR, "conf", "global_settings.py")
if not os.path.exists(_SETTINGS) and os.path.exists(_SETTINGS + ".example"):
    shutil.copyfile(_SETTINGS + ".example", _SETTINGS)


def make_mock_download_response(content: bytes = b"mock netcdf file content"):
    """
    Build a mock requests response for Harvester._stream_download.

    Behaves as a context manager whose streamed chunks match its declared
    Content-Length, so a download through it passes the size-verification check.
    Tests that want to simulate a truncated download can override .headers or
    .iter_content on the returned object.
    """
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.raise_for_status = MagicMock()
    response.headers = {"Content-Length": str(len(content))}
    response.iter_content = MagicMock(return_value=[content])
    return response


@pytest.fixture
def mock_solr():
    """Fixture to mock all Solr interactions."""
    with patch("harvesters.harvesterclasses.solr_utils.solr_query") as mock_query, \
         patch("harvesters.harvesterclasses.solr_utils.clean_solr") as mock_clean, \
         patch("harvesters.harvesterclasses.solr_utils.solr_update") as mock_update:

        mock_query.return_value = []
        mock_response = type('Response', (), {'status_code': 200})()
        mock_update.return_value = mock_response

        yield {
            "query": mock_query,
            "clean": mock_clean,
            "update": mock_update
        }


@pytest.fixture
def temp_output_dir():
    """Fixture to provide a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("harvesters.harvesterclasses.OUTPUT_DIR", tmpdir):
            yield tmpdir


@pytest.fixture
def base_config():
    """Fixture providing a base configuration dictionary."""
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
