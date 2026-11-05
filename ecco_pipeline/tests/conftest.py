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


def grid_block(name, datasets=None, exclude_ecco_variables=()):
    """
    Build a real baseclasses.GridBlock for tests.

    ``datasets=None`` means "all datasets" (the ``datasets: all`` form in
    conf/grid_coverage.yaml); pass a list for an explicit coverage list.
    """
    import baseclasses

    return baseclasses.GridBlock(
        name=name,
        datasets=None if datasets is None else list(datasets),
        excluded_ecco_variables=frozenset(exclude_ecco_variables),
    )


def mock_grid_coverage(mock_config, grids, coverage=None):
    """
    Wire a patched ``baseclasses.Config`` mock so ``grids_for`` behaves like the real
    classmethod.

    The factories patch Config wholesale, which would otherwise make grids_for return
    a MagicMock. Resolution delegates to the real GridBlock.covers, so coverage rules
    stay genuinely under test rather than being restated in the mock.

    ``coverage`` maps grid name to GridBlock (see grid_block); omit it when the test
    only cares that the enabled grids come through unfiltered.
    """
    grids = list(grids)
    mock_config.grids_to_use = grids

    if coverage is None:
        mock_config.grids_for.side_effect = lambda config: list(grids)
        return

    def grids_for(config):
        return [
            name
            for name in grids
            if coverage[name].covers(config.get("ds_name"), config.get("ecco_variable"))
        ]

    mock_config.grids_for.side_effect = grids_for


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
    with (
        patch("harvesters.harvesterclasses.solr_utils.solr_query") as mock_query,
        patch("harvesters.harvesterclasses.solr_utils.solr_update") as mock_update,
    ):
        mock_query.return_value = []
        mock_response = type("Response", (), {"status_code": 200})()
        mock_update.return_value = mock_response

        yield {"query": mock_query, "update": mock_update}


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
                "post_transformations": [],
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
        "notes": "",
    }
