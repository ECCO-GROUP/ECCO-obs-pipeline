"""
Unit tests for init_pipeline's Solr grid syncing.

Focused on update_solr_grid, whose three branches differ in a way that matters: a new
grid is inserted, a grid whose contents changed has all its transformations and
aggregations deleted, and a grid whose contents are identical must NOT lose that work
even when its file has moved.
"""

import unittest
from unittest.mock import MagicMock, patch

from utils.pipeline_utils import init_pipeline


def ok_response():
    return type("Response", (), {"status_code": 200})()


class UpdateSolrGridTestCase(unittest.TestCase):
    def setUp(self):
        patcher_query = patch(
            "utils.pipeline_utils.init_pipeline.solr_utils.solr_query"
        )
        patcher_update = patch(
            "utils.pipeline_utils.init_pipeline.solr_utils.solr_update"
        )
        patcher_md5 = patch("utils.pipeline_utils.init_pipeline.file_utils.md5")
        self.mock_query = patcher_query.start()
        self.mock_update = patcher_update.start()
        self.mock_md5 = patcher_md5.start()
        for patcher in (patcher_query, patcher_update, patcher_md5):
            self.addCleanup(patcher.stop)

        self.mock_update.return_value = ok_response()

    def existing_doc(self, **overrides):
        doc = {
            "id": "grid-doc-1",
            "grid_name_s": "ECCO_llc90",
            "grid_type_s": "llc",
            "grid_path_s": "grids/ECCO_llc90.nc",
            "grid_checksum_s": "abc123",
        }
        doc.update(overrides)
        return doc

    def sent_bodies(self):
        return [call.args[0] for call in self.mock_update.call_args_list]

    def test_new_grid_inserted(self):
        self.mock_query.return_value = []
        self.mock_md5.return_value = "abc123"

        init_pipeline.update_solr_grid("ECCO_llc540", "llc", "grids/ECCO_llc540.nc")

        (body,) = self.sent_bodies()
        self.assertEqual(body[0]["grid_name_s"], "ECCO_llc540")
        self.assertEqual(body[0]["grid_path_s"], "grids/ECCO_llc540.nc")
        self.assertEqual(body[0]["type_s"], "grid")

    def test_unchanged_grid_is_a_noop(self):
        self.mock_query.return_value = [self.existing_doc()]
        self.mock_md5.return_value = "abc123"

        init_pipeline.update_solr_grid("ECCO_llc90", "llc", "grids/ECCO_llc90.nc")

        self.mock_update.assert_not_called()

    def test_moved_grid_refreshes_path_without_deleting_work(self):
        """
        A content-identical grid whose file moved gets its grid_path_s refreshed, and
        keeps its transformations and aggregations.

        Aggregation opens the grid via grid_path_s, so a stale path breaks it; but the
        contents are unchanged, so deleting existing work (what the checksum-change
        branch does) would be destructive for no reason.
        """
        self.mock_query.return_value = [self.existing_doc()]
        self.mock_md5.return_value = "abc123"

        init_pipeline.update_solr_grid(
            "ECCO_llc90", "llc", "grids/subdir/ECCO_llc90.nc"
        )

        (body,) = self.sent_bodies()
        self.assertEqual(body[0]["id"], "grid-doc-1")
        self.assertEqual(body[0]["grid_path_s"], {"set": "grids/subdir/ECCO_llc90.nc"})
        # Checksum untouched, and nothing deleted.
        self.assertNotIn("grid_checksum_s", body[0])
        for sent in self.sent_bodies():
            self.assertNotIsInstance(sent, dict)  # delete bodies are dicts

    def test_changed_type_refreshes_without_deleting_work(self):
        self.mock_query.return_value = [self.existing_doc()]
        self.mock_md5.return_value = "abc123"

        init_pipeline.update_solr_grid("ECCO_llc90", "latlon", "grids/ECCO_llc90.nc")

        (body,) = self.sent_bodies()
        self.assertEqual(body[0]["grid_type_s"], {"set": "latlon"})
        self.assertNotIn("grid_path_s", body[0])

    def test_changed_checksum_deletes_transformations_and_aggregations(self):
        """Regression guard: a genuinely changed grid must still invalidate its work."""
        self.mock_query.side_effect = [
            [self.existing_doc()],  # initial grid docs query
            [self.existing_doc()],  # re-query for the doc id
        ]
        self.mock_md5.return_value = "different-checksum"

        init_pipeline.update_solr_grid("ECCO_llc90", "llc", "grids/ECCO_llc90.nc")

        bodies = self.sent_bodies()
        deletes = [b for b in bodies if isinstance(b, dict) and "delete" in b]
        self.assertEqual(len(deletes), 2)
        queries = " ".join(d["delete"]["query"] for d in deletes)
        self.assertIn("type_s:transformation", queries)
        self.assertIn("type_s:aggregation", queries)

        final = bodies[-1]
        self.assertEqual(final[0]["grid_checksum_s"], {"set": "different-checksum"})


class GridsToSolrTestCase(unittest.TestCase):
    def test_feeds_from_registry(self):
        """
        grids_to_solr syncs whatever the registry reports, so a grid whose `name`
        differs from its filename syncs under its name with its real path.
        """
        registry = {
            "polar_stereo_n_25km_demo": init_pipeline.grid_registry.GridFile(
                name="polar_stereo_n_25km_demo",
                path="grids/polar_stereo_n_25km.nc",
                type="latlon",
            )
        }

        with (
            patch(
                "utils.pipeline_utils.init_pipeline.grid_registry.load_registry",
                return_value=registry,
            ),
            patch(
                "utils.pipeline_utils.init_pipeline.update_solr_grid"
            ) as mock_update_grid,
            patch(
                "utils.pipeline_utils.init_pipeline.solr_utils.solr_query",
                return_value=[MagicMock()],
            ),
        ):
            missing = init_pipeline.grids_to_solr(["polar_stereo_n_25km_demo"])

        mock_update_grid.assert_called_once_with(
            "polar_stereo_n_25km_demo", "latlon", "grids/polar_stereo_n_25km.nc"
        )
        self.assertEqual(missing, [])

    def test_reports_grids_missing_from_solr(self):
        with (
            patch(
                "utils.pipeline_utils.init_pipeline.grid_registry.load_registry",
                return_value={},
            ),
            patch(
                "utils.pipeline_utils.init_pipeline.solr_utils.solr_query",
                return_value=[],
            ),
        ):
            missing = init_pipeline.grids_to_solr(["ECCO_llc540"])

        self.assertEqual(missing, ["ECCO_llc540"])
