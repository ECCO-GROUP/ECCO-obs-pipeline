"""
Unit tests for dataset-to-grid coverage: parsing conf/grid_coverage.yaml
(baseclasses.load_grid_coverage), resolving it per dataset (Config.grids_for), and
validating it against grids on disk and active dataset configs
(config_validator.validate_grid_coverage).

The last class exercises the real conf/grid_coverage.yaml and the real dataset configs,
so a coverage edit that drops a grid or names a retired dataset fails here.
"""

import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import baseclasses
from utils.pipeline_utils import config_validator, grid_registry


def write_coverage(body: str) -> str:
    """Write a grid_coverage.yaml to a temp file and return its path."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, prefix="grid_coverage_"
    )
    handle.write(textwrap.dedent(body))
    handle.close()
    return handle.name


class CoverageTestBase(unittest.TestCase):
    def setUp(self):
        # grids_for caches the parsed file on Config; keep tests independent.
        self.addCleanup(baseclasses.set_grid_coverage, None)
        self.addCleanup(baseclasses.set_grids, [])

    def load(self, body: str):
        path = write_coverage(body)
        self.addCleanup(os.remove, path)
        return baseclasses.load_grid_coverage(path)


class LoadGridCoverageTestCase(CoverageTestBase):
    def test_parses_all_and_explicit_lists(self):
        coverage = self.load(
            """
            grids:
              ECCO_llc90:
                datasets: all
              ECCO_llc540:
                datasets:
                  - DATASET_A
                  - DATASET_B
            """
        )

        self.assertIsNone(coverage["ECCO_llc90"].datasets)
        self.assertEqual(coverage["ECCO_llc540"].datasets, ["DATASET_A", "DATASET_B"])

    def test_parses_excluded_variables(self):
        coverage = self.load(
            """
            grids:
              TPOSE:
                datasets: all
                exclude_ecco_variables: [sea_ice]
            """
        )

        self.assertEqual(
            coverage["TPOSE"].excluded_ecco_variables, frozenset({"sea_ice"})
        )

    def test_empty_dataset_list_is_valid(self):
        """A grid declared but not yet populated covers nothing, which is legitimate."""
        coverage = self.load(
            """
            grids:
              ECCO_llc540:
                datasets: []
            """
        )

        self.assertEqual(coverage["ECCO_llc540"].datasets, [])
        self.assertFalse(coverage["ECCO_llc540"].covers("ANY_DATASET", "ssh"))

    def test_missing_file_raises(self):
        with self.assertRaises(baseclasses.GridCoverageError):
            baseclasses.load_grid_coverage(
                os.path.join(tempfile.gettempdir(), "definitely_not_here.yaml")
            )

    def test_missing_datasets_key_raises(self):
        with self.assertRaises(baseclasses.GridCoverageError) as ctx:
            self.load(
                """
                grids:
                  ECCO_llc90:
                    exclude_ecco_variables: [sea_ice]
                """
            )

        self.assertIn("datasets", str(ctx.exception))

    def test_bad_datasets_value_raises(self):
        with self.assertRaises(baseclasses.GridCoverageError) as ctx:
            self.load(
                """
                grids:
                  ECCO_llc90:
                    datasets: everything
                """
            )

        self.assertIn("everything", str(ctx.exception))

    def test_empty_grids_mapping_raises(self):
        with self.assertRaises(baseclasses.GridCoverageError):
            self.load("grids:\n")


class GridsForTestCase(CoverageTestBase):
    COVERAGE = """
        grids:
          ECCO_llc90:
            datasets: all
          ECCO_llc270:
            datasets: all
          ECCO_llc540:
            datasets:
              - COVERED_DATASET
          TPOSE:
            datasets: all
            exclude_ecco_variables: [sea_ice]
        """

    def setUp(self):
        super().setUp()
        baseclasses.set_grid_coverage(self.load(self.COVERAGE))
        baseclasses.set_grids(["ECCO_llc90", "ECCO_llc270", "ECCO_llc540", "TPOSE"])

    def test_all_grids_for_covered_dataset(self):
        grids = baseclasses.Config.grids_for(
            {"ds_name": "COVERED_DATASET", "ecco_variable": "ssh"}
        )

        self.assertEqual(grids, ["ECCO_llc90", "ECCO_llc270", "ECCO_llc540", "TPOSE"])

    def test_narrow_grid_dropped_for_unlisted_dataset(self):
        grids = baseclasses.Config.grids_for(
            {"ds_name": "OTHER_DATASET", "ecco_variable": "ssh"}
        )

        self.assertEqual(grids, ["ECCO_llc90", "ECCO_llc270", "TPOSE"])

    def test_excluded_variable_drops_grid(self):
        grids = baseclasses.Config.grids_for(
            {"ds_name": "SOME_SEA_ICE", "ecco_variable": "sea_ice"}
        )

        self.assertEqual(grids, ["ECCO_llc90", "ECCO_llc270"])

    def test_exclusion_does_not_depend_on_hemi_pattern(self):
        """
        The RDEFT4 case. Exclusion keys off ecco_variable, so a sea ice dataset with no
        hemispheric split is excluded too — the old hardcoded check keyed off
        hemi_pattern and let RDEFT4 through to TPOSE.
        """
        with_hemi = baseclasses.Config.grids_for(
            {
                "ds_name": "SPLIT_SEA_ICE",
                "ecco_variable": "sea_ice",
                "hemi_pattern": {"north": "_nh_", "south": "_sh_"},
            }
        )
        without_hemi = baseclasses.Config.grids_for(
            {"ds_name": "RDEFT4", "ecco_variable": "sea_ice"}
        )

        self.assertEqual(with_hemi, without_hemi)
        self.assertNotIn("TPOSE", without_hemi)

    def test_intersects_with_narrower_enabled_set(self):
        """--grids_to_use narrows the result; coverage cannot widen it back."""
        baseclasses.set_grids(["ECCO_llc540"])

        self.assertEqual(
            baseclasses.Config.grids_for(
                {"ds_name": "COVERED_DATASET", "ecco_variable": "ssh"}
            ),
            ["ECCO_llc540"],
        )
        self.assertEqual(
            baseclasses.Config.grids_for(
                {"ds_name": "OTHER_DATASET", "ecco_variable": "ssh"}
            ),
            [],
        )

    def test_preserves_enabled_grid_order(self):
        baseclasses.set_grids(["TPOSE", "ECCO_llc90"])

        self.assertEqual(
            baseclasses.Config.grids_for(
                {"ds_name": "COVERED_DATASET", "ecco_variable": "ssh"}
            ),
            ["TPOSE", "ECCO_llc90"],
        )

    def test_enabled_grid_without_block_raises(self):
        """
        init_pipeline aborts on this at startup; reaching grids_for means a caller
        bypassed it, so fail loudly rather than silently dropping or including a grid.
        """
        baseclasses.set_grids(["ECCO_llc90", "ECCO_llc1080"])

        with self.assertRaises(baseclasses.GridCoverageError) as ctx:
            baseclasses.Config.grids_for(
                {"ds_name": "COVERED_DATASET", "ecco_variable": "ssh"}
            )

        self.assertIn("ECCO_llc1080", str(ctx.exception))

    def test_empty_enabled_set_falls_back_to_declared_grids(self):
        """
        A bare --grids_to_use yields [], which must not fan a narrow grid across every
        dataset. Falls back to the grids declared in the coverage file, still filtered
        by each grid's own coverage.
        """
        baseclasses.set_grids([])

        self.assertEqual(
            baseclasses.Config.grids_for(
                {"ds_name": "OTHER_DATASET", "ecco_variable": "ssh"}
            ),
            ["ECCO_llc90", "ECCO_llc270", "TPOSE"],
        )


class ValidateGridCoverageTestCase(CoverageTestBase):
    REGISTRY = {"ECCO_llc90": object(), "ECCO_llc540": object(), "TPOSE": object()}

    def validate(self, body, grids_to_use):
        path = write_coverage(body)
        self.addCleanup(os.remove, path)
        with patch.object(baseclasses, "GRID_COVERAGE_PATH", path):
            return config_validator.validate_grid_coverage(grids_to_use, self.REGISTRY)

    def test_enabled_grid_without_block_reported(self):
        problems = self.validate(
            """
            grids:
              ECCO_llc90:
                datasets: all
            """,
            ["ECCO_llc90", "ECCO_llc540"],
        )

        self.assertEqual(len(problems), 1)
        self.assertIn("ECCO_llc540", problems[0])
        self.assertIn("no block", problems[0])

    def test_unknown_grid_name_reported(self):
        problems = self.validate(
            """
            grids:
              ECCO_llc90:
                datasets: all
              ECCO_llc1080:
                datasets: all
            """,
            ["ECCO_llc90"],
        )

        self.assertEqual(len(problems), 1)
        self.assertIn("ECCO_llc1080", problems[0])
        self.assertIn("no grid file", problems[0])

    def test_dangling_dataset_reported(self):
        problems = self.validate(
            """
            grids:
              ECCO_llc90:
                datasets: all
              ECCO_llc540:
                datasets:
                  - TOTALLY_MADE_UP_DATASET
            """,
            ["ECCO_llc90"],
        )

        self.assertEqual(len(problems), 1)
        self.assertIn("TOTALLY_MADE_UP_DATASET", problems[0])
        self.assertIn("no config", problems[0])

    def test_deprecated_dataset_gets_hint(self):
        """
        The load-bearing case for version bumps: moving a config into deprecated/ makes
        its coverage entry dangle in the same commit that adds the replacement, and the
        message says where it went.
        """
        problems = self.validate(
            """
            grids:
              ECCO_llc90:
                datasets: all
              ECCO_llc540:
                datasets:
                  - G02202_V5
            """,
            ["ECCO_llc90"],
        )

        self.assertEqual(len(problems), 1)
        self.assertIn("G02202_V5", problems[0])
        self.assertIn("deprecated", problems[0])

    def test_near_miss_dataset_gets_suggestion(self):
        problems = self.validate(
            """
            grids:
              ECCO_llc90:
                datasets: all
              ECCO_llc540:
                datasets:
                  - AVHRR_OI-NCEI-L4-GLOB-v9.9
            """,
            ["ECCO_llc90"],
        )

        self.assertEqual(len(problems), 1)
        self.assertIn("did you mean", problems[0])
        self.assertIn("AVHRR_OI-NCEI-L4-GLOB-v2.1", problems[0])

    def test_unknown_ecco_variable_reported(self):
        problems = self.validate(
            """
            grids:
              ECCO_llc90:
                datasets: all
              TPOSE:
                datasets: all
                exclude_ecco_variables: [seaice]
            """,
            ["ECCO_llc90"],
        )

        self.assertEqual(len(problems), 1)
        self.assertIn("seaice", problems[0])
        self.assertIn("sea_ice", problems[0])


class RealCoverageFileTestCase(CoverageTestBase):
    """
    The committed conf/grid_coverage.yaml must be valid against the committed grids and
    dataset configs, so a bad coverage edit fails in CI rather than at run time.
    """

    def test_real_coverage_file_is_valid(self):
        coverage = baseclasses.load_grid_coverage()
        registry = grid_registry.load_registry(use_cache=False)

        problems = config_validator.validate_grid_coverage(list(coverage), registry)

        self.assertEqual(problems, [])

    def test_sea_ice_datasets_excluded_from_tpose(self):
        baseclasses.set_grid_coverage(baseclasses.load_grid_coverage())
        baseclasses.set_grids(["ECCO_llc90", "TPOSE"])

        sea_ice = baseclasses.Config.grids_for(
            {"ds_name": "RDEFT4", "ecco_variable": "sea_ice"}
        )
        ssh = baseclasses.Config.grids_for(
            {"ds_name": "NASA_SSH_REF_ALONGTRACK_V11", "ecco_variable": "ssh"}
        )

        self.assertEqual(sea_ice, ["ECCO_llc90"])
        self.assertEqual(ssh, ["ECCO_llc90", "TPOSE"])
