"""
Unit tests for the grid registry, which owns grid identity (name, path, type).

Each test writes real (tiny) netCDF grid files to a temp directory rather than mocking
xarray: what is under test is how global attributes on disk map to registry entries,
and a mock would assert nothing about that.
"""

import os
import tempfile
import unittest

import xarray as xr

from utils.pipeline_utils import grid_registry


def write_grid(directory: str, filename: str, attrs: dict):
    """Write a minimal grid netCDF carrying the given global attributes."""
    ds = xr.Dataset({"effective_grid_radius": ("i", [1.0, 2.0])})
    ds.attrs.update(attrs)
    path = os.path.join(directory, filename)
    ds.to_netcdf(path)
    return path


class GridRegistryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.grids_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(grid_registry.clear_cache)
        grid_registry.clear_cache()

    def load(self):
        return grid_registry.load_registry(self.grids_dir, use_cache=False)

    def test_name_attribute_wins_over_filename(self):
        """
        A grid is identified by its ``name`` global attribute, not its filename.

        This is the whole point of the registry: grids/polar_stereo_n_25km.nc is named
        polar_stereo_n_25km_demo, and the transformation path used to rebuild paths as
        f"grids/{name}.nc", raising FileNotFoundError for exactly this case.
        """
        write_grid(
            self.grids_dir,
            "polar_stereo_n_25km.nc",
            {"name": "polar_stereo_n_25km_demo", "type": "latlon"},
        )

        registry = self.load()

        self.assertIn("polar_stereo_n_25km_demo", registry)
        self.assertNotIn("polar_stereo_n_25km", registry)
        entry = registry["polar_stereo_n_25km_demo"]
        self.assertTrue(entry.path.endswith("polar_stereo_n_25km.nc"))
        self.assertEqual(entry.type, "latlon")

    def test_multiple_grids_registered(self):
        write_grid(self.grids_dir, "a.nc", {"name": "ECCO_llc90", "type": "llc"})
        write_grid(self.grids_dir, "b.nc", {"name": "TPOSE", "type": "latlon"})

        registry = self.load()

        self.assertEqual(sorted(registry), ["ECCO_llc90", "TPOSE"])

    def test_duplicate_name_raises(self):
        """
        Two files claiming one name is rejected, naming both paths.

        Solr grid docs are keyed by name and update_solr_grid deletes every
        transformation and aggregation for a grid whose checksum changed, so a
        duplicate would make alternating runs wipe each other's work.
        """
        write_grid(self.grids_dir, "first.nc", {"name": "ECCO_llc90", "type": "llc"})
        write_grid(self.grids_dir, "second.nc", {"name": "ECCO_llc90", "type": "llc"})

        with self.assertRaises(grid_registry.GridRegistryError) as ctx:
            self.load()

        message = str(ctx.exception)
        self.assertIn("duplicate grid name", message)
        self.assertIn("first.nc", message)
        self.assertIn("second.nc", message)

    def test_missing_name_attribute_raises(self):
        """
        Previously a KeyError inside a caller that logged and continued, so the run
        proceeded with the grid silently absent.
        """
        write_grid(self.grids_dir, "nameless.nc", {"type": "llc"})

        with self.assertRaises(grid_registry.GridRegistryError) as ctx:
            self.load()

        self.assertIn("name", str(ctx.exception))

    def test_missing_type_attribute_raises(self):
        write_grid(self.grids_dir, "typeless.nc", {"name": "ECCO_llc90"})

        with self.assertRaises(grid_registry.GridRegistryError) as ctx:
            self.load()

        self.assertIn("type", str(ctx.exception))

    def test_invalid_type_raises(self):
        write_grid(
            self.grids_dir, "bad.nc", {"name": "ECCO_llc90", "type": "curvilinear"}
        )

        with self.assertRaises(grid_registry.GridRegistryError) as ctx:
            self.load()

        self.assertIn("curvilinear", str(ctx.exception))

    def test_empty_directory_returns_empty_registry(self):
        self.assertEqual(self.load(), {})

    def test_grid_path_resolves(self):
        write_grid(
            self.grids_dir, "weird_filename.nc", {"name": "TPOSE", "type": "latlon"}
        )

        path = grid_registry.grid_path("TPOSE", self.grids_dir)

        self.assertTrue(path.endswith("weird_filename.nc"))

    def test_grid_path_unknown_grid_raises_with_known_names(self):
        write_grid(self.grids_dir, "a.nc", {"name": "ECCO_llc90", "type": "llc"})

        with self.assertRaises(grid_registry.GridRegistryError) as ctx:
            grid_registry.grid_path("ECCO_llc1080", self.grids_dir)

        message = str(ctx.exception)
        self.assertIn("ECCO_llc1080", message)
        self.assertIn("ECCO_llc90", message)

    def test_cache_is_per_directory(self):
        write_grid(self.grids_dir, "a.nc", {"name": "ECCO_llc90", "type": "llc"})

        first = grid_registry.load_registry(self.grids_dir)
        second = grid_registry.load_registry(self.grids_dir)

        self.assertIs(first, second)

        grid_registry.clear_cache(self.grids_dir)
        third = grid_registry.load_registry(self.grids_dir)
        self.assertIsNot(first, third)
        self.assertEqual(sorted(third), sorted(first))


class RealGridsTestCase(unittest.TestCase):
    """The grid files actually committed to the repo must satisfy the registry."""

    def setUp(self):
        self.addCleanup(grid_registry.clear_cache)
        grid_registry.clear_cache()

    def test_repo_grids_load(self):
        # Absolute path rather than the default relative "grids": that default relies
        # on conf.global_settings having chdir'd to ROOT_DIR on import, which this
        # test does not depend on.
        grids_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "grids",
        )
        registry = grid_registry.load_registry(grids_dir, use_cache=False)

        for expected in ("ECCO_llc90", "ECCO_llc270", "ECCO_llc540"):
            self.assertIn(expected, registry)
            self.assertEqual(registry[expected].type, "llc")

        for entry in registry.values():
            self.assertTrue(os.path.isfile(entry.path))
