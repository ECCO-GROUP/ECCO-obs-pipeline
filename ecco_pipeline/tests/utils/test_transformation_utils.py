"""
Unit tests for transformation_utils module.
Pyresample calls are mocked where appropriate.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr

from utils.processing_utils.transformation_utils import (
    transform_to_target_grid,
    find_mappings_from_source_to_target,
    generalized_grid_product,
    along_track_factors,
    _EARTH_RADIUS_M,
)


class TransformToTargetGridTestCase(unittest.TestCase):
    """Tests for the transform_to_target_grid function."""

    def setUp(self):
        """Set up common test data."""
        # Simple 2x2 target grid
        self.target_shape = (2, 2)

        # Source indices mapping for each target cell
        self.source_indices = {
            0: np.array([0, 1]),
            1: np.array([2]),
            2: np.array([3, 4, 5]),
            3: np.array([]),  # No source points
        }

        # Count of source indices
        self.num_source_indices = np.array([2, 1, 3, 0])

        # Nearest neighbor mapping
        self.nearest_source = {3: 6}

        # Source field values (flat array with 7 elements)
        self.source_field = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])

    def test_mean_operation(self):
        """Test mean aggregation."""
        result = transform_to_target_grid(
            self.source_indices,
            self.num_source_indices,
            self.nearest_source,
            self.source_field,
            self.target_shape,
            operation="mean",
        )

        # Cell 0: mean of [1, 3] = 2.0
        self.assertEqual(result[0, 0], 2.0)
        # Cell 1: mean of [5] = 5.0
        self.assertEqual(result[0, 1], 5.0)
        # Cell 2: mean of [7, 9, 11] = 9.0
        self.assertEqual(result[1, 0], 9.0)
        # Cell 3: nearest neighbor = 13.0
        self.assertEqual(result[1, 1], 13.0)

    def test_nanmean_operation(self):
        """Test nanmean aggregation with NaN values."""
        source_with_nan = np.array([1.0, np.nan, 5.0, 7.0, 9.0, 11.0, 13.0])

        result = transform_to_target_grid(
            self.source_indices,
            self.num_source_indices,
            self.nearest_source,
            source_with_nan,
            self.target_shape,
            operation="nanmean",
        )

        # Cell 0: nanmean of [1, nan] = 1.0
        self.assertEqual(result[0, 0], 1.0)

    def test_median_operation(self):
        """Test median aggregation."""
        result = transform_to_target_grid(
            self.source_indices,
            self.num_source_indices,
            self.nearest_source,
            self.source_field,
            self.target_shape,
            operation="median",
        )

        # Cell 2: median of [7, 9, 11] = 9.0
        self.assertEqual(result[1, 0], 9.0)

    def test_nearest_operation(self):
        """Test nearest neighbor selection."""
        result = transform_to_target_grid(
            self.source_indices,
            self.num_source_indices,
            self.nearest_source,
            self.source_field,
            self.target_shape,
            operation="nearest",
        )

        # Cell 0: first element = 1.0
        self.assertEqual(result[0, 0], 1.0)

    def test_no_nearest_neighbor(self):
        """Test when nearest neighbor fallback is disabled."""
        result = transform_to_target_grid(
            self.source_indices,
            self.num_source_indices,
            {},  # No nearest neighbors
            self.source_field,
            self.target_shape,
            operation="mean",
            allow_nearest_neighbor=False,
        )

        # Cell 3 should be NaN (no source points and no fallback)
        self.assertTrue(np.isnan(result[1, 1]))

    def test_output_shape(self):
        """Test that output has correct shape."""
        # Use target shape that matches our source_indices (2x2 = 4 cells)
        result = transform_to_target_grid(
            self.source_indices,
            self.num_source_indices,
            self.nearest_source,
            self.source_field,
            self.target_shape,
            operation="mean",
        )

        self.assertEqual(result.shape, (2, 2))


class FindMappingsFromSourceToTargetTestCase(unittest.TestCase):
    """Tests for the find_mappings_from_source_to_target function."""

    @patch("utils.processing_utils.transformation_utils.pr.kd_tree.get_neighbour_info")
    def test_basic_mapping(self, mock_get_neighbour):
        """Test basic mapping creation."""
        # Create mock source and target grids
        mock_source = MagicMock()
        mock_source.size = 100

        mock_target = MagicMock()
        mock_target.size = 4

        target_radius = np.array([1000, 1000, 1000, 1000])

        # Mock pyresample return values
        # Returns: (input_index, valid_mask, index_array, distance_array)
        mock_get_neighbour.side_effect = [
            # First call for max target radius
            (
                None,
                np.array([True, True, True, False]),
                np.array([[0, 1], [2, 3], [4, 5], [99, 99]]),
                np.array([[100, 200], [150, 250], [120, 220], [9999, 9999]]),
            ),
            # Second call for nearest within source_grid_max_L
            (
                None,
                np.array([True, True, True, False]),
                np.array([0, 2, 4, 99]),
                np.array([100, 150, 120, 9999]),
            ),
        ]

        result = find_mappings_from_source_to_target(
            mock_source,
            mock_target,
            target_radius,
            source_grid_min_L=100,
            source_grid_max_L=500,
            neighbours=10,
            less_output=True,
        )

        self.assertEqual(len(result), 3)
        # source_indices, num_source_indices, nearest_source_index

    @patch("utils.processing_utils.transformation_utils.pr.kd_tree.get_neighbour_info")
    def test_neighbours_upper_bound_limiting(self, mock_get_neighbour):
        """Test that neighbours is limited to upper bound."""
        mock_source = MagicMock()
        mock_source.size = 100

        mock_target = MagicMock()
        mock_target.size = 1

        target_radius = np.array([1000])

        mock_get_neighbour.return_value = (
            None,
            np.array([True]),
            np.array([[0]]),
            np.array([[100]]),
        )

        # With large neighbours value that exceeds upper bound
        find_mappings_from_source_to_target(
            mock_source,
            mock_target,
            target_radius,
            source_grid_min_L=100,  # upper_bound = (2000/100)^2 = 400
            source_grid_max_L=500,
            neighbours=10000,  # Much larger than upper bound
            less_output=True,
        )

        # Verify the function executed without error
        self.assertTrue(mock_get_neighbour.called)


class GeneralizedGridProductTestCase(unittest.TestCase):
    """Tests for the generalized_grid_product function."""

    @patch("utils.processing_utils.transformation_utils.pr.geometry.SwathDefinition")
    @patch("utils.processing_utils.transformation_utils.pr.utils.check_and_wrap")
    @patch("utils.processing_utils.transformation_utils.pr.area_config.get_area_def")
    def test_basic_grid_generation(
        self, mock_get_area_def, mock_check_wrap, mock_swath
    ):
        """Test basic grid generation."""
        # Mock area definition
        mock_area = MagicMock()
        mock_lons = np.array([[0, 1], [0, 1]])
        mock_lats = np.array([[45, 45], [46, 46]])
        mock_area.get_lonlats.return_value = (mock_lons, mock_lats)
        mock_get_area_def.return_value = mock_area

        # Mock check_and_wrap
        mock_check_wrap.return_value = (mock_lons, mock_lats)

        # Mock SwathDefinition
        mock_swath_instance = MagicMock()
        mock_swath.return_value = mock_swath_instance

        proj_info = {
            "area_id": "test_area",
            "area_name": "Test Area",
            "proj_id": "test_proj",
            "proj4_args": "+proj=latlong +datum=WGS84",
        }

        result = generalized_grid_product(
            data_res=0.25,
            area_extent=[-180, -90, 180, 90],
            dims=[720, 1440],
            proj_info=proj_info,
        )

        self.assertEqual(len(result), 3)
        source_grid_min_L, source_grid_max_L, source_grid = result

        # Check that max_L is calculated correctly
        # max_L = 0.25 * 112e3 = 28000
        self.assertAlmostEqual(source_grid_max_L, 28000.0)

        # Check that min_L is calculated based on max latitude
        # At 46 degrees: cos(46) * 0.25 * 112e3
        expected_min_L = np.cos(np.deg2rad(46)) * 0.25 * 112e3
        self.assertAlmostEqual(source_grid_min_L, expected_min_L)

    @patch("utils.processing_utils.transformation_utils.pr.geometry.SwathDefinition")
    @patch("utils.processing_utils.transformation_utils.pr.utils.check_and_wrap")
    @patch("utils.processing_utils.transformation_utils.pr.area_config.get_area_def")
    def test_area_def_called_with_correct_params(
        self, mock_get_area_def, mock_check_wrap, mock_swath
    ):
        """Test that area definition is called with correct parameters."""
        mock_area = MagicMock()
        mock_area.get_lonlats.return_value = (np.zeros((10, 10)), np.zeros((10, 10)))
        mock_get_area_def.return_value = mock_area
        mock_check_wrap.return_value = (np.zeros((10, 10)), np.zeros((10, 10)))

        proj_info = {
            "area_id": "my_area",
            "area_name": "My Area",
            "proj_id": "my_proj",
            "proj4_args": "+proj=longlat",
        }

        generalized_grid_product(
            data_res=1.0,
            area_extent=[-10, -20, 30, 40],
            dims=[100, 200],
            proj_info=proj_info,
        )

        mock_get_area_def.assert_called_once_with(
            "my_area",
            "My Area",
            "my_proj",
            "+proj=longlat",
            100,
            200,
            (-10, -20, 30, 40),
        )


def _make_grid_ds(xc, yc, radius_m):
    """
    Minimal target grid dataset for along_track_factors: XC/YC cell centers (deg) and
    a per-cell radius (m) exposed as effective_grid_radius.
    """
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    radius_m = np.broadcast_to(np.asarray(radius_m, dtype=float), xc.shape)
    dims = [f"d{i}" for i in range(xc.ndim)]
    return xr.Dataset(
        {
            "XC": (dims, xc),
            "YC": (dims, yc),
            "effective_grid_radius": (dims, radius_m.copy()),
        }
    )


def _deg_radius_to_m(deg):
    """Great-circle arc length (m) for a given angular radius in degrees."""
    return np.deg2rad(deg) * _EARTH_RADIUS_M


class AlongTrackFactorsTestCase(unittest.TestCase):
    """Tests for along_track_factors nearest-cell binning (ADR 0002)."""

    def test_binning_and_nanmean_partition(self):
        """
        A straight track through a small known grid: each cell gets the nanmean of
        exactly the points nearest it; point-free cells stay NaN.
        """
        # 1x4 grid of cells at lon 0,1,2,3 (lat 0), generous radius so every point bins.
        xc = np.array([[0.0, 1.0, 2.0, 3.0]])
        yc = np.zeros((1, 4))
        grid_ds = _make_grid_ds(xc, yc, _deg_radius_to_m(0.6))

        # Points: two near cell 0, one near cell 2, none near cells 1 and 3.
        lons = np.array([-0.1, 0.1, 2.05])
        lats = np.array([0.0, 0.0, 0.0])
        vals = np.array([10.0, 20.0, 7.0])

        factors = along_track_factors(lons, lats, grid_ds)
        sidx, counts, nearest = factors

        self.assertEqual(nearest, {})
        # cell 0 got points 0 and 1; cell 2 got point 2.
        np.testing.assert_array_equal(sorted(sidx[0].tolist()), [0, 1])
        np.testing.assert_array_equal(sidx[2].tolist(), [2])
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[2], 1)
        self.assertEqual(counts[1], 0)
        self.assertEqual(counts[3], 0)

        out = transform_to_target_grid(
            *factors, vals, xc.shape, operation="nanmean", allow_nearest_neighbor=False
        )
        self.assertAlmostEqual(out[0, 0], 15.0)  # mean(10, 20)
        self.assertTrue(np.isnan(out[0, 1]))
        self.assertAlmostEqual(out[0, 2], 7.0)
        self.assertTrue(np.isnan(out[0, 3]))

    def test_indices_reference_full_array(self):
        """
        Surviving indices must reference the FULL field array, not a compacted one.
        With a NaN coordinate up front, a later point's stored index must still point
        at its original position so transform_to_target_grid averages the right sample.
        """
        xc = np.array([[0.0, 5.0]])
        yc = np.zeros((1, 2))
        grid_ds = _make_grid_ds(xc, yc, _deg_radius_to_m(0.6))

        # Point 0 has a NaN lon (dropped); point 1 is near cell 0. If indices were
        # compacted, cell 0 would store index 0 and average the wrong (point-0) value.
        lons = np.array([np.nan, 0.1])
        lats = np.array([0.0, 0.0])
        vals = np.array([999.0, 42.0])

        factors = along_track_factors(lons, lats, grid_ds)
        sidx, counts, _ = factors
        np.testing.assert_array_equal(sidx[0].tolist(), [1])  # original position, not 0

        out = transform_to_target_grid(
            *factors, vals, xc.shape, operation="nanmean", allow_nearest_neighbor=False
        )
        self.assertAlmostEqual(out[0, 0], 42.0)

    def test_per_cell_radius_cap(self):
        """A point just inside a cell's radius bins; just outside is discarded."""
        xc = np.array([[0.0]])
        yc = np.array([[0.0]])
        grid_ds = _make_grid_ds(xc, yc, _deg_radius_to_m(1.0))  # 1-degree radius

        # inside: 0.9 deg away; outside: 1.1 deg away (both along the equator).
        factors_in = along_track_factors(np.array([0.9]), np.array([0.0]), grid_ds)
        self.assertEqual(factors_in[1][0], 1)

        factors_out = along_track_factors(np.array([1.1]), np.array([0.0]), grid_ds)
        self.assertEqual(factors_out[1][0], 0)
        self.assertEqual(factors_out[0], {})

    def test_antimeridian(self):
        """
        A point near the antimeridian bins to the correct cell. A planar lon/lat
        kd-tree would read a 0.2-deg-apart pair as ~359.8 deg apart and mis-bin;
        the ECEF conversion gets it right.
        """
        # Two cells straddling the 180/-180 seam.
        xc = np.array([[179.9, -179.9]])
        yc = np.zeros((1, 2))
        grid_ds = _make_grid_ds(xc, yc, _deg_radius_to_m(0.3))

        # Point at 180.0 (== -180.0) is 0.1 deg from each cell; put it at 179.95 so
        # its true nearest is cell 0. Also test a point clearly nearest cell 1.
        f0 = along_track_factors(np.array([179.95]), np.array([0.0]), grid_ds)
        self.assertIn(0, f0[0])
        self.assertNotIn(1, f0[0])

        f1 = along_track_factors(np.array([-179.95]), np.array([0.0]), grid_ds)
        self.assertIn(1, f1[0])
        self.assertNotIn(0, f1[0])

    def test_high_latitude_nearest(self):
        """
        At high latitude the cos(lat) factor distorts planar lon distance. A point
        binned by true spherical distance goes to the geometrically nearest cell.
        """
        # Two cells 2 deg apart in longitude at 80N. In ECEF they are close together;
        # a point between them should bin to whichever is spherically nearer.
        xc = np.array([[10.0, 12.0]])
        yc = np.array([[80.0, 80.0]])
        # Radius large enough (in meters) to admit the point but small in degrees at
        # this latitude — exactly the regime where a degree-valued cap misbehaves.
        grid_ds = _make_grid_ds(xc, yc, _deg_radius_to_m(1.5))

        # Point at lon 10.4, closer to cell 0.
        f = along_track_factors(np.array([10.4]), np.array([80.0]), grid_ds)
        self.assertIn(0, f[0])
        self.assertNotIn(1, f[0])

    def test_all_nan_and_empty(self):
        """All-NaN coords / empty input -> all-NaN record, no crash."""
        xc = np.array([[0.0, 1.0]])
        yc = np.zeros((1, 2))
        grid_ds = _make_grid_ds(xc, yc, _deg_radius_to_m(0.6))

        # All-NaN coordinates.
        factors = along_track_factors(
            np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), grid_ds
        )
        self.assertEqual(factors[0], {})
        self.assertTrue(np.all(factors[1] == 0))
        out = transform_to_target_grid(
            *factors,
            np.array([1.0, 2.0]),
            xc.shape,
            operation="nanmean",
            allow_nearest_neighbor=False,
        )
        self.assertTrue(np.all(np.isnan(out)))

        # Empty input.
        factors_e = along_track_factors(np.array([]), np.array([]), grid_ds)
        self.assertEqual(factors_e[0], {})
        self.assertTrue(np.all(factors_e[1] == 0))

    def test_nan_values_not_masked_in_factors(self):
        """
        Factors depend on coordinates only; NaN field VALUES are not masked here (the
        downstream nanmean handles them), so a point with a valid coord but NaN value
        still appears in the factors and is dropped per-field by nanmean.
        """
        xc = np.array([[0.0]])
        yc = np.array([[0.0]])
        grid_ds = _make_grid_ds(xc, yc, _deg_radius_to_m(0.6))

        lons = np.array([0.0, 0.05])
        lats = np.array([0.0, 0.0])
        factors = along_track_factors(lons, lats, grid_ds)
        # Both points are in the factors regardless of value validity.
        self.assertEqual(factors[1][0], 2)

        vals = np.array([np.nan, 8.0])
        out = transform_to_target_grid(
            *factors, vals, xc.shape, operation="nanmean", allow_nearest_neighbor=False
        )
        self.assertAlmostEqual(out[0, 0], 8.0)  # nanmean drops the NaN value


if __name__ == "__main__":
    unittest.main()
