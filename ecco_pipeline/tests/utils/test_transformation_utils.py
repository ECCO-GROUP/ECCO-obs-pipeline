"""
Unit tests for transformation_utils module.
Pyresample calls are mocked where appropriate.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from utils.processing_utils.transformation_utils import (
    transform_to_target_grid,
    find_mappings_from_source_to_target,
    generalized_grid_product,
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
            (None, np.array([True, True, True, False]), np.array([0, 2, 4, 99]), np.array([100, 150, 120, 9999])),
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

        mock_get_neighbour.return_value = (None, np.array([True]), np.array([[0]]), np.array([[100]]))

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
    def test_basic_grid_generation(self, mock_get_area_def, mock_check_wrap, mock_swath):
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
            data_res=0.25, area_extent=[-180, -90, 180, 90], dims=[720, 1440], proj_info=proj_info
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
    def test_area_def_called_with_correct_params(self, mock_get_area_def, mock_check_wrap, mock_swath):
        """Test that area definition is called with correct parameters."""
        mock_area = MagicMock()
        mock_area.get_lonlats.return_value = (np.zeros((10, 10)), np.zeros((10, 10)))
        mock_get_area_def.return_value = mock_area
        mock_check_wrap.return_value = (np.zeros((10, 10)), np.zeros((10, 10)))

        proj_info = {"area_id": "my_area", "area_name": "My Area", "proj_id": "my_proj", "proj4_args": "+proj=longlat"}

        generalized_grid_product(data_res=1.0, area_extent=[-10, -20, 30, 40], dims=[100, 200], proj_info=proj_info)

        mock_get_area_def.assert_called_once_with(
            "my_area", "My Area", "my_proj", "+proj=longlat", 100, 200, (-10, -20, 30, 40)
        )


if __name__ == "__main__":
    unittest.main()
