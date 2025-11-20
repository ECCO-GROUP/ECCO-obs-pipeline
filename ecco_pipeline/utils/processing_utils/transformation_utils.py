import logging
from multiprocessing import current_process
from typing import Tuple
import warnings

import numpy as np
from pyresample.area_config import get_area_def
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info, resample_custom, resample_nearest
from pyresample.utils import check_and_wrap
from scipy.spatial import cKDTree

logger = logging.getLogger(str(current_process().pid))

GridProduct = Tuple[np.ndarray, float, SwathDefinition]


def transform_to_target_grid(
    source_indices_within_target_radius_i: dict,
    num_source_indices_within_target_radius_i: np.ndarray,
    nearest_source_index_to_target_index_i: dict,
    source_field: np.ndarray,
    target_grid_shape: tuple,
    operation: str = "mean",
    allow_nearest_neighbor: bool = True,
):
    """
    Vectorized transformation of source data to target grid.
    Missing target cells remain NaN.
    """

    # initialize full target grid
    source_on_target_grid = np.full(target_grid_shape, np.nan)
    tmp_r = source_on_target_grid.ravel()

    func_map = {
        "mean": np.mean,
        "nanmean": np.nanmean,
        "median": np.median,
        "nanmedian": np.nanmedian,
    }

    # Process cells that have source indices
    valid_cells = np.array(list(source_indices_within_target_radius_i.keys()))
    valid_counts = num_source_indices_within_target_radius_i[valid_cells] > 0

    if valid_counts.any():
        for i in valid_cells[valid_counts]:
            src_idx = source_indices_within_target_radius_i[i]
            vals = source_field.ravel()[src_idx]
            if operation in func_map:
                tmp_r[i] = func_map[operation](vals)
            elif operation == "nearest":
                tmp_r[i] = vals[0]

    # Fill remaining cells with nearest neighbor if allowed
    if allow_nearest_neighbor:
        nn_cells = np.array(list(nearest_source_index_to_target_index_i.keys()))
        tmp_r[nn_cells] = source_field.ravel()[np.array([nearest_source_index_to_target_index_i[i] for i in nn_cells])]

    return source_on_target_grid


def transform_along_track_to_grid(
    source_data: np.ndarray, source_def: SwathDefinition, target_def: SwathDefinition, lon_shape
):
    """ """

    def ones_weight(distances):
        return np.ones_like(distances, dtype=float)

    def gaussian_weight(distances, sigma=50000):  # sigma in meters
        return np.exp(-(distances**2) / (2 * sigma**2))

    ROI = 100000

    # Apply resampling
    source_resampled = resample_custom(
        source_def,
        source_data,
        target_def,
        radius_of_influence=ROI,
        neighbours=16,
        weight_funcs=ones_weight,
        fill_value=np.nan,
        reduce_data=False,
    )

    # Reshape back to tile structure
    source_on_target_grid = source_resampled.reshape(lon_shape)

    # Mask cells with no points within roi
    counts_flat = resample_nearest(
        source_def, np.ones_like(source_data), target_def, radius_of_influence=ROI, fill_value=0
    )
    counts = counts_flat.reshape(lon_shape)
    min_points = 1
    source_on_target_grid = np.where(counts >= min_points, source_on_target_grid, np.nan)
    return source_on_target_grid


def transform_along_track_to_target_grid(
    src_data: np.ndarray, src_lons: np.ndarray, src_lats: np.ndarray, tgt_xc: np.ndarray, tgt_yc: np.ndarray
) -> np.ndarray:
    valid_mask = ~np.isnan(src_data)
    src_lons_valid = src_lons[valid_mask]
    src_lats_valid = src_lats[valid_mask]
    src_values_valid = src_data[valid_mask]

    # Flatten LLC grid
    XC_flat = tgt_xc.ravel()
    YC_flat = tgt_yc.ravel()
    target_coords = np.column_stack((XC_flat, YC_flat))

    # Source coordinates
    source_coords = np.column_stack((src_lons_valid, src_lats_valid))

    # KDTree nearest neighbor mapping
    tree = cKDTree(target_coords)
    distances, indices = tree.query(source_coords, k=1)

    # Convert flat indices to (tile, j, i)
    tile_idx, j_idx, i_idx = np.unravel_index(indices, tgt_xc.shape)

    # Initialize accumulators
    accum = np.zeros_like(tgt_xc, dtype=float)
    count = np.zeros_like(tgt_xc, dtype=int)

    # Vectorized accumulation
    np.add.at(accum, (tile_idx, j_idx, i_idx), src_values_valid)
    np.add.at(count, (tile_idx, j_idx, i_idx), 1)

    # Fill only where there are points
    mean_grid = np.full_like(accum, np.nan, dtype=float)
    mask = count > 0
    mean_grid[mask] = accum[mask] / count[mask]


def find_mappings_from_source_to_target(
    target_grid: SwathDefinition,
    target_grid_radius: np.ndarray,
    grid_product: GridProduct,
    neighbours: int = 100,
    grid_name: str = "",
):
    """
    source grid, target_grid : area or grid defintion objects from pyresample

    target_grid_radius       : a vector indicating the radius of each
                               target grid cell (m)

    source_grid_min_l, source_grid_max_L : min and max distances
                               between adjacent source grid cells (m)

    neighbours     : Specifies number of neighbours to look for when getting
                     the neighbour info of a cell using pyresample.
                     Default is 100 to limit memory usage.
                     Value given must be a whole number greater than 0
    """
    source_grid, source_grid_min_L, source_grid_max_L = grid_product

    # # of element of the source and target grids
    len_source_grid = source_grid.size
    len_target_grid = target_grid.size

    # the maximum radius of the target grid
    max_target_grid_radius = np.nanmax(target_grid_radius)

    # the maximum number of neighbors to consider when doing the bin averaging
    # assuming that we have the largets target grid radius and the smallest
    # source grid length. (upper bound)

    # the ceiling is used to ensure the result is a whole number > 0
    neighbours_upper_bound = int((max_target_grid_radius * 2 / source_grid_min_L) ** 2)

    # compare provided and upper_bound value for neighbours.
    # limit neighbours to the upper_bound if the supplied neighbours value is larger
    # since you dont need more neighbours than exists within a cell.
    if neighbours > neighbours_upper_bound:
        print(
            "using more neighbours than upper bound.  limiting to the upper bound "
            f"of {int(neighbours_upper_bound)} neighbours"
        )
        neighbours = neighbours_upper_bound
    else:
        print(f"Only using {neighbours} nearest neighbours, but you may need up to {neighbours_upper_bound}")

    # make sure neighbours is an int for pyresample
    # neighbours_upper_bound is float, and user input can be float
    neighbours = int(neighbours)

    # FIRST FIND THE SET OF SOURCE GRID CELLS THAT FALL WITHIN THE SEARCH
    # RADIUS OF EACH TARGET GRID CELL

    # "target_grid_radius" is the half of the distance between
    # target grid cells.  No need to search for source
    # grid points more than halfway to the next target grid cell.

    # the get_neighbour_info returned from pyresample is quite useful.
    # Ax[1] seems to be a t/f array where true if there are any source grid points within target search radius (we think)
    # Ax[2] is the matrix of
    # closest SOURCE grid points for each TARGET grid point
    # Ax[3] is the actual distance in meters
    # also cool is that Ax[3] is sorted, first column is closest, last column
    # is furthest.
    # for some reason the radius of influence has to be in an int.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ax_max_target_grid_r = get_neighbour_info(
            source_grid,
            target_grid,
            radius_of_influence=int(max_target_grid_radius),
            neighbours=neighbours,
        )

    # define a dictionary, which will contain the list of SOURCE grid cells
    # that are within the search radius of each TARGET grid cell
    source_indices_within_target_radius_i = dict()

    # define a vector which is a COUNT of the # of SOURCE grid cells
    # that are within the search radius of each TARGET grid cell
    num_source_indices_within_target_radius_i = np.zeros((target_grid_radius.shape))

    # SECOND FIND THE SINGLE SOURCE GRID CELL THAT IS CLOSEST TO EACH
    # TARGET GRID CELL, BUT ONLY SEARCH AS FAR AS SOURCE_GRID_MAX_L

    # the kd_tree can also find the sigle nearest neighbor within the
    # radius 'source_grid_max_L'.  This second search is needed because sometimes
    # the source grid is finer than the target grid and therefore we may
    # end up in a situation where none of the centers of the SOURCE grid
    # fall within the small centers of the TARGET grid.
    # we'll look for the nearest SOURCE grid cell within 'source_grid_max_L'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ax_nearest_within_source_grid_max_L = get_neighbour_info(
            source_grid,
            target_grid,
            radius_of_influence=int(source_grid_max_L),
            neighbours=1,
        )

    # define a vector that will store the index of the source grid closest to
    # the target grid within the search radius 'source_grid_max_L'
    nearest_source_index_to_target_index_i = dict()

    # >> a list of i's to print debug statements for
    debug_is = np.linspace(0, len_target_grid, 20).astype(int)

    # loop through every TARGET grid cell, pull out the SOURCE grid cells
    # that fall within the TARGET grid radius and stick into the
    # 'source_indices_within_target_radius_i' dictionary
    # and then do the same for the nearest neighbour business.

    current_valid_target_i = 0

    for i in range(len_target_grid):
        if Ax_nearest_within_source_grid_max_L[1][i]:
            # Ax[2][i,:] are the closest source grid indices
            #            for target grid cell i
            # data_within_search_radius[i,:] is the T/F array for which
            #            of the closest 'neighbours' source grid indices are within
            #            the radius of this target grid cell i
            # -- so we're pulling out just those source grid indices
            #    that fall within the target grid cell radius

            # FIRST RECORD THE SOURCE POINTS THAT FALL WITHIN TARGET_GRID_RADIUS
            dist_from_src_to_target = Ax_max_target_grid_r[3][current_valid_target_i]

            dist_within_target_r = dist_from_src_to_target <= target_grid_radius[i]

            if neighbours > 1:
                src_indicies_here = Ax_max_target_grid_r[2][current_valid_target_i, :]
            else:
                src_indicies_here = Ax_max_target_grid_r[2][current_valid_target_i]

            source_indices_within_target_radius_i[i] = src_indicies_here[dist_within_target_r is True]

            # count the # source indices here
            num_source_indices_within_target_radius_i[i] = int(len(source_indices_within_target_radius_i[i]))

            # NOW RECORD THE NEAREST NEIGHBOR POINT WIHTIN SOURCE_GRID_MAX_L
            # when there is no source index within the search radius then
            # the 'get neighbour info' routine returns a dummy value of
            # the length of the source grid.  so we test to see if that's the
            # value that was returned.  If not, then we are good to go.
            # In other words...if the index in Ax_nearest... is the length of source grid, it is invalid
            # We think this should always because of the initial test for Ax[1] == True
            if Ax_nearest_within_source_grid_max_L[2][current_valid_target_i] < len_source_grid:
                nearest_source_index_to_target_index_i[i] = Ax_nearest_within_source_grid_max_L[2][
                    current_valid_target_i
                ]

            # increment this little bastard
            current_valid_target_i += 1

        # print progress.  always nice
        if i in debug_is:
            print(
                f"Creating {grid_name} mapping factors...{int(i / len_target_grid * 100)} %",
                end="\r",
            )
    print(f"Creating {grid_name} mapping factors...done.")
    return (
        source_indices_within_target_radius_i,
        num_source_indices_within_target_radius_i,
        nearest_source_index_to_target_index_i,
    )


def generalized_grid_product(
    data_res: float, area_extent: list[float], dims: list[float], proj_info: dict
) -> GridProduct:
    """
    Generates tuple containing (source_grid_min_L, source_grid_max_L, source_grid)

    https://pyresample.readthedocs.io/en/latest/api/pyresample.html#pyresample.area_config.create_area_def

    data_res: in degrees
    area_extent: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    dims: resolution of source grid
    proj_info: projection information
    return (source_grid_min_L, source_grid_max_L, source_grid)
    """

    # area_extent: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    areaExtent = (area_extent[0], area_extent[1], area_extent[2], area_extent[3])

    # Corresponds to resolution of grid from data
    cols = dims[0]
    rows = dims[1]

    # USE PYRESAMPLE TO GENERATE THE LAT/LON GRIDS
    # -- note we do not have to use pyresample for this, we could
    # have created it manually using the np.meshgrid or some other method
    # if we wanted.
    tmp_data_grid = get_area_def(
        proj_info["area_id"],
        proj_info["area_name"],
        proj_info["proj_id"],
        proj_info["proj4_args"],
        cols,
        rows,
        areaExtent,
    )

    data_grid_lons, data_grid_lats = tmp_data_grid.get_lonlats()

    # minimum Length of data product grid cells (km)
    source_grid_min_L = np.cos(np.deg2rad(np.nanmax(abs(data_grid_lats)))) * data_res * 112e3

    # maximum length of data roduct grid cells (km)
    # data product at equator has grid spacing of data_res*112e3 m
    source_grid_max_L = data_res * 112e3

    # Changes longitude bounds from 0-360 to -180-180, doesnt change if its already -180-180
    data_grid_lons, data_grid_lats = check_and_wrap(data_grid_lons, data_grid_lats)

    # Define the 'swath' (in the terminology of the pyresample module)
    # as the lats/lon pairs of the source observation grid
    # The routine needs the lats and lons to be one-dimensional vectors.
    source_grid = SwathDefinition(lons=data_grid_lons.ravel(), lats=data_grid_lats.ravel())

    return (source_grid_min_L, source_grid_max_L, source_grid)
