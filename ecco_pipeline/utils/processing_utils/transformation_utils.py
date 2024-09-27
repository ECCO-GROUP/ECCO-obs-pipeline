import logging
from multiprocessing import current_process
from typing import Iterable
import warnings

import numpy as np
import pyresample as pr

logger = logging.getLogger(str(current_process().pid))


def transform_to_target_grid(
    source_indices_within_target_radius_i: dict,
    num_source_indices_within_target_radius_i: list,
    nearest_source_index_to_target_index_i: dict,
    source_field: np.ndarray,
    target_grid_shape: tuple,
    operation: str = "mean",
    allow_nearest_neighbor: bool = True,
):
    """
    Transforms source data to target grid

    source_indices_within_target_radius_i
    num_source_indices_within_target_radius_i
    nearest_source_index_to_target_index_i
    source field: 2D field
    target_grid_shape : shape of target grid array (2D)
    operation : one of ['mean', 'nanmean', 'median', 'nanmedian', 'nearest']

    """

    source_field_r = source_field.ravel()

    # define array that will contain source_field mapped to target_grid
    source_on_target_grid = np.full((target_grid_shape), np.nan)

    # get a 1D version of source_on_target_grid
    tmp_r = source_on_target_grid.ravel()

    # loop through every target grid point
    for i in range(len(tmp_r)):
        # if the number of source_field points at target grid cell i > 0
        if num_source_indices_within_target_radius_i[i] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # average these values
                if operation == "mean":
                    tmp_r[i] = np.mean(
                        source_field_r[source_indices_within_target_radius_i[i]]
                    )

                # average of non-nan values (can be slow)
                elif operation == "nanmean":
                    tmp_r[i] = np.nanmean(
                        source_field_r[source_indices_within_target_radius_i[i]]
                    )

                # median of these values
                elif operation == "median":
                    tmp_r[i] = np.median(
                        source_field_r[source_indices_within_target_radius_i[i]]
                    )

                # median of non-nan values (can be slow)
                elif operation == "nanmedian":
                    tmp_r[i] = np.nanmedian(
                        source_field_r[source_indices_within_target_radius_i[i]]
                    )

                # nearest neighbor is the first element in source_indices
                elif operation == "nearest":
                    tmp = source_indices_within_target_radius_i[i]
                    tmp_r[i] = source_field_r[tmp[0]]

        # number source indices within target radius is 0, then we can potentially
        # search for a nearest neighbor.
        elif allow_nearest_neighbor:
            # there is a nearest neighbor within range
            if i in nearest_source_index_to_target_index_i.keys():
                tmp_r[i] = source_field_r[nearest_source_index_to_target_index_i[i]]

    return source_on_target_grid


def find_mappings_from_source_to_target(
    source_grid,
    target_grid,
    target_grid_radius,
    source_grid_min_L,
    source_grid_max_L,
    neighbours: int = 100,
    less_output=True,
    grid_name="",
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
        print(
            f"Only using {neighbours} nearest neighbours, but you may need up to {neighbours_upper_bound}"
        )

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
        Ax_max_target_grid_r = pr.kd_tree.get_neighbour_info(
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
        Ax_nearest_within_source_grid_max_L = pr.kd_tree.get_neighbour_info(
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

    if not less_output:
        print("length of target grid: ", len_target_grid)

    for i in range(len_target_grid):
        if not less_output:
            print("looping through all points of target grid: ", i)

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

            source_indices_within_target_radius_i[i] = src_indicies_here[
                dist_within_target_r is True
            ]

            # count the # source indices here
            num_source_indices_within_target_radius_i[i] = int(
                len(source_indices_within_target_radius_i[i])
            )

            # NOW RECORD THE NEAREST NEIGHBOR POINT WIHTIN SOURCE_GRID_MAX_L
            # when there is no source index within the search radius then
            # the 'get neighbour info' routine returns a dummy value of
            # the length of the source grid.  so we test to see if that's the
            # value that was returned.  If not, then we are good to go.
            # In other words...if the index in Ax_nearest... is the length of source grid, it is invalid
            # We think this should always because of the initial test for Ax[1] == True
            if (
                Ax_nearest_within_source_grid_max_L[2][current_valid_target_i]
                < len_source_grid
            ):
                nearest_source_index_to_target_index_i[i] = (
                    Ax_nearest_within_source_grid_max_L[2][current_valid_target_i]
                )

            # increment this little bastard
            current_valid_target_i += 1

        # print progress.  always nice
        if i in debug_is:
            print(
                f"Creating {grid_name} mapping factors...{int(i/len_target_grid*100)} %",
                end="\r",
            )
    print(f"Creating {grid_name} mapping factors...done.")
    return (
        source_indices_within_target_radius_i,
        num_source_indices_within_target_radius_i,
        nearest_source_index_to_target_index_i,
    )


def generalized_grid_product(
    data_res: float,
    area_extent: Iterable[float],
    dims: Iterable[float],
    proj_info: dict,
) -> Iterable[np.ndarray]:
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
    tmp_data_grid = pr.area_config.get_area_def(
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
    source_grid_min_L = (
        np.cos(np.deg2rad(np.nanmax(abs(data_grid_lats)))) * data_res * 112e3
    )

    # maximum length of data roduct grid cells (km)
    # data product at equator has grid spacing of data_res*112e3 m
    source_grid_max_L = data_res * 112e3

    # Changes longitude bounds from 0-360 to -180-180, doesnt change if its already -180-180
    data_grid_lons, data_grid_lats = pr.utils.check_and_wrap(
        data_grid_lons, data_grid_lats
    )

    # Define the 'swath' (in the terminology of the pyresample module)
    # as the lats/lon pairs of the source observation grid
    # The routine needs the lats and lons to be one-dimensional vectors.
    source_grid = pr.geometry.SwathDefinition(
        lons=data_grid_lons.ravel(), lats=data_grid_lats.ravel()
    )

    return (source_grid_min_L, source_grid_max_L, source_grid)
