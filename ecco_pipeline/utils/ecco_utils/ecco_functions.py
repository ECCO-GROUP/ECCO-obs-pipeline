import logging
import os
from datetime import datetime
from typing import List, Mapping, Tuple

import numpy as np
import pyresample as pr
import xarray as xr
from transformations.transformation import Transformation
from utils.ecco_utils import date_time, records


def transform_to_target_grid(source_indices_within_target_radius_i: dict,
                             num_source_indices_within_target_radius_i: list,
                             nearest_source_index_to_target_index_i: dict,
                             source_field, target_grid_shape, operation: str = 'mean',
                             allow_nearest_neighbor: bool = True):
    '''
    Transforms source data to target grid

    source_indices_within_target_radius_i
    num_source_indices_within_target_radius_i
    nearest_source_index_to_target_index_i
    source field: 2D field
    target_grid_shape : shape of target grid array (2D)
    operation : one of ['mean', 'nanmean', 'median', 'nanmedian', 'nearest']

    '''

    source_field_r = source_field.ravel()

    # define array that will contain source_field mapped to target_grid
    source_on_target_grid = np.zeros((target_grid_shape))*np.nan

    # get a 1D version of source_on_target_grid
    tmp_r = source_on_target_grid.ravel()

    # loop through every target grid point
    for i in range(len(tmp_r)):

        # if the number of source_field points at target grid cell i > 0
        if num_source_indices_within_target_radius_i[i] > 0:

            # average these values
            if operation == 'mean':
                tmp_r[i] = \
                    np.mean(source_field_r[source_indices_within_target_radius_i[i]])

            # average of non-nan values (can be slow)
            elif operation == 'nanmean':
                tmp_r[i] = \
                    np.nanmean(source_field_r[source_indices_within_target_radius_i[i]])

            # median of these values
            elif operation == 'median':
                tmp_r[i] = \
                    np.median(source_field_r[source_indices_within_target_radius_i[i]])

            # median of non-nan values (can be slow)
            elif operation == 'nanmedian':
                tmp_r[i] = \
                    np.nanmedian(source_field_r[source_indices_within_target_radius_i[i]])

            # nearest neighbor is the first element in source_indices
            elif operation == 'nearest':
                tmp = source_indices_within_target_radius_i[i]
                tmp_r[i] = source_field_r[tmp[0]]

        # number source indices within target radius is 0, then we can potentially
        # search for a nearest neighbor.
        elif allow_nearest_neighbor:
          # there is a nearest neighbor within range
            if i in nearest_source_index_to_target_index_i.keys():
                tmp_r[i] = source_field_r[nearest_source_index_to_target_index_i[i]]

    return source_on_target_grid


def find_mappings_from_source_to_target(source_grid, target_grid, target_grid_radius,
                                        source_grid_min_L, source_grid_max_L,
                                        neighbours: int = 100, less_output=True):
    '''
    source grid, target_grid : area or grid defintion objects from pyresample

    target_grid_radius       : a vector indicating the radius of each
                               target grid cell (m)

    source_grid_min_l, source_grid_max_L : min and max distances
                               between adjacent source grid cells (m)

    neighbours     : Specifies number of neighbours to look for when getting
                     the neighbour info of a cell using pyresample.
                     Default is 100 to limit memory usage.
                     Value given must be a whole number greater than 0
    '''

    # # of element of the source and target grids
    len_source_grid = source_grid.size
    len_target_grid = target_grid.size

    # the maximum radius of the target grid
    max_target_grid_radius = np.nanmax(target_grid_radius)

    # the maximum number of neighbors to consider when doing the bin averaging
    # assuming that we have the largets target grid radius and the smallest
    # source grid length. (upper bound)

    # the ceiling is used to ensure the result is a whole number > 0
    neighbours_upper_bound = int((max_target_grid_radius*2/source_grid_min_L)**2)

    # compare provided and upper_bound value for neighbours.
    # limit neighbours to the upper_bound if the supplied neighbours value is larger
    # since you dont need more neighbours than exists within a cell.
    if neighbours > neighbours_upper_bound:
        print('using more neighbours than upper bound.  limiting to the upper bound '
              f'of {int(neighbours_upper_bound)} neighbours')
        neighbours = neighbours_upper_bound
    else:
        print(f'Only using {neighbours} nearest neighbours, but you may need up to {neighbours_upper_bound}')

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

    Ax_max_target_grid_r = \
        pr.kd_tree.get_neighbour_info(source_grid,
                                      target_grid,
                                      radius_of_influence=int(max_target_grid_radius),
                                      neighbours=neighbours)

    # define a dictionary, which will contain the list of SOURCE grid cells
    # that are within the search radius of each TARGET grid cell
    source_indices_within_target_radius_i = dict()

    # define a vector which is a COUNT of the # of SOURCE grid cells
    # that are within the search radius of each TARGET grid cell
    num_source_indices_within_target_radius_i =\
        np.zeros((target_grid_radius.shape))

    # SECOND FIND THE SINGLE SOURCE GRID CELL THAT IS CLOSEST TO EACH
    # TARGET GRID CELL, BUT ONLY SEARCH AS FAR AS SOURCE_GRID_MAX_L

    # the kd_tree can also find the sigle nearest neighbor within the
    # radius 'source_grid_max_L'.  This second search is needed because sometimes
    # the source grid is finer than the target grid and therefore we may
    # end up in a situation where none of the centers of the SOURCE grid
    # fall within the small centers of the TARGET grid.
    # we'll look for the nearest SOURCE grid cell within 'source_grid_max_L'

    Ax_nearest_within_source_grid_max_L = \
        pr.kd_tree.get_neighbour_info(source_grid, target_grid,
                                      radius_of_influence=int(source_grid_max_L),
                                      neighbours=1)

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
        print('length of target grid: ', len_target_grid)

    for i in range(len_target_grid):

        if not less_output:
            print('looping through all points of target grid: ', i)

        if Ax_nearest_within_source_grid_max_L[1][i] == True:

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

            source_indices_within_target_radius_i[i] = \
                src_indicies_here[dist_within_target_r == True]

            # count the # source indices here
            num_source_indices_within_target_radius_i[i] = \
                int(len(source_indices_within_target_radius_i[i]))

            # NOW RECORD THE NEAREST NEIGHBOR POINT WIHTIN SOURCE_GRID_MAX_L
            # when there is no source index within the search radius then
            # the 'get neighbour info' routine returns a dummy value of
            # the length of the source grid.  so we test to see if that's the
            # value that was returned.  If not, then we are good to go.
            # In other words...if the index in Ax_nearest... is the length of source grid, it is invalid
            # We think this should always because of the initial test for Ax[1] == True
            if Ax_nearest_within_source_grid_max_L[2][current_valid_target_i] < len_source_grid:
                nearest_source_index_to_target_index_i[i] = Ax_nearest_within_source_grid_max_L[2][current_valid_target_i]

            # increment this little bastard
            current_valid_target_i += 1

        # print progress.  always nice
        if i in debug_is:
            print(str(int(i/len_target_grid*100)) + ' %')

    return source_indices_within_target_radius_i,\
        num_source_indices_within_target_radius_i,\
        nearest_source_index_to_target_index_i


def generalized_grid_product(data_res: float, data_max_lat: float, area_extent: List[float], dims: List[float], proj_info: dict) -> Tuple[np.ndarray]:
    '''
    Generates tuple containing (source_grid_min_L, source_grid_max_L, source_grid)

    data_res: in degrees
    area_extent: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    dims: resolution of source grid
    proj_info: projection information
    return (source_grid_min_L, source_grid_max_L, source_grid)
    '''

    # minimum Length of data product grid cells (km)
    source_grid_min_L = np.cos(np.deg2rad(data_max_lat))*data_res*112e3

    # maximum length of data roduct grid cells (km)
    # data product at equator has grid spacing of data_res*112e3 m
    source_grid_max_L = data_res*112e3

    #area_extent: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    areaExtent = (area_extent[0], area_extent[1],
                  area_extent[2], area_extent[3])

    # Corressponds to resolution of grid from data
    cols = dims[0]
    rows = dims[1]

    # USE PYRESAMPLE TO GENERATE THE LAT/LON GRIDS
    # -- note we do not have to use pyresample for this, we could
    # have created it manually using the np.meshgrid or some other method
    # if we wanted.
    tmp_data_grid = pr.area_config.get_area_def(proj_info['area_id'], proj_info['area_name'],
                                                proj_info['proj_id'], proj_info['proj4_args'],
                                                cols, rows, areaExtent)

    data_grid_lons, data_grid_lats = tmp_data_grid.get_lonlats()

    # Changes longitude bounds from 0-360 to -180-180, doesnt change if its already -180-180
    data_grid_lons, data_grid_lats = pr.utils.check_and_wrap(data_grid_lons,
                                                             data_grid_lats)

    # Define the 'swath' (in the terminology of the pyresample module)
    # as the lats/lon pairs of the source observation grid
    # The routine needs the lats and lons to be one-dimensional vectors.
    source_grid = pr.geometry.SwathDefinition(lons=data_grid_lons.ravel(),
                                              lats=data_grid_lats.ravel())

    return (source_grid_min_L, source_grid_max_L, source_grid)


def perform_mapping(T: Transformation, ds: xr.Dataset, factors: Tuple, data_field_info: Mapping, model_grid: xr.Dataset) -> xr.DataArray:
    '''
    Maps source data to target grid and applies metadata
    '''

    # initialize notes for this record
    record_notes = ''

    # set data info values
    data_field = data_field_info['name']

    # create empty data array
    data_DA = records.make_empty_record(T.date, model_grid, T.array_precision)

    # print(data_DA)

    # add some metadata to the newly formed data array object
    data_DA.attrs['long_name'] = data_field_info['long_name']
    data_DA.attrs['standard_name'] = data_field_info['standard_name']
    data_DA.attrs['units'] = data_field_info['units']
    data_DA.attrs['original_filename'] = T.file_name
    data_DA.attrs['original_field_name'] = data_field
    data_DA.attrs['interpolation_parameters'] = 'bin averaging'
    data_DA.attrs['interpolation_code'] = 'pyresample'
    data_DA.attrs['interpolation_date'] = str(np.datetime64(datetime.now(), 'D'))

    data_DA.time.attrs['long_name'] = 'center time of averaging period'

    data_DA.name = f'{data_field}_interpolated_to_{model_grid.name}'

    if T.transpose:
        orig_data = ds[data_field].values[0, :].T
    else:
        orig_data = ds[data_field].values

    # see if we have any valid data
    if np.sum(~np.isnan(orig_data)) > 0:
        data_model_projection = transform_to_target_grid(*factors, orig_data, model_grid.XC.shape,
                                                         operation=T.mapping_operation)

        # put the new data values into the data_DA array.
        # --where the mapped data are not nan, replace the original values
        # --where they are nan, just leave the original values alone
        data_DA.values = np.where(~np.isnan(data_model_projection), data_model_projection, data_DA.values)
    else:
        print(f' - CPU id {os.getpid()} empty granule for {T.file_name} (no data to transform to grid {model_grid.name})')
        record_notes = ' -- empty record -- '

    if T.time_bounds_var:
        if T.time_bounds_var in ds:
            time_start = str(ds[T.time_bounds_var].values.ravel()[0])
            time_end = str(ds[T.time_bounds_var].values.ravel()[0])
        else:
            logging.info(f'time_bounds_var {T.time_bounds_var} does not exist in file but is defined in config. \
                Using other method for obtaining start/end times.')

    else:
        time_start = T.date
        if T.data_time_scale.upper() == 'MONTHLY':
            month = str(np.datetime64(T.date, 'M') + 1)
            time_end = str(np.datetime64(month, 'ns'))
        elif T.data_time_scale.upper() == 'DAILY':
            time_end = str(np.datetime64(T.date, 'D') + np.timedelta64(1, 'D'))

    if '-' not in time_start:
        time_start = f'{time_start[0:4]}-{time_start[4:6]}-{time_start[6:8]}'
        time_end = f'{time_end[0:4]}-{time_end[4:6]}-{time_end[6:8]}'

    data_DA.time_start.values[0] = time_start.replace('Z', '')
    data_DA.time_end.values[0] = time_end.replace('Z', '')

    if 'time' in ds:
        data_DA.time.values[0] = ds['time'].values.ravel()[0]
    elif 'Time' in ds:
        data_DA.time.values[0] = ds['Time'].values.ravel()[0]
    else:
        data_DA.time.values[0] = T.date

    data_DA.attrs['notes'] = record_notes
    data_DA.attrs['original_time'] = str(data_DA.time.values[0])
    data_DA.attrs['original_time_start'] = str(data_DA.time_start.values[0])
    data_DA.attrs['original_time_end'] = str(data_DA.time_end.values[0])

    return data_DA


def monthly_aggregation(ds, var, year: str, A, uuid):
    attrs = ds.attrs
    mon_DS_year = []
    for month in range(1, 13):
        # to find the last day of the month, we go up one month and back one day
        # if Jan-Nov, then we'll go forward one month to Feb-Dec
        # for december we go up one year, and set month to january
        if month < 12:
            cur_mon_year = np.datetime64(f'{year}-{str(month+1).zfill(2)}-01', 'ns')
        else:
            cur_mon_year = np.datetime64(f'{int(year)+1}-01-01', 'ns')

        mon_str = str(year) + '-' + str(month).zfill(2)
        cur_mon = ds[var].sel(time=mon_str)

        if A.remove_nan_days_from_data:
            nonnan_days = []
            for i in range(len(cur_mon)):
                if(np.count_nonzero(~np.isnan(cur_mon[i].values)) > 0):
                    nonnan_days.append(cur_mon[i])
            if nonnan_days:
                cur_mon = xr.concat((nonnan_days), dim='time')

        # Compute monthly mean
        mon_DA = cur_mon.mean(axis=0, skipna=A.skipna_in_mean, keep_attrs=True)

        tb, ct = date_time.make_time_bounds_from_ds64(cur_mon_year, 'AVG_MON')

        mon_DA = mon_DA.assign_coords({'time': ct})
        mon_DA = mon_DA.expand_dims('time', axis=0)

        avg_center_time = mon_DA.time.copy(deep=True)
        avg_center_time.values[0] = ct

        # halfway through the approx 1M averaging period.
        mon_DA.time.values[0] = ct
        mon_DA.time.attrs['long_name'] = 'center time of 1M averaging period'

        mon_DS = mon_DA.to_dataset()

        mon_DS = mon_DS.assign_coords({'time_bnds': (('time', 'nv'), [tb])})
        mon_DS.time.attrs.update(bounds='time_bnds')

        mon_DS_year.append(mon_DS)

    mon_DS_year_merged = xr.concat((mon_DS_year), dim='time', combine_attrs='no_conflicts')

    attrs['time_coverage_duration'] = 'P1M'
    attrs['time_coverage_resolution'] = 'P1M'

    attrs['valid_min'] = np.nanmin(mon_DS_year_merged[var].values)
    attrs['valid_max'] = np.nanmax(mon_DS_year_merged[var].values)
    attrs['uuid'] = uuid

    mon_DS_year_merged.attrs = attrs

    return mon_DS_year_merged


def generalized_aggregate_and_save(DS_year_merged, data_var, do_monthly_aggregation,
                                   year, skipna_in_mean, filenames, fill_values,
                                   output_dirs, binary_dtype, model_grid_type,
                                   save_binary=True, save_netcdf=True,
                                   remove_nan_days_from_data=False, data_time_scale='DAILY',
                                   uuids=[]):
    '''
    deprecated in favor of monthly_aggregation()
    '''

    # # if everything comes back nans it means there were no files
    # # to load for the entire year.  don't bother saving the
    # # netcdf or binary files for this year
    # if np.sum(~np.isnan(DS_year_merged[data_var].values)) == 0:
    #     print('Empty year not writing to disk', year)
    #     return True

    global_attrs = DS_year_merged.attrs

    DS_year_merged.attrs['uuid'] = uuids[0]

    DS_year_merged.attrs['time_coverage_duration'] = 'P1Y'
    DS_year_merged.attrs['time_coverage_start'] = str(DS_year_merged.time_bnds.values[0][0])[0:19]
    DS_year_merged.attrs['time_coverage_end'] = str(DS_year_merged.time_bnds.values[-1][-1])[0:19]

    if data_time_scale.upper() == 'DAILY':
        DS_year_merged.attrs['time_coverage_resolution'] = 'P1D'
    elif data_time_scale.upper() == 'MONTHLY':
        DS_year_merged.attrs['time_coverage_resolution'] = 'P1M'

    if do_monthly_aggregation:
        mon_DS_year = []
        for month in range(1, 13):
            # to find the last day of the month, we go up one month and back one day
            # if Jan-Nov, then we'll go forward one month to Feb-Dec
            # for december we go up one year, and set month to january
            if month < 12:
                cur_mon_year = np.datetime64(f'{str(year)}-{str(month+1).zfill(2)}-01', 'ns')
            else:
                cur_mon_year = np.datetime64(f'{str(year+1)}-01-01', 'ns')

            mon_str = str(year) + '-' + str(month).zfill(2)
            cur_mon = DS_year_merged[data_var].sel(time=mon_str)

            if remove_nan_days_from_data:
                nonnan_days = []
                for i in range(len(cur_mon)):
                    if(np.count_nonzero(~np.isnan(cur_mon[i].values)) > 0):
                        nonnan_days.append(cur_mon[i])
                if nonnan_days:
                    cur_mon = xr.concat((nonnan_days), dim='time')

            # Compute monthly mean
            mon_DA = cur_mon.mean(axis=0, skipna=skipna_in_mean, keep_attrs=True)

            tb, ct = date_time.make_time_bounds_from_ds64(cur_mon_year, 'AVG_MON')

            mon_DA = mon_DA.assign_coords({'time': ct})
            mon_DA = mon_DA.expand_dims('time', axis=0)

            avg_center_time = mon_DA.time.copy(deep=True)
            avg_center_time.values[0] = ct

            # halfway through the approx 1M averaging period.
            mon_DA.time.values[0] = ct
            mon_DA.time.attrs['long_name'] = 'center time of 1M averaging period'

            mon_DS = mon_DA.to_dataset()

            mon_DS = mon_DS.assign_coords({'time_bnds': (('time', 'nv'), [tb])})
            mon_DS.time.attrs.update(bounds='time_bnds')

            mon_DS_year.append(mon_DS)

        mon_DS_year_merged = xr.concat((mon_DS_year), dim='time', combine_attrs='no_conflicts')

        # start_time = mon_DS_year_merged.time.values.min()
        # end_time = mon_DS_year_merged.time.values.max()

        # time_bnds = np.array([start_time, end_time], dtype='datetime64')
        # time_bnds = time_bnds.T
        # mon_DS_year_merged = mon_DS_year_merged.assign_coords(
        #     {'time_bnds': (('time','nv'), [time_bnds])})
        # mon_DS_year_merged.time.attrs.update(bounds='time_bnds')

        global_attrs['time_coverage_duration'] = 'P1M'
        global_attrs['time_coverage_resolution'] = 'P1M'

        global_attrs['valid_min'] = np.nanmin(mon_DS_year_merged[data_var].values)
        global_attrs['valid_max'] = np.nanmax(mon_DS_year_merged[data_var].values)
        global_attrs['uuid'] = uuids[1]

        mon_DS_year_merged.attrs = global_attrs

    DS_year_merged[data_var] = DS_year_merged[data_var].fillna(fill_values['netcdf'])

    if save_binary:
        records.save_binary(DS_year_merged, filenames['shortest'], fill_values['binary'],
                            output_dirs['binary'], binary_dtype, model_grid_type, data_var)
    if save_netcdf:
        records.save_netcdf(DS_year_merged, filenames['shortest'], fill_values['netcdf'],
                            output_dirs['netcdf'])

    if do_monthly_aggregation:
        mon_DS_year_merged[data_var] = mon_DS_year_merged[data_var].fill_na(fill_values['netcdf'])
        if save_binary:
            records.save_binary(mon_DS_year_merged, filenames['monthly'], fill_values['binary'],
                                output_dirs['binary'], binary_dtype, model_grid_type, data_var)
        if save_netcdf:
            records.save_netcdf(mon_DS_year_merged, filenames['monthly'], fill_values['netcdf'],
                                output_dirs['netcdf'])

    return False

# Preprocessing functions
# -----------------------------------------------------------------------------------------------------------------------------------------------

def ATL20_V004_monthly(file_path, config):
    vars = [field['name'] for field in config['fields']]

    ds = xr.open_dataset(file_path, decode_times=True)
    ds = ds[['grid_x', 'grid_y', 'crs']]
    
    var_ds = xr.open_dataset(file_path, group='monthly')[vars]
    merged_ds = xr.merge([ds, var_ds])
    return merged_ds


# Pre-transformation (on Datasets only)
# -----------------------------------------------------------------------------------------------------------------------------------------------


def RDEFT4_remove_negative_values(ds):
    '''
    Replaces negative values with nans for all data vars
    '''
    for field in ds.data_vars:
        if field in ['lat', 'lon']:
            continue
        ds[field].values = np.where(ds[field].values < 0, np.nan, ds[field].values)
    return ds


def G2202_mask_flagged_conc(ds):
    '''
    Masks out values greater than 1 in nsidc_nt_seaice_conc and cdr_seaice_conc
    '''
    logging.debug(f'G2202 masking flagged nt pre   : {np.sum(ds["nsidc_nt_seaice_conc"].values.ravel() > 1)}')
    tmpNT = np.where(ds["nsidc_nt_seaice_conc"].values.ravel() > 1, 1, 0)
    tmpCDR = np.where(ds["cdr_seaice_conc"].values.ravel() > 1, 1, 0)
    logging.debug(f'G2202 masking flagged NDR, CDR pre: {np.sum(tmpNT), np.sum(tmpCDR)}')

    ds['nsidc_nt_seaice_conc'] = ds['nsidc_nt_seaice_conc'].where(ds['nsidc_nt_seaice_conc'] <= 1)
    ds['cdr_seaice_conc'] = ds['cdr_seaice_conc'].where(ds['cdr_seaice_conc'] <= 1)

    # nan all spatial interpolation (removes  pole hole)
    ds['nsidc_nt_seaice_conc'] = ds['nsidc_nt_seaice_conc'].where(np.isnan(ds['spatial_interpolation_flag'].values))
    ds['cdr_seaice_conc'] = ds['cdr_seaice_conc'].where(np.isnan(ds['spatial_interpolation_flag'].values))

    tmpNT = np.where(ds["nsidc_nt_seaice_conc"].values.ravel() > 1, 1, 0)
    tmpCDR = np.where(ds["cdr_seaice_conc"].values.ravel() > 1, 1, 0)
    logging.debug(f'G2202 masking flagged NDR, CDR post: {np.sum(tmpNT), np.sum(tmpCDR)}')

    return ds


# Post-transformations (on DataArrays only)
# -----------------------------------------------------------------------------------------------------------------------------------------------


def kelvin_to_celsius(da):
    '''
    Converts Kelvin values to Celsius
    '''
    da.attrs['units'] = 'Celsius'
    da.values -= 273.15
    return da


def seaice_concentration_to_fraction(da):
    '''
    Converts seaice concentration values to a fraction by dividing them by 100
    '''
    da.attrs['units'] = "1"
    da.values /= 100.
    return da


def MEaSUREs_fix_time(da):
    '''
    time_start and time_end for MEaSUREs_1812 is not acceptable
    this function takes the provided center time, removes the hours:minutes:seconds.ns
    and sets the new time_start and time_end based on that new time
    '''
    cur_time = da.time.values

    # remove time from date
    cur_day = str(cur_time[0])[:10]

    new_start = str(np.datetime64(cur_day, 'ns'))

    # new end is the start date plus 1 day
    new_end = str(np.datetime64(str(np.datetime64(cur_day, 'D') + 1), 'ns'))

    da.time_start.values[0] = new_start
    da.time_end.values[0] = new_end

    return da
