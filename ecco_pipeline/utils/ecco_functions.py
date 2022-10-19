from typing import List, Tuple
import numpy as np
import pyresample as pr
import xarray as xr
from datetime import datetime
import os
from utils import records, mapping, date_time

# Generalized functions
# -----------------------------------------------------------------------------------------------------------------------------------------------

def generalized_grid_product(product_name: str, data_res: float, data_max_lat: float,
                             area_extent: List[float], dims: List[float], proj_info:str) -> Tuple[np.ndarray]:
    '''
    data_res: in degrees
    return (source_grid_min_L, source_grid_max_L, source_grid, data_grid_lons, data_grid_lats)
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

    return (source_grid_min_L, source_grid_max_L, source_grid, data_grid_lons, data_grid_lats)


def generalized_transform_to_model_grid_solr(data_field_info, record_date, model_grid,
                                             model_grid_type, array_precision,
                                             record_file_name, data_time_scale,
                                             extra_information, ds, factors,
                                             time_zone_included_with_time,
                                             model_grid_name):
    # initialize notes for this record
    record_notes = ''

    source_indices_within_target_radius_i, \
        num_source_indices_within_target_radius_i, \
        nearest_source_index_to_target_index_i = factors

    # set data info values
    data_field = data_field_info['name_s']
    standard_name = data_field_info['standard_name_s']
    long_name = data_field_info['long_name_s']
    units = data_field_info['units_s']

    # create empty data array
    data_DA = records.make_empty_record(standard_name, long_name, units,
                                   record_date,
                                   model_grid, model_grid_type,
                                   array_precision)

    # print(data_DA)

    # add some metadata to the newly formed data array object
    data_DA.attrs['original_filename'] = record_file_name
    data_DA.attrs['original_field_name'] = data_field
    data_DA.attrs['interpolation_parameters'] = 'bin averaging'
    data_DA.attrs['interpolation_code'] = 'pyresample'
    data_DA.attrs['interpolation_date'] = \
        str(np.datetime64(datetime.now(), 'D'))

    data_DA.time.attrs['long_name'] = 'center time of averaging period'

    data_DA.name = f'{data_field}_interpolated_to_{model_grid_name}'

    if 'transpose' in extra_information:
        orig_data = ds[data_field].values[0, :].T
    else:
        orig_data = ds[data_field].values

    # see if we have any valid data
    if np.sum(~np.isnan(orig_data)) > 0:

        data_model_projection = mapping.transform_to_target_grid(source_indices_within_target_radius_i,
                                                            num_source_indices_within_target_radius_i,
                                                            nearest_source_index_to_target_index_i,
                                                            orig_data, model_grid.XC.shape)

        # put the new data values into the data_DA array.
        # --where the mapped data are not nan, replace the original values
        # --where they are nan, just leave the original values alone
        data_DA.values = np.where(~np.isnan(data_model_projection),
                                  data_model_projection, data_DA.values)

    else:
        print(
            f' - CPU id {os.getpid()} empty granule for {record_file_name} (no data to transform to grid {model_grid_name})')
        record_notes = record_notes + ' -- empty record -- '

    # update time values
    if 'time_bounds_var' in extra_information:
        if 'Time_bounds' in ds.variables:
            data_DA.time_start.values[0] = ds.Time_bounds[0][0].values
            data_DA.time_end.values[0] = ds.Time_bounds[0][1].values
        elif 'time_bnds' in ds.variables:
            data_DA.time_start.values[0] = ds.time_bnds[0][0].values
            data_DA.time_end.values[0] = ds.time_bnds[0][1].values
        elif 'time_bounds' in ds.variables:
            try:
                data_DA.time_start.values[0] = ds.time_bounds[0][0].values
                data_DA.time_end.values[0] = ds.time_bounds[0][1].values
            except:
                data_DA.time_start.values[0] = ds.time_bounds[0].values
                data_DA.time_end.values[0] = ds.time_bounds[1].values
        elif 'timebounds' in ds.variables:
            data_DA.time_start.values[0] = ds.timebounds[0].values
            data_DA.time_end.values[0] = ds.timebounds[1].values

    elif 'no_time' in extra_information:
        # If no_time assume record_date is start date
        # The file may not provide the start date, but it is
        # determined in the harvesting code

        if data_time_scale.upper() == 'MONTHLY':
            month = str(np.datetime64(record_date, 'M') + 1)
            end_time = str(np.datetime64(month, 'ns'))
        elif data_time_scale.upper() == 'DAILY':
            end_time = str(np.datetime64(record_date, 'D') +
                           np.timedelta64(1, 'D'))

        data_DA.time_start.values[0] = record_date
        data_DA.time_end.values[0] = end_time
    elif 'no_time_dashes' in extra_information:
        new_start_time = f'{ds.time_coverage_start[0:4]}-{ds.time_coverage_start[4:6]}-{ds.time_coverage_start[6:8]}'
        new_end_time = f'{ds.time_coverage_end[0:4]}-{ds.time_coverage_end[4:6]}-{ds.time_coverage_end[6:8]}'
        data_DA.time_start.values[0] = new_start_time
        data_DA.time_end.values[0] = new_end_time
    elif time_zone_included_with_time:
        data_DA.time_start.values[0] = ds.time_coverage_start[:-1]
        data_DA.time_end.values[0] = ds.time_coverage_end[:-1]
    else:
        data_DA.time_start.values[0] = ds.time_coverage_start
        data_DA.time_end.values[0] = ds.time_coverage_end

    if 'time_var' in extra_information:
        if 'Time' in ds.variables:
            data_DA.time.values[0] = ds.Time[0].values
        elif 'time' in ds.variables:
            try:
                data_DA.time.values[0] = ds.time[0].values
            except:
                data_DA.time.values[0] = ds.time.values
    else:
        data_DA.time.values[0] = record_date

    data_DA.attrs['notes'] = record_notes

    data_DA.attrs['original_time'] = str(data_DA.time.values[0])
    data_DA.attrs['original_time_start'] = str(data_DA.time_start.values[0])
    data_DA.attrs['original_time_end'] = str(data_DA.time_end.values[0])

    return data_DA


def generalized_aggregate_and_save(DS_year_merged, data_var, do_monthly_aggregation,
                                   year, skipna_in_mean, filenames, fill_values,
                                   output_dirs, binary_dtype, model_grid_type,
                                   on_aws='', save_binary=True, save_netcdf=True,
                                   remove_nan_days_from_data=False, data_time_scale='DAILY',
                                   uuids=[]):

    # if everything comes back nans it means there were no files
    # to load for the entire year.  don't bother saving the
    # netcdf or binary files for this year
    if np.sum(~np.isnan(DS_year_merged[data_var].values)) == 0:
        print('Empty year not writing to disk', year)
        return True
    else:

        global_attrs = DS_year_merged.attrs

        DS_year_merged.attrs['uuid'] = uuids[0]

        if data_time_scale.upper() == 'DAILY':
            DS_year_merged.attrs['time_coverage_duration'] = 'P1Y'
            DS_year_merged.attrs['time_coverage_resolution'] = 'P1D'

            DS_year_merged.attrs['time_coverage_start'] = str(
                DS_year_merged.time_bnds.values[0][0])[0:19]
            DS_year_merged.attrs['time_coverage_end'] = str(
                DS_year_merged.time_bnds.values[-1][-1])[0:19]

        elif data_time_scale.upper() == 'MONTHLY':
            DS_year_merged.attrs['time_coverage_duration'] = 'P1Y'
            DS_year_merged.attrs['time_coverage_resolution'] = 'P1M'

            DS_year_merged.attrs['time_coverage_start'] = str(
                DS_year_merged.time_bnds.values[0][0])[0:19]
            DS_year_merged.attrs['time_coverage_end'] = str(
                DS_year_merged.time_bnds.values[-1][-1])[0:19]

        if do_monthly_aggregation:
            mon_DS_year = []
            for month in range(1, 13):
                # to find the last day of the month, we go up one month,
                # and back one day
                #   if Jan-Nov, then we'll go forward one month to Feb-Dec
                if month < 12:
                    cur_mon_year = np.datetime64(str(year) + '-' +
                                                 str(month+1).zfill(2) +
                                                 '-' + str(1).zfill(2), 'ns')
                    # for december we go up one year, and set month to january
                else:
                    cur_mon_year = np.datetime64(str(year+1) + '-' +
                                                 str('01') +
                                                 '-' + str(1).zfill(2), 'ns')

                mon_str = str(year) + '-' + str(month).zfill(2)
                cur_mon = DS_year_merged[data_var].sel(time=mon_str)

                if remove_nan_days_from_data:
                    nonnan_days = []
                    for i in range(len(cur_mon)):
                        if(np.count_nonzero(~np.isnan(cur_mon[i].values)) > 0):
                            nonnan_days.append(cur_mon[i])
                    if nonnan_days:
                        cur_mon = xr.concat((nonnan_days), dim='time')

                mon_DA = cur_mon.mean(
                    axis=0, skipna=skipna_in_mean, keep_attrs=True)

                tb, ct = date_time.make_time_bounds_from_ds64(cur_mon_year, 'AVG_MON')

                mon_DA = mon_DA.assign_coords({'time': ct})
                mon_DA = mon_DA.expand_dims('time', axis=0)

                avg_center_time = mon_DA.time.copy(deep=True)
                avg_center_time.values[0] = ct

                # halfway through the approx 1M averaging period.
                mon_DA.time.values[0] = ct
                mon_DA.time.attrs['long_name'] = 'center time of 1M averaging period'

                mon_DS = mon_DA.to_dataset()

                mon_DS = mon_DS.assign_coords(
                    {'time_bnds': (('time', 'nv'), [tb])})
                mon_DS.time.attrs.update(bounds='time_bnds')

                mon_DS_year.append(mon_DS)

            mon_DS_year_merged = xr.concat(
                (mon_DS_year), dim='time', combine_attrs='no_conflicts')

            # start_time = mon_DS_year_merged.time.values.min()
            # end_time = mon_DS_year_merged.time.values.max()

            # time_bnds = np.array([start_time, end_time], dtype='datetime64')
            # time_bnds = time_bnds.T
            # mon_DS_year_merged = mon_DS_year_merged.assign_coords(
            #     {'time_bnds': (('time','nv'), [time_bnds])})
            # mon_DS_year_merged.time.attrs.update(bounds='time_bnds')

            global_attrs['time_coverage_duration'] = 'P1M'
            global_attrs['time_coverage_resolution'] = 'P1M'

            global_attrs['valid_min'] = np.nanmin(
                mon_DS_year_merged[data_var].values)
            global_attrs['valid_max'] = np.nanmax(
                mon_DS_year_merged[data_var].values)
            global_attrs['uuid'] = uuids[1]

            mon_DS_year_merged.attrs = global_attrs

        save_netcdf = save_netcdf and not on_aws
        save_binary = save_binary or on_aws

        #######################################################
        ## BEGIN SAVE TO DISK                                ##

        DS_year_merged[data_var].values = \
            np.where(np.isnan(DS_year_merged[data_var].values),
                     fill_values['netcdf'], DS_year_merged[data_var].values)

        records.save_to_disk(DS_year_merged,
                        filenames['shortest'],
                        fill_values['binary'], fill_values['netcdf'],
                        output_dirs['netcdf'], output_dirs['binary'],
                        binary_dtype, model_grid_type, save_binary=save_binary,
                        save_netcdf=save_netcdf, data_var=data_var)

        if do_monthly_aggregation:
            mon_DS_year_merged[data_var].values = \
                np.where(np.isnan(mon_DS_year_merged[data_var].values),
                         fill_values['netcdf'], mon_DS_year_merged[data_var].values)

            records.save_to_disk(mon_DS_year_merged,
                            filenames['monthly'],
                            fill_values['binary'], fill_values['netcdf'],
                            output_dirs['netcdf'], output_dirs['binary'],
                            binary_dtype, model_grid_type, save_binary=save_binary,
                            save_netcdf=save_netcdf, data_var=data_var)
    return False

# Pre-transformation (on Datasets only)
# -----------------------------------------------------------------------------------------------------------------------------------------------


def RDEFT4_remove_negative_values(ds):
    for field in ds.data_vars:
        if field in ['lat', 'lon']:
            continue
        ds[field].values = np.where(
            ds[field].values < 0, np.nan, ds[field].values)
    return ds

# Post-transformations (on DataArrays only)
# -----------------------------------------------------------------------------------------------------------------------------------------------


def avhrr_sst_kelvin_to_celsius(da, field_name):
    if field_name == 'analysed_sst':
        da.attrs['units'] = 'Celsius'
        da.values -= 273.15
    return da


def seaice_concentration_to_fraction(da, field_name):
    if field_name == 'ice_conc':
        da.attrs['units'] = '1'
        da.values /= 100.
    return da

# time_start and time_end for MEaSUREs_1812 is not acceptable
# this function takes the provided center time, removes the hours:minutes:seconds.ns
# and sets the new time_start and time_end based on that new time
def MEaSUREs_fix_time(da, field_name):
    cur_time = da.time.values

    # remove time from date
    cur_day = str(cur_time[0])[:10]

    new_start = str(np.datetime64(cur_day, 'ns'))

    # new end is the start date plus 1 day
    new_end = str(np.datetime64(str(np.datetime64(cur_day, 'D') + 1), 'ns'))

    da.time_start.values[0] = new_start
    da.time_end.values[0] = new_end

    return da