from collections import defaultdict
import itertools
import json
import logging
from typing import List
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from conf.global_settings import OUTPUT_DIR
import netCDF4 as nc4

from utils.ecco_utils import date_time, ecco_functions, records
from utils import solr_utils


class Aggregation():

    def __init__(self, config: dict, grids_to_use: List[str]):
        self.dataset_name = config.get('ds_name')
        self.fields = config.get('fields')
        self.version = str(config.get('a_version', ''))
        self.precision = getattr(np, config.get('array_precision'))
        self.binary_dtype = '>f4' if self.precision == np.float32 else '>f8'
        self.nc_fill_val = nc4.default_fillvals[self.binary_dtype.replace('>', '')]
        self.bin_fill_val = -9999
        self.do_monthly_aggregation = config.get('do_monthly_aggregation', False)
        self.remove_nan_days_from_data = config.get('remove_nan_days_from_data', True)
        self.skipna_in_mean = config.get('skipna_in_mean', False)

        self.save_binary = config.get('save_binary', True)
        self.save_netcdf = config.get('save_netcdf', True)

        self.transformations = defaultdict(list)
        self._set_ds_meta()
        self._set_grids(grids_to_use)
        self._set_years()

    def _set_ds_meta(self):
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:dataset']
        ds_meta = solr_utils.solr_query(fq)[0]
        if 'start_date_dt' not in ds_meta:
            logging.info('No transformed granules to aggregate.')
            raise Exception('No transformed granules to aggregate.')
        self.ds_meta = ds_meta

    def _set_grids(self, grids_to_use):
        fq = ['type_s:grid']
        grids = [grid for grid in solr_utils.solr_query(fq)]
        if grids_to_use:
            grids = [grid for grid in grids if grid['grid_name_s'] in grids_to_use]
        self.grids = grids

    def _set_years(self):
        existing_agg_version = self.ds_meta.get('aggregation_version_s')
        if existing_agg_version != self.version:
            start_year = int(self.ds_meta.get('start_date_dt')[:4])
            end_year = int(self.ds_meta.get('end_date_dt')[:4])
            years = [str(year) for year in range(start_year, end_year + 1)]
            self.years = {grid.get('grid_name_s'): years for grid in self.grids}
        else:
            self.years = self._years_to_aggregate()

    def _years_to_aggregate(self) -> List[str]:
        '''
        We want to see if we need to aggregate this year again
        1. check if aggregation exists - if not add it to years to aggregate
        2. if it does - compare processing times:
        - if aggregation time is later than all transform times for that year
            no need to aggregate
        - if at least one transformation occured after agg time, year needs to
            be aggregated
        '''
        years = {}
        for grid in self.grids:
            grid_name = grid.get('grid_name_s')
            grid_years = []

            fq = [f'dataset_s:{self.dataset_name}',
                  'type_s:transformation', f'grid_name_s:{grid_name}']
            r = solr_utils.solr_query(fq)
            transformation_years = list(set([t['date_s'][:4] for t in r]))
            transformation_years.sort()
            transformation_docs = r

            # Years with transformations that exist for this dataset and this grid
            for year in transformation_years:
                fq = [f'dataset_s:{self.dataset_name}', 'type_s:aggregation',
                      f'grid_name_s:{grid_name}', f'year_s:{year}']
                r = solr_utils.solr_query(fq)

                if r:
                    agg_time = r[0]['aggregation_time_dt']
                    for t in transformation_docs:
                        if t['date_s'][:4] != year:
                            continue
                        if t['transformation_completed_dt'] > agg_time:
                            grid_years.append(year)
                            break
                else:
                    grid_years.append(year)
            years[grid_name] = grid_years
        return years

    def get_data_by_date(self, date, grid_name, field_name):
        # Query for date
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:transformation',
              f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'date_s:{date}*']

        docs = solr_utils.solr_query(fq)

        # If first of month is not found, query with 7 day tolerance only for monthly data
        if not docs and self.ds_meta.get('data_time_scale_s') == 'monthly':
            tolerance = int(self.ds_meta.get('monthly_tolerance', 8))
            start_month_date = datetime.strptime(date, '%Y-%m-%d')
            tolerance_days = []

            for i in range(1, tolerance):
                tolerance_days.append(datetime.strftime(start_month_date + timedelta(days=i), '%Y-%m-%d'))
                tolerance_days.append(datetime.strftime(start_month_date - timedelta(days=i), '%Y-%m-%d'))

            for tol_date in tolerance_days:
                fq = [f'dataset_s:{self.dataset_name}', 'type_s:transformation',
                      f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'date_s:{tol_date}*']
                docs = solr_utils.solr_query(fq)

                if docs:
                    return docs
        return docs

    def open_datasets(self, docs, field_name):
        # If transformed file is present for date, grid, and field combination
        # open the file, otherwise make empty record
        opened_datasets = []
        for doc in docs:
            data_DS = xr.open_dataset(doc['transformation_file_path_s'], decode_times=True)

            # get name of data variable in the dataset
            # to be used when accessing the values of the transformed data
            # since the transformed files only have one variable, we index at zero to get it
            # type is str
            data_var = list(data_DS.keys())[0]

            opened_datasets.append((data_DS, data_var))

            # Update JSON transformations list
            fq = [f'dataset_s:{self.dataset_name}', 'type_s:granule',
                  f'pre_transformation_file_path_s:"{doc["pre_transformation_file_path_s"]}"']
            harvested_metadata = solr_utils.solr_query(fq)

            transformation_metadata = doc
            transformation_metadata['harvested'] = harvested_metadata
            self.transformations[field_name].append(transformation_metadata)
        return opened_datasets

    def process_data_by_date(self, docs, field, grid, model_grid_ds, date):
        field_name = field.get('name')
        data_time_scale = self.ds_meta.get('data_time_scale_s')
        if docs:
            opened_datasets = self.open_datasets(docs, field_name)
        else:
            opened_datasets = []

        if len(opened_datasets) == 2:
            first_DS = opened_datasets[0][0]
            first_DS_name = opened_datasets[0][1]
            second_DS = opened_datasets[1][0]
            second_DS_name = opened_datasets[1][1]
            if ~np.isnan(first_DS[first_DS_name].values).all():
                data_var = opened_datasets[0][1]
                data_DS = first_DS.copy()
                data_DS[first_DS_name].values = np.where(np.isnan(data_DS[first_DS_name].values),
                                                         second_DS[second_DS_name].values, data_DS[first_DS_name].values)
                data_var = first_DS_name
            else:
                data_var = opened_datasets[1][1]
                data_DS = second_DS.copy()
                data_DS[second_DS_name].values = np.where(np.isnan(data_DS[second_DS_name].values),
                                                          first_DS[first_DS_name].values, data_DS[second_DS_name].values)
                data_var = second_DS_name
        elif len(opened_datasets) == 1:
            data_DS = opened_datasets[0][0]
            data_var = opened_datasets[0][1]
        else:
            data_var = f'{field_name}_interpolated_to_{grid.get("grid_name_s")}'
            data_DA = records.make_empty_record(date, model_grid_ds, self.precision)
            data_DA.attrs['long_name'] = field['long_name']
            data_DA.attrs['standard_name'] = field['standard_name']
            data_DA.attrs['units'] = field['units']
            data_DA.name = data_var

            empty_record_attrs = data_DA.attrs
            empty_record_attrs['original_field_name'] = field_name
            empty_record_attrs['interpolation_date'] = str(np.datetime64(datetime.now(), 'D'))
            data_DA.attrs = empty_record_attrs

            data_DS = data_DA.to_dataset()

            # add time_bnds coordinate
            # [start_time, end_time] dimensions
            # MONTHLY cannot use timedelta64 since it has a variable
            # number of ns/s/d. DAILY can so we use it.
            if data_time_scale.upper() == 'MONTHLY':
                end_time = str(data_DS.time_end.values[0])
                month = str(np.datetime64(end_time, 'M') + 1)
                end_time = [str(np.datetime64(month, 'ns'))]
            elif data_time_scale.upper() == 'DAILY':
                end_time = data_DS.time_end.values + np.timedelta64(1, 'D')

            _, ct = date_time.make_time_bounds_from_ds64(np.datetime64(end_time[0], 'ns'), 'AVG_MON')
            data_DS.time.values[0] = ct

            start_time = data_DS.time_start.values

            time_bnds = np.array([start_time, end_time], dtype='datetime64')
            time_bnds = time_bnds.T

            data_DS = data_DS.assign_coords({'time_bnds': (['time', 'nv'], time_bnds)})

            data_DS.time.attrs.update(bounds='time_bnds')

            data_DS = data_DS.drop('time_start')
            data_DS = data_DS.drop('time_end')
        return data_DS


def check_nan_da(da: xr.DataArray) -> bool:
    '''
    Check if da consists entirely of nans
    '''
    if np.sum(~np.isnan(da.values)) == 0:
        return True
    return False


def aggregation(config, grids_to_use=[]):
    """
    Aggregates data into annual files, saves them, and updates Solr
    """
    try:
        A = Aggregation(config, grids_to_use)
    except:
        return 'No aggregations performed'

    data_time_scale = A.ds_meta.get('data_time_scale_s')

    update_body = []

    aggregation_successes = True

    # =====================================================
    # Loop through grids
    # =====================================================
    for grid in A.grids:

        grid_path = grid['grid_path_s']
        grid_name = grid['grid_name_s']
        grid_type = grid['grid_type_s']

        years = A.years.get(grid_name)

        model_grid_ds = xr.open_dataset(grid_path, decode_times=True)

        for year, field in list(itertools.product(years, A.fields)):
            # Construct list of dates corresponding to data time scale
            if data_time_scale == 'daily':
                dates_in_year = np.arange(f'{year}-01-01', f'{int(year)+1}-01-01', dtype='datetime64[D]')
            elif data_time_scale == 'monthly':
                dates_in_year = np.arange(f'{year}-01', f'{int(year)+1}-01', dtype='datetime64[M]')
                dates_in_year = [f'{date}-01' for date in dates_in_year]

            field_name = field.get('name')

            logging.info(f'Aggregating {str(year)}_{grid_name}_{field_name}')

            daily_DS_year = []

            for date in dates_in_year:
                docs = A.get_data_by_date(date, grid_name, field_name)

                data_DS = A.process_data_by_date(docs, field, grid, model_grid_ds, date)

                # Append each day's data to annual list
                daily_DS_year.append(data_DS)

            # Concatenate all data files within annual list
            daily_annual_ds = xr.concat((daily_DS_year), dim='time')
            data_var = list(daily_annual_ds.keys())[0]

            daily_annual_ds.attrs['aggregation_version'] = config['a_version']
            daily_annual_ds[data_var].attrs['valid_min'] = np.nanmin(daily_annual_ds[data_var].values)
            daily_annual_ds[data_var].attrs['valid_max'] = np.nanmax(daily_annual_ds[data_var].values)

            remove_keys = []
            for (key, _) in daily_annual_ds[data_var].attrs.items():
                if ('original' in key and key != 'original_field_name'):
                    remove_keys.append(key)

            for key in remove_keys:
                del daily_annual_ds[data_var].attrs[key]

            # Create filenames based on date time scale
            # If data time scale is monthly, shortest_filename is monthly
            shortest_filename = f'{A.dataset_name}_{grid_name}_{data_time_scale.upper()}_{field_name}_{year}'
            monthly_filename = f'{A.dataset_name}_{grid_name}_MONTHLY_{field_name}_{year}'

            output_path = f'{OUTPUT_DIR}/{A.dataset_name}/transformed_products/{grid_name}/aggregated/{field_name}/'

            bin_output_dir = Path(output_path) / 'bin'
            bin_output_dir.mkdir(parents=True, exist_ok=True)

            netCDF_output_dir = Path(output_path) / 'netCDF'
            netCDF_output_dir.mkdir(parents=True, exist_ok=True)

            uuids = [str(uuid.uuid1()), str(uuid.uuid1())]

            success = True
            empty_year = False

            # Save
            if check_nan_da(daily_annual_ds[data_var]):
                empty_year = True
            else:
                if A.do_monthly_aggregation:
                    logging.info(f'Aggregating monthly {str(year)}_{grid_name}_{field_name}')
                    try:
                        mon_DS_year_merged = ecco_functions.monthly_aggregation(
                            daily_annual_ds, data_var, year, A, uuids[1])
                        mon_DS_year_merged[data_var] = mon_DS_year_merged[data_var].fillna(A.nc_fill_val)

                        if A.save_binary:
                            records.save_binary(mon_DS_year_merged, monthly_filename, A.bin_fill_val,
                                                bin_output_dir, A.binary_dtype, grid_type, data_var)
                        if A.save_netcdf:
                            records.save_netcdf(mon_DS_year_merged, monthly_filename, A.nc_fill_val, netCDF_output_dir)

                    except Exception as e:
                        logging.exception(f'Error aggregating {A.dataset_name}. {e}')
                        empty_year = True
                        success = False

                daily_annual_ds.attrs['uuid'] = uuids[0]
                daily_annual_ds.attrs['time_coverage_duration'] = 'P1Y'
                daily_annual_ds.attrs['time_coverage_start'] = str(daily_annual_ds.time_bnds.values[0][0])[0:19]
                daily_annual_ds.attrs['time_coverage_end'] = str(daily_annual_ds.time_bnds.values[-1][-1])[0:19]
                if data_time_scale.upper() == 'DAILY':
                    daily_annual_ds.attrs['time_coverage_resolution'] = 'P1D'
                elif data_time_scale.upper() == 'MONTHLY':
                    daily_annual_ds.attrs['time_coverage_resolution'] = 'P1M'

                daily_annual_ds[data_var] = daily_annual_ds[data_var].fillna(A.nc_fill_val)

                if A.save_binary:
                    records.save_binary(daily_annual_ds, shortest_filename, A.bin_fill_val,
                                        bin_output_dir, A.binary_dtype, grid_type, data_var)
                if A.save_netcdf:
                    records.save_netcdf(daily_annual_ds, shortest_filename, A.nc_fill_val,
                                        netCDF_output_dir)

            if empty_year:
                solr_output_filepaths = {'daily_bin': '',
                                         'daily_netCDF': '',
                                         'monthly_bin': '',
                                         'monthly_netCDF': ''}
            else:
                solr_output_filepaths = {'daily_bin': f'{output_path}bin/{shortest_filename}',
                                         'daily_netCDF': f'{output_path}netCDF/{shortest_filename}.nc',
                                         'monthly_bin': f'{output_path}bin/{monthly_filename}',
                                         'monthly_netCDF': f'{output_path}netCDF/{monthly_filename}.nc'}

            aggregation_successes = aggregation_successes and success
            empty_year = empty_year and success

            if empty_year:
                solr_output_filepaths = {'daily_bin': '',
                                         'daily_netCDF': '',
                                         'monthly_bin': '',
                                         'monthly_netCDF': ''}

            # Query Solr for existing aggregation
            fq = [f'dataset_s:{A.dataset_name}', 'type_s:aggregation',
                  f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'year_s:{year}']
            docs = solr_utils.solr_query(fq)

            # If aggregation exists, update using Solr entry id
            if len(docs) > 0:
                doc_id = docs[0]['id']
                update_body = [
                    {
                        "id": doc_id,
                        "aggregation_time_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                        "aggregation_version_s": {"set": A.version}
                    }
                ]
            else:
                update_body = [
                    {
                        "type_s": 'aggregation',
                        "dataset_s": A.dataset_name,
                        "year_s": year,
                        "grid_name_s": grid_name,
                        "field_s": field_name,
                        "aggregation_time_dt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "aggregation_success_b": success,
                        "aggregation_version_s": A.version
                    }
                ]

            # Update file paths according to the data time scale and do monthly aggregation config field
            if data_time_scale == 'daily':
                update_body[0]["aggregated_daily_bin_path_s"] = {"set": solr_output_filepaths['daily_bin']}
                update_body[0]["aggregated_daily_netCDF_path_s"] = {"set": solr_output_filepaths['daily_netCDF']}
                update_body[0]["daily_aggregated_uuid_s"] = {"set": uuids[0]}
                if A.do_monthly_aggregation:
                    update_body[0]["aggregated_monthly_bin_path_s"] = {"set": solr_output_filepaths['monthly_bin']}
                    update_body[0]["aggregated_monthly_netCDF_path_s"] = {
                        "set": solr_output_filepaths['monthly_netCDF']}
                    update_body[0]["monthly_aggregated_uuid_s"] = {"set": uuids[1]}
            elif data_time_scale == 'monthly':
                update_body[0]["aggregated_monthly_bin_path_s"] = {"set": solr_output_filepaths['monthly_bin']}
                update_body[0]["aggregated_monthly_netCDF_path_s"] = {"set": solr_output_filepaths['monthly_netCDF']}
                update_body[0]["monthly_aggregated_uuid_s"] = {"set": uuids[1]}

            if empty_year:
                update_body[0]["notes_s"] = {"set": 'Empty year (no data present in grid), not saving to disk.'}
            else:
                update_body[0]["notes_s"] = {"set": ''}

            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code != 200:
                logging.exception(
                    f'Failed to update Solr aggregation entry for {field_name} in {A.dataset_name} for {year} and grid {grid_name}')

            # Query for descendants entries from this year
            fq = ['type_s:descendants', f'dataset_s:{A.dataset_name}',
                  f'date_s:{year}*']
            existing_descendants_docs = solr_utils.solr_query(fq)

            # if descendants entries already exist, update them
            if len(existing_descendants_docs) > 0:
                for doc in existing_descendants_docs:
                    doc_id = doc['id']

                    update_body = [
                        {
                            "id": doc_id,
                            "all_aggregation_success_b": {"set": aggregation_successes}
                        }
                    ]

                    # Add aggregation file path fields to descendants entry
                    for key, value in solr_output_filepaths.items():
                        update_body[0][f'{grid_name}_{field_name}_aggregated_{key}_path_s'] = {
                            "set": value}

                    r = solr_utils.solr_update(update_body, r=True)

                    if r.status_code != 200:
                        logging.exception(
                            f'Failed to update Solr aggregation entry for {field_name} in {A.dataset_name} for {year} and grid {grid_name}')

            fq = [f'dataset_s:{A.dataset_name}', 'type_s:aggregation',
                  f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'year_s:{year}']
            docs = solr_utils.solr_query(fq)

            # Export annual descendants JSON file for each aggregation created
            logging.debug(f'Exporting {year} descendants for grid {grid_name} and field {field_name}')
            json_output = {}
            json_output['dataset'] = A.ds_meta
            json_output['aggregation'] = docs
            json_output['transformations'] = A.transformations[field_name]
            json_output_path = f'{OUTPUT_DIR}/{A.dataset_name}/transformed_products/{grid_name}/aggregated/{field_name}/{A.dataset_name}_{field_name}_{grid_name}_{year}_descendants'
            with open(json_output_path, 'w') as f:
                resp_out = json.dumps(json_output, indent=4)
                f.write(resp_out)
    # Query Solr for successful aggregation documents
    fq = [f'dataset_s:{A.dataset_name}',
          'type_s:aggregation', 'aggregation_success_b:true']
    successful_aggregations = solr_utils.solr_query(fq)

    # Query Solr for failed aggregation documents
    fq = [f'dataset_s:{A.dataset_name}',
          'type_s:aggregation', 'aggregation_success_b:false']
    failed_aggregations = solr_utils.solr_query(fq)

    aggregation_status = 'All aggregations successful'

    if not successful_aggregations and not failed_aggregations:
        aggregation_status = 'No aggregations performed'
    elif not successful_aggregations:
        aggregation_status = 'No successful aggregations'
    elif failed_aggregations:
        aggregation_status = f'{len(failed_aggregations)} aggregations failed'

    # Update Solr dataset entry status and years_updated to empty
    update_body = [
        {
            "id": A.ds_meta['id'],
            "aggregation_version_s": {"set": A.version},
            "aggregation_status_s": {"set": aggregation_status}
        }
    ]

    r = solr_utils.solr_update(update_body, r=True)

    if r.status_code == 200:
        logging.debug(
            f'Successfully updated Solr with aggregation information for {A.dataset_name}')
    else:
        logging.exception(
            f'Failed to update Solr dataset entry with aggregation information for {A.dataset_name}')

    return aggregation_status
