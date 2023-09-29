import itertools
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from aggregations.aggregation import Aggregation
from conf.global_settings import OUTPUT_DIR
from utils import solr_utils
from utils.ecco_utils import ecco_functions, records, date_time



def get_data_by_date(A, date, grid_name, field_name):
    # Query for date
    fq = [f'dataset_s:{A.dataset_name}', 'type_s:transformation',
            f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'date_s:{date}*']

    docs = solr_utils.solr_query(fq)

    # If first of month is not found, query with 7 day tolerance only for monthly data
    if not docs and A.ds_meta.get('data_time_scale_s') == 'monthly':
        tolerance = int(A.ds_meta.get('monthly_tolerance', 8))
        start_month_date = datetime.strptime(date, '%Y-%m-%d')
        tolerance_days = []

        for i in range(1, tolerance):
            tolerance_days.append(datetime.strftime(start_month_date + timedelta(days=i), '%Y-%m-%d'))
            tolerance_days.append(datetime.strftime(start_month_date - timedelta(days=i), '%Y-%m-%d'))
        for tol_date in tolerance_days:
            fq = [f'dataset_s:{A.dataset_name}', 'type_s:transformation',
                    f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'date_s:{tol_date}*']
            docs = solr_utils.solr_query(fq)
            if docs:
                return docs
    return docs

def open_datasets(A, docs, field_name):
    opened_datasets = []
    for doc in docs:
        data_DS = xr.open_dataset(doc['transformation_file_path_s'], decode_times=True)
        opened_datasets.append(data_DS)
        
        # Update JSON transformations list
        fq = [f'dataset_s:{A.dataset_name}', 'type_s:granule',
                f'pre_transformation_file_path_s:"{doc["pre_transformation_file_path_s"]}"']
        harvested_metadata = solr_utils.solr_query(fq)

        transformation_metadata = doc
        transformation_metadata['harvested'] = harvested_metadata
        A.transformations[field_name].append(transformation_metadata)
        
    if len(docs) == 2:
        first_DS = opened_datasets[0]
        var_1 = list(first_DS.keys())[0]
        
        second_DS = opened_datasets[1]
        var_2 = list(first_DS.keys())[0]
        if ~np.isnan(first_DS[var_1].values).all():
            ds = first_DS.copy()
            ds[var_1].values = np.where(np.isnan(ds[var_1].values), second_DS[var_2].values, ds[var_1].values)
        else:
            ds = second_DS.copy()
            ds[var_2].values = np.where(np.isnan(ds[var_2].values), first_DS[var_1].values, ds[var_2].values)
    else:
        ds = opened_datasets[0]
    return ds

def process_data_by_date(A, docs, field, grid, model_grid_ds, date):
    field_name = field.get('name')
    data_time_scale = A.data_time_scale

    if docs:
        data_ds = open_datasets(A, docs, field_name)
    else:
        data_da = records.make_empty_record(date, model_grid_ds, A.precision)
        data_da.attrs['long_name'] = field['long_name']
        data_da.attrs['standard_name'] = field['standard_name']
        data_da.attrs['units'] = field['units']
        data_da.name = f'{field_name}_interpolated_to_{grid.get("grid_name_s")}'

        empty_record_attrs = data_da.attrs
        empty_record_attrs['original_field_name'] = field_name
        empty_record_attrs['interpolation_date'] = str(np.datetime64(datetime.now(), 'D'))
        data_da.attrs = empty_record_attrs

        data_ds = data_da.to_dataset()

        # add time_bnds coordinate
        # [start_time, end_time] dimensions
        # MONTHLY cannot use timedelta64 since it has a variable
        # number of ns/s/d. DAILY can so we use it.
        if data_time_scale.upper() == 'MONTHLY':
            end_time = str(data_ds.time_end.values[0])
            month = str(np.datetime64(end_time, 'M') + 1)
            end_time = [str(np.datetime64(month, 'ns'))]
        elif data_time_scale.upper() == 'DAILY':
            end_time = data_ds.time_end.values + np.timedelta64(1, 'D')

        _, ct = date_time.make_time_bounds_from_ds64(np.datetime64(end_time[0], 'ns'), 'AVG_MON')
        data_ds.time.values[0] = ct

        start_time = data_ds.time_start.values

        time_bnds = np.array([start_time, end_time], dtype='datetime64')
        time_bnds = time_bnds.T

        data_ds = data_ds.assign_coords({'time_bnds': (['time', 'nv'], time_bnds)})

        data_ds.time.attrs.update(bounds='time_bnds')

        data_ds = data_ds.drop('time_start')
        data_ds = data_ds.drop('time_end')
    return data_ds
    

def check_nan_da(da: xr.DataArray) -> bool:
    '''
    Check if da consists entirely of nans
    '''
    if np.sum(~np.isnan(da.values)) == 0:
        return True
    return False


def generate_provenance(A, year, grid_name, field_name, solr_output_filepaths, aggregation_successes):
    # Query for descendants entries from this year
    fq = ['type_s:descendants', f'dataset_s:{A.dataset_name}', f'date_s:{year}*']
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
                update_body[0][f'{grid_name}_{field_name}_aggregated_{key}_path_s'] = {"set": value}

            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code != 200:
                logging.exception(f'Failed to update Solr aggregation entry for {field_name} in {A.dataset_name} for {year} and grid {grid_name}')

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


def aggregate(A: Aggregation, grid: dict, year: int, field: dict):
    aggregation_successes = True
    # Construct list of dates corresponding to data time scale
    grid_path = grid['grid_path_s']
    grid_name = grid['grid_name_s']
    grid_type = grid['grid_type_s']
    
    model_grid_ds = xr.open_dataset(grid_path, decode_times=True)
    
    data_time_scale = A.ds_meta.get('data_time_scale_s')
    if  data_time_scale == 'daily':
        dates_in_year = np.arange(f'{year}-01-01', f'{int(year)+1}-01-01', dtype='datetime64[D]')
    elif data_time_scale == 'monthly':
        dates_in_year = [f'{year}-{str(month).zfill(2)}-01' for month in range(1,13)]

    field_name = field.get('name')

    logging.info(f'Aggregating {str(year)}_{grid_name}_{field_name}')

    daily_DS_year = []
    for date in dates_in_year:
        docs = get_data_by_date(A, date, grid_name, field_name)
        data_DS = process_data_by_date(A, docs, field, grid, model_grid_ds, date)
        daily_DS_year.append(data_DS)

    # Concatenate all data files within annual list
    daily_annual_ds = xr.concat((daily_DS_year), dim='time')
    data_var = list(daily_annual_ds.keys())[0]

    daily_annual_ds.attrs['aggregation_version'] = A.version
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
                mon_DS_year_merged = ecco_functions.monthly_aggregation(daily_annual_ds, data_var, year, A, uuids[1])
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
            update_body[0]["aggregated_monthly_netCDF_path_s"] = {"set": solr_output_filepaths['monthly_netCDF']}
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
        logging.exception(f'Failed to update Solr aggregation entry for {field_name} in {A.dataset_name} for {year} and grid {grid_name}')

    generate_provenance(A, year, grid_name, field_name, solr_output_filepaths, aggregation_successes)
    
    

def daily_aggregation(A: Aggregation, grid: dict, year: int, field: dict):
    grid_path = grid['grid_path_s']
    grid_name = grid['grid_name_s']
    grid_type = grid['grid_type_s']
    
    model_grid_ds = xr.open_dataset(grid_path, decode_times=True)
    dates_in_year = np.arange(f'{year}-01-01', f'{int(year)+1}-01-01', dtype='datetime64[D]')
    
    field_name = field.get('name')

    logging.info(f'Aggregating {str(year)}_{grid_name}_{field_name}')

    daily_DS_year = []
    for date in dates_in_year:
        docs = get_data_by_date(A, date, grid_name, field_name)
        data_DS = A.process_data_by_date(docs, field, grid, model_grid_ds, date)
        daily_DS_year.append(data_DS)
    return daily_DS_year


def monthly_aggregation(A: Aggregation, grid: dict, year: int, field: dict):
    grid_path = grid['grid_path_s']
    grid_name = grid['grid_name_s']
    grid_type = grid['grid_type_s']
    
    model_grid_ds = xr.open_dataset(grid_path, decode_times=True)
    dates_in_year = np.arange(f'{year}-01', f'{int(year)+1}-01', dtype='datetime64[M]')
    dates_in_year = [f'{date}-01' for date in dates_in_year]
    
    field_name = field.get('name')

    logging.info(f'Aggregating {str(year)}_{grid_name}_{field_name}')

    monthly_year_ds = []
    for date in dates_in_year:
        docs = get_data_by_date(A, date, grid_name, field_name)
        data_DS = process_data_by_date(A, docs, field, grid, model_grid_ds, date)
        monthly_year_ds.append(data_DS)
    return monthly_year_ds

      
def monthly_avg_aggregation():
    pass

def aggregation(config, grids_to_use=[]):
    """
    Aggregates data into annual files, saves them, and updates Solr
    """
    try:
        A = Aggregation(config, grids_to_use)
    except:
        return 'No aggregations performed'

    update_body = []

    # =====================================================
    # Loop through grids
    # =====================================================
    for grid in A.grids:

        years = A.years.get(grid['grid_name_s'])

        for year, field in list(itertools.product(years, A.fields)):
            # if A.data_time_scale == 'daily':
            #     annual_ds = daily_aggregation(A, grid, year, field)
            #     # monthly_avg_aggregation(A, grid, year, field)
            # elif A.data_time_scale == 'monthly':
            #     annual_ds = monthly_aggregation(A, grid, year, field)
            #     # ecco_functions.monthly_aggregation(A, grid, year, field, str(uuid.uuid1()))
            aggregate(A, grid, year, field)
            
    aggregation_status = A.get_agg_status()
    
    # Update Solr dataset entry status
    update_body = [
        {
            "id": A.ds_meta['id'],
            "aggregation_version_s": {"set": A.version},
            "aggregation_status_s": {"set": aggregation_status}
        }
    ]

    r = solr_utils.solr_update(update_body, r=True)

    if r.status_code == 200:
        logging.debug(f'Successfully updated Solr with aggregation information for {A.dataset_name}')
    else:
        logging.exception(f'Failed to update Solr dataset entry with aggregation information for {A.dataset_name}')

    return aggregation_status