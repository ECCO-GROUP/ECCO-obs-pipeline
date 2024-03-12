import json
import logging
import os
import uuid
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import current_process
from typing import Iterable

import netCDF4 as nc4
import numpy as np
import xarray as xr
from baseclasses import Dataset, Field
from conf.global_settings import OUTPUT_DIR
from utils.pipeline_utils import solr_utils
from utils.processing_utils import records

logger = logging.getLogger(str(current_process().pid))

BINARY_FILL_VALUE = -9999
NETCDF_FILL_VALUE = nc4.default_fillvals['f4']

class Aggregation(Dataset):
    '''
    Aggregation class for aggregation job metadata - the specifics for a single aggregation task.
    '''
    def __init__(self, config: dict, grid: dict, year: str, field: Field):
        super().__init__(config)
        self.version: str = str(config.get('a_version', ''))
        self.do_monthly_aggregation: bool = config.get('do_monthly_aggregation', False)
        self.remove_nan_days_from_data: bool = config.get('remove_nan_days_from_data', True)
        self.skipna_in_mean: bool = config.get('skipna_in_mean', False)
        self.transformations: Iterable[dict] = defaultdict(list)
        self._set_ds_meta()
        self.grid: dict = grid
        self.year: str = year
        self.field: Field = field

    def __str__(self) -> str:
        return f'"{self.grid["grid_name_s"]} {self.field.name} {self.year}"'
    
    def _set_ds_meta(self):
        fq = [f'dataset_s:{self.ds_name}', 'type_s:dataset']
        ds_meta = solr_utils.solr_query(fq)[0]
        if 'start_date_dt' not in ds_meta:
            logger.info('No transformed granules to aggregate.')
            raise Exception('No transformed granules to aggregate.')
        self.ds_meta = ds_meta
        
    def make_empty_date(self, date: str, model_grid_ds: xr.Dataset, grid: dict) -> xr.Dataset:
        '''
        Creates "empty record" and fills in relevant metadata
        '''
        data_da = records.make_empty_record(date, model_grid_ds)
        data_da.name = f'{self.field.name}_interpolated_to_{grid.get("grid_name_s")}'
        data_ds = data_da.to_dataset()

        # add time_bnds coordinate
        # [start_time, end_time] dimensions
        if self.data_time_scale == 'monthly':
            period = 'AVG_MON'
        elif self.data_time_scale == 'daily':
            period = 'AVG_DAY'
        tb = records.TimeBound(rec_avg_start=data_ds.time_end.values[0], period=period)
            
        data_ds['time'] = [tb.center]
        data_ds = data_ds.assign_coords({'time_bnds': (['time', 'nv'], [tb.bounds.T])})
        data_ds.time.attrs.update({'bounds':'time_bnds'})
        data_ds = data_ds.drop(['time_start', 'time_end'])
        return data_ds
    
    def generate_provenance(self, grid_name, solr_output_filepaths, aggregation_successes):
        # Query for descendants entries from this year
        fq = ['type_s:descendants', f'dataset_s:{self.ds_name}', f'date_s:{self.year}*']
        existing_descendants_docs = solr_utils.solr_query(fq)

        # if descendants entries already exist, update them
        if len(existing_descendants_docs) > 0:
            update_body = []
            for doc in existing_descendants_docs:
                doc_id = doc['id']

                temp_doc = {
                        "id": doc_id,
                        "all_aggregation_success_b": {"set": aggregation_successes}
                    }

                # Add aggregation file path fields to descendants entry
                for key, value in solr_output_filepaths.items():
                    temp_doc[f'{grid_name}_{self.field.name}_aggregated_{key}_path_s'] = {"set": value}
                update_body.append(temp_doc)
                
            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code != 200:
                logger.exception(f'Failed to update Solr aggregation entry for {self.field.name} in {self.ds_name} for {self.year} and grid {grid_name}')

        fq = [f'dataset_s:{self.ds_name}', 'type_s:aggregation',
                f'grid_name_s:{grid_name}', f'field_s:{self.field.name}', f'year_s:{self.year}']
        docs = solr_utils.solr_query(fq)

        # Export annual descendants JSON file for each aggregation created
        logger.debug(f'Exporting {self.year} descendants for grid {grid_name} and field {self.field.name}')
        json_output = {}
        json_output['dataset'] = self.ds_meta
        json_output['aggregation'] = docs
        json_output['transformations'] = self.transformations[self.field.name]
        
        json_filename = f'{self.ds_name}_{self.field.name}_{grid_name}_{self.year}_descendants.json'
        json_output_path = os.path.join(OUTPUT_DIR, self.ds_name, 'transformed_products', grid_name, 'aggregated', self.field.name, json_filename)
        with open(json_output_path, 'w') as f:
            resp_out = json.dumps(json_output, indent=4)
            f.write(resp_out)
    
    def get_filepaths(self) -> dict:
        fq = [f'dataset_s:{self.ds_name}', 'type_s:transformation', 'success_b:True',
              f'grid_name_s:{self.grid["grid_name_s"]}', f'field_s:{self.field.name}', f'date_s:{self.year}*']
        docs = solr_utils.solr_query(fq)
        filepaths = defaultdict(list)
        for doc in docs:
            filepaths[doc['date_s']].append(doc['transformation_file_path_s'])
            
            # Update JSON transformations list
            fq = [f'dataset_s:{self.ds_name}', 'type_s:granule',
                    f'pre_transformation_file_path_s:"{doc["pre_transformation_file_path_s"]}"']
            harvested_metadata = solr_utils.solr_query(fq)

            transformation_metadata = doc
            transformation_metadata['harvested'] = harvested_metadata
            self.transformations[self.field.name].append(transformation_metadata)
        return filepaths
        
    def open_and_concat(self, filepaths: dict):
        opened_files = []
        dates = sorted(list(filepaths.keys()))
        for date in dates:
            files = filepaths[date]
            if len(files) == 1:
                opened_files.append(xr.open_dataset(files[0]))
            else:
                f1 = xr.open_dataset(files[0])
                f2 = xr.open_dataset(files[1])                
                var = list(f1.keys())[0]
                if np.isnan(f1[var].values).all():
                    if np.isnan(f2[var].values).all():
                        opened_files.append(f1)
                    else:
                        f2[var].values = np.where(np.isnan(f2[var].values), f1[var].values, f2[var].values)
                        opened_files.append(f2)
                else:        
                    f1[var].values = np.where(np.isnan(f1[var].values), f2[var].values, f1[var].values)
                    opened_files.append(f1)
        ds = xr.concat(opened_files, dim='time')
        return ds
        
    def get_missing_dates(self) -> Iterable[str]:
        fq = [f'dataset_s:{self.ds_name}', 'type_s:transformation', 'success_b:True',
              f'grid_name_s:{self.grid["grid_name_s"]}', f'field_s:{self.field.name}', f'date_s:{self.year}*']
        docs = solr_utils.solr_query(fq)
        doc_dates = sorted([doc['date_s'][:10] for doc in docs])
        
        data_time_scale = self.ds_meta.get('data_time_scale_s')

        if  data_time_scale == 'daily':
            dates_in_year = np.arange(f'{self.year}-01-01', f'{int(self.year)+1}-01-01', dtype='datetime64[D]').astype(str)
            missing_dates = sorted(list(set(dates_in_year) - set(doc_dates)))
        elif data_time_scale == 'monthly':
            dates_in_year = [f'{self.year}-{str(month).zfill(2)}-01' for month in range(1,13)]
            missing_dates = []
            for date in dates_in_year:
                if date in doc_dates:
                    continue
                tolerance_dates = []
                for i in range(1, 8):
                    tolerance_dates.append(datetime.strftime(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=i), '%Y-%m-%d'))
                    tolerance_dates.append(datetime.strftime(datetime.strptime(date, '%Y-%m-%d') - timedelta(days=i), '%Y-%m-%d'))
                if any(x in tolerance_dates for x in doc_dates):
                    continue
                missing_dates.append(date)
        return missing_dates       

    def monthly_aggregation(self, ds: xr.Dataset, var: str, uuid: str):
        attrs = ds.attrs
        mon_DS_year = []
        for month in range(1, 13):
            # to find the last day of the month, we go up one month and back one day
            # if Jan-Nov, then we'll go forward one month to Feb-Dec
            # for december we go up one year, and set month to january
            if month < 12:
                cur_mon_year = np.datetime64(f'{self.year}-{str(month+1).zfill(2)}-01', 'ns')
            else:
                cur_mon_year = np.datetime64(f'{int(self.year)+1}-01-01', 'ns')

            mon_str = str(self.year) + '-' + str(month).zfill(2)
            cur_mon = ds[var].sel(time=mon_str)

            if self.remove_nan_days_from_data:
                nonnan_days = []
                for i in range(len(cur_mon)):
                    if(np.count_nonzero(~np.isnan(cur_mon[i].values)) > 0):
                        nonnan_days.append(cur_mon[i])
                if nonnan_days:
                    cur_mon = xr.concat((nonnan_days), dim='time')

            # Compute monthly mean
            mon_DA = cur_mon.mean(axis=0, skipna=self.skipna_in_mean, keep_attrs=True)

            tb = records.TimeBound(rec_avg_end=cur_mon_year, period='AVG_MON')
            
            mon_DA = mon_DA.assign_coords({'time': tb.center})
            mon_DA = mon_DA.expand_dims('time', axis=0)

            avg_center_time = mon_DA.time.copy(deep=True)
            avg_center_time.values[0] = tb.center

            # halfway through the approx 1M averaging period.
            mon_DA.time.values[0] = tb.center
            mon_DA.time.attrs['long_name'] = 'center time of 1M averaging period'

            mon_DS = mon_DA.to_dataset()

            mon_DS = mon_DS.assign_coords({'time_bnds': (('time', 'nv'), [tb.bounds])})
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
    
    def aggregate(self):
        aggregation_successes = True
        # Construct list of dates corresponding to data time scale
        grid_path = self.grid['grid_path_s']
        grid_name = self.grid['grid_name_s']
        grid_type = self.grid['grid_type_s']
        
        model_grid_ds = xr.open_dataset(grid_path, decode_times=True)

        logger.info(f'Aggregating {str(self.year)}_{grid_name}_{self.field.name}')

        logger.info('Collecting filepaths from Solr')
        transformation_fps = self.get_filepaths()
        logger.info(f'Opening and concatenating {len(transformation_fps)} transformation files...')
        daily_annual_ds = self.open_and_concat(transformation_fps)
        
        logger.info('Polling for missing dates, creating ds of empty records, and combining with concatenated transformation files...')
        missing_dates = self.get_missing_dates()
        logger.debug(f'Making empty records for {missing_dates}')
        if missing_dates:
            empty_records = [self.make_empty_date(date, model_grid_ds, self.grid) for date in sorted(missing_dates)]
            missing_dates_ds = xr.concat(empty_records, dim='time')
            daily_annual_ds = xr.concat([daily_annual_ds, missing_dates_ds], dim='time')
        daily_annual_ds = daily_annual_ds.sortby(daily_annual_ds.time)
        
        data_var = list(daily_annual_ds.keys())[0]

        daily_annual_ds.attrs['aggregation_version'] = self.version
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            daily_annual_ds[data_var].attrs['valid_min'] = np.nanmin(daily_annual_ds[data_var].values)
            daily_annual_ds[data_var].attrs['valid_max'] = np.nanmax(daily_annual_ds[data_var].values)

        remove_keys = [k for k in daily_annual_ds[data_var].attrs.keys() if 'original' in k and k != 'original_field_name']
        for key in remove_keys:
            del daily_annual_ds[data_var].attrs[key]

        data_time_scale = self.ds_meta.get('data_time_scale_s')

        # Create filenames based on date time scale
        # If data time scale is monthly, shortest_filename is monthly
        shortest_filename = f'{self.ds_name}_{grid_name}_{data_time_scale.upper()}_{self.field.name}_{self.year}'
        monthly_filename = f'{self.ds_name}_{grid_name}_MONTHLY_{self.field.name}_{self.year}'

        output_path = f'{OUTPUT_DIR}/{self.ds_name}/transformed_products/{grid_name}/aggregated/{self.field.name}/'

        bin_output_dir = os.path.join(output_path, 'bin')
        os.makedirs(bin_output_dir, exist_ok=True)


        netCDF_output_dir = os.path.join(output_path, 'netCDF')
        os.makedirs(netCDF_output_dir, exist_ok=True)

        uuids = [str(uuid.uuid1()), str(uuid.uuid1())]

        success = True
        empty_year = False

        # Save
        if np.isnan(daily_annual_ds[data_var].values).all():
            empty_year = True
        else:
            if self.do_monthly_aggregation:
                logger.info(f'Aggregating monthly {str(self.year)}_{grid_name}_{self.field.name}')
                try:
                    mon_DS_year_merged = self.monthly_aggregation(daily_annual_ds, data_var, uuids[1])
                    mon_DS_year_merged[data_var] = mon_DS_year_merged[data_var].fillna(NETCDF_FILL_VALUE)

                    records.save_binary(mon_DS_year_merged, monthly_filename, bin_output_dir, grid_type, data_var)
                    records.save_netcdf(mon_DS_year_merged, f'{monthly_filename}.nc', netCDF_output_dir)

                except Exception as e:
                    logger.exception(f'Error aggregating {self.ds_name}. {e}')
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

            daily_annual_ds[data_var] = daily_annual_ds[data_var].fillna(NETCDF_FILL_VALUE)

            records.save_binary(daily_annual_ds, shortest_filename, bin_output_dir, grid_type, data_var)
            records.save_netcdf(daily_annual_ds, f'{shortest_filename}.nc', netCDF_output_dir)       

        aggregation_successes = aggregation_successes and success
        empty_year = empty_year and success

        if empty_year:
            solr_output_filepaths = {'daily_bin': '',
                                    'daily_netCDF': '',
                                    'monthly_bin': '',
                                    'monthly_netCDF': ''}
        else:
            solr_output_filepaths = {
                'daily_bin': os.path.join(output_path, 'bin', f'{shortest_filename}'),
                'daily_netCDF': os.path.join(output_path, 'net', f'{shortest_filename}.nc'),
                'monthly_bin': os.path.join(output_path, 'bin', f'{monthly_filename}'),
                'monthly_netCDF': os.path.join(output_path, 'net', f'{monthly_filename}.nc')
                }

        # Query Solr for existing aggregation
        fq = [f'dataset_s:{self.ds_name}', 'type_s:aggregation', 
              f'grid_name_s:{grid_name}', f'field_s:{self.field.name}', f'year_s:{self.year}']
        docs = solr_utils.solr_query(fq)

        # If aggregation exists, update using Solr entry id
        if len(docs) > 0:
            doc_id = docs[0]['id']
            update_doc = {
                    "id": doc_id,
                    "aggregation_time_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                    "aggregation_version_s": {"set": self.version}
                }
        else:
            update_doc = {
                    "type_s": 'aggregation',
                    "dataset_s": self.ds_name,
                    "year_s": self.year,
                    "grid_name_s": grid_name,
                    "field_s": self.field.name,
                    "aggregation_time_dt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "aggregation_success_b": success,
                    "aggregation_version_s": self.version
                }

        # Update file paths according to the data time scale and do monthly aggregation config field
        if data_time_scale == 'daily':
            update_doc["aggregated_daily_bin_path_s"] = {"set": solr_output_filepaths['daily_bin']}
            update_doc["aggregated_daily_netCDF_path_s"] = {"set": solr_output_filepaths['daily_netCDF']}
            update_doc["daily_aggregated_uuid_s"] = {"set": uuids[0]}
        if data_time_scale == 'monthly' or self.do_monthly_aggregation:
            update_doc["aggregated_monthly_bin_path_s"] = {"set": solr_output_filepaths['monthly_bin']}
            update_doc["aggregated_monthly_netCDF_path_s"] = {"set": solr_output_filepaths['monthly_netCDF']}
            update_doc["monthly_aggregated_uuid_s"] = {"set": uuids[1]}

        if empty_year:
            update_doc["notes_s"] = {"set": 'Empty year (no data present in grid), not saving to disk.'}
        else:
            update_doc["notes_s"] = {"set": ''}

        r = solr_utils.solr_update([update_doc], r=True)

        if r.status_code != 200:
            logger.exception(f'Failed to update Solr aggregation entry for {self.field.name} in {self.ds_name} for {self.year} and grid {grid_name}')

        self.generate_provenance(grid_name, solr_output_filepaths, aggregation_successes)
    
