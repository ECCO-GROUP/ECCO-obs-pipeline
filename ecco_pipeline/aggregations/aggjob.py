import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import uuid

import numpy as np
import xarray as xr
from conf.global_settings import OUTPUT_DIR
from aggregations.aggregation import Aggregation
from utils import solr_utils
from utils.ecco_utils import ecco_functions, records, date_time



class AggJob(Aggregation):

    def __init__(self, config: dict, grid: dict, year: str, field: dict):
        super().__init__(config)
        self.grid = grid
        self.year = year
        self.field = field

    def __str__(self) -> str:
        return f'{self.grid["grid_name_s"]}-{self.field["name"]}-{self.year}'

    def get_data_by_date(self, date, grid_name, field_name):
        # Query for date
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:transformation', 'success_b:True',
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
        opened_datasets = []
        for doc in docs:
            data_DS = xr.open_dataset(doc['transformation_file_path_s'], decode_times=True)
            opened_datasets.append(data_DS)
            
            # Update JSON transformations list
            fq = [f'dataset_s:{self.dataset_name}', 'type_s:granule',
                    f'pre_transformation_file_path_s:"{doc["pre_transformation_file_path_s"]}"']
            harvested_metadata = solr_utils.solr_query(fq)

            transformation_metadata = doc
            transformation_metadata['harvested'] = harvested_metadata
            self.transformations[field_name].append(transformation_metadata)
            
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
    
    def process_data_by_date(self, transformation_docs, field, grid, model_grid_ds, date):
        field_name = field.get('name')
        data_time_scale = self.data_time_scale

        make_empty_record = False
        if not transformation_docs:
            logging.info(f'No transformation docs for {str(date)}. Making empty record.')
            make_empty_record = True
        else:
            try:
                data_ds = self.open_datasets(transformation_docs, field_name)
            except Exception as e:
                logging.error(f'Error opening transformation file(s). Making empty record: {e}')
                make_empty_record = True
            
        if make_empty_record:
            data_da = records.make_empty_record(date, model_grid_ds, self.precision)
            data_da.attrs['long_name'] = field['long_name']
            data_da.attrs['standard_name'] = field['standard_name']
            data_da.attrs['units'] = field['units']
            data_da.name = f'{field_name}_interpolated_to_{grid.get("grid_name_s")}'
            data_da.attrs['original_field_name'] = field_name
            data_da.attrs['interpolation_date'] = str(np.datetime64(datetime.now(), 'D'))
            data_da.attrs['agg_notes'] = "empty record created during aggregation"

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
    
    def check_nan_da(self, da: xr.DataArray) -> bool:
        '''
        Check if da consists entirely of nans
        '''
        if np.sum(~np.isnan(da.values)) == 0:
            return True
        return False
    
    def generate_provenance(self, year, grid_name, field_name, solr_output_filepaths, aggregation_successes):
        # Query for descendants entries from this year
        fq = ['type_s:descendants', f'dataset_s:{self.dataset_name}', f'date_s:{year}*']
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
                    logging.exception(f'Failed to update Solr aggregation entry for {field_name} in {self.dataset_name} for {year} and grid {grid_name}')

        fq = [f'dataset_s:{self.dataset_name}', 'type_s:aggregation',
                f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'year_s:{year}']
        docs = solr_utils.solr_query(fq)

        # Export annual descendants JSON file for each aggregation created
        logging.debug(f'Exporting {year} descendants for grid {grid_name} and field {field_name}')
        json_output = {}
        json_output['dataset'] = self.ds_meta
        json_output['aggregation'] = docs
        json_output['transformations'] = self.transformations[field_name]
        json_output_path = f'{OUTPUT_DIR}/{self.dataset_name}/transformed_products/{grid_name}/aggregated/{field_name}/{self.dataset_name}_{field_name}_{grid_name}_{year}_descendants'
        with open(json_output_path, 'w') as f:
            resp_out = json.dumps(json_output, indent=4)
            f.write(resp_out)
    
    def aggregate(self):
        aggregation_successes = True
        # Construct list of dates corresponding to data time scale
        grid_path = self.grid['grid_path_s']
        grid_name = self.grid['grid_name_s']
        grid_type = self.grid['grid_type_s']
        
        model_grid_ds = xr.open_dataset(grid_path, decode_times=True)
        
        data_time_scale = self.ds_meta.get('data_time_scale_s')
        if  data_time_scale == 'daily':
            dates_in_year = np.arange(f'{self.year}-01-01', f'{int(self.year)+1}-01-01', dtype='datetime64[D]')
        elif data_time_scale == 'monthly':
            dates_in_year = [f'{self.year}-{str(month).zfill(2)}-01' for month in range(1,13)]

        field_name = self.field.get('name')

        logging.info(f'Aggregating {str(self.year)}_{grid_name}_{field_name}')

        daily_DS_year = []
        for date in dates_in_year:
            docs = self.get_data_by_date(date, grid_name, field_name)
            data_DS = self.process_data_by_date(docs, self.field, self.grid, model_grid_ds, date)
            daily_DS_year.append(data_DS)

        # Concatenate all data files within annual list
        daily_annual_ds = xr.concat((daily_DS_year), dim='time')
        data_var = list(daily_annual_ds.keys())[0]

        daily_annual_ds.attrs['aggregation_version'] = self.version
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
        shortest_filename = f'{self.dataset_name}_{grid_name}_{data_time_scale.upper()}_{field_name}_{self.year}'
        monthly_filename = f'{self.dataset_name}_{grid_name}_MONTHLY_{field_name}_{self.year}'

        output_path = f'{OUTPUT_DIR}/{self.dataset_name}/transformed_products/{grid_name}/aggregated/{field_name}/'

        bin_output_dir = Path(output_path) / 'bin'
        bin_output_dir.mkdir(parents=True, exist_ok=True)

        netCDF_output_dir = Path(output_path) / 'netCDF'
        netCDF_output_dir.mkdir(parents=True, exist_ok=True)

        uuids = [str(uuid.uuid1()), str(uuid.uuid1())]

        success = True
        empty_year = False

        # Save
        if self.check_nan_da(daily_annual_ds[data_var]):
            empty_year = True
        else:
            if self.do_monthly_aggregation:
                logging.info(f'Aggregating monthly {str(self.year)}_{grid_name}_{field_name}')
                try:
                    mon_DS_year_merged = ecco_functions.monthly_aggregation(daily_annual_ds, data_var, self.year, self, uuids[1])
                    mon_DS_year_merged[data_var] = mon_DS_year_merged[data_var].fillna(self.nc_fill_val)

                    if self.save_binary:
                        records.save_binary(mon_DS_year_merged, monthly_filename, self.bin_fill_val,
                                            bin_output_dir, self.binary_dtype, grid_type, data_var)
                    if self.save_netcdf:
                        records.save_netcdf(mon_DS_year_merged, monthly_filename, self.nc_fill_val, netCDF_output_dir)

                except Exception as e:
                    logging.exception(f'Error aggregating {self.dataset_name}. {e}')
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

            daily_annual_ds[data_var] = daily_annual_ds[data_var].fillna(self.nc_fill_val)

            if self.save_binary:
                records.save_binary(daily_annual_ds, shortest_filename, self.bin_fill_val,
                                    bin_output_dir, self.binary_dtype, grid_type, data_var)
            if self.save_netcdf:
                records.save_netcdf(daily_annual_ds, shortest_filename, self.nc_fill_val,
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
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:aggregation',
                f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'year_s:{self.year}']
        docs = solr_utils.solr_query(fq)

        # If aggregation exists, update using Solr entry id
        if len(docs) > 0:
            doc_id = docs[0]['id']
            update_body = [
                {
                    "id": doc_id,
                    "aggregation_time_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                    "aggregation_version_s": {"set": self.version}
                }
            ]
        else:
            update_body = [
                {
                    "type_s": 'aggregation',
                    "dataset_s": self.dataset_name,
                    "year_s": self.year,
                    "grid_name_s": grid_name,
                    "field_s": field_name,
                    "aggregation_time_dt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "aggregation_success_b": success,
                    "aggregation_version_s": self.version
                }
            ]

        # Update file paths according to the data time scale and do monthly aggregation config field
        if data_time_scale == 'daily':
            update_body[0]["aggregated_daily_bin_path_s"] = {"set": solr_output_filepaths['daily_bin']}
            update_body[0]["aggregated_daily_netCDF_path_s"] = {"set": solr_output_filepaths['daily_netCDF']}
            update_body[0]["daily_aggregated_uuid_s"] = {"set": uuids[0]}
            if self.do_monthly_aggregation:
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
            logging.exception(f'Failed to update Solr aggregation entry for {field_name} in {self.dataset_name} for {self.year} and grid {grid_name}')

        self.generate_provenance(self.year, grid_name, field_name, solr_output_filepaths, aggregation_successes)
    
