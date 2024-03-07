import logging
import os
import pickle
import warnings
from datetime import datetime
from multiprocessing import current_process
from typing import Iterable, Tuple

import netCDF4 as nc4
import numpy as np
import pyresample as pr
import xarray as xr
from baseclasses import Dataset, Field
from conf.global_settings import OUTPUT_DIR
from requests import HTTPError
from utils.pipeline_utils import file_utils, solr_utils
from utils.processing_utils import ds_functions, records, transformation_utils
from utils.processing_utils.ds_functions import PosttransformationFuncs, PreprocessingFuncs, PretransformationFuncs

logger = logging.getLogger(str(current_process().pid))

BINARY_DTYPE = 'f4'
NETCDF_FILL_VALUE = nc4.default_fillvals[BINARY_DTYPE]

class Transformation(Dataset):
    def __init__(self, config: dict, source_file_path: str, granule_date: str):
        super().__init__(config)
        self.file_name: str = os.path.splitext(source_file_path.split('/')[-1])[0]
        self.transformation_version: float = config.get('t_version')
        self.date: str = granule_date
        self.hemi: str = self._get_hemi(config)
        
        # Projection information
        self.data_res: float = self._compute_data_res(config)
        self.area_extent: Iterable = config.get(f'area_extent{self.hemi}')
        self.dims: Iterable = config.get(f'dims{self.hemi}')
        self.proj_info: dict = config.get(f'proj_info{self.hemi}')

        # Processing information
        self.time_bounds_var: str = config.get('time_bounds_var', None)
        self.transpose: bool = config.get('transpose', False)

        self.mapping_operation: str = config.get('mapping_operation', 'mean')

    def _compute_data_res(self, config):
        '''

        '''
        res = config.get('data_res')
        if type(res) is str:
            if '/' in res:
                num, den = res.replace(' ', '').split('/')
                res = float(num) / float(den)
            else:
                res = float(res)
        return res

    def _get_hemi(self, config):
        '''
        Extracts hemisphere from filename. One of either 'nh' or 'sh'
        '''
        if 'hemi_pattern' in config:
            if config['hemi_pattern']['north'] in self.file_name:
                return '_nh'
            elif config['hemi_pattern']['south'] in self.file_name:
                return '_sh'
        return ''

    def apply_funcs(self, data_object, funcs: Iterable):
        logger = logging.getLogger(str(current_process().pid))
        for func_to_run in funcs:
            logger.debug(f'Applying {func_to_run} to {self.file_name} data')
            try:
                callable_func = getattr(ds_functions, func_to_run)
                data_object = callable_func(data_object)
                logger.debug(f'{func_to_run} successfully ran on {self.file_name}')
            except:
                logger.exception(f'{func_to_run} failed to run on {self.file_name}')
                raise Exception(f'{func_to_run} failed to run on {self.file_name}')
        return data_object

    def make_factors(self, grid_ds: xr.Dataset) -> Tuple[dict, np.ndarray, dict]:
        '''
        Generate mappings from source to target grid

        Returns Tuple
        (source_indices_within_target_radius_i,
        num_source_indices_within_target_radius_i,
        nearest_source_index_to_target_index_i)

        '''
        logger = logging.getLogger(str(current_process().pid))
        
        grid_name = grid_ds.name
        factors_dir = f'{OUTPUT_DIR}/{self.ds_name}/transformed_products/{grid_name}/'
        factors_file = f'{grid_name}{self.hemi}_v{self.transformation_version}_factors'
        factors_path = f'{factors_dir}{factors_file}'

        if os.path.exists(factors_path):
            logger.debug(f'Loading {grid_name} factors')
            with open(factors_path, "rb") as f:
                factors = pickle.load(f)
                return factors
        else:
            logger.info(f'Creating {grid_name} factors for {self.ds_name}')

        # Use hemisphere specific variables if data is hemisphere specific
        source_grid_min_L, source_grid_max_L, source_grid = transformation_utils.generalized_grid_product(self.data_res, self.area_extent,
                                                                                                    self.dims, self.proj_info)

        # Define the 'swath' as the lats/lon pairs of the model grid
        target_grid = pr.geometry.SwathDefinition(lons=grid_ds.XC.values.ravel(),
                                                  lats=grid_ds.YC.values.ravel())

        # Retrieve target_grid_radius from model_grid file
        if 'effective_grid_radius' in grid_ds:
            target_grid_radius = grid_ds.effective_grid_radius.values.ravel()
        elif 'effective_radius' in grid_ds:
            target_grid_radius = grid_ds.effective_radius.values.ravel()
        elif 'RAD' in grid_ds:
            target_grid_radius = grid_ds.RAD.values.ravel()
        elif 'rA' in grid_ds:
            target_grid_radius = 0.5*np.sqrt(grid_ds.rA.values.ravel())
        else:
            logger.exception(f'Unable to extract grid radius from {grid_ds.name}. Grid not supported')

        factors = (transformation_utils.find_mappings_from_source_to_target(source_grid, target_grid, target_grid_radius,
                                                                      source_grid_min_L, source_grid_max_L))
        logger.debug(f'Saving {grid_name} factors')
        os.makedirs(factors_dir, exist_ok=True)
        with open(factors_path, 'wb') as f:
            pickle.dump(factors, f)
        return factors
    
    def perform_mapping(self, ds: xr.Dataset, factors: Tuple, field: Field, model_grid: xr.Dataset) -> xr.DataArray:
        '''
        Maps source data to target grid and applies metadata
        '''
        logger = logging.getLogger(str(current_process().pid))


        data_DA = records.make_empty_record(self.date, model_grid)

        # print(data_DA)

        # add some metadata to the newly formed data array object
        data_DA.attrs['long_name'] = field.long_name
        data_DA.attrs['standard_name'] = field.standard_name
        data_DA.attrs['units'] = field.units
        data_DA.attrs['original_filename'] = self.file_name
        data_DA.attrs['original_field_name'] = field.name
        data_DA.attrs['interpolation_parameters'] = 'bin averaging'
        data_DA.attrs['interpolation_code'] = 'pyresample'
        data_DA.attrs['interpolation_date'] = str(np.datetime64(datetime.now(), 'D'))

        data_DA.time.attrs['long_name'] = 'center time of averaging period'

        data_DA.name = f'{field.name}_interpolated_to_{model_grid.name}'

        if self.transpose:
            orig_data = ds[field.name].values[0, :].T
        else:
            orig_data = ds[field.name].values

        # see if we have any valid data
        if np.sum(~np.isnan(orig_data)) > 0:
            data_model_projection = transformation_utils.transform_to_target_grid(*factors, orig_data, model_grid.XC.shape,
                                                            operation=self.mapping_operation)

            # put the new data values into the data_DA array.
            # --where the mapped data are not nan, replace the original values
            # --where they are nan, just leave the original values alone
            data_DA.values = np.where(~np.isnan(data_model_projection), data_model_projection, data_DA.values)
            record_notes = ''
        else:
            logger.debug(f'Empty granule for {self.file_name} (no data to transform to grid {model_grid.name})')
            record_notes = ' -- empty record -- '

        if self.time_bounds_var:
            if self.time_bounds_var in ds:
                time_start = str(ds[self.time_bounds_var].values.ravel()[0])
                time_end = str(ds[self.time_bounds_var].values.ravel()[0])
            else:
                logger.info(f'time_bounds_var {self.time_bounds_var} does not exist in file but is defined in config. \
                    Using other method for obtaining start/end times.')

        else:
            time_start = self.date
            if self.data_time_scale.upper() == 'MONTHLY':
                month = str(np.datetime64(self.date, 'M') + 1)
                time_end = str(np.datetime64(month, 'ns'))
            elif self.data_time_scale.upper() == 'DAILY':
                time_end = str(np.datetime64(self.date, 'D') + np.timedelta64(1, 'D'))

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
            data_DA.time.values[0] = self.date

        data_DA.attrs['notes'] = record_notes
        data_DA.attrs['original_time'] = str(data_DA.time.values[0])
        data_DA.attrs['original_time_start'] = str(data_DA.time_start.values[0])
        data_DA.attrs['original_time_end'] = str(data_DA.time_end.values[0])

        return data_DA
    
    def transform(self, model_grid: xr.Dataset, factors: Tuple, ds: xr.Dataset) -> Iterable[Tuple[xr.Dataset, bool]]:
        """
        Function that actually performs the transformations. Returns a list of transformed
        xarray datasets, one dataset for each field being transformed for the given grid.
        """
        logger = logging.getLogger(str(current_process().pid))
        
        logger.info(f'Transforming {len(self.fields)} fields on {self.date} to {model_grid.name}')

        record_date = self.date.replace('Z', '')

        field_DSs = []
        
        # =====================================================
        # Loop through fields to transform
        # =====================================================
        for field in self.fields:
            logger.debug(f'Transforming {self.file_name} for field {field.name}')

            if field.pre_transformations:
                try:
                    func_machine = PretransformationFuncs()
                    ds = func_machine.call_functions(field.pre_transformations, ds)
                except Exception as e:
                    logger.exception(e)

            if field.name in ds.data_vars: 
                try:
                    field_DA = self.perform_mapping(ds, factors, field, model_grid)
                    mapping_success = True
                except Exception as e:
                    logger.exception(f'Transformation failed: {e}')
                    field_DA = records.make_empty_record(record_date, model_grid)
                    field_DA.attrs['long_name'] = field.long_name
                    field_DA.attrs['standard_name'] = field.standard_name
                    field_DA.attrs['units'] = field.units
                    field_DA.attrs['empty_record_note'] = 'Transformation failed'
                    mapping_success = False
            else:
                logger.error(f'Transformation failed: key {field.name} is missing from source data. Making empty record.')
                field_DA = records.make_empty_record(record_date, model_grid)
                field_DA.attrs['long_name'] = field.long_name
                field_DA.attrs['standard_name'] = field.standard_name
                field_DA.attrs['units'] = field.units
                field_DA.attrs['empty_record_note'] = f'{field.name} missing from source data'
                mapping_success = True

            # =====================================================
            # Post transformation functions
            # =====================================================
            if mapping_success:
                try:
                    func_machine = PosttransformationFuncs()
                    field_DA = func_machine.call_functions(field.post_transformations, field_DA)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        field_DA.attrs['valid_min'] = np.nanmin(field_DA.values)
                        field_DA.attrs['valid_max'] = np.nanmax(field_DA.values)
                except Exception as e:
                    logger.exception(f'Post-transformation failed: {e}')
                    field_DA = records.make_empty_record(record_date, model_grid)
                    field_DA.attrs['long_name'] = field.long_name
                    field_DA.attrs['standard_name'] = field.standard_name
                    field_DA.attrs['units'] = field.units
                    field_DA.attrs['empty_record_note'] = 'Post transformation(s) failed'
                    mapping_success = False

            field_DA.values = np.where(np.isnan(field_DA.values), NETCDF_FILL_VALUE, field_DA.values)

            # Make dataarray into dataset
            field_DS = field_DA.to_dataset()

            # Dataset metadata
            ds_meta = {
                'interpolated_grid': model_grid.name,
                'model_grid_type': model_grid.type,
                'original_dataset_title': self.og_ds_metadata.get('original_dataset_title'),
                'original_dataset_short_name': self.og_ds_metadata.get('original_dataset_short_name'),
                'original_dataset_url': self.og_ds_metadata.get('original_dataset_url'),
                'original_dataset_reference': self.og_ds_metadata.get('original_dataset_reference'),
                'original_dataset_doi': self.og_ds_metadata.get('original_dataset_doi'),
                'interpolated_grid_id': model_grid.name,
                'transformation_version': self.transformation_version,
                'notes': self.note
            }

            field_DS = field_DS.assign_attrs(ds_meta)

            # add time_bnds coordinate
            # [start_time, end_time] dimensions
            start_time = field_DS.time_start.values
            end_time = field_DS.time_end.values

            time_bnds = np.array([start_time, end_time], dtype='datetime64')
            time_bnds = time_bnds.T
            field_DS = field_DS.assign_coords({'time_bnds': (['time', 'nv'], time_bnds)})

            field_DS.time.attrs.update(bounds='time_bnds')

            # time stuff
            data_time_scale = self.data_time_scale
            if data_time_scale == 'daily':
                period = 'AVG_DAY'
                rec_end = field_DS.time_bnds.values[0][1]
            elif data_time_scale == 'monthly':
                period = 'AVG_MON'
                cur_year = int(self.date[:4])
                cur_month = int(self.date[5:7])

                if cur_month < 12:
                    rec_end = np.datetime64(f'{cur_year}-{str(cur_month+1).zfill(2)}-01', 'ns')
                else:
                    rec_end = np.datetime64(f'{cur_year+1}-01-01', 'ns')

            if 'DEBIAS_LOCEAN' in self.ds_name:
                rec_end = field_DS.time.values[0] + np.timedelta64(1, 'D')

            tb = records.TimeBound(rec_avg_end=rec_end, period=period)
            field_DS.time.values[0] = tb.center
            field_DS.time_bnds.values[0][0] = tb.bounds[0]
            field_DS.time_bnds.values[0][1] = tb.bounds[1]

            field_DS = field_DS.drop('time_start')
            field_DS = field_DS.drop('time_end')
            field_DSs.append((field_DS, mapping_success))

        return field_DSs

    def load_file(self, source_file_path: str) -> xr.Dataset:
        if self.preprocessing_function:
            func_machine = PreprocessingFuncs()
            ds = func_machine.call_function(self.preprocessing_function, source_file_path, self.fields)
        else:
            ds = xr.open_dataset(source_file_path, decode_times=True)
        ds.attrs['original_file_name'] = self.file_name
        return ds

    def prepopulate_solr(self, source_file_path: str, grid_name: str):
        '''
        Populate Solr with transformation entries prior to attempting transformation
        '''
        update_body = []
        for field in self.fields:
            logger.debug(f'Transforming {field.name}')

            # Query if grid/field combination transformation entry exists
            query_fq = [f'dataset_s:{self.ds_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field.name}', f'pre_transformation_file_path_s:"{source_file_path}"']
            docs = solr_utils.solr_query(query_fq)
            transform = {}

            # If grid/field combination transformation exists, update transformation status
            # Otherwise initialize new transformation entry
            if len(docs) > 0:
                # Reset status fields
                transform['id'] = docs[0]['id']
                transform['transformation_in_progress_b'] = {"set": True}
                transform['success_b'] = {"set": False}
            else:
                # Query for granule entry to get checksum
                query_fq = [f'dataset_s:{self.ds_name}', 'type_s:granule',
                            f'pre_transformation_file_path_s:"{source_file_path}"']
                docs = solr_utils.solr_query(query_fq)

                # Initialize new transformation entry
                transform['type_s'] = 'transformation'
                transform['date_s'] = self.date
                transform['dataset_s'] = self.ds_name
                transform['pre_transformation_file_path_s'] = source_file_path
                transform['hemisphere_s'] = self.hemi.replace('_', '')
                transform['origin_checksum_s'] = docs[0]['checksum_s']
                transform['grid_name_s'] = grid_name
                transform['field_s'] = field.name
                transform['transformation_in_progress_b'] = True
                transform['success_b'] = False
            update_body.append(transform)
        r = solr_utils.solr_update(update_body, r=True)
        try:
            r.raise_for_status()
        except HTTPError:
            logger.exception(f'Failed to update Solr transformation status for {self.ds_name} on {self.date}')
            raise HTTPError

def transform(source_file_path: str, tx_jobs: dict, config: dict, granule_date: str):
    """
    Performs and saves locally all remaining transformations for a given source granule
    Updates Solr with transformation entries and updates descendants, and dataset entries
    """
    T = Transformation(config, source_file_path, granule_date)

    transformation_successes = True
    transformation_file_paths = {}
    grids_updated = []

    logger.debug(f'Loading {T.file_name} data')
    ds = T.load_file(source_file_path)
    
    grid_fields = [[f'({grid_name}, {field})' for field in tx_jobs[grid_name]] for grid_name in tx_jobs.keys()]
    logger.debug(f'{T.file_name} needs to transform: {grid_fields} ')

    # Iterate through grids in remaining_transformations
    for grid_name in tx_jobs.keys():
        fields: Iterable[Field] = tx_jobs[grid_name]

        logger.debug(f'Loading {grid_name} model grid')
        grid_ds = xr.open_dataset(f'grids/{grid_name}.nc').reset_coords()
        factors = T.make_factors(grid_ds)
        T.prepopulate_solr(source_file_path, grid_name)

        # =====================================================
        # Run transformation
        # =====================================================
        logger.debug(f'Running transformations for {T.file_name}')

        # Returns list of transformed DSs, one for each field in fields
        field_DSs = T.transform(grid_ds, factors, ds)
            
        # =====================================================
        # Save the output in netCDF format
        # =====================================================
        # Save each transformed granule for the current field
        for field, (field_DS, success) in zip(fields, field_DSs):
            output_filename = f'{grid_name}_{field.name}_{T.file_name[:-3]}.nc'
            output_filename = f'{grid_name}_{field.name}_{T.file_name}.nc'
            
            output_path = f'{OUTPUT_DIR}/{T.ds_name}/transformed_products/{grid_name}/transformed/{field.name}/'
            transformed_location = f'{output_path}{output_filename}'

            os.makedirs(output_path, exist_ok=True)

            # save field_DS
            records.save_netcdf(field_DS, output_filename, output_path)

            # Query Solr for transformation entry
            query_fq = [f'dataset_s:{T.ds_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field.name}', f'pre_transformation_file_path_s:"{source_file_path}"']

            doc_id = solr_utils.solr_query(query_fq)[0]['id']

            transformation_successes = transformation_successes and success
            transformation_file_paths[f'{grid_name}_{field.name}_transformation_file_path_s'] = transformed_location

            # Update Solr transformation entry with file paths and status
            update_body = [
                {
                    "id": doc_id,
                    "filename_s": {"set": output_filename},
                    "transformation_file_path_s": {"set": transformed_location},
                    "transformation_completed_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                    "transformation_in_progress_b": {"set": False},
                    "success_b": {"set": success},
                    "transformation_checksum_s": {"set": file_utils.md5(transformed_location)},
                    "transformation_version_f": {"set": T.transformation_version}
                }
            ]
            
            if success and 'Default empty model grid record' in field_DS.variables:
                update_body[0]['transformation_note'] = {"set": 'Field not found in source data. Defaulting to empty record.'}

            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code != 200:
                logger.exception(f'Failed to update Solr transformation entry for {field["name"]} in {T.ds_name} on {T.date}')

            if success and grid_name not in grids_updated:
                grids_updated.append(grid_name)

        logger.debug(f'CPU id {os.getpid()} saving {T.file_name} output file for grid {grid_name}')