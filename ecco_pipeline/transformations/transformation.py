import logging
import os
from typing import List, Tuple
import numpy as np
import netCDF4 as nc4
from datetime import datetime
import xarray as xr
import pyresample as pr
import pickle
from conf.global_settings import OUTPUT_DIR


from utils.ecco_utils import ecco_functions, records, date_time
from utils import file_utils


class Transformation():

    def __init__(self, config: dict, source_file_path: str, granule_date: str):
        self.file_name = os.path.splitext(source_file_path.split('/')[-1])[0]
        self.dataset_name = config.get('ds_name')
        self.transformation_version = config.get('t_version')
        self.date = granule_date
        self.hemi = self._get_hemi(config)

        self.array_precision = getattr(np, config.get('array_precision'))
        self.binary_dtype = '>f4' if self.array_precision == np.float32 else '>f8'
        self.fill_values = {'binary': -9999, 'netcdf': nc4.default_fillvals[self.binary_dtype.replace('>', '')]}

        self.og_ds_metadata = {k: v for k, v in config.items() if 'original' in k}
        self.data_time_scale = config.get('data_time_scale')

        # Projection information
        self.data_res = self._compute_data_res(config)
        self.area_extent = config.get(f'area_extent{self.hemi}')
        self.dims = config.get(f'dims{self.hemi}')
        self.proj_info = config.get(f'proj_info{self.hemi}')

        # Processing information
        self.time_bounds_var = config.get('time_bounds_var', None)
        self.transpose = config.get('transpose', False)

        self.mapping_operation = config.get('mapping_operation', 'mean')

    def _compute_data_res(self, config):
        '''

        '''
        res = config.get('data_res')
        if type(res) is str and '/' in res:
            num, den = res.replace(' ', '').split('/')
            res = float(num) / float(den)
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

    def apply_funcs(self, data_object, funcs: List):
        for func_to_run in funcs:
            logging.info(f'Applying {func_to_run} to data')
            try:
                callable_func = getattr(ecco_functions, func_to_run)
                data_object = callable_func(data_object)
                logging.debug(f'{func_to_run} successfully ran')
            except:
                logging.exception(f'{func_to_run} failed to run')
                raise Exception(f'{func_to_run} failed to run')
        return data_object

    def make_factors(self, grid_ds: xr.Dataset) -> Tuple[dict, np.ndarray, dict]:
        '''
        Generate mappings from source to target grid

        Returns Tuple
        (source_indices_within_target_radius_i,
        num_source_indices_within_target_radius_i,
        nearest_source_index_to_target_index_i)

        '''
        grid_name = grid_ds.name
        factors_dir = f'{OUTPUT_DIR}/{self.dataset_name}/transformed_products/{grid_name}/'
        factors_file = f'{grid_name}{self.hemi}_v{self.transformation_version}_factors'
        factors_path = f'{factors_dir}{factors_file}'

        if os.path.exists(factors_path):
            logging.debug(f' - Loading {grid_name} factors')
            with open(factors_path, "rb") as f:
                factors = pickle.load(f)
                return factors
        else:
            logging.info(f'Creating {grid_name} factors for {self.dataset_name}')

        # Use hemisphere specific variables if data is hemisphere specific
        source_grid_min_L, source_grid_max_L, source_grid = ecco_functions.generalized_grid_product(self.data_res, self.area_extent,
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
            logging.exception(f'Unable to extract grid radius from {grid_ds.name}. Grid not supported')

        factors = (ecco_functions.find_mappings_from_source_to_target(source_grid, target_grid, target_grid_radius,
                                                                      source_grid_min_L, source_grid_max_L))
        logging.debug(f' - Saving {grid_name} factors')
        os.makedirs(factors_dir, exist_ok=True)
        with open(factors_path, 'wb') as f:
            pickle.dump(factors, f)
        return factors

    def transform(self, model_grid: xr.Dataset, factors: Tuple, ds: xr.Dataset, config: dict) -> List[Tuple[xr.Dataset, bool]]:
        """
        Function that actually performs the transformations. Returns a list of transformed
        xarray datasets, one dataset for each field being transformed for the given grid.
        """
        logging.info(f'Transforming {self.date} to {model_grid.name}')

        record_date = self.date.replace('Z', '')

        field_DSs = []

        fields = config.get('fields')

        # =====================================================
        # Loop through fields to transform
        # =====================================================
        for data_field_info in fields:
            field_name = data_field_info['name']
            standard_name = data_field_info['standard_name']
            long_name = data_field_info['long_name']
            units = data_field_info['units']
            pre_transformations = data_field_info.get('pre_transformations', [])
            try:
                self.apply_funcs(ds, pre_transformations)
            except Exception as e:
                logging.exception(e)

            post_transformations = data_field_info.get('post_transformations', [])

            logging.debug(f'Transforming {self.file_name} for field {field_name}')

            try:
                field_DA = ecco_functions.perform_mapping(self, ds, factors, data_field_info, model_grid)
                success = True
            except Exception as e:
                logging.exception(f'Transformation failed: {e}')
                field_DA = records.make_empty_record(record_date, model_grid, self.array_precision)
                field_DA.attrs['long_name'] = long_name
                field_DA.attrs['standard_name'] = standard_name
                field_DA.attrs['units'] = units
                success = False

            # =====================================================
            # Post transformation functions
            # =====================================================
            if success:
                try:
                    field_DA = self.apply_funcs(field_DA, post_transformations)
                    field_DA.attrs['valid_min'] = np.nanmin(field_DA.values)
                    field_DA.attrs['valid_max'] = np.nanmax(field_DA.values)
                except Exception as e:
                    logging.exception(f'Post-transformation failed: {e}')
                    field_DA = records.make_empty_record(record_date, model_grid, self.array_precision)
                    field_DA.attrs['long_name'] = long_name
                    field_DA.attrs['standard_name'] = standard_name
                    field_DA.attrs['units'] = units
                    success = False

            field_DA.values = np.where(np.isnan(field_DA.values), self.fill_values['netcdf'], field_DA.values)

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
                'notes': config['notes']
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
            data_time_scale = config.get('data_time_scale')
            if data_time_scale == 'daily':
                output_freq_code = 'AVG_DAY'
                rec_end = field_DS.time_bnds.values[0][1]
            elif data_time_scale == 'monthly':
                output_freq_code = 'AVG_MON'
                cur_year = int(self.date[:4])
                cur_month = int(self.date[5:7])

                if cur_month < 12:
                    cur_mon_year = np.datetime64(f'{cur_year}-{str(cur_month+1).zfill(2)}-01', 'ns')

                    # for december we go up one year, and set month to january
                else:
                    cur_mon_year = np.datetime64(f'{cur_year+1}-01-01', 'ns')

                rec_end = cur_mon_year

            if 'DEBIAS_LOCEAN' in self.dataset_name:
                rec_end = field_DS.time.values[0] + np.timedelta64(1, 'D')

            tb, ct = date_time.make_time_bounds_from_ds64(rec_end, output_freq_code)

            field_DS.time.values[0] = ct
            field_DS.time_bnds.values[0][0] = tb[0]
            field_DS.time_bnds.values[0][1] = tb[1]

            # field_DS.time_bnds.attrs['long_name'] = 'time bounds'

            field_DS = field_DS.drop('time_start')
            field_DS = field_DS.drop('time_end')
            # print(field_DS)
            field_DSs.append((field_DS, success))

        return field_DSs