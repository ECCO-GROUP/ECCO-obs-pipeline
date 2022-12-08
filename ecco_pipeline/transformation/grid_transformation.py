import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pyresample as pr
import xarray as xr
import netCDF4 as nc4
from utils.ecco_utils import ecco_functions, records, date_time, mapping
from utils import file_utils, solr_utils
from conf.global_settings import OUTPUT_DIR

np.warnings.filterwarnings('ignore')


class Transformation():

    def __init__(self, config: dict, source_file_path: str, remaining_transformations: dict):
        self.file_name = source_file_path.split('/')[-1].replace('.nc4', '.nc')
        self.dataset_name = config.get('ds_name')
        self.transformation_version = config.get('t_version')
        self.remaining_transformations = remaining_transformations
        self.data_res = self._compute_data_res(config)
        self.array_precision = getattr(np, config.get('array_precision'))
        self.binary_dtype = '>f4' if self.array_precision == np.float32 else '>f8'
        self.fill_values = {'binary': -9999,
                            'netcdf': nc4.default_fillvals[self.binary_dtype.replace('>', '')]}
        self.og_ds_metadata = {k: v for k,
                               v in config.items() if 'original' in k}
        self._set_granule_meta(source_file_path)
        self._ds_meta()

        ts_keys = ['extra_information', 'time_zone_included_with_time',
                   'pre_transformation_steps', 'post_transformation_steps']

    def _compute_data_res(self, config):
        res = config.get('data_res')
        if type(res) is str and '/' in res:
            num, den = res.replace(' ', '').split('/')
            res = float(num) / float(den)
        return res

    def _set_granule_meta(self, source_file_path):
        query_fq = [f'dataset_s:{self.dataset_name}', 'type_s:granule',
                    f'pre_transformation_file_path_s:"{source_file_path}"']
        granule_meta = solr_utils.solr_query(query_fq)[0]
        self.origin_checksum = granule_meta['checksum_s']
        self.date = granule_meta['date_s']
        self.hemi = f'_{granule_meta.get("hemisphere_s")}' if 'hemisphere_s' in granule_meta else ''

    def _ds_meta(self):
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:dataset']
        ds_meta = solr_utils.solr_query(fq)[0]
        self.ds_meta = ds_meta

    def get_factors_path(self, grid_factors):
        return self.ds_meta[grid_factors]

    @staticmethod
    def get_grids(grid_name):
        fq = ['type_s:grid', f'grid_name_s:{grid_name}']
        grid_metadata = solr_utils.solr_query(fq)[0]
        return grid_metadata['grid_path_s'], grid_metadata['grid_type_s']

    @staticmethod
    def execute_funcs(data_obj, transformations: List):
        for func_to_run in transformations:
            callable_func = getattr(ecco_functions, func_to_run)
            data_obj = callable_func(data_obj)
        return data_obj

    def transform(self, model_grid, grid_name, grid_type, fields, factors, ds, config):
        """
        Function that actually performs the transformations. Returns a list of transformed
        xarray datasets, one dataset for each field being transformed for the given grid.
        """
        logging.info(f'Transforming {self.date} to {grid_name}')

        record_date = self.date.replace('Z', '')
        record_file_name = ds.attrs['original_file_name']
        extra_information = config['extra_information']
        time_zone_included_with_time = config['time_zone_included_with_time']

        field_DSs = []

        pre_transformations = config.get('pre_transformation_steps', [])
        post_transformations = config.get('post_transformation_steps', [])

        # =====================================================
        # Pre transformation functions
        # =====================================================
        try:
            ds = self.execute_funcs(ds, pre_transformations)
        except Exception as e:
            logging.exception(f'Transformation failed: {e}')
            return []

        # =====================================================
        # Loop through fields to transform
        # =====================================================
        for data_field_info in fields:
            field_name = data_field_info['name']
            standard_name = data_field_info['standard_name']
            long_name = data_field_info['long_name']
            units = data_field_info['units']

            logging.debug(
                f'Transforming {record_file_name} for field {field_name}')

            try:
                field_DA = ecco_functions.generalized_transform_to_model_grid_solr(data_field_info, self.date, model_grid, grid_type,
                                                                                   self.array_precision, record_file_name,
                                                                                   self.ds_meta.get(
                                                                                       'data_time_scale_s'),
                                                                                   extra_information, ds, factors, time_zone_included_with_time,
                                                                                   grid_name)
                success = True
            except Exception as e:
                logging.exception(f'Transformation failed: {e}')
                field_DA = records.make_empty_record(standard_name, long_name, units,
                                                     record_date, model_grid,
                                                     grid_type, self.array_precision)
                success = False

            # =====================================================
            # Post transformation functions
            # =====================================================
            if success:
                try:
                    field_DA = self.execute_funcs(
                        field_DA, post_transformations)
                    field_DA.attrs['valid_min'] = np.nanmin(field_DA.values)
                    field_DA.attrs['valid_max'] = np.nanmax(field_DA.values)
                except Exception as e:
                    logging.exception(f'Post-transformation failed: {e}')
                    field_DA = records.make_empty_record(standard_name, long_name, units,
                                                         record_date, model_grid,
                                                         grid_type, self.array_precision)
                    success = False

            field_DA.values = np.where(
                np.isnan(field_DA.values), self.fill_values['netcdf'], field_DA.values)

            # Make dataarray into dataset
            field_DS = field_DA.to_dataset()

            ds_meta = {}

            # Dataset metadata
            if 'title' in model_grid:
                ds_meta['interpolated_grid'] = model_grid.title
            else:
                ds_meta['interpolated_grid'] = model_grid.name
            ds_meta['model_grid_type'] = grid_type
            ds_meta['original_dataset_title'] = self.og_ds_metadata.get(
                'original_dataset_title')
            ds_meta['original_dataset_short_name'] = self.og_ds_metadata.get(
                'original_dataset_short_name')
            ds_meta['original_dataset_url'] = self.og_ds_metadata.get(
                'original_dataset_url')
            ds_meta['original_dataset_reference'] = self.og_ds_metadata.get(
                'original_dataset_reference')
            ds_meta['original_dataset_doi'] = self.og_ds_metadata.get(
                'original_dataset_doi')
            ds_meta['interpolated_grid_id'] = grid_name
            ds_meta['transformation_version'] = self.transformation_version
            ds_meta['notes'] = config['notes']
            field_DS = field_DS.assign_attrs(ds_meta)

            # add time_bnds coordinate
            # [start_time, end_time] dimensions
            start_time = field_DS.time_start.values
            end_time = field_DS.time_end.values

            time_bnds = np.array(
                [start_time, end_time], dtype='datetime64')
            time_bnds = time_bnds.T
            field_DS = field_DS.assign_coords(
                {'time_bnds': (['time', 'nv'], time_bnds)})

            field_DS.time.attrs.update(bounds='time_bnds')

            field_DSs.append((field_DS, success))

        return field_DSs


def main(source_file_path, remaining_transformations, config):
    """
    Performs and saves locally all remaining transformations for a given source granule
    Updates Solr with transformation entries and updates descendants, and dataset entries
    """
    T = Transformation(config, source_file_path, remaining_transformations)

    transformation_successes = True
    transformation_file_paths = {}
    grids_updated = []

    # =====================================================
    # Load file to transform
    # =====================================================
    logging.debug(f'Loading {T.file_name} data')

    ds = xr.open_dataset(source_file_path, decode_times=True)
    ds.attrs['original_file_name'] = T.file_name

    # Iterate through grids in remaining_transformations
    for grid_name in T.remaining_transformations.keys():
        fields = remaining_transformations[grid_name]

        grid_path, grid_type = T.get_grids(grid_name)

        # =====================================================
        # Load grid
        # =====================================================
        logging.debug(f' - Loading {grid_name} model grid')
        model_grid = xr.open_dataset(grid_path).reset_coords()

        # =====================================================
        # Make model grid factors if not present locally
        # =====================================================
        grid_factors = f'{grid_name}{T.hemi}_factors_path_s'
        grid_factors_version = f'{grid_name}{T.hemi}_factors_version_f'

        # check to see if there is 'grid_factors_version' key in the
        # dataset and whether the transformation version matches with the
        # current version
        if T.ds_meta.get(grid_factors_version) and T.transformation_version == T.ds_meta.get(grid_factors_version):
            factors_path = T.get_factors_path(grid_factors)

            logging.debug(f' - Loading {grid_name} factors')
            with open(factors_path, "rb") as f:
                factors = pickle.load(f)

        else:
            logging.info(f'Creating {grid_name} factors for {T.dataset_name}')

            # Use hemisphere specific variables if data is hemisphere specific
            source_grid_min_L, source_grid_max_L, source_grid, \
                _, _ = ecco_functions.generalized_grid_product(T.ds_meta.get('short_name_s'), T.data_res, config[f'data_max_lat{T.hemi}'],
                                                               config[f'area_extent{T.hemi}'], config[f'dims{T.hemi}'],
                                                               config[f'proj_info{T.hemi}'])

            # Define the 'swath' as the lats/lon pairs of the model grid
            target_grid = pr.geometry.SwathDefinition(lons=model_grid.XC.values.ravel(),
                                                      lats=model_grid.YC.values.ravel())

            # Retrieve target_grid_radius from model_grid file
            if 'effective_grid_radius' in model_grid:
                target_grid_radius = model_grid.effective_grid_radius.values.ravel()
            elif 'effective_radius' in model_grid:
                target_grid_radius = model_grid.effective_radius.values.ravel()
            elif 'RAD' in model_grid:
                target_grid_radius = model_grid.RAD.values.ravel()
            elif 'rA' in model_grid:
                target_grid_radius = 0.5*np.sqrt(model_grid.rA.values.ravel())
            else:
                logging.exception(f'{grid_name} grid not supported')
                continue

            # Compute the mapping between the data and model grid
            source_indices_within_target_radius_i,\
                num_source_indices_within_target_radius_i,\
                nearest_source_index_to_target_index_i = \
                mapping.find_mappings_from_source_to_target(source_grid,
                                                            target_grid,
                                                            target_grid_radius,
                                                            source_grid_min_L,
                                                            source_grid_max_L)

            factors = (source_indices_within_target_radius_i,
                       num_source_indices_within_target_radius_i,
                       nearest_source_index_to_target_index_i)

            logging.debug(f' - Saving {grid_name} factors')
            factors_path = f'{OUTPUT_DIR}/{T.dataset_name}/transformed_products/{grid_name}/'

            # Create directory if needed and save factors
            if not os.path.exists(factors_path):
                os.makedirs(factors_path)

            factors_path += f'{grid_name}{T.hemi}_factors'

            with open(factors_path, 'wb') as f:
                pickle.dump(factors, f)

            logging.debug(' - Updating Solr with factors')

            # Update Solr dataset entry with factors metadata
            update_body = [
                {
                    "id": T.ds_meta.get('id'),
                    f'{grid_factors}': {"set": factors_path},
                    f'{grid_name}{T.hemi}_factors_stored_dt': {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                    f'{grid_factors_version}': {"set": T.transformation_version}
                }
            ]

            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code == 200:
                logging.debug(
                    'Successfully updated Solr with factors information')
            else:
                logging.exception(
                    'Failed to update Solr with factors information')

        update_body = []

        # Iterate through remaining transformation fields
        for field in fields:
            field_name = field["name"]

            # Query if grid/field combination transformation entry exists
            query_fq = [f'dataset_s:{T.dataset_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field_name}', f'pre_transformation_file_path_s:"{source_file_path}"']
            docs = solr_utils.solr_query(query_fq)
            update_body = []
            transform = {}

            # If grid/field combination transformation exists, update transformation status
            # Otherwise initialize new transformation entry
            if len(docs) > 0:
                # Reset status fields
                transform['id'] = docs[0]['id']
                transform['transformation_in_progress_b'] = {"set": True}
                transform['success_b'] = {"set": False}
                update_body.append(transform)
                r = solr_utils.solr_update(update_body, r=True)
            else:
                # Initialize new transformation entry
                transform['type_s'] = 'transformation'
                transform['date_s'] = T.date
                transform['dataset_s'] = T.dataset_name
                transform['pre_transformation_file_path_s'] = source_file_path
                if '_' in T.hemi:
                    transform['hemisphere_s'] = T.hemi[1:]
                transform['origin_checksum_s'] = T.origin_checksum
                transform['grid_name_s'] = grid_name
                transform['field_s'] = field_name
                transform['transformation_in_progress_b'] = True
                transform['success_b'] = False
                update_body.append(transform)
                r = solr_utils.solr_update(update_body, r=True)

            if r.status_code != 200:
                logging.exception(
                    f'Failed to update Solr transformation status for {T.dataset_name} on {T.date}')

        # =====================================================
        # Run transformation
        # =====================================================
        logging.debug(f'Running transformations for {T.file_name}')

        # Returns list of transformed DSs, one for each field in fields
        field_DSs = T.transform(model_grid, grid_name, grid_type, fields,
                                factors, ds, config)

        # =====================================================
        # Save the output in netCDF format
        # =====================================================

        # Save each transformed granule for the current field
        for field, (field_DS, success) in zip(fields, field_DSs):
            field_name = field["name"]

            # time stuff
            data_time_scale = T.ds_meta.get('data_time_scale_s')
            if data_time_scale == 'daily':
                output_freq_code = 'AVG_DAY'
                rec_end = field_DS.time_bnds.values[0][1]
            elif data_time_scale == 'monthly':
                output_freq_code = 'AVG_MON'
                cur_year = int(T.date[:4])
                cur_month = int(T.date[5:7])

                if cur_month < 12:
                    cur_mon_year = np.datetime64(str(cur_year) + '-' +
                                                 str(cur_month+1).zfill(2) +
                                                 '-' + str(1).zfill(2), 'ns')
                    # for december we go up one year, and set month to january
                else:
                    cur_mon_year = np.datetime64(str(cur_year+1) + '-' +
                                                 str('01') +
                                                 '-' + str(1).zfill(2), 'ns')
                rec_end = cur_mon_year

            if 'DEBIAS_LOCEAN' in T.dataset_name:
                rec_end = field_DS.time.values[0] + np.timedelta64(1, 'D')

            tb, ct = date_time.make_time_bounds_from_ds64(
                rec_end, output_freq_code)

            field_DS.time.values[0] = ct
            field_DS.time_bnds.values[0][0] = tb[0]
            field_DS.time_bnds.values[0][1] = tb[1]

            # field_DS.time_bnds.attrs['long_name'] = 'time bounds'

            field_DS = field_DS.drop('time_start')
            field_DS = field_DS.drop('time_end')

            output_filename = f'{grid_name}_{field_name}_{T.file_name}'
            output_path = f'{OUTPUT_DIR}/{T.dataset_name}/transformed_products/{grid_name}/transformed/{field_name}/'
            transformed_location = f'{output_path}{output_filename}'

            Path(output_path).mkdir(parents=True, exist_ok=True)

            # save field_DS
            records.save_to_disk(field_DS, output_filename[:-3], T.fill_values.get('binary'),
                                 T.fill_values.get(
                                     'netcdf'), Path(output_path),
                                 Path(output_path), T.binary_dtype, grid_type, save_binary=False)

            # Query Solr for transformation entry
            query_fq = [f'dataset_s:{T.dataset_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field_name}', f'pre_transformation_file_path_s:"{source_file_path}"']

            docs = solr_utils.solr_query(query_fq)
            doc_id = solr_utils.solr_query(query_fq)[0]['id']

            transformation_successes = transformation_successes and success
            transformation_file_paths[f'{grid_name}_{field_name}_transformation_file_path_s'] = transformed_location

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

            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code != 200:
                logging.exception(
                    f'Failed to update Solr transformation entry for {field["name"]} in {T.dataset_name} on {T.date}')

            if success and grid_name not in grids_updated:
                grids_updated.append(grid_name)

        logging.debug(
            f'CPU id {os.getpid()} saving {T.file_name} output file for grid {grid_name}')

    return grids_updated, T.date[:4]
