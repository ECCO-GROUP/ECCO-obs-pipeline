import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr
from utils.ecco_utils import ecco_functions, records, date_time
from utils import file_utils, solr_utils
from conf.global_settings import OUTPUT_DIR

from transformations.transformation import Transformation


def get_grids(grid_name):
    fq = ['type_s:grid', f'grid_name_s:{grid_name}']
    grid_metadata = solr_utils.solr_query(fq)[0]
    return grid_metadata['grid_path_s'], grid_metadata['grid_type_s']


def transform(source_file_path, remaining_transformations, config, granule_date):
    """
    Performs and saves locally all remaining transformations for a given source granule
    Updates Solr with transformation entries and updates descendants, and dataset entries
    """
    T = Transformation(config, source_file_path, granule_date)

    transformation_successes = True
    transformation_file_paths = {}
    grids_updated = []

    # =====================================================
    # Load file to transform
    # =====================================================
    logging.debug(f'Loading {T.file_name} data')

    if 'preprocessing' in config:
        preprocessing_func = config['preprocessing']
        callable_func = getattr(ecco_functions, preprocessing_func)
        ds = callable_func(source_file_path, config)
    else:
        ds = xr.open_dataset(source_file_path, decode_times=True)
    ds.attrs['original_file_name'] = T.file_name

    # Iterate through grids in remaining_transformations
    for grid_name in remaining_transformations.keys():
        fields = remaining_transformations[grid_name]

        grid_path, grid_type = get_grids(grid_name)

        # =====================================================
        # Load grid
        # =====================================================
        logging.debug(f' - Loading {grid_name} model grid')
        model_grid = xr.open_dataset(grid_path).reset_coords()

        # =====================================================
        # Make model grid factors if not present locally
        # =====================================================
        factors = T.make_factors(model_grid)

        # Iterate through remaining transformation fields
        for field in fields:
            logging.debug(f'Transforming {field}')
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
            else:
                # Query for granule entry to get checksum
                query_fq = [f'dataset_s:{T.dataset_name}', 'type_s:granule',
                            f'pre_transformation_file_path_s:"{source_file_path}"']
                docs = solr_utils.solr_query(query_fq)

                # Initialize new transformation entry
                transform['type_s'] = 'transformation'
                transform['date_s'] = T.date
                transform['dataset_s'] = T.dataset_name
                transform['pre_transformation_file_path_s'] = source_file_path
                transform['hemisphere_s'] = T.hemi.replace('_', '')
                transform['origin_checksum_s'] = docs[0]['checksum_s']
                transform['grid_name_s'] = grid_name
                transform['field_s'] = field_name
                transform['transformation_in_progress_b'] = True
                transform['success_b'] = False

            update_body.append(transform)
            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code != 200:
                logging.exception(f'Failed to update Solr transformation status for {T.dataset_name} on {T.date}')

        # =====================================================
        # Run transformation
        # =====================================================
        logging.debug(f'Running transformations for {T.file_name}')

        # Returns list of transformed DSs, one for each field in fields
        field_DSs = T.transform(model_grid, factors, ds, config)
        # =====================================================
        # Save the output in netCDF format
        # =====================================================
        # Save each transformed granule for the current field
        for field, (field_DS, success) in zip(fields, field_DSs):
            field_name = field["name"]
            output_filename = f'{grid_name}_{field_name}_{T.file_name[:-3]}.nc'
            output_filename = f'{grid_name}_{field_name}_{T.file_name}.nc'
            
            output_path = f'{OUTPUT_DIR}/{T.dataset_name}/transformed_products/{grid_name}/transformed/{field_name}/'
            transformed_location = f'{output_path}{output_filename}'

            os.makedirs(output_path, exist_ok=True)

            # save field_DS
            records.save_to_disk(field_DS, output_filename[:-3], T.fill_values.get('binary'),
                                 T.fill_values.get('netcdf'), Path(output_path),
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

        logging.debug(f'CPU id {os.getpid()} saving {T.file_name} output file for grid {grid_name}')

    return grids_updated, T.date[:4]