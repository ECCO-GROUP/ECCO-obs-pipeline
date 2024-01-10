import logging
from multiprocessing import current_process
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable
from requests import HTTPError
import xarray as xr

from field import Field
from utils.ecco_utils import ecco_functions, records
from utils import file_utils, solr_utils
from conf.global_settings import OUTPUT_DIR
from transformations.transformation import Transformation

logger = logging.getLogger(str(current_process().pid))


def load_file(source_file_path: str, T: Transformation) -> xr.Dataset:
    if T.preprocessing_function:
        callable_func = getattr(ecco_functions, T.preprocessing_function)
        ds = callable_func(source_file_path, T)
    else:
        ds = xr.open_dataset(source_file_path, decode_times=True)
    ds.attrs['original_file_name'] = T.file_name
    return ds

def prepopulate_solr(T: Transformation, source_file_path: str, grid_name: str):
    '''
    Populate Solr with transformation entries prior to attempting transformation
    '''
    update_body = []
    for field in T.fields:
        logger.debug(f'Transforming {field.name}')

        # Query if grid/field combination transformation entry exists
        query_fq = [f'dataset_s:{T.ds_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
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
            query_fq = [f'dataset_s:{T.ds_name}', 'type_s:granule',
                        f'pre_transformation_file_path_s:"{source_file_path}"']
            docs = solr_utils.solr_query(query_fq)

            # Initialize new transformation entry
            transform['type_s'] = 'transformation'
            transform['date_s'] = T.date
            transform['dataset_s'] = T.ds_name
            transform['pre_transformation_file_path_s'] = source_file_path
            transform['hemisphere_s'] = T.hemi.replace('_', '')
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
        logger.exception(f'Failed to update Solr transformation status for {T.ds_name} on {T.date}')
        raise HTTPError


def transform(source_file_path, remaining_transformations, config, granule_date, loaded_factors, loaded_grids):
    """
    Performs and saves locally all remaining transformations for a given source granule
    Updates Solr with transformation entries and updates descendants, and dataset entries
    """    
    T = Transformation(config, source_file_path, granule_date)

    transformation_successes = True
    transformation_file_paths = {}
    grids_updated = []

    logger.debug(f'Loading {T.file_name} data')
    ds = load_file(source_file_path, T)
    
    grid_fields = [[f'({grid_name}, {field})' for field in remaining_transformations[grid_name]] for grid_name in remaining_transformations.keys()]
    logger.debug(f'{T.file_name} needs to transform: {grid_fields} ')

    # Iterate through grids in remaining_transformations
    for grid_name in remaining_transformations.keys():
        fields: Iterable[Field] = remaining_transformations[grid_name]

        logger.debug(f'Loading {grid_name} model grid')
        grid_ds = getattr(loaded_grids, grid_name).reset_coords()

        # =====================================================
        # Pull factors from preloaded object
        # =====================================================
        factors_file = f'{grid_ds.name}{T.hemi}_v{T.transformation_version}_factors'
        factors = getattr(loaded_factors, factors_file)

        prepopulate_solr(T, source_file_path, grid_name)

        # =====================================================
        # Run transformation
        # =====================================================
        logger.debug(f'Running transformations for {T.file_name}')

        # Returns list of transformed DSs, one for each field in fields
        field_DSs = T.transform(grid_ds, factors, ds, fields, config)
            
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
            records.save_netcdf(field_DS, output_filename[:-3], T.fill_values.get('netcdf'), Path(output_path))

            # Query Solr for transformation entry
            query_fq = [f'dataset_s:{T.ds_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field.name}', f'pre_transformation_file_path_s:"{source_file_path}"']

            docs = solr_utils.solr_query(query_fq)
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