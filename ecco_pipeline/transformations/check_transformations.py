import logging
from collections import defaultdict
from multiprocessing import current_process, Pool
import os
from typing import Iterable
import xarray as xr

from dataset import Dataset
from utils import solr_utils, log_config
from transformations.grid_transformation import transform
from transformations.transformation import Transformation
import utils.log_config as log_config


logger = logging.getLogger('pipeline')


def need_to_update(granule, tx, ds_tx_version):
    '''
    Triple if:
    1. do we have a version entry,
    2. compare transformation version number and current transformation version number
    3. compare checksum of harvested file (currently in solr) and checksum
       of the harvested file that was previously transformed (recorded in transformation entry)
    '''
    if tx.get('success_b') and tx.get('transformation_version_f') == ds_tx_version and tx['origin_checksum_s'] == granule['checksum_s']:
        return False
    return True

def get_tx_jobs(harvested_granules: Iterable, dataset: Dataset, grids: Iterable):
    fq = [f'dataset_s:{dataset.ds_name}', 'type_s:transformation']
    solr_txs = solr_utils.solr_query(fq)
    tx_dict = defaultdict(list)
    for tx in solr_txs:
        tx_dict[tx['pre_transformation_file_path_s'].split('/')[-1]].append(tx)
    
    all_jobs = []
    for granule in harvested_granules:
        grid_fields = {}
        for grid in grids:
            fields_for_grid = []
            for field in dataset.fields:
                update = True
                for tx in tx_dict[granule['filename_s']]:
                    if tx['field_s'] == field.name and tx['grid_name_s'] == grid:
                        update = need_to_update(granule, tx, dataset.t_version)
                        break
                if update:
                    fields_for_grid.append(field)
            if fields_for_grid:
                grid_fields[grid] = fields_for_grid
        if grid_fields:
            all_jobs.append((granule, grid_fields))
    return all_jobs

def multiprocess_transformation(config: dict, granule: dict, tx_jobs: dict, log_level: str, log_dir: str):
    """
    Callable function that performs the actual transformation on a granule.
    """
    try:
        logger = log_config.mp_logging(str(current_process().pid), log_level, log_dir)
    except Exception as e:
        print(e)

    granule_filepath = granule.get('pre_transformation_file_path_s')
    granule_date = granule.get('date_s')

    # Skips granules that weren't harvested properly
    if not granule_filepath or granule.get('file_size_l') < 100:
        logger.exception(f'Granule {granule_filepath} was not harvested properly. Skipping.')
        return

    # Perform remaining transformations
    try:
        logger.info(f'{sum([len(v) for v in tx_jobs.values()])} remaining transformations for {granule_filepath.split("/")[-1]}')
        transform(granule_filepath, tx_jobs, config, granule_date)
    except Exception as e:
        logger.exception(f'Error transforming {granule_filepath}: {e}')


def find_data_for_factors(config: dict, harvested_granules: Iterable[dict]) -> Iterable[dict]:
    '''
    Returns Solr granule entry (two in the case of hemispherical data) to be used
    to generate factors
    '''
    data_for_factors = []
    nh_added = False
    sh_added = False
    # Find appropriate granule(s) to use for factor calculation
    for granule in harvested_granules:
        if 'hemisphere_s' in granule.keys():
            hemi = f'_{granule["hemisphere_s"]}_'
        else:
            hemi = ''
        if granule.get('pre_transformation_file_path_s'):
            if hemi:
                # Get one of each
                if hemi == config['hemi_pattern']['north'] and not nh_added:
                    data_for_factors.append(granule)
                    nh_added = True
                elif hemi == config['hemi_pattern']['south'] and not sh_added:
                    data_for_factors.append(granule)
                    sh_added = True
                if nh_added and sh_added:
                    return data_for_factors
            else:
                data_for_factors.append(granule)
                return data_for_factors


def pregenerate_factors(config: dict, grids: Iterable[str], harvested_granules: Iterable[dict]):
    '''
    Generates factors for all grids used for the given transformation version. Loads them into
    Factors object which is used to reduce I/O.
    '''
    for grid in grids:
        # loaded_grids.set_grid(f'grids/{grid}.nc')
        for granule in find_data_for_factors(config, harvested_granules):
            grid_ds = xr.open_dataset(f'grids/{grid}.nc')
            T = Transformation(config, granule['pre_transformation_file_path_s'], '1972-01-01')
            T.make_factors(grid_ds)
            
            # factors_file = f'{grid_ds.name}{T.hemi}_v{T.transformation_version}_factors'
            # factors_path = os.path.join(OUTPUT_DIR, T.ds_name, 'transformed_products', grid_ds.name, factors_file)
            
            # loaded_factors.set_factors(factors_path)


def generate_jobs(config: dict, harvested_granules: Iterable, dataset: Dataset, grids: Iterable):
    logger.info('Generating jobs...')
    log_level = logging.getLevelName(logging.getLogger('pipeline').level)
    log_dir = os.path.dirname(logging.getLogger('pipeline').handlers[0].baseFilename)
    log_dir = os.path.join(log_dir[log_dir.find('logs/'):], f'tx_{dataset.ds_name}')
    
    all_jobs = get_tx_jobs(harvested_granules, dataset, grids)

    new_jobs = []
    for (granule, grid_fields) in all_jobs:
        job_params = (config, granule, grid_fields, log_level, log_dir)
        new_jobs.append(job_params)
        
    # jobs = []
    # for granule in harvested_granules:
    #     granule_filepath = granule.get('pre_transformation_file_path_s')
    #     remaining_transformations = get_remaining_transformations(dataset, granule_filepath, grids)
    #     print(granule['filename_s'], remaining_transformations)
    #     if remaining_transformations:
    #         job_params = (config, granule, remaining_transformations, log_level, log_dir)
    #         jobs.append(job_params)
    #     else:
    #         logger.debug(f'No remaining transformations for {granule_filepath.split("/")[-1]}')
    return new_jobs


def main(config: dict, user_cpus: int = 1, grids_to_use: Iterable[str]=[]) -> str:
    """
    This function performs all remaining grid/field transformations for all harvested
    granules for a dataset. It also makes use of multiprocessing to perform multiple
    transformations at the same time. After all transformations have been attempted,
    the Solr dataset entry is updated with additional metadata.
    """
    dataset = Dataset(config)

    # Get all harvested granules for this dataset
    fq = [f'dataset_s:{dataset.ds_name}', 'type_s:granule', 'harvest_success_b:true']
    harvested_granules = solr_utils.solr_query(fq)

    if not harvested_granules:
        logger.info(f'No harvested granules found in solr for {dataset.ds_name}')
        return f'No transformations performed'

    # Query for grids
    if not grids_to_use:
        fq = ['type_s:grid']
        docs = solr_utils.solr_query(fq)
        grids = [doc['grid_name_s'] for doc in docs]
    else:
        grids = grids_to_use
        
    pregenerate_factors(config, grids, harvested_granules)

    job_params = generate_jobs(config, harvested_granules, dataset, grids)
    logger.info(f'{len(job_params)} harvested granules with remaining transformations.')
    if job_params:
        # BEGIN MULTIPROCESSING
        if user_cpus == 1:
            logger.info('Not using multiprocessing to do transformation')
            for job_param in job_params:
                multiprocess_transformation(*job_param)
        else:
            logger.info(f'Using {user_cpus} CPUs to do multiprocess transformation')
                
            with Pool(processes=user_cpus) as pool:
                pool.starmap_async(multiprocess_transformation, job_params)
                pool.close()
                pool.join()
                
    # Query Solr for dataset metadata
    fq = [f'dataset_s:{dataset.ds_name}', 'type_s:dataset']
    dataset_metadata = solr_utils.solr_query(fq)[0]

    # Query Solr for successful transformation documents
    fq = [f'dataset_s:{dataset.ds_name}', 'type_s:transformation', 'success_b:true']
    successful_transformations = solr_utils.solr_query(fq)

    # Query Solr for failed transformation documents
    fq = [f'dataset_s:{dataset.ds_name}', 'type_s:transformation', 'success_b:false']
    failed_transformations = solr_utils.solr_query(fq)

    transformation_status = f'All transformations successful'

    if not successful_transformations and not failed_transformations:
        transformation_status = f'No transformations performed'
    elif not successful_transformations:
        transformation_status = f'No successful transformations'
    elif failed_transformations:
        transformation_status = f'{len(failed_transformations)} transformations failed'

    # Update Solr dataset entry status to transformed
    update_body = [{
        "id": dataset_metadata['id'],
        "transformation_status_s": {"set": transformation_status},
    }]

    r = solr_utils.solr_update(update_body, r=True)

    if r.status_code == 200:
        logger.debug(f'Successfully updated Solr with transformation information for {dataset.ds_name}')
    else:
        logger.exception(f'Failed to update Solr with transformation information for {dataset.ds_name}')

    return transformation_status