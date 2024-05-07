from argparse import Namespace
import os
from glob import glob
import logging
from typing import Iterable
import requests
import netrc
import xarray as xr
from datetime import datetime
from utils.pipeline_utils import solr_utils, config_validator, log_config, file_utils, status_report

try:
    import conf.global_settings as global_settings
    from conf.global_settings import OUTPUT_DIR, SOLR_COLLECTION, GRIDS
except ImportError:
    raise ImportError('Missing global_settings.py file. See ecco_pipeline/conf/global_settings.py.example for more info.')

def setup_logger(args):
    logger = log_config.mp_logging('pipeline', args.log_level)
    # Set package logging level to WARNING
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger

def validate_output_dir():
    logger = logging.getLogger('pipeline')
    # Verify output directory is valid
    if not os.path.isdir(OUTPUT_DIR):
        logger.fatal('Missing output directory. Please fill in. Exiting.')
        exit()
    logger.debug(f'Using output directory: {OUTPUT_DIR}')


def validate_solr():
    logger = logging.getLogger('pipeline')
    
    logger.debug(f'Using Solr collection: {SOLR_COLLECTION}')

    # Verify solr is running
    try:
        solr_utils.ping_solr()
    except requests.ConnectionError:
        logger.fatal('Solr is not currently running! Start Solr and try again.')
        exit()

    if not solr_utils.core_check():
        logger.fatal(
            f'Solr core {SOLR_COLLECTION} does not exist. Add a core using "bin/solr create -c {{collection_name}}".')
        exit()


def wipe_factors():
    logger = logging.getLogger('pipeline')
    
    logger.info('Removing all factors')
    all_factors = glob(f'{OUTPUT_DIR}/**/transformed_products/**/*_factors')
    for factors_file in all_factors:
        try:
            os.remove(factors_file)
        except:
            logger.error(f'Error removing {factors_file}')
        logger.info('Successfully removed all factors')


def validate_netrc():
    logger = logging.getLogger('pipeline')
    
    try: 
        nrc = netrc.netrc()
        if 'urs.earthdata.nasa.gov' not in nrc.hosts.keys():
            logger.fatal('Earthdata login required in netrc file.')
            exit()
    except FileNotFoundError:
        logger.fatal('No netrc found. Please create one and add Earthdata login credentials.')
        exit()
        
def update_solr_grid(grid_name:str, grid_type:str, grid_file_path:str):
    '''
    Update solr grid docs with latest versions of grids
    '''
    logger = logging.getLogger('pipeline')
    
    fq = ['type_s:grid']
    grid_docs = solr_utils.solr_query(fq)

    update_body = []

    grids_in_solr = [doc['grid_name_s'] for doc in grid_docs]

    # Add grid to solr
    if grid_name not in grids_in_solr:
        grid_meta = {
            'type_s': 'grid',
            'grid_type_s': grid_type,
            'grid_name_s': grid_name,
            'grid_path_s': str(grid_file_path),
            'date_added_dt': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'grid_checksum_s': file_utils.md5(grid_file_path)
        }
        update_body.append(grid_meta)
    # Verify grid in solr matches grid file
    else:
        current_checksum = file_utils.md5(grid_file_path)

        for doc in grid_docs:
            if doc['grid_name_s'] == grid_name:
                solr_checksum = doc['grid_checksum_s']

        if current_checksum != solr_checksum:
            # Delete previous grid's transformations from Solr
            update_body = {
                "delete": {
                    "query": f'type_s:transformation AND grid_name_s:{grid_name}'
                }
            }

            r = solr_utils.solr_update(update_body, r=True)
            if r.status_code == 200:
                logger.debug(f'Successfully deleted Solr transformation documents for {grid_name}')
            else:
                logger.exception(f'Failed to delete Solr transformation documents for {grid_name}')

            # Delete previous grid's aggregations from Solr
            update_body = {
                "delete": {
                    "query": f'type_s:aggregation AND grid_name_s:{grid_name}'
                }
            }

            r = solr_utils.solr_update(update_body, r=True)
            if r.status_code == 200:
                logger.debug(f'Successfully deleted Solr aggregation documents for {grid_name}')
            else:
                logger.exception(f'Failed to delete Solr aggregation documents for {grid_name}')

            # Update grid on Solr
            fq = [f'grid_name_s:{grid_name}', 'type_s:grid']
            grid_metadata = solr_utils.solr_query(fq)[0]

            update_body = [
                {
                    "id": grid_metadata['id'],
                    "grid_type_s": {"set": grid_type},
                    "grid_name_s": {"set": grid_name},
                    "grid_checksum_s": {"set": current_checksum},
                    "date_added_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
                }
            ]
        else:
            return

    r = solr_utils.solr_update(update_body, r=True)
    if r.status_code == 200:
        logger.debug(f'Successfully updated {grid_name} Solr grid document')
    else:
        logger.error(f'Failed to update Solr {grid_name} grid document')


def grids_to_solr(grids_to_use: Iterable[str]=[]):
    logger = logging.getLogger('pipeline')

    grid_files = glob('grids/*.nc')
    
    grids = []
    for grid_file_path in grid_files:
        grid_file_name = os.path.basename(grid_file_path)
        ds = xr.open_dataset(grid_file_path)
        # Assumes grids conform to metadata standard (see documentation)
        grid_name = ds.attrs['name']
        grid_type = ds.attrs['type']
        grids.append((grid_name, grid_type, grid_file_name))
        logger.debug(f'Loaded {grid_name} {grid_type} {grid_file_name}')

    # Create Solr grid-type document for each missing grid type
    for grid_name, grid_type, grid_file_name in grids:
        logger.debug(f'Uploading solr grid {grid_name} {grid_type} {grid_file_name}')
        update_solr_grid(grid_name, grid_type, grid_file_path)

    # Verify grid names supplied exist on Solr
    grids_not_in_solr = []
    logger.debug(f'Grids to use: {grids_to_use}')
    for grid_name in grids_to_use:
        fq = ['type_s:grid', f'grid_name_s:{grid_name}']
        docs = solr_utils.solr_query(fq)
        if not docs:
            grids_not_in_solr.append(grid_name)

    return grids_not_in_solr

def init_pipeline(args: Namespace):
    if args.status_report:
        status_report.generate_reports()
        exit()
    
    logger = setup_logger(args)
    validate_output_dir()
    validate_solr()
    config_validator.validate_configs()
    validate_netrc()
    
    if args.harvested_entry_validation:
        solr_utils.validate_granules()

    if args.wipe_transformations:
        logger.info('Removing transformations with out of sync version numbers from Solr and disk')
        solr_utils.delete_mismatch_transformations()
        pass

    if isinstance(args.grids_to_use, list):
        grids_to_use = args.grids_to_use
    else:
        grids_to_use = GRIDS

    if args.grids_to_solr or solr_utils.check_grids():
        try:
            grids_not_in_solr = []
            grids_not_in_solr = grids_to_solr(grids_to_use)
            if grids_not_in_solr:
                for name in grids_not_in_solr:
                    logger.exception(
                        f'Grid "{name}" not in Solr. Ensure it\'s file name is present in grids_config.yaml and run pipeline with the --grids_to_solr argument')
                exit()
            logger.info('Successfully updated grids on Solr.')
        except Exception as e:
            logger.exception(e)

    if args.wipe_factors:
        wipe_factors()

    user_cpus = args.multiprocesses
    logger.debug(f'Using {user_cpus} processes for multiprocess transformations')

    return grids_to_use, user_cpus