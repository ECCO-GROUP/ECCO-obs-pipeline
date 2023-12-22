import os
from pathlib import Path
from glob import glob
import logging
import requests
import netrc
from utils import log_config, solr_utils, grids_to_solr
from utils.config_validator import validate_configs

try:
    from conf.global_settings import OUTPUT_DIR, SOLR_COLLECTION, GRIDS
    import conf.global_settings as global_settings
except ImportError:
    raise ImportError('Missing global_settings.py file. See ecco_pipeline/conf/global_settings.py.example for more info.')

def setup_logger(args):
    filename = log_config.configure_logging(True, args.log_level, args.wipe_logs)

    # Set package logging level to WARNING
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return filename

def validate_output_dir():
    # Verify output directory is valid
    if not Path.is_dir(OUTPUT_DIR):
        logging.fatal('Missing output directory. Please fill in. Exiting.')
        exit()
    logging.debug(f'Using output directory: {OUTPUT_DIR}')


def validate_solr():
    logging.debug(f'Using Solr collection: {SOLR_COLLECTION}')

    # Verify solr is running
    try:
        solr_utils.ping_solr()
    except requests.ConnectionError:
        logging.fatal('Solr is not currently running! Start Solr and try again.')
        exit()

    if not solr_utils.core_check():
        logging.fatal(
            f'Solr core {SOLR_COLLECTION} does not exist. Add a core using "bin/solr create -c {{collection_name}}".')
        exit()


def wipe_factors():
    logging.info('Removing all factors')
    all_factors = glob(f'{OUTPUT_DIR}/**/transformed_products/**/*_factors')
    for factors_file in all_factors:
        try:
            os.remove(factors_file)
        except:
            logging.error(f'Error removing {factors_file}')
        logging.info('Successfully removed all factors')


def validate_netrc():
    try: 
        nrc = netrc.netrc()
        if 'urs.earthdata.nasa.gov' not in nrc.hosts.keys():
            logging.fatal('Earthdata login required in netrc file.')
            exit()
    except FileNotFoundError:
        logging.fatal('No netrc found. Please create one and add Earthdata login credentials.')
        exit()

def init_pipeline(args):
    log_filename = setup_logger(args)
    validate_output_dir()
    validate_solr()
    validate_configs()
    validate_netrc()

    if args.harvested_entry_validation:
        solr_utils.validate_granules()

    if args.wipe_transformations:
        logging.info('Removing transformations with out of sync version numbers from Solr and disk')
        solr_utils.delete_mismatch_transformations()
        pass

    if isinstance(args.grids_to_use, list):
        grids_to_use = args.grids_to_use
    else:
        grids_to_use = GRIDS

    if args.grids_to_solr or solr_utils.check_grids():
        try:
            grids_not_in_solr = []
            grids_not_in_solr = grids_to_solr.main(grids_to_use)
            if grids_not_in_solr:
                for name in grids_not_in_solr:
                    logging.exception(
                        f'Grid "{name}" not in Solr. Ensure it\'s file name is present in grids_config.yaml and run pipeline with the --grids_to_solr argument')
                exit()
            logging.info('Successfully updated grids on Solr.')
        except Exception as e:
            logging.exception(e)

    if args.wipe_factors:
        wipe_factors()

    user_cpus = args.multiprocesses
    logging.debug(f'Using {user_cpus} processes for multiprocess transformations')

    return grids_to_use, user_cpus, log_filename