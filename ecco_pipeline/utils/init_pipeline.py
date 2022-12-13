from pathlib import Path
import logging
import requests
from conf.global_settings import OUTPUT_DIR, SOLR_COLLECTION, GRIDS

from utils import log_config, solr_utils, grids_to_solr

def setup_logger(args):
    log_config.configure_logging(False, args.log_level)

    # Set package logging level to WARNING
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

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
        logging.fatal(f'Solr core {SOLR_COLLECTION} does not exist. Add a core using "bin/solr create -c {{collection_name}}".')
        exit()

def init_pipeline(args):
    print(' === init pipeline === ')
    print(' --- setup logger --- ')
    setup_logger(args)
    print(' --- validate output dir --- ')
    validate_output_dir()
    print(' --- validate solr --- ')
    validate_solr()

    if args.harvested_entry_validation:
        print(' --- validate granules --- ')
        solr_utils.validate_granules()

    if args.wipe_transformations:
        print(' --- wipe transformations --- ')
        # Wipe transformations
        logging.info('Removing transformations with out of sync version numbers from Solr and disk')
        solr_utils.delete_mismatch_transformations()
        pass

    if isinstance(args.grids_to_use, list):
        print(' ... grids_to_use is a list: ', args.grids_to_use)
        grids_to_use = args.grids_to_use
    else:
        print(' ... grids_to_use is not a list:', GRIDS)
        grids_to_use = GRIDS

    if args.grids_to_solr or solr_utils.check_grids():
        try:
            grids_not_in_solr = []
            grids_not_in_solr = grids_to_solr.main(grids_to_use)
            if grids_not_in_solr:
                for name in grids_not_in_solr:
                    logging.exception(f'Grid "{name}" not in Solr. Ensure it\'s file name is present in grids_config.yaml and run pipeline with the --grids_to_solr argument')
                exit()
            logging.info('Successfully updated grids on Solr.')
        except Exception as e:
            logging.exception(e)

    user_cpus = args.multiprocesses
    logging.debug(f'Using {user_cpus} processes for multiprocess transformations')
    

    return grids_to_use, user_cpus
