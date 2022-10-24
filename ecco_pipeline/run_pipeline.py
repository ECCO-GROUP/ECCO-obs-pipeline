import argparse
import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import requests
import yaml

import grids_to_solr
from aggregation import aggregation
from conf.global_settings import OUTPUT_DIR, SOLR_COLLECTION
from grid_transformation import check_transformations
from utils import solr_utils, log_config

###########
# Perform set up and verify system elements
###########

log_config.configure_logging(False)

# Set package logging level to WARNING
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Verify output directory is valid
if not Path.is_dir(OUTPUT_DIR):
    logging.fatal('Missing output directory. Please fill in. Exiting.')
    exit()

logging.debug(f'Using output directory: {OUTPUT_DIR}')
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

ds_status = defaultdict(list)


def create_parser():
    """
    Creates command line argument parser

    Returns:
        parser (ArgumentParser): the ArgumentParser object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--grids_to_solr', default=False, action='store_true',
                        help='updates Solr with grids in grids_config')

    parser.add_argument('--single_processing', default=False, action='store_true',
                        help='turns off the use of multiprocessing during transformation')

    parser.add_argument('--multiprocesses', type=int, choices=range(1, cpu_count()+1),
                        default=int(cpu_count()/2), metavar=f'[1, {cpu_count()}]',
                        help=f'sets the number of multiprocesses used during transformation with a \
                            system max of {cpu_count()} with default set to half of system max')

    parser.add_argument('--harvested_entry_validation', default=False, action='store_true',
                        help='verifies each Solr granule entry points to a valid file.')

    parser.add_argument('--wipe_transformations', default=False, action='store_true',
                        help='deletes transformations with version number different than what is \
                            currently in transformation_config')

    parser.add_argument('--grids_to_use', default=False, nargs='*',
                        help='Names of grids to use during the pipeline')

    return parser


def run_harvester(datasets, grids_to_use):
    for ds in datasets:
        try:
            logging.info(f'Beginning harvesting {ds}')

            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            try:
                harvester_type = config['harvester_type']
            except:
                logging.fatal(f'Harvester type missing from {ds} config. Exiting.')
                exit()

            if harvester_type == 'cmr':
                from harvesters.cmr_harvester import harvester
            elif harvester_type == 'osisaf_ftp':
                from harvesters.osisaf_ftp_harvester import harvester
            elif harvester_type == 'nsidc_ftp':
                from harvesters.nsidc_ftp_harvester import harvester
            elif harvester_type == 'ifremer_ftp':
                from harvesters.ifremer_ftp_harvester import harvester
            elif harvester_type == 'rdeft4':
                from harvesters.rdeft4_harvester import harvester
            else:
                logging.fatal(f'{harvester_type} is not a supported harvester type.')
                exit()

            status = harvester(config, grids_to_use)
            logging.info(f'{ds} harvesting complete. {status}')
        except Exception as e:
            logging.exception(f'{ds} harvesting failed. {e}')


def run_transformation(datasets, multiprocessing, user_cpus, wipe, grids_to_use):
    for ds in datasets:
        try:
            logging.info(f'Beginning transformations on {ds}')
            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            status = check_transformations.main(config, multiprocessing, user_cpus,
                                                wipe, grids_to_use)
            ds_status[ds].append(status)
            logging.info(f'{ds} transformation complete. {status}')
        except:
            logging.exception(f'{ds} transformation failed.')


def run_aggregation(datasets: List[str], grids_to_use):
    for ds in datasets:
        try:
            logging.info(f'Beginning aggregation on {ds}')
            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            status = aggregation(config, grids_to_use)
            ds_status[ds].append(status)

            logging.info(f'{ds} aggregation complete. {status}')
        except Exception as e:
            logging.info(f'{ds} aggregation failed: {e}')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # ------------------- Harvested Entry Validation -------------------
    if args.harvested_entry_validation:
        solr_utils.validate_granules()

    # ------------------- Grids to Use -------------------
    if isinstance(args.grids_to_use, list):
        grids_to_use = args.grids_to_use
        verify_grids = True
    else:
        grids_to_use = []
        verify_grids = False

    # ------------------- Grids to Solr -------------------
    if args.grids_to_solr or verify_grids or solr_utils.check_grids():
        try:
            grids_not_in_solr = []
            grids_not_in_solr = grids_to_solr.main(grids_to_use, verify_grids)

            if grids_not_in_solr:
                for name in grids_not_in_solr:
                    logging.exception(f'Grid "{name}" not in Solr. Ensure it\'s file name is present in grids_config.yaml and run pipeline with the --grids_to_solr argument')
                exit()
            logging.info('Successfully updated grids on Solr.')
        except Exception as e:
            logging.exception(e)

    # ------------------- Multiprocessing -------------------
    multiprocessing = not args.single_processing
    user_cpus = args.multiprocesses

    if multiprocessing:
        logging.debug(f'Using {user_cpus} processes for multiprocess transformations')
    else:
        logging.debug('Using single process transformations')

    # ------------------- Run pipeline -------------------
    while True:
        print('\n===== ECCO PREPROCESSING PIPELINE =====')
        print('\n------------- OPTIONS -------------')
        print('1) Run all')
        print('2) Harvesters only')
        print('3) Up to aggregation')
        print('4) Dataset input')
        chosen_option = input('Enter option number: ')

        if chosen_option in ['1', '2', '3', '4']:
            break
        else:
            print(f'Unknown option entered, "{chosen_option}", please enter a valid option\n')

    datasets = [os.path.splitext(ds)[0] for ds in os.listdir(
        'conf/ds_configs') if ds != '.DS_Store' and 'tpl' not in ds]
    datasets.sort()

    wipe = args.wipe_transformations

    # Run all
    if chosen_option == '1':
        for ds in datasets:
            run_harvester([ds], grids_to_use)
            run_transformation([ds], multiprocessing, user_cpus, wipe, grids_to_use)
            run_aggregation([ds], grids_to_use)

    # Run harvester
    elif chosen_option == '2':
        run_harvester(datasets, grids_to_use)

    # Run up through transformation
    elif chosen_option == '3':
        for ds in datasets:
            run_harvester([ds], grids_to_use)
            run_transformation([ds], multiprocessing, user_cpus, wipe, grids_to_use)

    # Manually enter dataset and pipeline step(s)
    elif chosen_option == '4':
        ds_dict = {i: ds for i, ds in enumerate(datasets, start=1)}
        while True:
            print(f'\nAvailable datasets:\n')
            for i, dataset in ds_dict.items():
                print(f'{i}) {dataset}')
            ds_index = input('\nEnter dataset number: ')

            if not ds_index.isdigit() or int(ds_index) not in range(1, len(datasets)+1):
                print(
                    f'Invalid dataset, "{ds_index}", please enter a valid selection')
            else:
                break

        wanted_ds = ds_dict[int(ds_index)]
        print(f'\nUsing {wanted_ds} dataset')

        steps = ['harvest', 'transform', 'aggregate',
                 'harvest and transform', 'transform and aggregate', 'all']
        steps_dict = {i: step for i, step in enumerate(steps, start=1)}
        while True:
            print(f'\nAvailable steps:\n')
            for i, step in steps_dict.items():
                print(f'{i}) {step}')
            steps_index = input('\nEnter pipeline step(s) number: ')

            if not steps_index.isdigit() or int(steps_index) not in range(1, len(steps)+1):
                print(
                    f'Invalid step(s), "{steps_index}", please enter a valid selection')
            else:
                break

        wanted_steps = steps_dict[int(steps_index)]

        if 'harvest' in wanted_steps:
            run_harvester([wanted_ds], grids_to_use)
        if 'transform' in wanted_steps:
            run_transformation([wanted_ds], multiprocessing, user_cpus, wipe, grids_to_use)
        if 'aggregate' in wanted_steps:
            run_aggregation([wanted_ds], grids_to_use)
        if wanted_steps == 'all':
            run_harvester([wanted_ds], grids_to_use)
            run_transformation([wanted_ds], multiprocessing, user_cpus, wipe, grids_to_use)
            run_aggregation([wanted_ds], grids_to_use)
