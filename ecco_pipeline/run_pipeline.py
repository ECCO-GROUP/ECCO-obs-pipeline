import argparse
import importlib
import logging
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import yaml

from aggregation import aggregation
from grid_transformation import check_transformations
from utils import init_pipeline


def create_parser() -> argparse.ArgumentParser:
    """
    Creates command line argument parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--grids_to_solr', default=False, action='store_true',
                        help='updates Solr with grids in grids_config')

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

    parser.add_argument('--log_level', default='INFO', help='sets the log level')

    return parser


def show_menu(grids_to_use: List[str], user_cpus: int):
    while True:
        print('\n===== ECCO PREPROCESSING PIPELINE =====')
        print('\n------------- OPTIONS -------------')
        print('1) Run all')
        print('2) Harvesters only')
        print('3) Up to aggregation')
        print('4) Dataset input')
        try:
            chosen_option = input('Enter option number: ')
        except KeyboardInterrupt:
            exit()

        if chosen_option in ['1', '2', '3', '4']:
            break
        else:
            print(f'Unknown option entered, "{chosen_option}", please enter a valid option\n')

    datasets = [os.path.splitext(ds)[0] for ds in os.listdir(
        'conf/ds_configs') if ds != '.DS_Store' and 'tpl' not in ds]
    datasets.sort()

    # Run all
    if chosen_option == '1':
        for ds in datasets:
            run_harvester([ds], grids_to_use)
            run_transformation([ds], user_cpus, grids_to_use)
            run_aggregation([ds], grids_to_use)

    # Run harvester
    elif chosen_option == '2':
        run_harvester(datasets, grids_to_use)

    # Run up through transformation
    elif chosen_option == '3':
        for ds in datasets:
            run_harvester([ds], grids_to_use)
            run_transformation([ds], user_cpus, grids_to_use)

    # Manually enter dataset and pipeline step(s)
    elif chosen_option == '4':
        ds_dict = {i: ds for i, ds in enumerate(datasets, start=1)}
        while True:
            print(f'\nAvailable datasets:\n')
            for i, dataset in ds_dict.items():
                print(f'{i}) {dataset}')
            try:
                ds_index = input('\nEnter dataset number: ')
            except KeyboardInterrupt:
                exit()

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
            run_transformation([wanted_ds], user_cpus, grids_to_use)
        if 'aggregate' in wanted_steps:
            run_aggregation([wanted_ds], grids_to_use)
        if wanted_steps == 'all':
            run_harvester([wanted_ds], grids_to_use)
            run_transformation([wanted_ds], user_cpus, grids_to_use)
            run_aggregation([wanted_ds], grids_to_use)


def run_harvester(datasets: List[str], grids_to_use: List[str]):
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

            try:
                harvester = importlib.import_module(f'harvesters.{harvester_type}_harvester')
            except:
                logging.fatal(f'{harvester_type} is not a supported harvester type.')
                exit()

            status = harvester.harvester(config, grids_to_use)
            logging.info(f'{ds} harvesting complete. {status}')
        except Exception as e:
            logging.exception(f'{ds} harvesting failed. {e}')


def run_transformation(datasets: List[str], user_cpus: int, grids_to_use: List[str]):
    for ds in datasets:
        try:
            logging.info(f'Beginning transformations on {ds}')
            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            status = check_transformations.main(config, user_cpus, grids_to_use)
            logging.info(f'{ds} transformation complete. {status}')
        except:
            logging.exception(f'{ds} transformation failed.')


def run_aggregation(datasets: List[str], grids_to_use: List[str]):
    for ds in datasets:
        try:
            logging.info(f'Beginning aggregation on {ds}')
            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            status = aggregation(config, grids_to_use)
            logging.info(f'{ds} aggregation complete. {status}')
        except Exception as e:
            logging.info(f'{ds} aggregation failed: {e}')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    grids_to_use, user_cpus = init_pipeline.init_pipeline(args)

    show_menu(grids_to_use, user_cpus)