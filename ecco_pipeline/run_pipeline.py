import argparse
import importlib
import logging
import os
from glob import glob
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import yaml
from aggregations.aggregation_factory import AgJobFactory
from transformations.transformation_factory import TxJobFactory
from utils.pipeline_utils import init_pipeline, log_config


def create_parser() -> argparse.Namespace:
    """
    Creates command line argument parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--menu', default=False,
                        action='store_true', help='Show interactive menu')
    parser.add_argument('--grids_to_solr', default=False, action='store_true',
                        help='updates Solr with grids in grids_config')
    parser.add_argument('--multiprocesses', type=int, choices=range(1, cpu_count()+1), default=int(cpu_count()/2),
                        metavar=f'[1, {cpu_count()}]', help=f'sets the number of multiprocesses used during transformation with a system max of {cpu_count()} \
                            with default set to half of system max (aggregation has a stricter cap due to high I/O volume)')
    parser.add_argument('--harvested_entry_validation', default=False, action='store_true',
                        help='verifies each Solr granule entry points to a valid file.')
    parser.add_argument('--wipe_transformations', default=False, action='store_true',
                        help='deletes transformations with version number different than what is currently in transformation_config')
    parser.add_argument('--grids_to_use', default=False, nargs='*',
                        help='Names of grids to use during the pipeline')
    parser.add_argument('--log_level', default='INFO',
                        help='sets the log level')
    parser.add_argument('--wipe_factors', default=False,
                        action='store_true', help='removes all stored factors')
    parser.add_argument('--wipe_logs', default=False,
                        action='store_true', help='removes all prior log files')

    args, _ = parser.parse_known_args()

    if not args.menu:
        parser.add_argument('--dataset', help='Specify the dataset option', type=str)
        parser.add_argument('--step', help='Specify the step option', type=str,
                            default='all', choices=['harvest', 'transform', 'aggregate', 'all'])
        args = parser.parse_args()

        if args.dataset:
            # Check if dataset is valid option
            valid_datasets = [os.path.splitext(os.path.basename(f))[0] for f in glob('conf/ds_configs/*.yaml')]
            if args.dataset not in valid_datasets:
                raise ValueError(
                    f'{args.dataset} is not valid dataset. Check spelling.')
    return args


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

    # Load all dataset configuration YAML names
    datasets = [os.path.splitext(os.path.basename(f))[0] for f in glob(f'conf/ds_configs/*.yaml')]
    datasets.sort()

    # Run all
    if chosen_option == '1':
        for ds in datasets:
            run_harvester([ds])
            run_transformation([ds], user_cpus, grids_to_use)
            run_aggregation([ds], user_cpus, grids_to_use)

    # Run harvester
    elif chosen_option == '2':
        run_harvester(datasets)

    # Run up through transformation
    elif chosen_option == '3':
        for ds in datasets:
            run_harvester([ds])
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
                print(f'Invalid dataset, "{ds_index}", please enter a valid selection')
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
                print(f'Invalid step(s), "{steps_index}", please enter a valid selection')
            else:
                break

        wanted_steps = steps_dict[int(steps_index)]

        if 'harvest' in wanted_steps:
            run_harvester([wanted_ds])
        if 'transform' in wanted_steps:
            run_transformation([wanted_ds], user_cpus, grids_to_use)
        if 'aggregate' in wanted_steps:
            run_aggregation([wanted_ds], user_cpus, grids_to_use)
        if wanted_steps == 'all':
            run_harvester([wanted_ds])
            run_transformation([wanted_ds], user_cpus, grids_to_use)
            run_aggregation([wanted_ds], user_cpus, grids_to_use)


def run_harvester(datasets: List[str]):
    for ds in datasets:
        try:
            logger.info(f'Beginning harvesting {ds}')
            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)
            harvester_type = config.get('harvester_type')
            if not harvester_type:
                raise ValueError(f'Harvester type missing from {ds} config. Exiting.')
            
            harvester = importlib.import_module(f'harvesters.{harvester_type}_harvester')
            status = harvester.harvester(config)
            logger.info(f'{ds} harvesting complete. {status}')
        except Exception as e:
            logger.exception(f'{ds} harvesting failed. {e}')


def run_transformation(datasets: List[str], user_cpus: int, grids_to_use: List[str]):
    for ds in datasets:
        try:
            logger.info(f'Beginning transformations on {ds}')
            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)
            status = TxJobFactory(config, user_cpus, grids_to_use).start_factory()
            logger.info(f'{ds} transformation complete. {status}')
        except:
            logger.exception(f'{ds} transformation failed.')


def run_aggregation(datasets: List[str], user_cpus: int, grids_to_use: List[str]):
    for ds in datasets:
        try:
            logger.info(f'Beginning aggregation on {ds}')
            with open(Path(f'conf/ds_configs/{ds}.yaml'), 'r') as stream:
                config = yaml.load(stream, yaml.Loader)
            status = AgJobFactory(config, user_cpus, grids_to_use).start_factory()
            logger.info(f'{ds} aggregation complete. {status}')
        except Exception as e:
            logger.exception(f'{ds} aggregation failed: {e}')


def start_pipeline(args: argparse.Namespace, grids_to_use: List[str], user_cpus: int):
    if args.dataset:
        datasets = [args.dataset]
    else:
        # Load all dataset configuration YAML names
        datasets = sorted([os.path.splitext(os.path.basename(f))[0] for f in glob(f'conf/ds_configs/*.yaml')])
    
    if args.step == 'harvest':
        run_harvester(datasets)
    elif args.step == 'transform':
        run_transformation(datasets, user_cpus, grids_to_use)
    elif args.step == 'aggregate':
        run_aggregation(datasets, user_cpus, grids_to_use)
    else:
        for dataset in datasets:
            run_harvester([dataset])
            run_transformation([dataset], user_cpus, grids_to_use)
            run_aggregation([dataset], user_cpus, grids_to_use)

if __name__ == '__main__':
    args = create_parser()
    grids_to_use, user_cpus = init_pipeline.init_pipeline(args)
    logger = logging.getLogger('pipeline')
    if args.menu:
        show_menu(grids_to_use, user_cpus)
    else:
        start_pipeline(args, grids_to_use, user_cpus)