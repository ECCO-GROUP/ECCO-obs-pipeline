import itertools
import logging
from logging.handlers import QueueHandler
from collections import defaultdict
from multiprocessing import Queue, current_process, Pool, Manager
import xarray as xr

from utils import solr_utils, log_config
from transformations.grid_transformation import transform
from transformations.transformation import Transformation

import conf.global_settings as global_settings

def logging_process(queue: Queue, log_filename: str):
    '''
    Process to run during multiprocessing. Allows for logging of each process.
    '''
    log_config.configure_logging(True, log_filename=log_filename)

    # report that the logging process is running
    logging.debug(f'Logger process running.')
    # run forever
    while True:
        # consume a log message, block until one arrives
        message = queue.get()
        # check for shutdown
        if message is None:
            logging.debug(f'Logger process shutting down.')
            break
        # log the message
        logging.getLogger().handle(message)

def get_remaining_transformations(config, granule_file_path, grids):
    """
    Given a single granule, the function uses Solr to find all combinations of
    grids and fields that have yet to be transformed. It returns a dictionary
    where the keys are grids and the values are lists of fields.

    if a transformation entry exists for this granule, check to see if the
    checksum of the harvested granule matches the checksum recorded in the
    transformation entry for this granule, if not then we have to retransform
    also check to see if the version of the transformation code recorded in
    the entry matches the current version of the transformation code, if not
    redo the transformation.

    these checks are made for each grid/field pair associated with the harvested granule
    """

    dataset_name = config['ds_name']
    fields = config['fields']

    # Cartesian product of grid/field combinations
    grid_field_combinations = list(itertools.product(grids, fields))

    # Build dictionary of remaining transformations
    # -- grid_field_dict has grid key, entries is list of fields
    grid_field_dict = defaultdict(list)

    # Query for existing transformations
    fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
          f'pre_transformation_file_path_s:"{granule_file_path}"']
    docs = solr_utils.solr_query(fq)

    # No transformations exist yet, so all must be done
    if not docs:
        for grid, field in grid_field_combinations:
            grid_field_dict[grid].append(field)
        logging.info(f'{sum([len(v) for v in grid_field_dict.values()])} remaining transformations for {granule_file_path.split("/")[-1]}')
        return dict(grid_field_dict)

    # Dictionary where key is grid, field tuple and value is harvested granule checksum
    # For existing transformations pulled from Solr
    existing_transformations = {(doc['grid_name_s'], doc['field_s']): doc['origin_checksum_s'] for doc in docs}

    drop_list = []

    # Loop through all grid/field combos, checking if processing is required
    for (grid, field) in grid_field_combinations:
        field_name = field['name']

        # If transformation exists, must compare checksums and versions for updates
        if (grid, field_name) in existing_transformations:

            # Query for harvested granule checksum
            fq = [f'dataset_s:{dataset_name}', 'type_s:granule',
                  f'pre_transformation_file_path_s:"{granule_file_path}"']
            harvested_checksum = solr_utils.solr_query(fq)[0]['checksum_s']

            origin_checksum = existing_transformations[(grid, field_name)]

            # Query for existing transformation
            fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
                  f'pre_transformation_file_path_s:"{granule_file_path}"',
                  f'field_s:{field_name}']
            transformation = solr_utils.solr_query(fq)[0]

            # Triple if:
            # 1. do we have a version entry,
            # 2. compare transformation version number and current transformation version number
            # 3. compare checksum of harvested file (currently in solr) and checksum
            #    of the harvested file that was previously transformed (recorded in transformation entry)
            if ('success_b' in transformation.keys() and transformation['success_b'] == True) and \
                ('transformation_version_f' in transformation.keys() and transformation['transformation_version_f'] == config['t_version']) and \
                    origin_checksum == harvested_checksum:
                logging.debug(f'No need to transform {granule_file_path} for grid {grid} and field {field_name}')
                # all tests passed, we do not need to redo the transformation
                # for this grid/field pair

                # Add grid/field combination to drop_list
                drop_list.append((grid, field))

    # Remove drop_list grid/field combinations from list of remaining transformations
    grid_field_combinations = [combo for combo in grid_field_combinations if combo not in drop_list]

    for grid, field in grid_field_combinations:
        grid_field_dict[grid].append(field)
    logging.info(f'{sum([len(v) for v in grid_field_dict.values()])} remaining transformations for {granule_file_path.split("/")[-1]}')
    return dict(grid_field_dict)


def multiprocess_transformation(granule, config, grids, log_filename: str):
    """
    Callable function that performs the actual transformation on a granule.
    """
    log_config.configure_logging(True, log_filename=log_filename)
       
    process = current_process()
    granule_filepath = granule.get('pre_transformation_file_path_s')
    granule_date = granule.get('date_s')

    # Skips granules that weren't harvested properly
    if not granule_filepath or granule.get('file_size_l') < 100:
        logging.exception(f'Granule was not harvested properly. Skipping.')
        return ('', '')
    # logging.info(f'{process.name} performing transformations for {granule_filepath.split("/")[-1]}')

    # Get transformations to be completed for this file
    remaining_transformations = get_remaining_transformations(config, granule_filepath, grids)
    
    # Perform remaining transformations
    if remaining_transformations:
        try:
            transform(granule_filepath, remaining_transformations, config, granule_date)
        except Exception as e:
            logging.exception(f'Error transforming {granule_filepath}: {e}')
    else:
        logging.debug(f'No new transformations for {granule["filename_s"]}')
        
    # logging.info(f'{process.name} trasnsformations complete for {granule_filepath.split("/")[-1]}')
    # if queue:
    #     logging.getLogger().removeHandler(queue_handler)

def find_data_for_factors(harvested_granules):
    data_for_factors = []
    nh_added = False
    sh_added = False
    # Find appropriate granule(s) to use for factor calculation
    for granule in harvested_granules:
        if 'hemisphere_s' in granule.keys():
            hemi = f'_{granule["hemisphere_s"]}'
        else:
            hemi = ''

        file_path = granule.get('pre_transformation_file_path_s', '')
        if file_path:
            if hemi:
                # Get one of each
                if hemi == '_nh' and not nh_added:
                    data_for_factors.append(granule)
                    nh_added = True
                elif hemi == '_sh' and not sh_added:
                    data_for_factors.append(granule)
                    sh_added = True
                if nh_added and sh_added:
                    return data_for_factors
            else:
                data_for_factors.append(granule)
                return data_for_factors


def main(config, user_cpus=1, grids_to_use=[]):
    """
    This function performs all remaining grid/field transformations for all harvested
    granules for a dataset. It also makes use of multiprocessing to perform multiple
    transformations at the same time. After all transformations have been attempted,
    the Solr dataset entry is updated with additional metadata.
    """
    log_filename = global_settings.log_filename
    dataset_name = config['ds_name']

    # Get all harvested granules for this dataset
    fq = [f'dataset_s:{dataset_name}', 'type_s:granule', 'harvest_success_b:true']
    harvested_granules = solr_utils.solr_query(fq)

    if not harvested_granules:
        logging.info(f'No harvested granules found in solr for {dataset_name}')
        return f'No transformations performed'

    # Query for grids
    if not grids_to_use:
        fq = ['type_s:grid']
        docs = solr_utils.solr_query(fq)
        grids = [doc['grid_name_s'] for doc in docs]
    else:
        grids = grids_to_use

    # PRE GENERATE FACTORS TO ACCOMODATE MULTIPROCESSING

    for grid in grids:
        data_for_factors = find_data_for_factors(harvested_granules)

        for granule in data_for_factors:
            grid_ds = xr.open_dataset(f'grids/{grid}.nc')
            T = Transformation(config, granule['pre_transformation_file_path_s'], '1972-04-20')
            T.make_factors(grid_ds)

    # END PRE GENERATE FACTORS TO ACCOMODATE MULTIPROCESSING

    # BEGIN MULTIPROCESSING
    # Create list of tuples of function arguments (necessary for using pool.starmap)
    multiprocess_parameters = [(granule, config, grids) for granule in harvested_granules]

    # for grid in grids:
    logging.info(f'Running transformations for {grids} grids')

    if user_cpus == 1:
        logging.info('Not using multiprocessing to do transformation')
        for (granule, config, grids) in multiprocess_parameters:
            multiprocess_transformation(granule, config, grids)
    else:
        logging.info(f'Using {user_cpus} CPUs to do multiprocess transformation')
        with Manager() as manager:
            queue = manager.Queue()
            # add a handler that uses the shared queue
            queue_handler = QueueHandler(queue)
            logging.getLogger().addHandler(queue_handler)
            queue_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
            
            with Pool(processes=user_cpus) as pool:               
                # issue a long running task to receive logging messages
                _ = pool.apply_async(logging_process, args=(queue,log_filename,))

                pool.starmap(multiprocess_transformation, [(granule, config, grids, log_filename) for granule in harvested_granules])
                queue.put(None)
                pool.close()
                pool.join()
            logging.getLogger().removeHandler(queue_handler)


    # Query Solr for dataset metadata
    fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
    dataset_metadata = solr_utils.solr_query(fq)[0]

    # Query Solr for successful transformation documents
    fq = [f'dataset_s:{dataset_name}', 'type_s:transformation', 'success_b:true']
    successful_transformations = solr_utils.solr_query(fq)

    # Query Solr for failed transformation documents
    fq = [f'dataset_s:{dataset_name}', 'type_s:transformation', 'success_b:false']
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
        logging.debug(f'Successfully updated Solr with transformation information for {dataset_name}')
    else:
        logging.exception(f'Failed to update Solr with transformation information for {dataset_name}')

    return transformation_status