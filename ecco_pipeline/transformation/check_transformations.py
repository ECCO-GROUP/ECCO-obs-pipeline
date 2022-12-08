import itertools
import logging
import os
from collections import defaultdict
from multiprocessing import Pool

from utils import solr_utils

from transformation import grid_transformation


def get_remaining_transformations(config, granule_file_path, grids):
    """
    Given a single granule, the function uses Solr to find all combinations of
    grids and fields that have yet to be transformed. It returns a dictionary
    where the keys are grids and the values are lists of fields.
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

    if not docs:
        for grid, field in grid_field_combinations:
            grid_field_dict[grid].append(field)

        return dict(grid_field_dict)

    # if a transformation entry exists for this granule, check to see if the
    # checksum of the harvested granule matches the checksum recorded in the
    # transformation entry for this granule, if not then we have to retransform
    # also check to see if the version of the transformation code recorded in
    # the entry matches the current version of the transformation code, if not
    # redo the transformation.

    # these checks are made for each grid/field pair associated with the
    # harvested granule)

    # Dictionary where key is grid, field tuple and value is harvested granule checksum
    # For existing transformations pulled from Solr
    existing_transformations = {
        (doc['grid_name_s'], doc['field_s']): doc['origin_checksum_s'] for doc in docs}

    drop_list = []

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
                  f'pre_transformation_file_path_s:"{granule_file_path}"']
            transformation = solr_utils.solr_query(fq)[0]

            # Triple if:
            # 1. do we have a version entry,
            # 2. compare transformation version number and current transformation version number
            # 3. compare checksum of harvested file (currently in solr) and checksum
            #    of the harvested file that was previously transformed (recorded in transformation entry)
            if ('success_b' in transformation.keys() and transformation['success_b'] == True) and \
                ('transformation_version_f' in transformation.keys() and transformation['transformation_version_f'] == config['t_version']) and \
                    origin_checksum == harvested_checksum:
                logging.debug(f'No need to transform {granule_file_path}')
                # all tests passed, we do not need to redo the transformation
                # for this grid/field pair

                # Add grid/field combination to drop_list
                drop_list.append((grid, field))

    # Remove drop_list grid/field combinations from list of remaining transformations
    grid_field_combinations = [
        combo for combo in grid_field_combinations if combo not in drop_list]

    for grid, field in grid_field_combinations:
        grid_field_dict[grid].append(field)

    return dict(grid_field_dict)


def multiprocess_transformation(granule, config, grids):
    """
    Callable function that performs the actual transformation on a granule.
    """

    # f is file path to granule from solr
    f = granule.get('pre_transformation_file_path_s', '')

    # Skips granules that weren't harvested properly
    if f == '':
        logging.exception("pre transformation path doesn't exist")
        return ('', '')

    # Get transformations to be completed for this file
    remaining_transformations = get_remaining_transformations(config, f, grids)

    # Perform remaining transformations
    if remaining_transformations:
        grids_updated, year = grid_transformation.main(
            f, remaining_transformations, config)

        return (grids_updated, year)
    else:
        logging.debug(
            f'CPU id {os.getpid()} no new transformations for {granule["filename_s"]}')
        return ('', '')


def main(config, user_cpus=1, grids_to_use=[]):
    """
    This function performs all remaining grid/field transformations for all harvested
    granules for a dataset. It also makes use of multiprocessing to perform multiple
    transformations at the same time. After all transformations have been attempted,
    the Solr dataset entry is updated with additional metadata.
    """

    dataset_name = config['ds_name']
    transformation_version = config['t_version']

    # Get all harvested granules for this dataset
    fq = [f'dataset_s:{dataset_name}',
          'type_s:granule', 'harvest_success_b:true']
    harvested_granules = solr_utils.solr_query(fq)

    if not harvested_granules:
        logging.exception(
            f'No harvested granules found in solr for {dataset_name}')
        return f'No successful transformations'

    years_updated = defaultdict(list)

    # Query for grids
    if not grids_to_use:
        fq = ['type_s:grid']
        docs = solr_utils.solr_query(fq)
        grids = [doc['grid_name_s'] for doc in docs]
    else:
        grids = grids_to_use

    # PRE GENERATE FACTORS TO ACCOMODATE MULTIPROCESSING
    # Query for dataset metadata
    fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
    try:
        dataset_metadata = solr_utils.solr_query(fq)[0]
    except:
        logging.exception(f'No dataset found in solr for {dataset_name}')
        exit()

    # Precompute grid factors using one dataset data file
    # (or one from each hemisphere, if data is hemispherical) before running main loop
    data_for_factors = []
    for grid in grids:
        data_for_factors = []
        nh_added = False
        sh_added = False

        # Find appropriate granule(s) to use for factor calculation
        for granule in harvested_granules:
            if 'hemisphere_s' in granule.keys():
                hemi = f'_{granule["hemisphere_s"]}'
            else:
                hemi = ''

            grid_factors = f'{grid}{hemi}_factors_path_s'
            grid_factors_version = f'{grid}{hemi}_factors_version_f'

            if grid_factors in dataset_metadata.keys() and transformation_version == dataset_metadata[grid_factors_version]:
                continue

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
                        break
                else:
                    data_for_factors.append(granule)
                    break

    # Actually perform transformation on chosen granule(s)
    # This will generate factors and avoid redundant calculations when using multiprocessing
    for granule in data_for_factors:
        file_path = granule['pre_transformation_file_path_s']

        # Get transformations to be completed for this file
        remaining_transformations = get_remaining_transformations(
            config, file_path, grids)

        grids_updated, year = grid_transformation.main(
            file_path, remaining_transformations, config)

        for grid in grids_updated:
            if year not in years_updated[grid]:
                years_updated[grid].append(year)
    # END PRE GENERATE FACTORS TO ACCOMODATE MULTIPROCESSING

    # BEGIN MULTIPROCESSING
    # Create list of tuples of function arguments (necessary for using pool.starmap)
    multiprocess_tuples = [(granule, config, grids)
                           for granule in harvested_granules]

    grid_years_list = []

    # for grid in grids:
    logging.info(f'Running transformations for {grids} grids')

    with Pool(processes=user_cpus) as pool:
        grid_years_list = pool.starmap(
            multiprocess_transformation, multiprocess_tuples)
        pool.close()
        pool.join()

    for (grids, year) in grid_years_list:
        if grids and year:
            for grid in grids:
                if year not in years_updated[grid]:
                    years_updated[grid].append(year)

    # Query Solr for dataset metadata
    fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
    dataset_metadata = solr_utils.solr_query(fq)[0]

    # Query Solr for successful transformation documents
    fq = [f'dataset_s:{dataset_name}',
          'type_s:transformation', 'success_b:true']
    successful_transformations = solr_utils.solr_query(fq)

    # Query Solr for failed transformation documents
    fq = [f'dataset_s:{dataset_name}',
          'type_s:transformation', 'success_b:false']
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
        logging.debug(
            f'Successfully updated Solr with transformation information for {dataset_name}')
    else:
        logging.exception(
            f'Failed to update Solr with transformation information for {dataset_name}')

    return transformation_status
