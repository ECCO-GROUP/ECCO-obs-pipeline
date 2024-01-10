import logging
from multiprocessing import current_process, Pool, get_logger
import os
from typing import Iterable
from aggregations.aggjob import AggJob
from aggregations.aggregation import Aggregation
from field import Field
from utils import solr_utils, log_config


def get_grids(grids_to_use: Iterable[str]) -> Iterable[dict]:
    '''
    Queries for grids on Solr and filters based on grids to use
    '''
    fq = ['type_s:grid']
    grids = [grid for grid in solr_utils.solr_query(fq)]
    if grids_to_use:
        grids = [grid for grid in grids if grid['grid_name_s'] in grids_to_use]
    return grids

def get_solr_ds_metadata(ds_name: str) -> dict:
    '''
    Gets type_s:dataset Solr document for a given dataset
    '''
    fq = [f'dataset_s:{ds_name}', 'type_s:dataset']
    ds_meta = solr_utils.solr_query(fq)[0]
    if 'start_date_dt' not in ds_meta:
        logging.error('No transformed granules to aggregate.')
        raise Exception('No transformed granules to aggregate.')
    return ds_meta

def make_jobs(ds_name: str, grids: Iterable[dict], fields: Iterable[Field], config: dict) -> Iterable[AggJob]:
    '''
    Generates list of AggJob objects that define the grid/field/year aggregations to be performed.
    Checks if aggregation exists for a given grid/field/year combo and if so if it needs to be reprocessed.
    '''
    all_jobs = []
    for grid in grids:
        for field in fields:
            grid_name = grid.get('grid_name_s')

            # Get grid / field transformation documents from Solr
            fq = [f'dataset_s:{ds_name}', f'field_s:{field.name}',
                    'type_s:transformation', f'grid_name_s:{grid_name}']
            transformation_docs = solr_utils.solr_query(fq)
            transformation_years = list(set([t['date_s'][:4] for t in transformation_docs]))
            transformation_years.sort()
            years_to_aggregate = []
            for year in transformation_years:
                # Check for successful aggregation doc for this combo of grid / field / year 
                fq = [f'dataset_s:{ds_name}', 'type_s:aggregation', 'aggregation_success_b:true', 
                      f'field_s:{field.name}', f'grid_name_s:{grid_name}', f'year_s:{year}']
                aggregation_docs = solr_utils.solr_query(fq)

                # If aggregation was previously done compare transformation time with aggregation time
                if aggregation_docs:
                    agg_time = aggregation_docs[0]['aggregation_time_dt']
                    for t in transformation_docs:
                        if t['date_s'][:4] != year:
                            continue
                        if t['transformation_completed_dt'] > agg_time:
                            years_to_aggregate.append(year)
                            break
                else:
                    years_to_aggregate.append(year)            
            all_jobs.extend([AggJob(config, grid, year, field) for year in years_to_aggregate])
    return all_jobs

def make_jobs_all_years(ds_metadata: dict, grids: Iterable[dict], fields: Iterable[Field], config: dict) -> Iterable[AggJob]:
    '''
    Makes AggJob objects for all years for all grids for all fields
    '''
    start_year = int(ds_metadata.get('start_date_dt')[:4])
    end_year = int(ds_metadata.get('end_date_dt')[:4])
    years = [str(year) for year in range(start_year, end_year + 1)]
    jobs = []
    for grid in grids:
        for field in fields:
            jobs.extend([AggJob(config, grid, year, field) for year in years])
    return jobs

def get_jobs(config: dict, grids: Iterable[dict], fields: Iterable[Field]) -> Iterable[AggJob]:
    '''
    Gets list of AggJob objects for each annual aggregation to be performed for each grid / field combination
    '''
    ds_metadata = get_solr_ds_metadata(config['ds_name'])
    existing_agg_version = ds_metadata.get('aggregation_version_s')
    
    if existing_agg_version and existing_agg_version != str(config.get('a_version', '')):
        logging.debug('Making jobs for all years')
        agg_jobs = make_jobs_all_years(ds_metadata, grids, fields, config)
    else:
        logging.debug('Determining jobs')
        agg_jobs = make_jobs(config['ds_name'], grids, fields, config)
    return agg_jobs

def get_agg_status(ds_name: str) -> str:
    '''
    Queries Solr for dataset aggregations. 
    Returns overall status of aggregation, not specific to this particular execution.
    '''
    # Query Solr for successful aggregation documents
    fq = [f'dataset_s:{ds_name}', 'type_s:aggregation', 'aggregation_success_b:true']
    successful_aggregations = solr_utils.solr_query(fq)

    # Query Solr for failed aggregation documents
    fq = [f'dataset_s:{ds_name}', 'type_s:aggregation', 'aggregation_success_b:false']
    failed_aggregations = solr_utils.solr_query(fq)

    aggregation_status = 'All aggregations successful'

    if not successful_aggregations and not failed_aggregations:
        aggregation_status = 'No aggregations performed'
    elif not successful_aggregations:
        aggregation_status = 'No successful aggregations'
    elif failed_aggregations:
        aggregation_status = f'{len(failed_aggregations)} aggregations failed'
    return aggregation_status


def multiprocess_aggregate(job: AggJob, log_dir: str):
    '''
    Function used to execute by multiprocessing to execute a single grid/year/field aggregation
    '''
    log_subdir = os.path.join(log_dir, f'ag_{job.ds_name}')
    logger = log_config.mp_logging(str(current_process().pid), log_subdir, get_logger().level)
    
    logger.info(f'Beginning aggregation for {job.grid["grid_name_s"]}, {job.year}, {job.field.name}')
    job.aggregate()
    
def update_solr_ds(aggregation_status: str, config: dict):
    # Update Solr dataset entry with new aggregation status
    update_body = [
        {
            "id": Aggregation(config).ds_meta['id'],
            "aggregation_version_s": {"set": str(config['a_version'])},
            "aggregation_status_s": {"set": aggregation_status}
        }
    ]

    r = solr_utils.solr_update(update_body, r=True)

    if r.status_code == 200:
        logging.debug(f'Successfully updated Solr with aggregation information for {config["ds_name"]}')
    else:
        logging.exception(f'Failed to update Solr dataset entry with aggregation information for {config["ds_name"]}')

def aggregation(config: dict, user_cpus: int, grids_to_use: Iterable[str]=[], log_dir: str = '') -> str:
    """
    Aggregates data into annual files (either daily, monthly, or both), saves them, and updates Solr
    """    
    grids = get_grids(grids_to_use)
    agg_jobs = get_jobs(config, grids, Aggregation(config).fields)
    
    logging.info(f'Executing jobs: {", ".join([str(job) for job in agg_jobs])}')
    
    if user_cpus == 1:
        logging.info('Not using multiprocessing to do aggregations')
        for agg_job in agg_jobs:
            multiprocess_aggregate(agg_job, log_dir)
    else:
        logging.info(f'Using {user_cpus} CPUs to do multiprocess aggregation')
        with Pool(processes=user_cpus) as pool:               
            pool.starmap(multiprocess_aggregate, [(agg_job, log_dir) for agg_job in agg_jobs])
            pool.close()
            pool.join()
        
    aggregation_status = get_agg_status(config['ds_name'])
    update_solr_ds(aggregation_status, config)
    return aggregation_status