import itertools
import logging
from multiprocessing import Pool
import numpy as np
import xarray as xr
from aggregations.aggjob import AggJob
from aggregations.aggregation import Aggregation
from conf.global_settings import OUTPUT_DIR
from utils import solr_utils

logger = logging.getLogger()

def get_grids(grids_to_use):
    fq = ['type_s:grid']
    grids = [grid for grid in solr_utils.solr_query(fq)]
    if grids_to_use:
        grids = [grid for grid in grids if grid['grid_name_s'] in grids_to_use]
    return grids

def get_solr_ds_metadata(ds_name):
    fq = [f'dataset_s:{ds_name}', 'type_s:dataset']
    ds_meta = solr_utils.solr_query(fq)[0]
    if 'start_date_dt' not in ds_meta:
        logging.error('No transformed granules to aggregate.')
        raise Exception('No transformed granules to aggregate.')
    return ds_meta

def years_to_aggregate(ds_name, grids) -> dict[str]:
    '''
    We want to see if we need to aggregate this year again
    1. check if aggregation exists - if not add it to years to aggregate
    2. if it does - compare processing times:
    - if aggregation time is later than all transform times for that year
        no need to aggregate
    - if at least one transformation occured after agg time, year needs to
        be aggregated
    '''
    years = {}
    for grid in grids:
        grid_name = grid.get('grid_name_s')
        grid_years = []

        fq = [f'dataset_s:{ds_name}',
                'type_s:transformation', f'grid_name_s:{grid_name}']
        r = solr_utils.solr_query(fq)
        transformation_years = list(set([t['date_s'][:4] for t in r]))
        transformation_years.sort()
        transformation_docs = r

        # Years with transformations that exist for this dataset and this grid
        for year in transformation_years:
            fq = [f'dataset_s:{ds_name}', 'type_s:aggregation',
                    f'grid_name_s:{grid_name}', f'year_s:{year}']
            r = solr_utils.solr_query(fq)

            if r:
                agg_time = r[0]['aggregation_time_dt']
                for t in transformation_docs:
                    if t['date_s'][:4] != year:
                        continue
                    if t['transformation_completed_dt'] > agg_time:
                        grid_years.append(year)
                        break
            else:
                grid_years.append(year)
        years[grid_name] = grid_years
    return years

def get_years(config, grids):
    ds_metadata = get_solr_ds_metadata(config['ds_name'])
    existing_agg_version = ds_metadata.get('aggregation_version_s')
    
    if existing_agg_version and existing_agg_version != str(config.get('a_version', '')):
        start_year = int(ds_metadata.get('start_date_dt')[:4])
        end_year = int(ds_metadata.get('end_date_dt')[:4])
        years = [str(year) for year in range(start_year, end_year + 1)]
        grid_years = {grid.get('grid_name_s'): years for grid in grids}
    else:
        grid_years = years_to_aggregate(config['ds_name'], grids)
    return grid_years

def get_agg_status(ds_name: str) -> str:
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


def make_agg_job_and_aggregate(config, grid, year, field):
    logging.info(f'Beggining aggregation for {grid["grid_name_s"]}, {year}, {field["name"]}')
    
    agg = AggJob(config, grid, year, field)
    agg.aggregate()

def aggregation(config, user_cpus=2, grids_to_use=[]):
    """
    Aggregates data into annual files, saves them, and updates Solr
    """
    
    grids = get_grids(grids_to_use)
    grid_years_to_agg = get_years(config, grids)

    all_combos = []
    
    for grid in get_grids(grids_to_use):
        all_combos.extend(list(itertools.product([config], [grid], grid_years_to_agg[grid.get('grid_name_s')], config.get('fields'))))
    
    with Pool(processes=user_cpus) as pool:
        pool.starmap(make_agg_job_and_aggregate, all_combos)
        pool.close()
        pool.join()
        
    update_body = []
    aggregation_status = get_agg_status(config['ds_name'])
    
    ds_meta = Aggregation(config).ds_meta
    
    # Update Solr dataset entry status
    update_body = [
        {
            "id":ds_meta['id'],
            "aggregation_version_s": {"set": str(config['a_version'])},
            "aggregation_status_s": {"set": aggregation_status}
        }
    ]

    r = solr_utils.solr_update(update_body, r=True)

    if r.status_code == 200:
        logging.debug(f'Successfully updated Solr with aggregation information for {config["ds_name"]}')
    else:
        logging.exception(f'Failed to update Solr dataset entry with aggregation information for {config["ds_name"]}')

    return aggregation_status