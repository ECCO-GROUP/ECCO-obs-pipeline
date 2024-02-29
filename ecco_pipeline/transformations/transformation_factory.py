import logging
from collections import defaultdict
from multiprocessing import cpu_count, current_process, Pool
import os
from typing import Iterable
import xarray as xr

from baseclasses import Dataset
from utils.pipeline_utils import solr_utils, log_config
from transformations.grid_transformation import Transformation, transform


logger = logging.getLogger('pipeline')


def multiprocess_transformation(config: dict, granule: dict, tx_jobs: dict, log_level: str, log_dir: str):
    """
    Callable function that performs the actual transformation on a granule.
    """
    try:
        logger = log_config.mp_logging(str(current_process().pid), log_level, log_dir)
    except Exception as e:
        print(e)

    granule_filepath = granule.get('pre_transformation_file_path_s')
    granule_date = granule.get('date_s')

    # Skips granules that weren't harvested properly
    if not granule_filepath or granule.get('file_size_l') < 100:
        logger.exception(f'Granule {granule_filepath} was not harvested properly. Skipping.')
        return

    # Perform remaining transformations
    try:
        logger.info(f'{sum([len(v) for v in tx_jobs.values()])} remaining transformations for {granule_filepath.split("/")[-1]}')
        transform(granule_filepath, tx_jobs, config, granule_date)
    except Exception as e:
        logger.exception(f'Error transforming {granule_filepath}: {e}')
        
        
class TxJobFactory(Dataset):
    
    def __init__(self, config: dict, user_cpus: int = 1, grids_to_use: Iterable[str]=[]) -> None:
        super().__init__(config)
        self.config = config
        self.user_cpus = user_cpus
        self.harvested_granules = solr_utils.solr_query([f'dataset_s:{self.ds_name}', 'type_s:granule', 'harvest_success_b:true'])

        if not grids_to_use:
            fq = ['type_s:grid']
            docs = solr_utils.solr_query(fq)
            self.grids = [doc['grid_name_s'] for doc in docs]
        else:
            self.grids = grids_to_use
        
    def start_factory(self) -> str:
        if not self.harvested_granules:
            logger.info(f'No harvested granules found in solr for {self.ds_name}')
            return 'No transformations performed'
        self.initialize_jobs()
        if self.job_params:
            self.execute_jobs()
        else:
            return 'No transformations performed'
    
        pipeline_status = self.pipeline_cleanup()
        return pipeline_status
        
    def initialize_jobs(self):
        self.pregenerate_factors()
        self.job_params = self.generate_jobs()
        logger.info(f'{len(self.job_params)} harvested granules with remaining transformations.')
        
    def execute_jobs(self):
        if self.job_params:
            if self.user_cpus == 1:
                logger.info('Not using multiprocessing to do transformation')
                for job_param in self.job_params:
                    multiprocess_transformation(*job_param)
            else:
                user_cpus = min(self.user_cpus, int(cpu_count()/4), len(self.job_params))
                logger.info(f'Using {user_cpus} CPUs to do {len(self.job_params)} multiprocess transformation jobs')
                    
                with Pool(processes=user_cpus) as pool:
                    pool.starmap_async(multiprocess_transformation, self.job_params)
                    pool.close()
                    pool.join()
                    
    def pipeline_cleanup(self) -> str:
        # Query Solr for dataset metadata
        fq = [f'dataset_s:{self.ds_name}', 'type_s:dataset']
        dataset_metadata = solr_utils.solr_query(fq)[0]

        # Query Solr for successful transformation documents
        fq = [f'dataset_s:{self.ds_name}', 'type_s:transformation', 'success_b:true']
        successful_transformations = solr_utils.solr_query(fq)

        # Query Solr for failed transformation documents
        fq = [f'dataset_s:{self.ds_name}', 'type_s:transformation', 'success_b:false']
        failed_transformations = solr_utils.solr_query(fq)

        transformation_status = 'All transformations successful'

        if not successful_transformations and not failed_transformations:
            transformation_status = 'No transformations performed'
        elif not successful_transformations:
            transformation_status = 'No successful transformations'
        elif failed_transformations:
            transformation_status = f'{len(failed_transformations)} transformations failed'

        # Update Solr dataset entry status to transformed
        update_body = [{
            "id": dataset_metadata['id'],
            "transformation_status_s": {"set": transformation_status},
        }]

        r = solr_utils.solr_update(update_body, r=True)

        if r.status_code == 200:
            logger.debug(f'Successfully updated Solr with transformation information for {self.ds_name}')
        else:
            logger.exception(f'Failed to update Solr with transformation information for {self.ds_name}')

        return transformation_status
    
    def pregenerate_factors(self):
        '''
        Generates mapping factors for all grids used for the given transformation version. Loads them into
        Factors object which is used to reduce I/O.
        '''
        for grid in self.grids:
            for granule in self.find_data_for_factors():
                grid_ds = xr.open_dataset(f'grids/{grid}.nc')
                T = Transformation(self.config, granule['pre_transformation_file_path_s'], '1972-01-01')
                T.make_factors(grid_ds)
                
    def find_data_for_factors(self) -> Iterable[dict]:
        '''
        Returns Solr granule entry (two in the case of hemispherical data) to be used
        to generate factors
        '''
        data_for_factors = []
        nh_added = False
        sh_added = False
        # Find appropriate granule(s) to use for factor calculation
        for granule in self.harvested_granules:
            if 'hemisphere_s' in granule.keys():
                hemi = f'_{granule["hemisphere_s"]}_'
            else:
                hemi = ''
            if granule.get('pre_transformation_file_path_s'):
                if hemi:
                    # Get one of each
                    if hemi == self.hemi_pattern['north'] and not nh_added:
                        data_for_factors.append(granule)
                        nh_added = True
                    elif hemi == self.hemi_pattern['south'] and not sh_added:
                        data_for_factors.append(granule)
                        sh_added = True
                    if nh_added and sh_added:
                        return data_for_factors
                else:
                    data_for_factors.append(granule)
                    return data_for_factors
                
    def generate_jobs(self):
        logger.info('Generating jobs...')
        log_level = logging.getLevelName(logging.getLogger('pipeline').level)
        log_dir = os.path.dirname(logging.getLogger('pipeline').handlers[0].baseFilename)
        log_dir = os.path.join(log_dir[log_dir.find('logs/'):], f'tx_{self.ds_name}')
        
        all_jobs = self.get_tx_jobs()

        new_jobs = []
        for (granule, grid_fields) in all_jobs:
            job_params = (self.config, granule, grid_fields, log_level, log_dir)
            new_jobs.append(job_params)
        return new_jobs
    
    def get_tx_jobs(self):
        fq = [f'dataset_s:{self.ds_name}', 'type_s:transformation']
        solr_txs = solr_utils.solr_query(fq)
        tx_dict = defaultdict(list)
        for tx in solr_txs:
            tx_dict[tx['pre_transformation_file_path_s'].split('/')[-1]].append(tx)
        
        all_jobs = []
        for granule in self.harvested_granules:
            grid_fields = {}
            for grid in self.grids:
                fields_for_grid = []
                for field in self.fields:
                    update = True
                    for tx in tx_dict[granule['filename_s']]:
                        if tx['field_s'] == field.name and tx['grid_name_s'] == grid:
                            update = self.need_to_update(granule, tx)
                            break
                    if update:
                        fields_for_grid.append(field)
                if fields_for_grid:
                    grid_fields[grid] = fields_for_grid
            if grid_fields:
                all_jobs.append((granule, grid_fields))
        return all_jobs
    
    def need_to_update(self, granule: dict, tx: dict) -> bool:
        '''
        Triple if:
        1. do we have a version entry,
        2. compare transformation version number and current transformation version number
        3. compare checksum of harvested file (currently in solr) and checksum
        of the harvested file that was previously transformed (recorded in transformation entry)
        '''
        if tx.get('success_b') and tx.get('transformation_version_f') == self.t_version and tx['origin_checksum_s'] == granule['checksum_s']:
            return False
        return True