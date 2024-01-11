import os
import logging
from datetime import datetime
from harvesters.granule import Granule
from dataset import Dataset
from utils import solr_utils
from conf.global_settings import OUTPUT_DIR

logger = logging.getLogger('pipeline')

class Harvester(Dataset):
    
    solr_format:str = "%Y-%m-%dT%H:%M:%SZ"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._harvester_parsing(config)
        self.target_dir: str = f'{OUTPUT_DIR}/{self.ds_name}/harvested_granules/'        
        self.updated_solr_docs: list = []
        
        self.ensure_target_dir()
        solr_utils.clean_solr(config)
        
        self.solr_docs, self.descendant_docs = self.get_solr_docs()
        self.config: dict = config
    
    def _harvester_parsing(self, config: dict):
        if self.harvester_type == 'cmr':
            self.cmr_concept_id = config.get('cmr_concept_id')
            self.provider = config.get('provider')
        elif self.harvester_type == 'osisaf_ftp':       
            self.host = config.get('host')
            self.user = config.get('user')
            self.ddir = config.get('ddir')
            self.filename_filter = config.get('filename_filter')
        elif self.harvester_type == 'nsidc_ftp':
            self.host = config.get('host')
            self.user = config.get('user')
            self.ddir = config.get('ddir')
        elif self.harvester_type == 'ifremer_ftp':
            self.host = config.get('host')
            self.user = config.get('user')
            self.password = config.get('password')
            self.ddir = config.get('ddir')
            self.filename_filter = config.get('filename_filter')
      
    def fetch(self):
        raise NotImplementedError
    
    def get_mod_time(self):
        raise NotImplementedError
    
    def dl_file(self):
        raise NotImplementedError
        
    def ensure_target_dir(self):
        os.makedirs(self.target_dir, exist_ok=True)
       
    def get_solr_docs(self) -> list:
        docs = {}
        descendants_docs = {}

        # Query for existing harvested docs
        fq = ['type_s:granule', f'dataset_s:{self.ds_name}']
        harvested_docs = solr_utils.solr_query(fq)

        # Dictionary of existing harvested docs
        # harvested doc filename : solr entry for that doc
        if len(harvested_docs) > 0:
            for doc in harvested_docs:
                docs[doc['filename_s']] = doc

        # Query for existing descendants docs
        fq = ['type_s:descendants', f'dataset_s:{self.ds_name}']
        existing_descendants_docs = solr_utils.solr_query(fq)

        # Dictionary of existing descendants docs
        # descendant doc date : solr entry for that doc
        if len(existing_descendants_docs) > 0:
            for doc in existing_descendants_docs:
                if 'hemisphere_s' in doc.keys() and doc['hemisphere_s']:
                    key = (doc['date_s'], doc['hemisphere_s'])
                else:
                    key = doc['date_s']
                descendants_docs[key] = doc

        return docs, descendants_docs
    
    def make_ds_doc(self, source: str, chk_time: str):
        ds_meta = {}
        ds_meta['type_s'] = 'dataset'
        ds_meta['dataset_s'] = self.ds_name
        ds_meta['short_name_s'] = self.og_ds_metadata['original_dataset_short_name']
        ds_meta['source_s'] = source
        ds_meta['data_time_scale_s'] = self.data_time_scale
        ds_meta['last_checked_dt'] = chk_time
        ds_meta['original_dataset_title_s'] = self.og_ds_metadata['original_dataset_title']
        ds_meta['original_dataset_short_name_s'] = self.og_ds_metadata['original_dataset_short_name']
        ds_meta['original_dataset_url_s'] = self.og_ds_metadata['original_dataset_url']
        ds_meta['original_dataset_reference_s'] = self.og_ds_metadata['original_dataset_reference']
        ds_meta['original_dataset_doi_s'] = self.og_ds_metadata['original_dataset_doi']
        return ds_meta
    
    def check_update(self, filename, mod_time):
        return (not filename in self.solr_docs.keys()) or \
            (not self.solr_docs[filename]['harvest_success_b']) or \
            (self.solr_docs[filename]['download_time_dt'] < str(mod_time))
    
    def need_to_download(self, granule: Granule) -> bool:
        if not os.path.exists(granule.local_fp):
            return True
        # If file exists locally, but is out of date, download it
        elif datetime.fromtimestamp(os.path.getmtime(granule.local_fp)) <= granule.modified_time:
            return True
        return False
    
    def post_fetch(self) -> str:
        check_time = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
        
        if self.updated_solr_docs:
            r = solr_utils.solr_update(self.updated_solr_docs, r=True)
            if r.status_code == 200:
                logger.debug(
                    'Successfully created or updated Solr harvested documents')
            else:
                logger.exception('Failed to create Solr harvested documents')
        else:
            logger.debug('No downloads required.')
            
        harvesting_status = self.harvester_status()
        
        # Query for Solr dataset level document
        fq = ['type_s:dataset', f'dataset_s:{self.ds_name}']
        ds_doc = solr_utils.solr_query(fq)
        
        if ds_doc:
            # -----------------------------------------------------
            # Update Solr dataset entry
            # -----------------------------------------------------
            dataset_metadata = ds_doc[0]

            # Query for dates of all harvested docs
            fq = [f'dataset_s:{self.ds_name}', 'type_s:granule', 'harvest_success_b:true']
            dates_query = solr_utils.solr_query(fq, fl='date_s')
            dates = [x['date_s'] for x in dates_query]

            # Build update document body
            ds_meta = {}
            ds_meta['id'] = dataset_metadata['id']
            ds_meta['last_checked_dt'] = {"set": check_time}
            if dates:
                ds_meta['start_date_dt'] = {"set": min(dates)}
                ds_meta['end_date_dt'] = {"set": max(dates)}

            if self.updated_solr_docs:
                ds_meta['harvest_status_s'] = {"set": harvesting_status}
                dl_solr_docs = [doc for doc in self.updated_solr_docs if 'download_time_dt' in doc.keys()]
                last_dl_item = sorted(dl_solr_docs, key=lambda d: d['download_time_dt'])[-1]
                ds_meta['last_download_dt'] = {"set": last_dl_item['download_time_dt']}
        else:
            # -----------------------------------------------------
            # Create Solr Dataset-level Document if doesn't exist
            # -----------------------------------------------------
            source = f'https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/{self.ds_name}/'
            ds_meta = self.make_ds_doc(source, check_time)

            # Only include start_date and end_date if there was at least one successful download
            if self.updated_solr_docs:
                ds_meta['harvest_status_s'] = {"set": harvesting_status}
                dl_solr_docs = [doc for doc in self.updated_solr_docs if 'download_time_dt' in doc.keys()]
                dl_items = sorted(dl_solr_docs, key=lambda d: d['date_s'])
                ds_meta['start_date_dt'] = dl_items[0]['date_s']
                ds_meta['end_date_dt'] = dl_items[-1]['date_s']
                ds_meta['last_download_dt'] = sorted(dl_solr_docs, key=lambda d: d['download_time_dt'])[-1]['download_time_dt']

            ds_meta['harvest_status_s'] = harvesting_status
            
        # Update Solr with modified dataset entry
        r = solr_utils.solr_update([ds_meta], r=True)

        if r.status_code == 200:
            logger.debug('Successfully updated Solr dataset document')
        else:
            logger.exception('Failed to update Solr dataset document')
        return harvesting_status
    
    def harvester_status(self) -> str:
        # Query for Solr failed harvest documents
        fq = ['type_s:granule', f'dataset_s:{self.ds_name}', f'harvest_success_b:false']
        failed_harvesting = solr_utils.solr_query(fq)

        # Query for Solr successful harvest documents
        fq = ['type_s:granule', f'dataset_s:{self.ds_name}', f'harvest_success_b:true']
        successful_harvesting = solr_utils.solr_query(fq)

        harvest_status = f'All granules successfully harvested'
        
        if not successful_harvesting:
            harvest_status = f'No usable granules harvested (either all failed or no data collected)'
        elif failed_harvesting:
            harvest_status = f'{len(failed_harvesting)} harvested granules failed'
            
        return harvest_status
    
    def ds_doc_update(self) -> bool:
        # Query for Solr dataset level document
        fq = ['type_s:dataset', f'dataset_s:{self.ds_name}']
        dataset_query = solr_utils.solr_query(fq)

        # If dataset entry exists on Solr
        return len(dataset_query) == 1