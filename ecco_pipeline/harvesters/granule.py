import os
from datetime import datetime

from utils.file_utils import md5

class Granule():
    
    def __init__(self, ds_name: str, local_fp: str, date: str, modified_time: datetime, url: str):
        self.ds_name = ds_name
        self.local_fp = local_fp
        self.filename = local_fp.split('/')[-1]
        self.datetime = date
        self.modified_time = modified_time 
        self.url = url
        self.gen_granule_doc()
        self.gen_descendant_doc()

    def gen_granule_doc(self):
        item = {}
        item['type_s'] = 'granule'
        item['date_s'] = datetime.strftime(self.datetime, "%Y-%m-%dT00:00:00Z")
        item['dataset_s'] = self.ds_name
        item['filename_s'] = self.filename
        item['source_s'] = self.url
        item['modified_time_dt'] = self.modified_time.strftime("%Y-%m-%dT00:00:00Z")
        self.solr_item = item
        
    def gen_descendant_doc(self):
        descendant_item = {}
        descendant_item['type_s'] = 'descendants'
        descendant_item['date_s'] = datetime.strftime(self.datetime, "%Y-%m-%dT00:00:00Z")
        descendant_item['dataset_s'] = self.ds_name
        descendant_item['filename_s'] = self.filename
        descendant_item['source_s'] = self.url
        self.descendant_item = descendant_item
        
    def update_item(self, solr_docs, success):
        if self.filename in solr_docs.keys():
            self.solr_item['id'] = solr_docs[self.filename]['id']
            
        if success:
            # calculate checksum and expected file size
            self.solr_item['checksum_s'] = md5(self.local_fp)
            self.solr_item['pre_transformation_file_path_s'] = self.local_fp
            self.solr_item['granule_file_path_s'] = self.local_fp
            self.solr_item['harvest_success_b'] = True
            self.solr_item['file_size_l'] = os.path.getsize(self.local_fp)
        else:
            self.solr_item['harvest_success_b'] = False
            self.solr_item['filename_s'] = ''
            self.solr_item['pre_transformation_file_path_s'] = ''
            self.solr_item['file_size_l'] = 0
        self.solr_item['download_time_dt'] = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
        
    def update_descendant(self, descendants_docs, success):
        # Update Solr entry using id if it exists
        key = self.descendant_item['date_s']

        if key in descendants_docs.keys():
            self.descendant_item['id'] = descendants_docs[key]['id']

        self.descendant_item['harvest_success_b'] = success
        self.descendant_item['pre_transformation_file_path_s'] = self.solr_item.get('pre_transformation_file_path_s')

    def get_solr_docs(self):
        return [self.solr_item, self.descendant_item]