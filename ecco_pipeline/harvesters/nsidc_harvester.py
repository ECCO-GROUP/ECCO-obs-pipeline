import logging
import os
from datetime import datetime
from typing import Iterable

import requests
from harvesters.enumeration.nsidc_enumerator import NSIDCGranule, search_nsidc
from harvesters.harvesterclasses import Granule, Harvester
from utils.pipeline_utils.file_utils import get_date

logger = logging.getLogger('pipeline')


class NSIDC_Harvester(Harvester):
    
    def __init__(self, config: dict):
        Harvester.__init__(self, config)
        self.nsidc_granules: Iterable[NSIDCGranule] = search_nsidc(self)
    
    def fetch(self):
        for nsidc_granule in self.nsidc_granules:
            filename = nsidc_granule.url.split('/')[-1]
            # Get date from filename and convert to dt object
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start <= dt) and (self.end >= dt):
                continue
            
            year = str(dt.year)

            local_fp = f'{self.target_dir}{year}/{filename}'

            if not os.path.exists(f'{self.target_dir}{year}/'):
                os.makedirs(f'{self.target_dir}{year}/')
                
            if self.check_update(filename, nsidc_granule.mod_time):
                success = True
                granule = Granule(self.ds_name, local_fp, dt, nsidc_granule.mod_time, nsidc_granule.url)
                
                if self.need_to_download(granule):
                    logger.info(f'Downloading {filename} to {local_fp}')
                    try:
                        self.dl_file(nsidc_granule.url, local_fp)
                    except:
                        success = False
                else:
                    logger.debug(f'{filename} already downloaded and up to date')
                    
                granule.update_item(self.solr_docs, success)
                granule.update_descendant(self.descendant_docs, success)
                self.updated_solr_docs.extend(granule.get_solr_docs())
        logger.info(f'Downloading {self.ds_name} complete')
        
    
    def dl_file(self, src: str, dst: str):
        r = requests.get(src)
        r.raise_for_status()
        open(dst, 'wb').write(r.content)


def harvester(config: dict) -> str:
    """
    Uses CMR search to find granules within date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, and descendants.
    """

    harvester = NSIDC_Harvester(config)    
    harvester.fetch()
    harvesting_status = harvester.post_fetch()
    return harvesting_status