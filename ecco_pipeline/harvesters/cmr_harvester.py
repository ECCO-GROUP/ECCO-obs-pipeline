import logging
import os
from datetime import datetime

import numpy as np
import requests
import xarray as xr
from harvesters.enumeration.cmr_enumerator import cmr_search
from harvesters.granule import Granule
from harvesters.harvester import Harvester
from utils import file_utils, harvesting_utils


class CMR_Harvester(Harvester):
    
    def __init__(self, config):
        Harvester.__init__(self, config)
        if config['end'] == 'NOW':
            config['end'] = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")
        self.urls = cmr_search(config)
    
    def fetch(self):
        for url, id, updated in self.urls:
            filename = url.split('/')[-1]
            # Get date from filename and convert to dt object
            date = file_utils.get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start_time_dt <= dt) and (self.end_time_dt >= dt):
                continue
            
            year = str(dt.year)

            local_fp = f'{self.target_dir}{year}/{filename}'

            if not os.path.exists(f'{self.target_dir}{year}/'):
                os.makedirs(f'{self.target_dir}{year}/')

            if not updated:
                modified_time = self.get_mod_time(id)
            else:
                modified_time = datetime.strptime(f'{updated.split(".")[0]}Z', self.solr_format)
                
            if harvesting_utils.check_update(self.solr_docs, filename, modified_time):
                success = True
                granule = Granule(self.ds_name, local_fp, dt, modified_time, url)
                
                if self.need_to_download(granule):
                    logging.info(f'Downloading {filename} to {local_fp}')
                    try:
                        self.dl_file(url, local_fp)
                    except:
                        success = False
                else:
                    logging.debug(f'{filename} already downloaded and up to date')
                    
                granule.update_item(self.solr_docs, success)
                granule.update_descendant(self.descendant_docs, success)
                self.updated_solr_docs.extend(granule.get_solr_docs())
        logging.info(f'Downloading {self.ds_name} complete')
            
    def get_mod_time(self, id: str) -> datetime:
        meta_url = f'https://cmr.earthdata.nasa.gov/search/concepts/{id}.json'
        r = requests.get(meta_url)
        meta = r.json()
        modified_time = datetime.strptime(f'{meta["updated"].split(".")[0]}Z', self.solr_format)
        return modified_time
    
    def dl_file(self, src: str, dst: str):
        r = requests.get(src)
        r.raise_for_status()
        open(dst, 'wb').write(r.content)


    def fetch_atl_daily(self):
        for url, id, updated in self.urls:
            filename = url.split('/')[-1]
            # Get date from filename and convert to dt object
            date = file_utils.get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start_time_dt <= dt) and (self.end_time_dt >= dt):
                continue
            
            year = str(dt.year)
            month = str(dt.month)

            local_fp = f'{self.target_dir}{year}/{filename}'

            if not os.path.exists(f'{self.target_dir}{year}/'):
                os.makedirs(f'{self.target_dir}{year}/')

            if not updated:
                modified_time = self.get_mod_time(id)
            else:
                modified_time = datetime.strptime(f'{updated.split(".")[0]}Z', self.solr_format)
                
            if harvesting_utils.check_update(self.solr_docs, filename, modified_time):
                native_granule = Granule(self.ds_name, local_fp, dt, modified_time, url)

                if self.need_to_download(native_granule):
                    logging.info(f'Downloading {filename} to {local_fp}')
                    try:
                        self.dl_file(url, local_fp)
                        ds = xr.open_dataset(local_fp, decode_times=True)
                        ds = ds[['grid_x', 'grid_y', 'crs']]
                        
                        for i in range(1,32):
                            success = True
                            day_number = str(i).zfill(2)
                            try:
                                var_ds = xr.open_dataset(local_fp, group=f'daily/day{day_number}')
                            except:
                                continue
                            mid_date = (var_ds.delta_time_beg.values[0] + ((var_ds.delta_time_end.values[0] - var_ds.delta_time_beg.values[0]) / 2)).astype(str)[:10]
                            date = np.datetime64(mid_date)
                            dt = datetime(int(year), int(month), i)
                            time_var_ds = var_ds.expand_dims({'time': [date]})
                            time_var_ds = time_var_ds[[field['name'] for field in self.config['fields']]]
                            merged_ds = xr.merge([ds, time_var_ds])
                            
                            daily_filename = filename[:9] + year + month + day_number + filename[-10:-3] + '.nc'
                            daily_local_fp = f'{self.target_dir}{year}/{daily_filename}'
                            merged_ds.to_netcdf(daily_local_fp)
                            
                            daily_granule = Granule(self.ds_name, daily_local_fp, dt, modified_time, url)
                            daily_granule.update_item(self.solr_docs, success)
                            daily_granule.update_descendant(self.descendant_docs, success)
                            self.updated_solr_docs.extend(daily_granule.get_solr_docs())
                    except Exception as e:
                        print(e)
                        success = False
                else:
                    logging.debug(f'{filename} already downloaded and up to date')
        logging.info(f'Downloading {self.ds_name} complete')


def harvester(config: dict) -> str:
    """
    Uses CMR search to find granules within date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, and descendants.
    """

    harvester = CMR_Harvester(config)
    if harvester.ds_name == 'ATL20_V004_daily':
        harvester.fetch_atl_daily()
    else:
        harvester.fetch()
    harvesting_status = harvester.post_fetch()
    return harvesting_status