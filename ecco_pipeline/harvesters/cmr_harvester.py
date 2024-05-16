import calendar
import logging
import os
from datetime import datetime, timedelta
import time
from typing import Iterable

import numpy as np
import requests
import xarray as xr
from harvesters.enumeration.cmr_enumerator import CMRGranule, CMRQuery
from harvesters.harvesterclasses import Granule, Harvester
from utils.pipeline_utils.file_utils import get_date
from utils.processing_utils.records import TimeBound

logger = logging.getLogger('pipeline')

class CMR_Harvester(Harvester):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.cmr_granules: Iterable[CMRGranule] = CMRQuery(self).query()           
    
    def fetch(self):
        for cmr_granule in self.cmr_granules:
            filename = cmr_granule.url.split('/')[-1]
            if 'NRT' in filename:
                continue
            # Get date from filename and convert to dt object
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start <= dt) and (self.end >= dt):
                continue
            
            year = str(dt.year)

            local_fp = os.path.join(self.target_dir, year, filename)
            os.makedirs(os.path.join(self.target_dir, year), exist_ok=True)
                
            if self.check_update(filename, cmr_granule.mod_time):
                success = True
                granule = Granule(self.ds_name, local_fp, dt, cmr_granule.mod_time, cmr_granule.url)
                
                if self.need_to_download(granule):
                    logger.info(f'Downloading {filename} to {local_fp}')
                    try:
                        self.dl_file(cmr_granule.url, local_fp)
                    except:
                        success = False
                else:
                    logger.debug(f'{filename} already downloaded and up to date')
                    
                granule.update_item(self.solr_docs, success)
                granule.update_descendant(self.descendant_docs, success)
                self.updated_solr_docs.extend(granule.get_solr_docs())
        logger.info(f'Downloading {self.ds_name} complete')
    
    def dl_file(self, src: str, dst: str):
        try:
            r = requests.get(src)
            r.raise_for_status()
            with open(dst, 'wb') as f:
                f.write(r.content)
        except:
            time.sleep(5)
            r = requests.get(src)
            r.raise_for_status()
            with open(dst, 'wb') as f:
                f.write(r.content)

    def fetch_atl_daily(self):
        for cmr_granule in self.cmr_granules:
            filename = cmr_granule.url.split('/')[-1]
            if 'NRT' in filename:
                continue
            # Get date from filename and convert to dt object
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start <= dt) and (self.end >= dt):
                continue
            
            year = str(dt.year)
            month = str(dt.month)

            local_fp = f'{self.target_dir}{year}/{filename}'

            if not os.path.exists(f'{self.target_dir}{year}/'):
                os.makedirs(f'{self.target_dir}{year}/')
                
            native_granule = Granule(self.ds_name, local_fp, dt, cmr_granule.mod_time, cmr_granule.url)

            if self.need_to_download(native_granule):
                logger.info(f'Downloading {filename} to {local_fp}')
                try:
                    self.dl_file(cmr_granule.url, local_fp)
                except Exception as e:
                    logger.warning(e)
                    success = False
            else:
                logger.info(f'{year}-{str(month).zfill(2)} monthly file up to date. Slicing to ensure entries in Solr...')
            
            if success:        
                base_ds = xr.open_dataset(local_fp, decode_times=True)
                base_ds = base_ds[['grid_x', 'grid_y', 'crs']]
                
                # Pull out daily slices from monthly granule
                for i in range(1,32):
                    
                    if not self.check_update(filename, cmr_granule.mod_time):
                        continue
                    
                    success = True
                    day_number = str(i).zfill(2)
                    try:
                        datetime(dt.year,dt.month,i)
                    except:
                        continue
                    daily_filename = filename[:9] + year + str(month).zfill(2) + day_number + filename[-10:-3] + '.nc'
                    daily_local_fp = f'{self.target_dir}{year}/{daily_filename}'

                    try:
                        var_ds = xr.open_dataset(local_fp, group=f'daily/day{day_number}')
                    except:
                        continue
                    mid_date = (var_ds.delta_time_beg.values[0] + ((var_ds.delta_time_end.values[0] - var_ds.delta_time_beg.values[0]) / 2)).astype(str)[:10]
                    date = np.datetime64(mid_date).astype('datetime64[ns]')
                    time_var_ds = var_ds.expand_dims({'time': [date]})
                    time_var_ds = time_var_ds[[field.name for field in self.fields]]
                    merged_ds = xr.merge([base_ds, time_var_ds])
                    merged_ds.to_netcdf(daily_local_fp)
                    
                    try:
                        daily_dt = datetime(int(year), int(month), i)
                        daily_granule = Granule(self.ds_name, daily_local_fp, daily_dt, cmr_granule.mod_time, cmr_granule.url)
                        daily_granule.update_item(self.solr_docs, success)
                        daily_granule.update_descendant(self.descendant_docs, success)
                        self.updated_solr_docs.extend(daily_granule.get_solr_docs())
                    except:
                        logger.debug(f'{year}-{str(month).zfill(2)}-{day_number} unable to be sliced. Daily data likely missing in monthly file.')
        logger.info(f'Downloading {self.ds_name} complete')


    def fetch_tellus_grac_grfo(self):
        for cmr_granule in self.cmr_granules:
            filename = cmr_granule.url.split('/')[-1]
            if 'NRT' in filename:
                continue
            # Get date from filename and convert to dt object
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start <= dt) and (self.end >= dt):
                continue

            local_fp = f'{self.target_dir}/{filename}'

            if not os.path.exists(f'{self.target_dir}/'):
                os.makedirs(f'{self.target_dir}/')

            if not os.path.exists(local_fp) or datetime.fromtimestamp(os.path.getmtime(local_fp)) < cmr_granule.mod_time:
                logger.info(f'Downloading {filename} to {local_fp}')
                self.dl_file(cmr_granule.url, local_fp)
                downloaded = True
            else:
                logger.info(f'File up to date. Slicing to ensure entries in Solr...')
                downloaded = False
            
            ds = xr.open_dataset(local_fp, decode_times=True)
            
            # Extract time coverage from file metadata
            time_start = np.datetime64(ds.attrs['time_coverage_start'][:-1]).astype('datetime64[M]')
            time_end = np.datetime64(ds.attrs['time_coverage_end'][:-1]).astype('datetime64[M]') + 1
            
            # Construct months within time coverage
            months = np.arange(time_start, time_end, 1, dtype='datetime64[M]')
            # Compute monthly centertimes
            monthly_cts = [TimeBound(rec_avg_start=month, period='AVG_MON').center for month in months]
            
            logger.info('Slicing aggregated granule into monthly granules...')

            for monthly_center in monthly_cts:
                try:
                    success = True
                    sub_ds = ds.sel(time=np.datetime64(monthly_center), method = 'nearest')
                    sub_ds_time = sub_ds.time.values
                    
                    # Check if slice is within +/- 7 day tolerance
                    if not (sub_ds_time >= monthly_center - np.timedelta64(7, 'D') and sub_ds_time <= monthly_center + np.timedelta64(7, 'D')):
                        logger.info(f'No slice found within 7 day tolerance for {monthly_center}')
                        continue

                    time_dt = datetime.strptime(str(monthly_center.astype('datetime64[M]')), "%Y-%m")
                    filename_time = str(time_dt)[:10].replace('-', '')

                    slice_filename = f'{self.ds_name}_{filename_time}.nc'
                    slice_local_fp = f'{self.target_dir}{str(time_dt.year)}/{slice_filename}'
                    os.makedirs(f'{self.target_dir}{str(time_dt.year)}', exist_ok=True)
                    if downloaded:
                        sub_ds.to_netcdf(slice_local_fp)
                    
                except Exception as e:
                    logger.error(f'Error making granule slice: {e}')
                    success = False
                    
                logger.debug(f'Monthly center: {monthly_center}, data_slice_time: {sub_ds_time}, time for solr: {str(time_dt)}')
                monthly_granule = Granule(self.ds_name, slice_local_fp, time_dt, cmr_granule.mod_time, cmr_granule.url)
                monthly_granule.update_item(self.solr_docs, success)
                monthly_granule.update_descendant(self.descendant_docs, success)
                self.updated_solr_docs.extend(monthly_granule.get_solr_docs())

        logger.info(f'Downloading {self.ds_name} complete')


    def fetch_rdeft4(self):
        
        # Data filenames contain end of time coverage. To get data covering an
        # entire calendar month, we only look at files with the last day of the
        # month in the filename.
        
        # Consider looking for closest date (within 2 or 3 days) to end of month if end of month is unavailable
        
        years = np.arange(int(self.start.year), int(self.end.year) + 1)
        end_of_month = [datetime(year, month, calendar.monthrange(year,month)[1]) for year in years for month in range(1,13)]
        url_dict = {granule.url.split('RDEFT4_')[-1].split('.')[0]: granule for granule in self.cmr_granules} # granule end date:url

        end_of_month_granules = []
        for month_end in end_of_month:
            month_end_str = month_end.strftime(self.config['filename_date_fmt'])
            if month_end_str in url_dict.keys():
                end_of_month_granules.append(url_dict[month_end_str])
            else:
                for tolerance_num in range(-1,-3,-1):
                    new_date = month_end + timedelta(days=tolerance_num)
                    month_end_str = new_date.strftime(self.config['filename_date_fmt'])
                    if month_end_str in url_dict.keys():
                        end_of_month_granules.append(url_dict[month_end_str])
                        break

        self.cmr_granules: Iterable[CMRGranule] = end_of_month_granules

        for cmr_granule in self.cmr_granules:
            filename = cmr_granule.url.split('/')[-1]
            if 'NRT' in filename:
                continue
            # Get date from filename and convert to dt object
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            
            # Force date to be first of the month
            dt = dt.replace(day=1)
            
            if not (self.start <= dt) and (self.end >= dt):
                continue
            
            year = str(dt.year)

            local_fp = f'{self.target_dir}{year}/{filename}'

            if not os.path.exists(f'{self.target_dir}{year}/'):
                os.makedirs(f'{self.target_dir}{year}/')
                
            if self.check_update(filename, cmr_granule.mod_time):
                success = True
                granule = Granule(self.ds_name, local_fp, dt, cmr_granule.mod_time, cmr_granule.url)
                
                if self.need_to_download(granule):
                    logger.info(f'Downloading {filename} to {local_fp}')
                    try:
                        self.dl_file(cmr_granule.url, local_fp)
                    except:
                        success = False
                else:
                    logger.debug(f'{filename} already downloaded and up to date')
                    
                granule.update_item(self.solr_docs, success)
                granule.update_descendant(self.descendant_docs, success)
                self.updated_solr_docs.extend(granule.get_solr_docs())
        logger.info(f'Downloading {self.ds_name} complete')
                

    def fetch_tolerance_filter(self):
        sorted_granules = sorted(self.cmr_granules, key=lambda x: x.url)
        sorted_granule_dict = {np.datetime64(str(datetime.strptime(get_date(self.filename_date_regex, g.url.split('/')[-1]), self.filename_date_fmt))[:10]) : g for g in sorted_granules}

        filename_time = get_date(self.filename_date_regex, sorted_granules[0].url.split('/')[-1])
        time_start = datetime.strptime(filename_time, self.filename_date_fmt)
        
        filename_time = get_date(self.filename_date_regex, sorted_granules[-1].url.split('/')[-1])
        time_end = datetime.strptime(filename_time, self.filename_date_fmt)
        
        # Construct months within time coverage
        months = np.arange(time_start.strftime('%Y-%m-01'), time_end.strftime('%Y-%m-01'), 1, dtype='datetime64[M]')
        months = [m.astype('datetime64[D]') for m in months]
        
        granules_to_use = []
        for month in months:
            delta = [abs(x - month) for x in list(sorted_granule_dict.keys())]
            idx = np.argmin(delta)
            nearest_key = list(sorted_granule_dict.keys())[idx]
            
            if nearest_key >= month - np.timedelta64(7, 'D') and nearest_key <= month + np.timedelta64(7, 'D'):
                granules_to_use.append(sorted_granule_dict[nearest_key])
            else:
                logger.info(f'Granule nearest to {month} ({nearest_key}) is outside of tolerance window. Skipping.')
        
        for cmr_granule in self.cmr_granules:
            filename = cmr_granule.url.split('/')[-1]
            if 'NRT' in filename:
                continue
            # Get date from filename and convert to dt object
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start <= dt) and (self.end >= dt):
                continue
            
            year = str(dt.year)

            local_fp = f'{self.target_dir}{year}/{filename}'

            if not os.path.exists(f'{self.target_dir}{year}/'):
                os.makedirs(f'{self.target_dir}{year}/')
                
            if self.check_update(filename, cmr_granule.mod_time):
                success = True
                granule = Granule(self.ds_name, local_fp, dt, cmr_granule.mod_time, cmr_granule.url)
                
                if self.need_to_download(granule):
                    logger.info(f'Downloading {filename} to {local_fp}')
                    try:
                        self.dl_file(cmr_granule.url, local_fp)
                    except:
                        success = False
                else:
                    logger.debug(f'{filename} already downloaded and up to date')
                    
                granule.update_item(self.solr_docs, success)
                granule.update_descendant(self.descendant_docs, success)
                self.updated_solr_docs.extend(granule.get_solr_docs())
        logger.info(f'Downloading {self.ds_name} complete')

def harvester(config: dict) -> str:
    """
    Uses CMR search to find granules within date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, and descendants.
    """

    harvester = CMR_Harvester(config)
        
    if harvester.ds_name == 'ATL20_V004_daily' or harvester.ds_name == 'ATL21_V003_daily':
        harvester.fetch_atl_daily()
    elif harvester.ds_name == 'TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3':
        harvester.fetch_tellus_grac_grfo()
    elif harvester.ds_name == 'RDEFT4':
        harvester.fetch_rdeft4()
    elif 'TELLUS_GR' in harvester.ds_name:
        harvester.fetch_tolerance_filter()
    else:
        harvester.fetch()
    harvesting_status = harvester.post_fetch()
    return harvesting_status