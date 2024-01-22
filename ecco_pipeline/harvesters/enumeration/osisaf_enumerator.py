from dataclasses import dataclass
import logging
import requests
import os
import lxml
from bs4 import BeautifulSoup
from datetime import datetime

from harvesters.harvester import Harvester

OSISAF_URL = 'https://thredds.met.no/thredds/catalog/osisaf/met.no/reprocessed/ice/'
    
def search_osisaf(harvester: Harvester):
    harvester.data_time_scale
    logging.info(f'Searching OSISAF for {harvester.ds_name} granules...')
    
    if harvester.data_time_scale == 'monthly':
        base_url = os.path.join(OSISAF_URL, harvester.ddir, 'monthly')
    else:
        base_url = os.path.join(OSISAF_URL, harvester.ddir)

    session = requests.session()
    r = session.get(os.path.join(base_url, 'catalog.xml'))
    data = BeautifulSoup(r.text, 'xml')
    all_granules = []
    for year_dir in data.find_all('catalogref'):
        if 'monthly' in year_dir['xlink:title']:
            continue
        r = session.get(os.path.join(base_url, year_dir['xlink:title'], 'catalog.xml'))
        year_data = BeautifulSoup(r.text, 'xml')

        if harvester.data_time_scale == 'daily':
            for month_dir in year_data.find_all('catalogref'):
                month_url = os.path.join(base_url, year_dir['xlink:title'], month_dir['xlink:title'], 'catalog.xml')
                r = session.get(month_url)
                month_data = BeautifulSoup(r.text, 'xml')
                for dataset in month_data.find_all('dataset'):
                    for granule in dataset.find_all('dataset'):
                        url = os.path.join('https://thredds.met.no/thredds/fileServer/', granule['urlpath'])
                        mod_time = datetime.strptime(granule.find('date').text, '%Y-%m-%dT%H:%M:%SZ')
                        all_granules.append(OSISAFGranule(url, mod_time))
        else:
            for dataset in year_data.find_all('dataset'):
                for granule in dataset.find_all('dataset'):
                    url = os.path.join('https://thredds.met.no/thredds/fileServer/', granule['urlpath'])
                    mod_time = datetime.strptime(granule.find('date').text, '%Y-%m-%dT%H:%M:%SZ')
                    all_granules.append(OSISAFGranule(url, mod_time))
    logging.info(f'Found {len(all_granules)} possible granules')
    return all_granules
    

@dataclass
class OSISAFGranule():
    url: str
    mod_time: datetime