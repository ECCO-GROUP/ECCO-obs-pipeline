from dataclasses import dataclass
import logging
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime

from harvesters.harvester import Harvester

NSIDC_URL = 'https://noaadata.apps.nsidc.org/NOAA/'
    
def search_nsidc(harvester: Harvester):
    logging.info(f'Searching NSIDC for {harvester.ds_name} granules...')
    date_range = range(harvester.start.year, harvester.end.year + 1)
    all_granules = []
    for hemi in ['north', 'south']:
        ds_hemi_url = os.path.join(NSIDC_URL, harvester.ds_name, hemi, 'daily')
        r = requests.get(ds_hemi_url)
        data = BeautifulSoup(r.text, "html.parser")
        years = [l["href"].replace('/', '') for l in data.find_all("a")[1:]]
        for year in years:
            if int(year) not in date_range:
                continue
            ds_hemi_year_url = os.path.join(ds_hemi_url, year)
            r = requests.get(ds_hemi_year_url)
            data = BeautifulSoup(r.text, "html.parser")
            for l in data.find_all("a")[1:]:
                url = os.path.join(ds_hemi_year_url, l["href"])
                tokens = l.next_sibling.split()
                mod_time = datetime.strptime(tokens[0] + ' ' + tokens[1], '%d-%b-%Y %H:%M')
                size = tokens[2]
                all_granules.append(NSIDCGranule(url, mod_time, size))
    logging.info(f'Found {len(all_granules)} possible granules')
    return all_granules

@dataclass
class NSIDCGranule():
    url: str
    mod_time: datetime
    size: int