from dataclasses import dataclass
import logging
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime

from harvesters.harvester import Harvester

logger = logging.getLogger('pipeline')


CATDS_URL = 'https://data.catds.fr/cecos-locean/Ocean_products/'
    
def search_catds(harvester: Harvester):
    logger.info(f'Searching CATDS for {harvester.ds_name} granules...')   
    all_granules = []
    ds_url = os.path.join(CATDS_URL, harvester.ddir)
    r = requests.get(ds_url)
    data = BeautifulSoup(r.text, "html.parser")
    for l in data.find_all("tr")[3:-1]:
        tokens = l.find_all("td")
        url = os.path.join(ds_url, tokens[1].find("a")['href'])
        mod_time = datetime.strptime(tokens[2].text, '%Y-%m-%d %H:%M  ')
        all_granules.append(CATDSGranule(url, mod_time))
    logger.info(f'Found {len(all_granules)} possible granules')
    return all_granules

@dataclass
class CATDSGranule():
    url: str
    mod_time: datetime