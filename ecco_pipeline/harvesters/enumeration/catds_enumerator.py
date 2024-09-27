import logging
import os
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from harvesters.harvesterclasses import Harvester

logger = logging.getLogger("pipeline")


CATDS_URL = "https://data.catds.fr/cecos-locean/Ocean_products/"


def search_catds(harvester: Harvester):
    logger.info(f"Searching CATDS for {harvester.ds_name} granules...")
    all_granules = []
    ds_url = os.path.join(CATDS_URL, harvester.ddir)
    r = requests.get(ds_url)
    data = BeautifulSoup(r.text, "html.parser")
    for row in data.find_all("tr")[3:-1]:
        tokens = row.find_all("td")
        url = os.path.join(ds_url, tokens[1].find("a")["href"])
        mod_time = datetime.strptime(tokens[2].text, "%Y-%m-%d %H:%M  ")
        all_granules.append(CATDSGranule(url, mod_time))
    logger.info(f"Found {len(all_granules)} possible granules. Checking for updates...")
    return all_granules


@dataclass
class CATDSGranule:
    url: str
    mod_time: datetime
