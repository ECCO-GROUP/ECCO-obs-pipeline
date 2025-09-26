import logging
import os
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from harvesters.harvesterclasses import Harvester

logger = logging.getLogger("pipeline")


NSIDC_URL = "https://noaadata.apps.nsidc.org/NOAA/"


def search_nsidc(harvester: Harvester):
    logger.info(f"Searching NSIDC for {harvester.ds_name} granules...")
    date_range = range(harvester.start.year, harvester.end.year + 1)
    all_granules = []
    for hemi in ["north", "south"]:
        if harvester.ddir:
            ds_hemi_url = os.path.join(NSIDC_URL, harvester.ds_name, harvester.ddir, hemi, "daily")
        else:
            ds_hemi_url = os.path.join(NSIDC_URL, harvester.ds_name, hemi, "daily")
        r = requests.get(ds_hemi_url)
        data = BeautifulSoup(r.text, "html.parser")
        years = [link["href"].replace("/", "") for link in data.find_all("a")[1:]]
        for year in years:
            if int(year) not in date_range:
                continue
            ds_hemi_year_url = os.path.join(ds_hemi_url, year)
            r = requests.get(ds_hemi_year_url)
            data = BeautifulSoup(r.text, "html.parser")
            for link in data.find_all("a")[1:]:
                url = os.path.join(ds_hemi_year_url, link["href"])
                tokens = link.next_sibling.split()
                mod_time = datetime.strptime(tokens[0] + " " + tokens[1], "%d-%b-%Y %H:%M")
                all_granules.append(NSIDCGranule(url, mod_time))
    logger.info(f"Found {len(all_granules)} possible granules. Checking for updates...")
    return all_granules


@dataclass
class NSIDCGranule:
    url: str
    mod_time: datetime
