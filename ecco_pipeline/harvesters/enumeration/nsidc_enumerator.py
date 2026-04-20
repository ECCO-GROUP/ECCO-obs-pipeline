import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from harvesters.harvesterclasses import Harvester

logger = logging.getLogger("pipeline")


NSIDC_URL = "https://noaadata.apps.nsidc.org/NOAA/"
MAX_WORKERS = 4


def _fetch_year_granules(ds_hemi_url: str, year: str) -> list:
    ds_hemi_year_url = os.path.join(ds_hemi_url, year)
    r = requests.get(ds_hemi_year_url)
    r.raise_for_status()
    data = BeautifulSoup(r.text, "html.parser")
    granules = []
    for link in data.find_all("a")[1:]:
        url = os.path.join(ds_hemi_year_url, link["href"])
        tokens = link.next_sibling.split()
        mod_time = datetime.strptime(tokens[0] + " " + tokens[1], "%d-%b-%Y %H:%M")
        granules.append(NSIDCGranule(url, mod_time))
    return granules


def search_nsidc(harvester: Harvester):
    logger.info(f"Searching NSIDC for {harvester.ds_name} granules...")
    date_range = range(harvester.start.year, harvester.end.year + 1)
    all_granules = []

    # Collect (hemi_url, year) pairs to fetch concurrently
    hemi_year_pairs = []
    for hemi in ["north", "south"]:
        if harvester.ddir:
            ds_hemi_url = os.path.join(
                NSIDC_URL, harvester.ds_name, harvester.ddir, hemi, "daily"
            )
        else:
            ds_hemi_url = os.path.join(NSIDC_URL, harvester.ds_name, hemi, "daily")

        r = requests.get(ds_hemi_url)
        r.raise_for_status()
        data = BeautifulSoup(r.text, "html.parser")
        years = [link["href"].replace("/", "") for link in data.find_all("a")[1:]]
        for year in years:
            if int(year) in date_range:
                hemi_year_pairs.append((ds_hemi_url, year))

    # Fetch all (hemi, year) directory listings concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_year_granules, ds_hemi_url, year): (ds_hemi_url, year)
            for ds_hemi_url, year in hemi_year_pairs
        }
        for future in as_completed(futures):
            all_granules.extend(future.result())

    logger.info(f"Found {len(all_granules)} possible granules. Checking for updates...")
    return all_granules


@dataclass
class NSIDCGranule:
    url: str
    mod_time: datetime
