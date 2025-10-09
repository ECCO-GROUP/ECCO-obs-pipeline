import logging
import os
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from ecco_pipeline.harvesters.harvesterclasses import Harvester

logger = logging.getLogger("pipeline")


OSISAF_URL = "https://thredds.met.no/thredds/catalog/osisaf/met.no/"


def search_osisaf(harvester: Harvester):
    harvester.data_time_scale
    logger.info(f"Searching OSISAF for {harvester.ds_name} granules...")

    if harvester.data_time_scale == "monthly":
        base_url = os.path.join(OSISAF_URL, harvester.ddir, "monthly")
    else:
        base_url = os.path.join(OSISAF_URL, harvester.ddir)

    session = requests.session()
    r = session.get(os.path.join(base_url, "catalog.xml"))
    data = BeautifulSoup(r.text, "xml")

    all_granules = []
    for year_dir in data.find_all("catalogRef"):
        if "monthly" in year_dir["xlink:title"]:
            continue
        r = session.get(os.path.join(base_url, year_dir["xlink:title"], "catalog.xml"))
        year_data = BeautifulSoup(r.text, "xml")
        if harvester.data_time_scale == "daily":
            for month_dir in year_data.find_all("catalogRef"):
                month_url = os.path.join(
                    base_url,
                    year_dir["xlink:title"],
                    month_dir["xlink:title"],
                    "catalog.xml",
                )
                r = session.get(month_url)
                month_data = BeautifulSoup(r.text, "xml")
                for dataset in month_data.find_all("dataset"):
                    for granule in dataset.find_all("dataset"):
                        url = os.path.join(
                            "https://thredds.met.no/thredds/fileServer/",
                            granule["urlPath"],
                        )
                        mod_time = datetime.strptime(
                            granule.find("date").text, "%Y-%m-%dT%H:%M:%SZ"
                        )
                        all_granules.append(OSISAFGranule(url, mod_time))
        else:
            for dataset in year_data.find_all("dataset"):
                for granule in dataset.find_all("dataset"):
                    url = os.path.join(
                        "https://thredds.met.no/thredds/fileServer/", granule["urlPath"]
                    )
                    mod_time = datetime.strptime(
                        granule.find("date").text, "%Y-%m-%dT%H:%M:%SZ"
                    )
                    all_granules.append(OSISAFGranule(url, mod_time))
    logger.info(f"Found {len(all_granules)} possible granules. Checking for updates...")
    return all_granules


@dataclass
class OSISAFGranule:
    url: str
    mod_time: datetime
