import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from harvesters.harvesterclasses import Harvester

logger = logging.getLogger("pipeline")


OSISAF_URL = "https://thredds.met.no/thredds/catalog/osisaf/met.no/"
MAX_WORKERS = 10


def _fetch_xml(url: str) -> BeautifulSoup:
    r = requests.get(url)
    r.raise_for_status()
    return BeautifulSoup(r.text, "xml")


def _granules_from_dataset(data: BeautifulSoup) -> list:
    granules = []
    for dataset in data.find_all("dataset"):
        for granule in dataset.find_all("dataset"):
            url = os.path.join(
                "https://thredds.met.no/thredds/fileServer/", granule["urlPath"]
            )
            mod_time = datetime.strptime(
                granule.find("date").text, "%Y-%m-%dT%H:%M:%SZ"
            )
            granules.append(OSISAFGranule(url, mod_time))
    return granules


def _fetch_month_granules(base_url: str, year_title: str, month_dir) -> list:
    month_url = os.path.join(
        base_url, year_title, month_dir["xlink:title"], "catalog.xml"
    )
    return _granules_from_dataset(_fetch_xml(month_url))


def search_osisaf(harvester: Harvester):
    logger.info(f"Searching OSISAF for {harvester.ds_name} granules...")

    if harvester.data_time_scale == "monthly":
        base_url = os.path.join(OSISAF_URL, harvester.ddir, "monthly")
    else:
        base_url = os.path.join(OSISAF_URL, harvester.ddir)

    root_data = _fetch_xml(os.path.join(base_url, "catalog.xml"))

    start_year = harvester.start.year
    end_year = harvester.end.year

    year_dirs = []
    for d in root_data.find_all("catalogRef"):
        title = d["xlink:title"]
        if "monthly" in title:
            continue
        try:
            year = int(title)
        except ValueError:
            year_dirs.append(d)
            continue
        if start_year <= year <= end_year:
            year_dirs.append(d)

    all_granules = []

    if harvester.data_time_scale == "daily":
        # Fetch all year catalogs concurrently to get month listings
        def fetch_month_dirs(year_dir):
            year_title = year_dir["xlink:title"]
            year_data = _fetch_xml(
                os.path.join(base_url, year_title, "catalog.xml")
            )
            month_dirs = year_data.find_all("catalogRef")
            # Filter months to only those within the requested date range
            try:
                year = int(year_title)
                start_month = harvester.start.month if year == start_year else 1
                end_month = harvester.end.month if year == end_year else 12
                month_dirs = [
                    md for md in month_dirs
                    if start_month <= int(md["xlink:title"]) <= end_month
                ]
            except (ValueError, KeyError):
                pass
            return year_title, month_dirs

        year_month_pairs = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_month_dirs, yd): yd for yd in year_dirs}
            for future in as_completed(futures):
                year_title, month_dirs = future.result()
                for md in month_dirs:
                    year_month_pairs.append((year_title, md))

        # Fetch all month catalogs concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(_fetch_month_granules, base_url, year_title, month_dir)
                for year_title, month_dir in year_month_pairs
            ]
            for future in as_completed(futures):
                all_granules.extend(future.result())
    else:
        # Monthly: fetch all year catalogs concurrently
        def fetch_year_granules(year_dir):
            year_data = _fetch_xml(
                os.path.join(base_url, year_dir["xlink:title"], "catalog.xml")
            )
            return _granules_from_dataset(year_data)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(fetch_year_granules, yd) for yd in year_dirs
            ]
            for future in as_completed(futures):
                all_granules.extend(future.result())

    logger.info(f"Found {len(all_granules)} possible granules. Checking for updates...")
    return all_granules


@dataclass
class OSISAFGranule:
    url: str
    mod_time: datetime
