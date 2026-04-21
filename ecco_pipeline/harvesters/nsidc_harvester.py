import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable

import requests
from harvesters.enumeration.nsidc_enumerator import MAX_WORKERS, NSIDCGranule, search_nsidc
from harvesters.harvesterclasses import Granule, Harvester
from utils.pipeline_utils.file_utils import get_date

logger = logging.getLogger("pipeline")

CHUNK_SIZE = 1024 * 1024  # 1 MB


class NSIDC_Harvester(Harvester):
    def __init__(self, config: dict):
        Harvester.__init__(self, config)
        self.nsidc_granules: Iterable[NSIDCGranule] = search_nsidc(self)

    def fetch(self):
        # Pre-filter granules to only those within the date range
        to_process = []
        for nsidc_granule in self.nsidc_granules:
            filename = nsidc_granule.url.split("/")[-1]
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not (self.start <= dt <= self.end):
                continue
            to_process.append((nsidc_granule, filename, dt))

        for _, _, dt in to_process:
            os.makedirs(os.path.join(self.target_dir, str(dt.year)), exist_ok=True)

        lock = threading.Lock()

        def process_granule(nsidc_granule: NSIDCGranule, filename: str, dt: datetime):
            year = str(dt.year)
            local_fp = os.path.join(self.target_dir, year, filename)

            if not self.check_update(filename, nsidc_granule.mod_time):
                return []

            success = True
            granule = Granule(
                self.ds_name, local_fp, dt, nsidc_granule.mod_time, nsidc_granule.url
            )

            if self.need_to_download(granule):
                logger.info(f"Downloading {filename} to {local_fp}")
                try:
                    self.dl_file(nsidc_granule.url, local_fp)
                except Exception:
                    success = False
            else:
                logger.debug(f"{filename} already downloaded and up to date")

            granule.update_item(self.solr_docs, success)
            granule.update_descendant(self.descendant_docs, success)
            return granule.get_solr_docs()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_granule, *args) for args in to_process
            ]
            for future in as_completed(futures):
                docs = future.result()
                with lock:
                    self.updated_solr_docs.extend(docs)

        logger.info(f"Downloading {self.ds_name} complete")

    def dl_file(self, src: str, dst: str):
        with requests.get(src, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)


def harvester(config: dict) -> str:
    """
    Uses CMR search to find granules within date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, and descendants.
    """

    harvester = NSIDC_Harvester(config)
    harvester.fetch()
    source = f"https://noaadata.apps.nsidc.org/NOAA/{harvester.ds_name}"
    harvesting_status = harvester.post_fetch(source)
    return harvesting_status
