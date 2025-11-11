import logging
import os
from datetime import datetime
import time
from typing import Iterable

import requests
from harvesters.enumeration.cmems_enumerator import CMEMSGranule, CMEMSQuery
from harvesters.harvesterclasses import Granule, Harvester
from utils.pipeline_utils.file_utils import get_date

logger = logging.getLogger("pipeline")


class CMEMS_Harvester(Harvester):
    def __init__(self, config: dict):
        super().__init__(config)
        self.cmems_granules: Iterable[CMEMSGranule] = CMEMSQuery(self).query()

    def fetch(self):
        for cmems_granule in self.cmems_granules:
            filename = cmems_granule.filename

            # Get date from filename and convert to dt object
            date = get_date(self.filename_date_regex, filename)
            dt = datetime.strptime(date, self.filename_date_fmt)
            if not ((self.start <= dt) and (self.end >= dt)):
                continue

            year = str(dt.year)

            local_fp = os.path.join(self.target_dir, year, filename)
            os.makedirs(os.path.join(self.target_dir, year), exist_ok=True)

            if self.check_update(filename, cmems_granule.mod_time):
                success = True
                granule = Granule(
                    self.ds_name, local_fp, dt, cmems_granule.mod_time, cmems_granule.url
                )

                if self.need_to_download(granule):
                    logger.info(f"Downloading {filename} to {local_fp}")
                    try:
                        self.dl_file(cmems_granule.url, local_fp)
                    except Exception:
                        success = False
                        
                else:
                    logger.debug(f"{filename} already downloaded and up to date")

                granule.update_item(self.solr_docs, success)
                granule.update_descendant(self.descendant_docs, success)
                self.updated_solr_docs.extend(granule.get_solr_docs())
        logger.info(f"Downloading {self.ds_name} complete")

    def dl_file(self, src: str, dst: str):
        try:
            r = requests.get(src)
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)
        except Exception:
            time.sleep(5)
            r = requests.get(src)
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)


def harvester(config: dict) -> str:
    """
    Uses CMR search to find granules within date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, and descendants.
    """

    harvester = CMEMS_Harvester(config)

    harvester.fetch()
    source = f"CMEMS copernicusmarine API"
    harvesting_status = harvester.post_fetch(source)
    return harvesting_status
