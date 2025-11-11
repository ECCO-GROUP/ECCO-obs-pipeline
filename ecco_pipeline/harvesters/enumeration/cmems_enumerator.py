import logging
import time
from datetime import datetime
from typing import Iterable

import copernicusmarine as cm

from harvesters.harvesterclasses import Harvester

logger = logging.getLogger("pipeline")


class URLNotFound(Exception):
    """Raise for non S3 URL not available in CMEMS metadata exception"""


class CMEMSGranule:
    url: str
    id: str
    mod_time: datetime

    def __init__(self, query_result: cm.FileGet):
        self.url = query_result.https_url
        self.file_size = query_result.file_size
        self.filename = query_result.filename
        # self.id = query_result.get("id")
        self.mod_time = datetime.fromisoformat(query_result.last_modified_datetime.split("+")[0])
        # self.collection_id: str = collection

    def extract_url(self, links: Iterable[dict], provider: str) -> str:
        for link in links:
            if "rel" in link and link["rel"] == "http://esipfed.org/ns/fedsearch/1.1/data#":
                if provider in link["href"]:
                    return link["href"]
        raise URLNotFound()


class CMEMSQuery:
    def __init__(self, harvester: Harvester) -> None:
        self.concept_id: str = harvester.cmems_concept_id
        self.start_date: datetime = harvester.start
        self.end_date: datetime = harvester.end

    def granule_query_with_wait(self):
        max_retries = 3
        attempt = 1
        while attempt <= max_retries:
            time.sleep(attempt * 15)
            try:
                query_results = cm.get(dataset_id=self.concept_id, dry_run=True, disable_progress_bar=True)
                return query_results
            except RuntimeError:
                attempt += 1
        logger.error("Unable to query CMR")
        raise RuntimeError

    def query(self) -> Iterable[CMEMSGranule]:

        try:
            query_results = cm.get(dataset_id=self.concept_id, dry_run=True, disable_progress_bar=True)
        except RuntimeError:
            query_results = self.granule_query_with_wait()
            
        cmr_granules = [CMEMSGranule(result) for result in query_results.files]
        logger.info(f"Found {len(cmr_granules)} granule(s) from CMR query. Checking for updates...")
        return cmr_granules
