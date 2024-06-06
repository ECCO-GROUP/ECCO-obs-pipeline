import logging
import time
from datetime import datetime
from typing import Iterable

from cmr import GranuleQuery
from harvesters.harvesterclasses import Harvester

logger = logging.getLogger('pipeline')


class URLNotFound(Exception):
    """Raise for non S3 URL not available in CMR metadata exception"""
    
class CMRGranule():
    url: str
    id: str
    mod_time: datetime
    
    def __init__(self, query_result: dict, provider: str):
        self.url = self.extract_url(query_result['links'], provider)
        self.id = query_result.get('id')
        self.mod_time = datetime.strptime(query_result.get('updated'), '%Y-%m-%dT%H:%M:%S.%fZ')
        self.collection_id: str = query_result.get('collection_concept_id')
        
    def extract_url(self, links: Iterable[dict], provider: str) -> str:
        for link in links:
            if 'rel' in link and link['rel'] == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                if provider in link['href']:
                    return link['href']
        raise URLNotFound()

class CMRQuery():
    def __init__(self, harvester: Harvester) -> None:
        self.concept_id: str = harvester.cmr_concept_id
        self.start_date: datetime = harvester.start
        self.end_date: datetime = harvester.end
        self.provider: str = harvester.provider
        
    def granule_query_with_wait(self):
        api = GranuleQuery()
        max_retries = 3
        attempt = 1
        while attempt <= max_retries:
            time.sleep(attempt * 15)
            try:
                query_results = api.concept_id(self.concept_id).temporal(self.start_date, self.end_date).get_all()
                return query_results
            except RuntimeError:
                attempt += 1
        logger.error('Unable to query CMR')
        raise RuntimeError

    def query(self) -> Iterable[CMRGranule]:     
        api = GranuleQuery()
        try:
            query_results = api.concept_id(self.concept_id).temporal(self.start_date, self.end_date).get_all()
        except RuntimeError:
            query_results = self.granule_query_with_wait()
        cmr_granules = [CMRGranule(result, self.provider) for result in query_results]
        logger.info(f'Found {len(cmr_granules)} granule(s) from CMR query. Checking for updates...')
        return cmr_granules