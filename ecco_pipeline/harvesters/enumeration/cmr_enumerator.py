from datetime import datetime
import json
import logging
import ssl
from urllib.request import Request, urlopen

import requests

from harvesters.harvester import Harvester

logger = logging.getLogger('pipeline')

CMR_GRANULE_URL = 'https://cmr.earthdata.nasa.gov/search/granules.json?&sort_key[]=start_date&sort_key[]=producer_granule_id&scroll=true&page_size=2000'

def build_cmr_query_url(time_start: str, time_end: str, filename_filter: str, concept_id: str) -> str:
    query_url = f'{CMR_GRANULE_URL}&temporal[]={time_start},{time_end}&concept_id={concept_id}'
    if filename_filter:
        query_url += f'&producer_granule_id[]={filename_filter}&options[producer_granule_id][pattern]=true'
    return query_url

def get_mod_time(id: str) -> datetime:
    meta_url = f'https://cmr.earthdata.nasa.gov/search/concepts/{id}.json'
    r = requests.get(meta_url)
    meta = r.json()
    modified_time = datetime.strptime(f'{meta["updated"].split(".")[0]}Z', "%Y-%m-%dT%H:%M:%SZ")
    return modified_time

def cmr_filter_urls(search_results, provider):
    """Select only the desired data files from CMR response."""
    if 'feed' not in search_results or 'entry' not in search_results['feed']:
        return []

    entries = []
    for e in search_results['feed']['entry']:
        links = e['links'] if 'links' in e else None
        id = e['id']
        if 'updated' in e:
            updated = datetime.strptime(f'{e["updated"].split(".")[0]}Z', "%Y-%m-%dT%H:%M:%SZ")
        else:
            updated = get_mod_time()
        entries.append((links, id, updated))

    granules = []
    unique_filenames = set()
    for link_list, id, update in entries:
        for link in link_list:
            if 'href' not in link:
                continue
            if 'inherited' in link and link['inherited'] is True:
                continue
            if provider not in link['href']:
                continue

            if link['href'].endswith('.html'):
                link['href'] = link['href'].replace('.html', '.nc4')

            filename = link['href'].split('/')[-1]

            bad_formats = ['md5', 'tif', 'txt', 'png', 'xml', 'jpg', 'dmrpp', 's3credentials', 'NRT']
            if any(s in filename for s in bad_formats):
                continue
            if filename in unique_filenames:
                continue

            unique_filenames.add(filename)
            granule = CMRGranule(link['href'], id, update)
            granules.append(granule)
    return granules


def cmr_search(harvester: Harvester, filename_filter=''):
    logger.info(f'Querying CMR for concept id {harvester.cmr_concept_id}')
    time_start = harvester.start.strftime('%Y-%m-%dT%H:%M:%SZ')
    time_end = harvester.end.strftime('%Y-%m-%dT%H:%M:%SZ')

    cmr_query_url = build_cmr_query_url(time_start, time_end, filename_filter, harvester.cmr_concept_id)
    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    granules = []
    while True:
        req = Request(cmr_query_url)
        if cmr_scroll_id:
            req.add_header('cmr-scroll-id', cmr_scroll_id)
        response = urlopen(req, context=ctx)
        if not cmr_scroll_id:
            # Python 2 and 3 have different case for the http headers
            headers = {k.lower(): v for k, v in dict(response.info()).items()}
            cmr_scroll_id = headers['cmr-scroll-id']
            hits = int(headers['cmr-hits'])

        search_page = response.read()
        search_page = json.loads(search_page.decode('utf-8'))
        url_scroll_results = cmr_filter_urls(search_page, harvester.provider)
        if not url_scroll_results:
            break
        granules += url_scroll_results
    return granules

class CMRGranule():
    url: str
    id: str
    mod_time: str
    
    def __init__(self, url, id, mod_time: datetime):
        self.url = url
        self.id = id
        self.mod_time = mod_time
            
