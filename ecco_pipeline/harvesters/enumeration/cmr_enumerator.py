import base64
import json
import logging
import os
import ssl
import sys
from datetime import datetime
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import requests
from conf.global_settings import OUTPUT_DIR
from utils import file_utils, solr_utils, harvesting_utils


CMR_URL = 'https://cmr.earthdata.nasa.gov'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = ('{0}/search/granules.json?'
                '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                '&scroll=true&page_size={1}'.format(CMR_URL, CMR_PAGE_SIZE))


def build_cmr_query_url(short_name, time_start, time_end,
                        filename_filter=None, concept_id=None):
    params = '&short_name={0}'.format(short_name)
    params += '&temporal[]={0},{1}'.format(time_start, time_end)
    if filename_filter:
        option = '&options[producer_granule_id][pattern]=true'
        params += '&producer_granule_id[]={0}{1}'.format(filename_filter, option)
    if concept_id:
        params += f'&concept_id={concept_id}'
    return CMR_FILE_URL + params


def cmr_filter_urls(search_results, provider):
    """Select only the desired data files from CMR response."""
    if 'feed' not in search_results or 'entry' not in search_results['feed']:
        return []

    entries = []
    for e in search_results['feed']['entry']:
        links = e['links'] if 'links' in e else None
        id = e['id']
        updated = e['updated'] if 'updated' in e else None
        entries.append((links, id, updated))

    urls = []
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

            bad_formats = ['md5', 'tif', 'txt', 'png', 'xml', 'jpg', 'dmrpp', 's3credentials']
            if any(s in filename for s in bad_formats):
                continue
            if filename in unique_filenames:
                continue

            unique_filenames.add(filename)

            urls.append((link['href'], id, update))
    return urls


def cmr_search(config, filename_filter=''):
    """Perform a scrolling CMR query for files matching input criteria."""
    short_name = config['cmr_short_name']
    concept_id = config.get('cmr_concept_id')
    provider = config.get('provider')
    time_start = '-'.join([config['start'][:4], config['start'][4:6], config['start'][6:]])
    time_end = '-'.join([config['end'][:4], config['end'][4:6], config['end'][6:]])

    cmr_query_url = build_cmr_query_url(short_name, time_start, time_end, filename_filter, concept_id)
    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    urls = []
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
        url_scroll_results = cmr_filter_urls(search_page, provider)
        if not url_scroll_results:
            break
        if hits > CMR_PAGE_SIZE:
            print('.', end='')
            sys.stdout.flush()
        urls += url_scroll_results

    if hits > CMR_PAGE_SIZE:
        print()
    return urls