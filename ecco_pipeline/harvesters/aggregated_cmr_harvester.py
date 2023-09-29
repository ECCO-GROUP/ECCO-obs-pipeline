import base64
import json
import logging
import os
import ssl
import sys
import xarray as xr
import numpy as np
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


def get_credentials():
    username = 'ecco_access'
    password = 'ECCOAccess1'
    credentials = f'{username}:{password}'
    credentials = base64.b64encode(credentials.encode('utf-8')).decode()
    return credentials


def build_cmr_query_url(short_name, time_start, time_end,
                        bounding_box=None, polygon=None,
                        filename_filter=None):
    params = '&short_name={0}'.format(short_name)
    params += '&temporal[]={0},{1}'.format(time_start, time_end)
    if polygon:
        params += '&polygon={0}'.format(polygon)
    elif bounding_box:
        params += '&bounding_box={0}'.format(bounding_box)
    if filename_filter:
        option = '&options[producer_granule_id][pattern]=true'
        params += '&producer_granule_id[]={0}{1}'.format(
            filename_filter, option)
    return CMR_FILE_URL + params


def cmr_filter_urls(search_results, provider):
    """Select only the desired data files from CMR response."""
    if 'feed' not in search_results or 'entry' not in search_results['feed']:
        return []

    entries = [e['links'] for e in search_results['feed']['entry']
               if 'links' in e]
    ids = [e['id'] for e in search_results['feed']['entry']]

    urls = []
    unique_filenames = set()
    for link_list, id in zip(entries, ids):
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

            if 'md5' in filename or 'tif' in filename or 'txt' in filename or 'png' in filename or 'xml' in filename:
                continue
            if 's3credentials' in filename:
                continue
            if filename in unique_filenames:
                continue

            unique_filenames.add(filename)

            urls.append((link['href'], id))
    return urls


def cmr_search(config, bounding_box='', polygon='', filename_filter=''):
    """Perform a scrolling CMR query for files matching input criteria."""
    short_name = config['cmr_short_name']
    provider = config.get('provider')
    time_start = '-'.join([config['start'][:4],
                          config['start'][4:6], config['start'][6:]])
    time_end = '-'.join([config['end'][:4], config['end']
                        [4:6], config['end'][6:]])

    cmr_query_url = build_cmr_query_url(short_name=short_name,
                                        time_start=time_start, time_end=time_end,
                                        bounding_box=bounding_box,
                                        polygon=polygon, filename_filter=filename_filter)
    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        urls = []
        while True:
            req = Request(cmr_query_url)
            if cmr_scroll_id:
                req.add_header('cmr-scroll-id', cmr_scroll_id)
            response = urlopen(req, context=ctx)
            if not cmr_scroll_id:
                # Python 2 and 3 have different case for the http headers
                headers = {k.lower(): v for k, v in dict(
                    response.info()).items()}
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
    except KeyboardInterrupt:
        quit()


def get_mod_time(id, solr_format):
    meta_url = f'https://cmr.earthdata.nasa.gov/search/concepts/{id}.json'
    r = requests.get(meta_url)
    meta = r.json()
    modified_time = datetime.strptime(
        f'{meta["updated"].split(".")[0]}Z', solr_format)
    return modified_time


def dl_file(src, dst):
    credentials = get_credentials()
    req = Request(src)
    req.add_header('Authorization',
                   'Basic {0}'.format(credentials))
    opener = build_opener(HTTPCookieProcessor())
    data = opener.open(req).read()
    open(dst, 'wb').write(data)


def harvester(config):
    """
    Uses CMR search to find granules within date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, and descendants.
    """

    # =====================================================
    # Read harvester_config.yaml and setup variables
    # =====================================================
    dataset_name = config['ds_name']

    if config['end'] == 'NOW':
        config['end'] = datetime.utcnow().strftime('%Y%m%dT%H:%M:%SZ')
    start_time_dt = datetime.strptime(config['start'], "%Y%m%dT%H:%M:%SZ")
    end_time_dt = datetime.strptime(config['end'], "%Y%m%dT%H:%M:%SZ")

    solr_format = "%Y-%m-%dT%H:%M:%SZ"
    last_success_item = {}
    start_times = []
    end_times = []
    chk_time = datetime.utcnow().strftime(solr_format)

    target_dir = f'{OUTPUT_DIR}/{dataset_name}/harvested_granules/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    solr_utils.clean_solr(config)
    logging.info(f'Downloading {dataset_name} files to {target_dir}')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================
    docs, descendants_docs = harvesting_utils.get_solr_docs(dataset_name)

    # =====================================================
    # Setup EDSC loop variables
    # =====================================================
    url_list = cmr_search(config)

    entries_for_solr = []
    updating = False
    for url, id in url_list:
        # Date in filename is end date of 30 day period
        filename = url.split('/')[-1]
        agg_local_fp = f'{target_dir}{filename}'
        # updating = False
        modified_time = get_mod_time(id, solr_format)
        logging.debug(agg_local_fp)
        if ~os.path.exists(agg_local_fp) or modified_time > datetime.fromtimestamp(os.path.getmtime(local_fp)):
            logging.info(f'Downloading aggregated {filename} to {agg_local_fp}')
            try:
                dl_file(url, agg_local_fp)
                updating = True
            except Exception as e:
                logging.exception(e)
    if updating:
        for time in ['day01', 'day02']:
            ds = xr.open_dataset(agg_local_fp, group=f'daily/{time}')

        for time in ds.time.values:
            logging.debug(f'Slicing on {time}')
            time_dt = datetime.strptime(
                str(time)[:-3], "%Y-%m-%dT%H:%M:%S.%f")
            if time_dt < start_time_dt or time_dt > end_time_dt:
                continue

            if config['data_time_scale'].upper() == 'MONTHLY':
                if not time_dt.day == 1:
                    time_dt = time_dt.replace(day=1)
            year = str(time_dt.year)
            filename_time = str(time_dt)[:10].replace('-', '')

            file_name = f'{dataset_name}_{filename_time}.nc'
            local_fp = f'{target_dir}{year}/{file_name}'
            time_s = datetime.strftime(time_dt, solr_format)

            if not os.path.exists(f'{target_dir}{year}'):
                os.makedirs(f'{target_dir}{year}')

            # Granule metadata used for Solr harvested entries
            item = {}
            item['type_s'] = 'granule'
            item['date_s'] = time_s
            item['dataset_s'] = dataset_name
            item['filename_s'] = file_name
            item['source_s'] = url
            item['modified_time_dt'] = modified_time.strftime(solr_format)
            item['download_time_dt'] = chk_time

            # Granule metadata used for initializing Solr descendants entries
            descendants_item = {}
            descendants_item['type_s'] = 'descendants'
            descendants_item['dataset_s'] = item['dataset_s']
            descendants_item['date_s'] = item["date_s"]
            descendants_item['source_s'] = item['source_s']

            try:
                sub_ds = ds.sel(time=time)

                sub_ds.to_netcdf(path=local_fp)
                # Create checksum for file
                item['checksum_s'] = file_utils.md5(local_fp)
                item['pre_transformation_file_path_s'] = local_fp
                item['harvest_success_b'] = True
                item['file_size_l'] = os.path.getsize(local_fp)
            except:
                item['harvest_success_b'] = False
                item['pre_transformation_file_path_s'] = ''
                item['file_size_l'] = 0
                item['checksum_s'] = ''

            fq = ['type_s:granule', f'dataset_s:{dataset_name}',
                  f'date_s:{time_s[:10]}*']
            granule = solr_utils.solr_query(fq)

            if granule:
                item['id'] = granule[0]['id']

            if time_s in descendants_docs.keys():
                descendants_item['id'] = descendants_docs[time_s]['id']

            entries_for_solr.append(item)
            entries_for_solr.append(descendants_item)

            start_times.append(time_dt)
            end_times.append(time_dt)

            if item['harvest_success_b']:
                last_success_item = item

    logging.info(f'Downloading {dataset_name} complete')

    # Only update Solr harvested entries if there are fresh downloads
    if entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        r = solr_utils.solr_update(entries_for_solr, r=True)
        if r.status_code == 200:
            logging.debug(
                'Successfully created or updated Solr harvested documents')
        else:
            logging.exception('Failed to create Solr harvested documents')
    else:
        logging.debug('No downloads required.')

    # Query for Solr failed harvest documents
    fq = ['type_s:granule',
          f'dataset_s:{dataset_name}', f'harvest_success_b:false']
    failed_harvesting = solr_utils.solr_query(fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:granule',
          f'dataset_s:{dataset_name}', f'harvest_success_b:true']
    successful_harvesting = solr_utils.solr_query(fq)

    harvest_status = f'All granules successfully harvested'

    if not successful_harvesting:
        harvest_status = f'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'

    overall_start = min(start_times) if len(start_times) > 0 else None
    overall_end = max(end_times) if len(end_times) > 0 else None

    # Query for Solr dataset level document
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_query = solr_utils.solr_query(fq)

    # If dataset entry exists on Solr
    update = (len(dataset_query) == 1)

    # =====================================================
    # Solr dataset entry
    # =====================================================
    if not update:
        # -----------------------------------------------------
        # Create Solr Dataset-level Document if doesn't exist
        # -----------------------------------------------------
        source = f'https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/{dataset_name}/'
        ds_meta = harvesting_utils.make_ds_doc(config, source, chk_time)

        # Only include start_date and end_date if there was at least one successful download
        if not overall_start is None:
            ds_meta['start_date_dt'] = overall_start.strftime(solr_format)
            ds_meta['end_date_dt'] = overall_end.strftime(solr_format)

        # Only include last_download_dt if there was at least one successful download
        if updating:
            ds_meta['last_download_dt'] = last_success_item['download_time_dt']

        ds_meta['harvest_status_s'] = harvest_status

        # Update Solr with dataset metadata
        r = solr_utils.solr_update([ds_meta], r=True)

        if r.status_code == 200:
            logging.debug('Successfully created Solr dataset document')
        else:
            logging.exception('Failed to create Solr dataset document')

    # if dataset entry exists, update download time, converage start date, coverage end date
    else:
        # -----------------------------------------------------
        # Update Solr dataset entry
        # -----------------------------------------------------
        dataset_metadata = dataset_query[0]

        # Query for dates of all harvested docs
        fq = [f'dataset_s:{dataset_name}',
              'type_s:granule', 'harvest_success_b:true']
        dates_query = solr_utils.solr_query(fq, fl='date_s')
        dates = [x['date_s'] for x in dates_query]

        # Build update document body
        update_doc = {}
        update_doc['id'] = dataset_metadata['id']
        update_doc['last_checked_dt'] = {"set": chk_time}
        if dates:
            update_doc['start_date_dt'] = {"set": min(dates)}
            update_doc['end_date_dt'] = {"set": max(dates)}

        if entries_for_solr:
            update_doc['harvest_status_s'] = {"set": harvest_status}

            if 'download_time_dt' in last_success_item.keys():
                update_doc['last_download_dt'] = {
                    "set": last_success_item['download_time_dt']}

        # Update Solr with modified dataset entry
        r = solr_utils.solr_update([update_doc], r=True)

        if r.status_code == 200:
            logging.debug('Successfully updated Solr dataset document')
        else:
            logging.exception('Failed to update Solr dataset document')
    return harvest_status