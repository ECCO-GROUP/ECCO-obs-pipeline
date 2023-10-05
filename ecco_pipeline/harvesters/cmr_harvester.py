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
    try:
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
    except KeyboardInterrupt:
        quit()


def get_mod_time(id, solr_format):
    meta_url = f'https://cmr.earthdata.nasa.gov/search/concepts/{id}.json'
    r = requests.get(meta_url)
    meta = r.json()
    modified_time = datetime.strptime(f'{meta["updated"].split(".")[0]}Z', solr_format)
    return modified_time


def dl_file(src, dst):
    r = requests.get(src)
    r.raise_for_status()
    open(dst, 'wb').write(r.content)


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
    logging.debug(url_list)
    entries_for_solr = []

    for url, id, updated in url_list:
        # Date in filename is end date of 30 day period
        filename = url.split('/')[-1]

        # Get date from filename and convert to dt object
        date = file_utils.get_date(config['filename_date_regex'], filename)
        dt = datetime.strptime(date, config['filename_date_fmt'])

        if not (start_time_dt <= dt) and (end_time_dt >= dt):
            continue

        new_date_format = datetime.strftime(dt, "%Y-%m-%dT00:00:00Z")
        year = new_date_format[:4]

        local_fp = f'{target_dir}{year}/{filename}'

        if not os.path.exists(f'{target_dir}{year}/'):
            os.makedirs(f'{target_dir}{year}/')

        if not updated:
            modified_time = get_mod_time(id, solr_format)
        else:
            modified_time = datetime.strptime(f'{updated.split(".")[0]}Z', solr_format)

        item = {}
        item['type_s'] = 'granule'
        item['date_s'] = new_date_format
        item['dataset_s'] = dataset_name
        item['filename_s'] = filename
        item['source_s'] = url
        item['modified_time_dt'] = str(modified_time)

        descendants_item = {}
        descendants_item['type_s'] = 'descendants'
        descendants_item['date_s'] = item["date_s"]
        descendants_item['dataset_s'] = item['dataset_s']
        descendants_item['filename_s'] = filename
        descendants_item['source_s'] = item['source_s']

        updating = False

        try:
            updating = harvesting_utils.check_update(docs, filename, modified_time)

            # If updating, download file if necessary
            if updating:
                # If file doesn't exist locally, download it
                if not os.path.exists(local_fp):
                    logging.info(f'Downloading {filename} to {local_fp}')
                    dl_file(url, local_fp)
                # If file exists locally, but is out of date, download it
                elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= modified_time:
                    logging.info(f'Updating {filename} and downloading to {local_fp}')
                    dl_file(url, local_fp)
                else:
                    logging.debug(f'{filename} already downloaded and up to date')

                if filename in docs.keys():
                    item['id'] = docs[filename]['id']

                # calculate checksum and expected file size
                item['checksum_s'] = file_utils.md5(local_fp)
                item['pre_transformation_file_path_s'] = local_fp
                item['granule_file_path_s'] = local_fp
                item['harvest_success_b'] = True
                item['file_size_l'] = os.path.getsize(local_fp)

            else:
                logging.debug(f'{filename} already downloaded and up to date')

        except Exception as e:
            logging.exception(e)
            if updating:
                logging.debug(f'{filename} failed to download')

                item['harvest_success_b'] = False
                item['filename'] = ''
                item['pre_transformation_file_path_s'] = ''
                item['file_size_l'] = 0

        if updating:
            item['download_time_dt'] = chk_time

            # Update Solr entry using id if it exists
            key = descendants_item['date_s']

            if key in descendants_docs.keys():
                descendants_item['id'] = descendants_docs[key]['id']

            descendants_item['harvest_success_b'] = item['harvest_success_b']
            descendants_item['pre_transformation_file_path_s'] = item['pre_transformation_file_path_s']
            entries_for_solr.append(descendants_item)

            start_times.append(datetime.strptime(new_date_format, solr_format))
            end_times.append(datetime.strptime(new_date_format, solr_format))

            # add item to metadata json
            entries_for_solr.append(item)
            # store meta for last successful download
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
        ds_meta = {}
        ds_meta['id'] = dataset_metadata['id']
        ds_meta['last_checked_dt'] = {"set": chk_time}
        if dates:
            ds_meta['start_date_dt'] = {"set": min(dates)}
            ds_meta['end_date_dt'] = {"set": max(dates)}

        if entries_for_solr:
            ds_meta['harvest_status_s'] = {"set": harvest_status}

            if 'download_time_dt' in last_success_item.keys():
                ds_meta['last_download_dt'] = {
                    "set": last_success_item['download_time_dt']}

    # Update Solr with modified dataset entry
    r = solr_utils.solr_update([ds_meta], r=True)

    if r.status_code == 200:
        logging.debug('Successfully updated Solr dataset document')
    else:
        logging.exception('Failed to update Solr dataset document')
    return harvest_status