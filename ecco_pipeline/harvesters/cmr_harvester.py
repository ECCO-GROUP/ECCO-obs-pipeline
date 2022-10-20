import base64
import json
import logging
import os
import ssl
import sys
from datetime import datetime
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import requests
from utils import file_utils, solr_utils


CMR_URL = 'https://cmr.earthdata.nasa.gov'
URS_URL = 'https://urs.earthdata.nasa.gov'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = ('{0}/search/granules.json?'
                '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                '&scroll=true&page_size={1}'.format(CMR_URL, CMR_PAGE_SIZE))


def get_credentials(url):
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

            if 'md5' in filename or 'tif' in filename or 'txt' in filename:
                continue
            if 's3credentials' in filename:
                continue
            if filename in unique_filenames:
                continue

            unique_filenames.add(filename)
            
            urls.append((link['href'], id))
    return urls


def cmr_search(short_name, provider, time_start, time_end,
               bounding_box='', polygon='', filename_filter=''):
    """Perform a scrolling CMR query for files matching input criteria."""
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

def harvester(config, output_path, grids_to_use=[]):
    """
    Uses CMR search to find granules within date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, fields,
    and descendants.
    """

    # =====================================================
    # Read harvester_config.yaml and setup variables
    # =====================================================
    dataset_name = config['ds_name']
    date_regex = config['date_regex']
    start_time = config['start']
    end_time = config['end']
    regex = config['regex']
    date_format = config['date_format']
    data_time_scale = config['data_time_scale']

    if end_time == 'NOW':
        end_time = datetime.utcnow().strftime('%Y%m%dT%H:%M:%SZ')

    target_dir = f'{output_path}/{dataset_name}/harvested_granules/'

    time_format = "%Y-%m-%dT%H:%M:%SZ"
    entries_for_solr = []
    last_success_item = {}
    start_times = []
    end_times = []
    chk_time = datetime.utcnow().strftime(date_regex)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    solr_utils.clean_solr(config, grids_to_use)
    logging.info(f'Downloading {dataset_name} files to {target_dir}')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================
    docs = {}
    descendants_docs = {}

    # Query for existing harvested docs
    fq = ['type_s:granule', f'dataset_s:{dataset_name}']
    harvested_docs = solr_utils.solr_query(fq)

    # Dictionary of existing harvested docs
    # harvested doc filename : solr entry for that doc
    if len(harvested_docs) > 0:
        for doc in harvested_docs:
            docs[doc['filename_s']] = doc

    # Query for existing descendants docs
    fq = ['type_s:descendants', f'dataset_s:{dataset_name}']
    existing_descendants_docs = solr_utils.solr_query(fq)

    # Dictionary of existing descendants docs
    # descendant doc date : solr entry for that doc
    if len(existing_descendants_docs) > 0:
        for doc in existing_descendants_docs:
            if 'hemisphere_s' in doc.keys() and doc['hemisphere_s']:
                key = (doc['date_s'], doc['hemisphere_s'])
            else:
                key = doc['date_s']
            descendants_docs[key] = doc

    # =====================================================
    # Setup EDSC loop variables
    # =====================================================
    short_name = config['cmr_short_name']
    provider = config.get('provider')
    filename_date_fmt = config['filename_date_fmt']

    item = {}

    start_time_dt = datetime.strptime(start_time, "%Y%m%dT%H:%M:%SZ")
    end_time_dt = datetime.strptime(end_time, "%Y%m%dT%H:%M:%SZ")
    start_string = datetime.strftime(start_time_dt, "%Y-%m-%dT00:00:00Z")
    end_string = datetime.strftime(end_time_dt, "%Y-%m-%dT00:00:00Z")

    url_list = cmr_search(short_name, provider, start_string, end_string)

    for url, id in url_list:
        # Date in filename is end date of 30 day period
        filename = url.split('/')[-1]

        date = file_utils.get_date(regex, filename)
        dt = datetime.strptime(date, filename_date_fmt)
        new_date_format = datetime.strftime(dt, "%Y-%m-%dT00:00:00Z")
        
        year = new_date_format[:4]

        local_fp = f'{target_dir}{year}/{filename}'

        if not os.path.exists(f'{target_dir}{year}/'):
            os.makedirs(f'{target_dir}{year}/')

        # Extract metadata from xml file associated with .nc file
        meta_url = f'https://cmr.earthdata.nasa.gov/search/concepts/{id}.json'
        r = requests.get(meta_url)
        meta = r.json()
        modified_time = datetime.strptime(f'{meta["updated"].split(".")[0]}Z', time_format)   

        # check if file in download date range
        if (start_time_dt <= dt) and (end_time_dt >= dt):
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
                updating = (not filename in docs.keys()) or \
                            (not docs[filename]['harvest_success_b']) or \
                            (docs[filename]['download_time_dt'] < str(modified_time))

                # If updating, download file if necessary
                if updating:
                    # If file doesn't exist locally, download it
                    if not os.path.exists(local_fp):
                        logging.info(f'Downloading {filename} to {local_fp}')
                        credentials = get_credentials(url)
                        req = Request(url)
                        req.add_header('Authorization',
                                        'Basic {0}'.format(credentials))
                        opener = build_opener(HTTPCookieProcessor())
                        data = opener.open(req).read()
                        open(local_fp, 'wb').write(data)

                    # If file exists locally, but is out of date, download it
                    elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= modified_time:
                        logging.info(f'Updating {filename} and downloading to {local_fp}')

                        credentials = get_credentials(url)
                        req = Request(url)
                        req.add_header('Authorization',
                                        'Basic {0}'.format(credentials))
                        opener = build_opener(HTTPCookieProcessor())
                        data = opener.open(req).read()
                        open(local_fp, 'wb').write(data)

                    else:
                        logging.debug(f'{filename} already downloaded and up to date')

                    if filename in docs.keys():
                        item['id'] = docs[filename]['id']

                    # calculate checksum and expected file size
                    item['checksum_s'] = file_utils.md5(local_fp)
                    item['pre_transformation_file_path_s'] = local_fp
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

                start_times.append(datetime.strptime(
                    new_date_format, date_regex))
                end_times.append(datetime.strptime(
                    new_date_format, date_regex))

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
            logging.debug('Successfully created or updated Solr harvested documents')
        else:
            logging.exception('Failed to create Solr harvested documents')

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
        ds_meta = {}
        ds_meta['type_s'] = 'dataset'
        ds_meta['dataset_s'] = dataset_name
        ds_meta['short_name_s'] = config['original_dataset_short_name']
        ds_meta['source_s'] = url_list[0][:-30]
        ds_meta['data_time_scale_s'] = data_time_scale
        ds_meta['date_format_s'] = date_format
        ds_meta['last_checked_dt'] = chk_time
        ds_meta['original_dataset_title_s'] = config['original_dataset_title']
        ds_meta['original_dataset_short_name_s'] = config['original_dataset_short_name']
        ds_meta['original_dataset_url_s'] = config['original_dataset_url']
        ds_meta['original_dataset_reference_s'] = config['original_dataset_reference']
        ds_meta['original_dataset_doi_s'] = config['original_dataset_doi']

        # Only include start_date and end_date if there was at least one successful download
        if overall_start != None:
            ds_meta['start_date_dt'] = overall_start.strftime(time_format)
            ds_meta['end_date_dt'] = overall_end.strftime(time_format)

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

        # If the dataset entry needs to be created, so do the field entries

        # -----------------------------------------------------
        # Create Solr dataset field entries
        # -----------------------------------------------------

        # Query for Solr field documents
        fq = ['type_s:field', f'dataset_s:{dataset_name}']
        field_query = solr_utils.solr_query(fq)

        body = []
        for field in config['fields']:
            field_obj = {}
            field_obj['type_s'] = {'set': 'field'}
            field_obj['dataset_s'] = {'set': dataset_name}
            field_obj['name_s'] = {'set': field['name']}
            field_obj['long_name_s'] = {'set': field['long_name']}
            field_obj['standard_name_s'] = {'set': field['standard_name']}
            field_obj['units_s'] = {'set': field['units']}

            for solr_field in field_query:
                if field['name'] == solr_field['name_s']:
                    field_obj['id'] = {'set': solr_field['id']}

            body.append(field_obj)

        if body:
            # Update Solr with dataset fields metadata
            r = solr_utils.solr_update(body, r=True)

            if r.status_code == 200:
                logging.debug('Successfully created Solr field documents')
            else:
                logging.exception('Failed to create Solr field documents')

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
