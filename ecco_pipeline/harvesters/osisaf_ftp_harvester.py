import logging
import os
from datetime import datetime
from ftplib import FTP
import numpy as np

from dateutil import parser
from conf.global_settings import OUTPUT_DIR
from utils import file_utils, solr_utils


SOLR_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class OsisafHarvester():
    
    def __init__(self, config: dict) -> None:
        self.dataset_name = config['ds_name']
        self.start_time = config['start']
        self.end_time = config['end']
        self.host = config['host']
        self.ddir = config['ddir']

        if self.end_time == 'NOW':
            self.end_time = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")
            
        self.years = np.arange(int(self.start_time[:4]), int(self.end_time[:4]) + 1)

        self.target_dir = f'{OUTPUT_DIR}/{self.dataset_name}/harvested_granules/'

        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        self.entries_for_solr = []
        self.last_success_item = {}
        self.granule_dates = []
        self.chk_time = datetime.utcnow().strftime(SOLR_FORMAT)
        self.now = datetime.utcnow()
        self.updating = False

        solr_utils.clean_solr(config)
        logging.info(f'Downloading {self.dataset_name} files to {self.target_dir}')
        
        self.docs_from_solr()

    def docs_from_solr(self):
        self.docs = {}
        self.descendants_docs = {}

        # Query for existing harvested docs
        fq = ['type_s:granule', f'dataset_s:{self.dataset_name}']
        query_docs = solr_utils.solr_query(fq)

        if len(query_docs) > 0:
            for doc in query_docs:
                self.docs[doc['filename_s']] = doc

        # Query for existing descendants docs
        fq = ['type_s:descendants', f'dataset_s:{self.dataset_name}']
        existing_descendants_docs = solr_utils.solr_query(fq)

        if len(existing_descendants_docs) > 0:
            for doc in existing_descendants_docs:
                if doc['hemisphere_s']:
                    key = (doc['date_s'], doc['hemisphere_s'])
                else:
                    key = doc['date_s']
                self.descendants_docs[key] = doc
                
    def download_from_ddir(self, ftp, data_dir: str, config: dict):
        try:
            files = []
            ftp.dir(data_dir, files.append)
            file_meta = {}
            for f in files:
                tokens = f.split()
                fname = tokens[-1]
                mod_date = ' '.join(tokens[5:8])
                mod_dt = parser.parse(mod_date)
                if config["filename_filter"] in fname and \
                    file_utils.valid_date(fname, config) and \
                        fname.endswith('.nc'):
                    file_meta[fname] = mod_dt

        except:
            logging.exception(f'Error finding files at {data_dir}. Check harvester config.')

        for filename, mod_dt in file_meta.items():
            hemi = file_utils.get_hemi(filename)

            if not hemi or not any(ext in filename for ext in ['.nc', '.bz2', '.gz']):
                continue

            date = file_utils.get_date(config['filename_date_regex'], filename)
            date_time = datetime.strptime(date, config['filename_date_fmt'])
            new_date_format = date_time.strftime('%Y-%m-%dT00:00:00Z')

            self.granule_dates.append(datetime.strptime(new_date_format, SOLR_FORMAT))

            url = f'{data_dir}/{filename}'

            # Granule metadata used for Solr harvested entries
            item = {}
            item['type_s'] = 'granule'
            item['date_s'] = new_date_format
            item['dataset_s'] = self.dataset_name
            item['filename_s'] = filename
            item['hemisphere_s'] = hemi
            item['source_s'] = f'ftp://{self.host}/{url}'

            # Granule metadata used for initializing Solr descendants entries
            descendants_item = {}
            descendants_item['type_s'] = 'descendants'
            descendants_item['date_s'] = item["date_s"]
            descendants_item['dataset_s'] = item['dataset_s']
            descendants_item['filename_s'] = filename
            descendants_item['hemisphere_s'] = hemi
            descendants_item['source_s'] = item['source_s']

            updating = False

            mod_time = mod_dt.strftime(SOLR_FORMAT)
            item['modified_time_dt'] = mod_time

            updating = granule_update_check(self.docs, filename, mod_dt, SOLR_FORMAT)

            if updating:
                year = date[:4]
                local_fp = f'{self.target_dir}{year}/{filename}'

                if not os.path.exists(f'{self.target_dir}{year}/'):
                    os.makedirs(f'{self.target_dir}{year}/')
                try:
                    # If file doesn't exist locally, download it
                    if not os.path.exists(local_fp):
                        logging.info(f'Downloading {filename} to {local_fp}')
                        with open(local_fp, 'wb') as f:
                            ftp.retrbinary('RETR '+url, f.write, blocksize=262144)

                    # If file exists, but is out of date, download it
                    elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_dt:
                        logging.info(f'Updating {filename} and downloading to {local_fp}')
                        with open(local_fp, 'wb') as f:
                            ftp.retrbinary('RETR '+url, f.write, blocksize=262144)
                    else:
                        logging.debug(f'{filename} already downloaded and up to date')
                    # Create checksum for file
                    item['harvest_success_b'] = True
                    item['file_size_l'] = os.path.getsize(local_fp)
                    item['checksum_s'] = file_utils.md5(local_fp)
                    item['pre_transformation_file_path_s'] = local_fp
                    item['download_time_dt'] = self.chk_time

                except Exception as e:
                    logging.exception(e)
                    item['harvest_success_b'] = False
                    item['filename'] = ''
                    item['pre_transformation_file_path_s'] = ''
                    item['file_size_l'] = 0

                # Update descendant item
                if hemi:
                    key = (descendants_item['date_s'], hemi)
                else:
                    key = descendants_item['date_s']

                if key in self.descendants_docs.keys():
                    descendants_item['id'] = self.descendants_docs[key]['id']

                descendants_item['harvest_success_b'] = item['harvest_success_b']
                descendants_item['pre_transformation_file_path_s'] = item['pre_transformation_file_path_s']

                self.entries_for_solr.append(item)
                self.entries_for_solr.append(descendants_item)

                self.last_success_item = item
            else:
                logging.debug(f'{filename} already downloaded and up to date')

def granule_update_check(docs: dict, filename: str, mod_date_time: datetime, time_format: str):
    key = filename.replace('.NRT', '')

    # Granule hasn't been harvested yet
    if key not in docs.keys():
        return True

    entry = docs[key]

    # Granule failed harvesting previously
    if not entry['harvest_success_b']:
        return True

    # Granule has been updated since last harvest
    if datetime.strptime(entry['download_time_dt'], time_format) <= mod_date_time:
        return True

    # Granule is replacing NRT version
    if '.NRT' in entry['filename_s'] and '.NRT' not in filename:
        return True

    # Granule is up to date
    return False



def harvester(config: dict):
    """
    Pulls data files for OSISAF FTP id and date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, and descendants.
    """
    osisaf = OsisafHarvester(config)
    # =====================================================
    # OSISAF loop
    # =====================================================
    try:
        ftp = FTP(osisaf.host)
        ftp.login(config['user'])
        logging.debug('Connected to FTP')
    except Exception as e:
        logging.exception(f'Harvesting failed. Unable to connect to FTP. {e}')
        return 'Harvesting failed. Unable to connect to FTP.'

    years_on_ftp = [int(y.split('/')[-1]) for y in ftp.nlst(osisaf.ddir) if not y.endswith('monthly')]
    logging.debug(f'The following years are available on ftp: {years_on_ftp}')
    
    for year in osisaf.years:
        if config['data_time_scale'] == 'daily':
            for month in [f'{x:02}' for x in list(range(1, 13))]:
                data_dir = f'{osisaf.ddir}{year}/{month}'
                osisaf.download_from_ddir(ftp, data_dir, config)
        else:
            data_dir = f'{osisaf.ddir}{year}'
            osisaf.download_from_ddir(ftp, data_dir, config)
            
    logging.info(f'Downloading {osisaf.dataset_name} complete')

    ftp.quit()

    # Only update Solr harvested entries if there are fresh downloads
    if osisaf.entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        r = solr_utils.solr_update(osisaf.entries_for_solr, r=True)

        if r.status_code == 200:
            logging.debug('Successfully created or updated Solr harvested documents')
        else:
            logging.exception('Failed to create Solr harvested documents')

    # Query for Solr failed harvest documents
    fq = ['type_s:granule', f'dataset_s:{osisaf.dataset_name}',
          f'harvest_success_b:false']
    failed_harvesting = solr_utils.solr_query(fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:granule', f'dataset_s:{osisaf.dataset_name}',
          f'harvest_success_b:true']
    successful_harvesting = solr_utils.solr_query(fq)

    harvest_status = f'All granules successfully harvested'

    if not successful_harvesting:
        harvest_status = f'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'

    overall_start = min(osisaf.granule_dates) if osisaf.granule_dates else None
    overall_end = max(osisaf.granule_dates) if osisaf.granule_dates else None

    # Query for Solr Dataset-level Document
    fq = ['type_s:dataset', f'dataset_s:{osisaf.dataset_name}']
    dataset_query = solr_utils.solr_query(fq)

    # If dataset entry exists on Solr
    update = (len(dataset_query) == 1)

    # =====================================================
    # Solr dataset entry
    # =====================================================
    if not update:
        # -----------------------------------------------------
        # Create Solr dataset entry
        # -----------------------------------------------------
        ds_meta = {}
        ds_meta['type_s'] = 'dataset'
        ds_meta['dataset_s'] = osisaf.dataset_name
        ds_meta['short_name_s'] = config['original_dataset_short_name']
        ds_meta['source_s'] = f'ftp://{osisaf.host}/{osisaf.ddir}'
        ds_meta['data_time_scale_s'] = config['data_time_scale']
        ds_meta['last_checked_dt'] = osisaf.chk_time
        ds_meta['original_dataset_title_s'] = config['original_dataset_title']
        ds_meta['original_dataset_short_name_s'] = config['original_dataset_short_name']
        ds_meta['original_dataset_url_s'] = config['original_dataset_url']
        ds_meta['original_dataset_reference_s'] = config['original_dataset_reference']
        ds_meta['original_dataset_doi_s'] = config['original_dataset_doi']

        # Only include start_date and end_date if there was at least one successful download
        if overall_start != None:
            ds_meta['start_date_dt'] = overall_start.strftime(SOLR_FORMAT)
            ds_meta['end_date_dt'] = overall_end.strftime(SOLR_FORMAT)

        # Only include last_download_dt if there was at least one successful download
        if osisaf.last_success_item:
            ds_meta['last_download_dt'] = osisaf.last_success_item['download_time_dt']

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
        fq = [f'dataset_s:{osisaf.dataset_name}',
              'type_s:granule', 'harvest_success_b:true']
        dates_query = solr_utils.solr_query(fq, fl='date_s')
        dates = [x['date_s'] for x in dates_query]

        # Build update document body
        update_doc = {}
        update_doc['id'] = dataset_metadata['id']
        update_doc['last_checked_dt'] = {"set": osisaf.chk_time}
        if dates:
            update_doc['start_date_dt'] = {"set": min(dates)}
            update_doc['end_date_dt'] = {"set": max(dates)}

        if osisaf.entries_for_solr:
            update_doc['harvest_status_s'] = {"set": harvest_status}

            if 'download_time_dt' in osisaf.last_success_item.keys():
                update_doc['last_download_dt'] = {"set": osisaf.last_success_item['download_time_dt']}

        # Update Solr with modified dataset entry
        r = solr_utils.solr_update([update_doc], r=True)

        if r.status_code == 200:
            logging.debug('Successfully updated Solr dataset document')
        else:
            logging.exception('Failed to update Solr dataset document')
    return harvest_status
