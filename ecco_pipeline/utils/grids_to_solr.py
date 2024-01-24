import logging
import os
from datetime import datetime
import xarray as xr
from utils import file_utils, solr_utils, log_config

logger = logging.getLogger('pipeline')

def update_solr_grid(grid_name, grid_type, grid_file):
    '''
    Update solr grid docs with latest versions of grids
    '''

    # =====================================================
    # Query for Solr Grid-type Documents
    # =====================================================
    fq = ['type_s:grid']
    grid_docs = solr_utils.solr_query(fq)

    grid_path = f'grids/{grid_file}'
    update_body = []

    grids_in_solr = [doc['grid_name_s'] for doc in grid_docs]

    # Add grid to solr
    if grid_name not in grids_in_solr:
        grid_meta = {}
        grid_meta['type_s'] = 'grid'
        grid_meta['grid_type_s'] = grid_type
        grid_meta['grid_name_s'] = grid_name
        grid_meta['grid_path_s'] = str(grid_path)
        grid_meta['date_added_dt'] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        grid_meta['grid_checksum_s'] = file_utils.md5(grid_path)
        update_body.append(grid_meta)
    # Verify grid in solr matches grid file
    else:
        current_checksum = file_utils.md5(grid_path)

        for doc in grid_docs:
            if doc['grid_name_s'] == grid_name:
                solr_checksum = doc['grid_checksum_s']

        if current_checksum != solr_checksum:
            # Delete previous grid's transformations from Solr
            update_body = {
                "delete": {
                    "query": f'type_s:transformation AND grid_name_s:{grid_name}'
                }
            }

            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code == 200:
                logger.debug(f'Successfully deleted Solr transformation documents for {grid_name}')
            else:
                logger.exception(f'Failed to delete Solr transformation documents for {grid_name}')

            # Delete previous grid's aggregations from Solr
            update_body = {
                "delete": {
                    "query": f'type_s:aggregation AND grid_name_s:{grid_name}'
                }
            }

            r = solr_utils.solr_update(update_body, r=True)

            if r.status_code == 200:
                logger.debug(f'Successfully deleted Solr aggregation documents for {grid_name}')
            else:
                logger.exception(f'Failed to delete Solr aggregation documents for {grid_name}')

            # Update grid on Solr
            fq = [f'grid_name_s:{grid_name}', 'type_s:grid']
            grid_metadata = solr_utils.solr_query(fq)[0]

            update_body = [
                {
                    "id": grid_metadata['id'],
                    "grid_type_s": {"set": grid_type},
                    "grid_name_s": {"set": grid_name},
                    "grid_checksum_s": {"set": current_checksum},
                    "date_added_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
                }
            ]

    r = solr_utils.solr_update(update_body, r=True)

    if r.status_code == 200:
        logger.debug(f'Successfully updated {grid_name} Solr grid document')
    else:
        logger.error(f'Failed to update Solr {grid_name} grid document')


def main(grids_to_use=[]):
    # =====================================================
    # Scan directory for grid types
    # =====================================================
    grid_files = [f for f in os.listdir('grids/') if f.endswith('nc')]

    # =====================================================
    # Extract grid names from netCDF
    # =====================================================
    grids = []

    # Assumes grids conform to metadata standard (see documentation)
    for grid_file in grid_files:
        ds = xr.open_dataset(f'grids/{grid_file}')

        grid_name = ds.attrs['name']
        grid_type = ds.attrs['type']
        grids.append((grid_name, grid_type, grid_file))
        logger.debug(f'Loaded {grid_name} {grid_type} {grid_file}')

    # =====================================================
    # Create Solr grid-type document for each missing grid type
    # =====================================================
    for grid_name, grid_type, grid_file in grids:
        logger.debug(f'Uploading solr grid {grid_name} {grid_type} {grid_file}')
        update_solr_grid(grid_name, grid_type, grid_file)

    # =====================================================
    # Verify grid names supplied exist on Solr
    # =====================================================
    grids_not_in_solr = []
    logger.debug(f'Grids to use: {grids_to_use}')
    for grid_name in grids_to_use:
        fq = ['type_s:grid', f'grid_name_s:{grid_name}']
        docs = solr_utils.solr_query(fq)

        if docs:
            continue
        else:
            grids_not_in_solr.append(grid_name)

    return grids_not_in_solr