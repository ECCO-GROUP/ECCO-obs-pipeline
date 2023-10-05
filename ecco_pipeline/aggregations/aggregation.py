import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List

import netCDF4 as nc4
import numpy as np
import xarray as xr
from utils import solr_utils


class Aggregation():

    def __init__(self, config: dict, grids_to_use: List[str]):
        self.dataset_name = config.get('ds_name')
        self.fields = config.get('fields')
        self.version = str(config.get('a_version', ''))
        self.precision = getattr(np, config.get('array_precision'))
        self.binary_dtype = '>f4' if self.precision == np.float32 else '>f8'
        self.nc_fill_val = nc4.default_fillvals[self.binary_dtype.replace('>', '')]
        self.bin_fill_val = -9999
        self.data_time_scale = config.get('data_time_scale')
        self.do_monthly_aggregation = config.get('do_monthly_aggregation', False)
        self.remove_nan_days_from_data = config.get('remove_nan_days_from_data', True)
        self.skipna_in_mean = config.get('skipna_in_mean', False)

        self.save_binary = config.get('save_binary', True)
        self.save_netcdf = config.get('save_netcdf', True)

        self.transformations = defaultdict(list)
        self._set_ds_meta()
        self._set_grids(grids_to_use)
        self._set_years()

    def _set_ds_meta(self):
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:dataset']
        ds_meta = solr_utils.solr_query(fq)[0]
        if 'start_date_dt' not in ds_meta:
            logging.info('No transformed granules to aggregate.')
            raise Exception('No transformed granules to aggregate.')
        self.ds_meta = ds_meta

    def _set_grids(self, grids_to_use):
        fq = ['type_s:grid']
        grids = [grid for grid in solr_utils.solr_query(fq)]
        if grids_to_use:
            grids = [grid for grid in grids if grid['grid_name_s'] in grids_to_use]
        self.grids = grids

    def _set_years(self):
        existing_agg_version = self.ds_meta.get('aggregation_version_s')
        if existing_agg_version != self.version:
            start_year = int(self.ds_meta.get('start_date_dt')[:4])
            end_year = int(self.ds_meta.get('end_date_dt')[:4])
            years = [str(year) for year in range(start_year, end_year + 1)]
            self.years = {grid.get('grid_name_s'): years for grid in self.grids}
        else:
            self.years = self._years_to_aggregate()

    def _years_to_aggregate(self) -> List[str]:
        '''
        We want to see if we need to aggregate this year again
        1. check if aggregation exists - if not add it to years to aggregate
        2. if it does - compare processing times:
        - if aggregation time is later than all transform times for that year
            no need to aggregate
        - if at least one transformation occured after agg time, year needs to
            be aggregated
        '''
        years = {}
        for grid in self.grids:
            grid_name = grid.get('grid_name_s')
            grid_years = []

            fq = [f'dataset_s:{self.dataset_name}',
                  'type_s:transformation', f'grid_name_s:{grid_name}']
            r = solr_utils.solr_query(fq)
            transformation_years = list(set([t['date_s'][:4] for t in r]))
            transformation_years.sort()
            transformation_docs = r

            # Years with transformations that exist for this dataset and this grid
            for year in transformation_years:
                fq = [f'dataset_s:{self.dataset_name}', 'type_s:aggregation',
                      f'grid_name_s:{grid_name}', f'year_s:{year}']
                r = solr_utils.solr_query(fq)

                if r:
                    agg_time = r[0]['aggregation_time_dt']
                    for t in transformation_docs:
                        if t['date_s'][:4] != year:
                            continue
                        if t['transformation_completed_dt'] > agg_time:
                            grid_years.append(year)
                            break
                else:
                    grid_years.append(year)
            years[grid_name] = grid_years
        return years

    def get_agg_status(self):
        # Query Solr for successful aggregation documents
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:aggregation', 'aggregation_success_b:true']
        successful_aggregations = solr_utils.solr_query(fq)

        # Query Solr for failed aggregation documents
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:aggregation', 'aggregation_success_b:false']
        failed_aggregations = solr_utils.solr_query(fq)

        aggregation_status = 'All aggregations successful'

        if not successful_aggregations and not failed_aggregations:
            aggregation_status = 'No aggregations performed'
        elif not successful_aggregations:
            aggregation_status = 'No successful aggregations'
        elif failed_aggregations:
            aggregation_status = f'{len(failed_aggregations)} aggregations failed'
        return aggregation_status