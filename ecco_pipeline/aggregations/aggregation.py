import numpy as np

import netCDF4 as nc4
from collections import defaultdict
from utils import solr_utils
import logging

class Aggregation():
    
    def __init__(self, config: dict) -> None:
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

    def _set_ds_meta(self):
        fq = [f'dataset_s:{self.dataset_name}', 'type_s:dataset']
        ds_meta = solr_utils.solr_query(fq)[0]
        if 'start_date_dt' not in ds_meta:
            logging.info('No transformed granules to aggregate.')
            raise Exception('No transformed granules to aggregate.')
        self.ds_meta = ds_meta