from multiprocessing import current_process
from typing import Iterable
import numpy as np

import netCDF4 as nc4
from collections import defaultdict
from dataset import Dataset
from utils import solr_utils
import logging

logger = logging.getLogger(str(current_process().pid))


class Aggregation(Dataset):
    '''
    Aggregation class for containing all dataset level metadata. Values come from
    dataset config and the state of the dataset in Solr. 
    '''
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.version: str = str(config.get('a_version', ''))
        self.precision: float = getattr(np, config.get('array_precision'))
        self.binary_dtype: str = '>f4' if self.precision == np.float32 else '>f8'
        self.nc_fill_val: float = nc4.default_fillvals[self.binary_dtype.replace('>', '')]
        self.bin_fill_val: int = -9999
        self.do_monthly_aggregation: bool = config.get('do_monthly_aggregation', False)
        self.remove_nan_days_from_data: bool = config.get('remove_nan_days_from_data', True)
        self.skipna_in_mean: bool = config.get('skipna_in_mean', False)

        self.save_binary: bool = config.get('save_binary', True)
        self.save_netcdf: bool = config.get('save_netcdf', True)

        self.transformations: Iterable[dict] = defaultdict(list)
        self._set_ds_meta()

    def _set_ds_meta(self):
        fq = [f'dataset_s:{self.ds_name}', 'type_s:dataset']
        ds_meta = solr_utils.solr_query(fq)[0]
        if 'start_date_dt' not in ds_meta:
            logger.info('No transformed granules to aggregate.')
            raise Exception('No transformed granules to aggregate.')
        self.ds_meta = ds_meta