from datetime import datetime
from typing import Iterable
from field import Field

class Dataset():
    '''
    Base class for working with a dataset's config values. Contains 
    '''
    def __init__(self, config: dict) -> None:
        self.ds_name: str = config.get('ds_name')
        self.start: datetime = datetime.strptime(config.get('start'), '%Y%m%dT%H:%M:%SZ')
        self.end: datetime = datetime.now() if config.get('end') == 'NOW' else datetime.strptime(config.get('end'), '%Y%m%dT%H:%M:%SZ')
        self.harvester_type: str = config.get('harvester_type')
        self.filename_date_fmt: str = config.get('filename_date_fmt')
        self.filename_date_regex: str = config.get('filename_date_regex')        
        self.data_time_scale: str = config.get('data_time_scale')
        self.hemi_pattern:dict = config.get('hemi_pattern')        
        self.fields: Iterable[Field] = [Field(**config_field) for config_field in config.get('fields')]        
        self.og_ds_metadata: dict = {k: v for k, v in config.items() if 'original' in k}