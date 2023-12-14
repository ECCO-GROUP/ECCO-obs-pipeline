from glob import glob
import logging
from jsonschema import validate, ValidationError
import yaml

def validate_configs():
    configs = glob('conf/ds_configs/*.yaml')
    configs.sort()
    
    with open('conf/ds_configs/ds_schema.json', 'r') as f:
        schema = yaml.load(f, Loader=yaml.Loader)

    for filepath in configs:
        logging.debug(f'Validating {filepath.split("/")[-1]}')
        with open(filepath, 'r') as f:
            config = yaml.load(f, Loader= yaml.Loader)
            try:
                validate(config, schema)
            except ValidationError as e:
                logging.error(e)
                raise ValidationError(f'{filepath.split("/")[-1]} is malformed.')

if __name__ == '__main__':
    validate_configs()