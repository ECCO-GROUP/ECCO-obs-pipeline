from glob import glob
import logging
from jsonschema import validate, ValidationError
import yaml
import os

logger = logging.getLogger("pipeline")


def validate_configs():
    configs = glob("conf/ds_configs/*.yaml")
    configs.sort()

    with open("conf/ds_configs/ds_schema.json", "r") as f:
        schema = yaml.load(f, Loader=yaml.Loader)

    for filepath in configs:
        logger.debug(f"Validating {os.path.basename(filepath)}")
        with open(filepath, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
            try:
                validate(config, schema)
            except ValidationError as e:
                logger.error(e)
                raise ValidationError(f"{os.path.basename(filepath)} is malformed.")
