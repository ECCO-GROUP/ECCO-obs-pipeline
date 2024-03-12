# Configuration files

## Dataset configs

Config file per dataset contains all information needed for a dataset to be run through the pipeline from harvesting through aggregation. See README in `ecco_pipeline/conf/ds_configs` for more information. `ecco_pipeline/conf/ds_configs/deprecated` contains configs for datasets no longer supported, typically because they have been supplanted by a newer version.

## global_settings.py

Script that contains some settings that are used globally throughout the pipeline, such as the location of the output directory. This file must be manually set up after cloning the repo.