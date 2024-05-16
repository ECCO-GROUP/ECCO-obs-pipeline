# ECCO-PIPELINE

## Summary
The ECCO_OBS_PIPELINE exists as a framework to regularly harvest and transform datasets used as inputs for the ECCO model. 

Data is harvested from multiple sources (PO.DAAC, NSIDC, OSISAF, iFremer) and transformed to a variety of grids and formats (binary and netCDF, daily and monthly average). A Solr database is utilized for maintaining the state of a dataset between pipeline runs.

## Documentation
Documentation is in the process of being overhauled. Legacy documentation can be found on the repo's wiki: https://github.com/ECCO-GROUP/ECCO-obs-pipeline/wiki. 

## Setup

### Requirements
- Solr (metadata server/database)
- Conda (package management)
- .netrc file containing valid Earthdata login credentials

### Standup Solr Server
Follow steps in the Solr deployment guide to download Solr package:
https://solr.apache.org/guide/solr/latest/deployment-guide/installing-solr.html


### Start Solr and Setup core
```
cd <path/to/solr/directory>
bin/solr start
bin/solr create -c ecco_datasets
```

### Clone repo
```
git clone https://github.com/ECCO-GROUP/ECCO-obs-pipeline.git
```

### Install dependencies via Conda envrionment 
```
cd <path/to/cloned/repo>
conda env create -f environment.yml
conda activate ecco_pipeline
```

### Setup global_settings.py
```
cp ecco_pipeline/conf/global_settings.py.example ecco_pipeline/conf/global_settings.py
```
Fill in variables.

### Running pipeline
```
python ecco_pipeline/run_pipeline.py
```
The above command will execute the pipeline by iterating through all currently supported datasets, running the harvesting, transformation, and aggregation steps for each. If you want to run the pipeline on a single dataset or step, you can use the `--dataset` and `--step` flags, respectively. ex:

```
python ecco_pipeline/run_pipeline.py --dataset G02202_V4 --step harvest
```

You can also run the pipeline via an interactive menu where you can select from a list of the supported datasets and the steps of the pipeline to run by providing the `--menu` flag. 

In either case, the pipeline will default to using the list of grids provided in `ecco_pipeline/conf/global_settings.py`, but can be overridden for a specific list of grids with the `--grids_to_use` argument. ex:

```
python ecco_pipeline/run_pipeline.py --grids_to_use ECCO_llc90
```

The default logging level is set to `info` but is adjustable via the `--log_level` flag when running the pipeline. ex:
```
python ecco_pipeline/run_pipeline.py --log_level DEBUG
```
