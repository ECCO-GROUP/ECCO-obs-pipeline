# Generating a dataset config file

The recommended approach to adding support for a new dataset is to start with an existing config or one of the mostly blank templates in `ecco_pipeline/conf/ds_configs/templates`. The best way to obtain the information for these fields is through a combination of looking at a sample data granule and looking at the dataset's documentation. Here we'll walk through the config for `AMSR-2_OSI-408`, looking at the various sections.

## Dataset

```
ds_name: AMSR-2_OSI-408 # Name for dataset
start: "19800101T00:00:01Z" # yyyymmddThh:mm:ssZ
end: "NOW" # yyyymmddThh:mm:ssZ for specific date or "NOW" for...now
```

- `ds_name` is the internal name for the dataset. We recommend using the dataset's shortname or something similar if shortname is not available.
- `start` and `end` are the isoformatted (with Z!) date ranges that the pipeline should try to process. The `end` field can also be the string "NOW" which will set the end field to the current datetime at runtime.

## Harvester
This section contains fields that dictate which harvester the data should be pulled from. There are different fields required depending on the harvester. These are the fields consistent across all harvesters.
```
harvester_type: osisaf
filename_date_fmt: "%Y%m%d"
filename_date_regex: '\d{8}'
```
- `harvester_type` can be one of `cmr`, `osisaf`, `nisdc`, or `catds`.
- `filename_date_fmt` is the string format of the date in the filename.
- `filename_date_regex` is the regex format of the date in filename.

The section where `cmr` is the harvester_type includes the following:
```
cmr_concept_id: C2491756442-POCLOUD
provider: "archive.podaac"
```
- `cmr_concept_id` is the unique concept_id identifier in CMR for the datatset
- `provider` is the provider of the data, used to select the download URLs. Typically this is set to "archive.podaac" although it will be different for NSIDC datasets.


The section `osisaf` or `catds` is the harvester includes:
```
ddir: "ice/amsr2_conc"
```
- `ddir` is the subdirectory where the specific data can be found. This is only required for non-cmr harvesters, and is not required for the non-cmr NSIDC harvester.

## Metadata
This section includes details specific to the data.
```
data_time_scale: "daily" # daily or monthly
hemi_pattern:
  north: "_nh_"
  south: "_sh_"
fields:
  - name: ice_conc
    long_name: Sea ice concentration
    standard_name: sea_ice_area_fraction
    units: " "
    pre_transformations: []
    post_transformations: ["seaice_concentration_to_fraction"]
original_dataset_title: "Global Sea Ice Concentration (AMSR-2)"
original_dataset_short_name: "Global Sea Ice Concentration (AMSR-2)"
original_dataset_url: "https://osi-saf.eumetsat.int/products/osi-408"
original_dataset_reference: "https://osisaf-hl.met.no/sites/osisaf-hl.met.no/files/user_manuals/osisaf_cdop2_ss2_pum_amsr2-ice-conc_v1p1.pdf"
original_dataset_doi: "OSI-408"
```
- `data_time_scale` is the time scale of the data, either daily or monthly. Monthly data is considered data averaged per month - all other data is considered daily.
- `hemi_pattern` sets the filename pattern for data split by hemisphere. This section can be omitted for datasets that don't do this.
- `fields` is the list of data variables that should be transformed as part of the pipeline. You need to manually provide the field's `name`, `long_name`, `standard_name`, and `units`. The `pre_transformations` and `post_transformations` are the names of functions to be applied to the specific data field. Some examples are units conversion, or data masking. Functions are defined in `ecco_pipeline/utils/processing_utils/ds_functions.py`. 
- The five `original_*` fields are dataset level metadata that will be included in transformed file metadata.

## Transformation
This section contains fields required for transformation and contain information on data resolution etc. For hemispherical data, `area_extent`, `dims`, and `proj_info` must be defined for each hemispher as below. It is not unusual for the values in this section to be determined iteratively. The testing notebooks in `tests/quicklook_notebooks/` are a useful tool in determining the validity of the values provided.
```
t_version: 2.0 # Update this value if any changes are made to this file
data_res: 10/111 # Resolution of dataset

# Values for non split datasets (for datasets split into nh/sh, append '_nh'/'_sh')
area_extent_nh: [-3845000, -5345000, 3745000, 5845000]
area_extent_sh: [-3950000, -3950000, 3945000, 4340000]
dims_nh: [760, 1120]
dims_sh: [790, 830]
proj_info_nh:
  area_id: "3411"
  area_name: "polar_stereographic"
  proj_id: "3411"
  proj4_args: "+init=EPSG:3411"
proj_info_sh:
  area_id: "3412"
  area_name: "polar_stereographic"
  proj_id: "3412"
  proj4_args: "+init=EPSG:3412"

notes: ""
```
- `t_version` is a metadata field used internally in the pipeline. Modifying the value will trigger retransformation.
- `data_res` is the spatial resolution of the dataset in degrees
- `area_extent` is the area extent specific to this data in the form: lower_left_x, lower_left_y, upper_right_x, upper_right_y
- `dims` is the size of longitude or x coordinate, latitude or y coordinate
- `proj_info` contains projection information used by pyresample
- `notes` is an optional string to include in global metadata in output files

## Aggregation
This section contains fields required for aggregating data.
```
a_version: 1.3 # Update this value if any changes are made to this file
remove_nan_days_from_data: False # Remove empty days from data when aggregating
do_monthly_aggregation: True
skipna_in_mean: True # Controls skipna when calculating monthly mean
```
- `a_version` is a metadata field used internally in the pipeline. Modifying the value will trigger reaggregation.
- `remove_nan_days_from_data` will remove nan days from aggregated outputs
- `do_monthly_aggregation` will also compute monthly averages when aggregating annual files
- `skipna_in_mean` is used when calculating the monthly mean