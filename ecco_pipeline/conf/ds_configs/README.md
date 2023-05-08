# Generating a dataset config file

1. Get shortname ex: `OISSS_L4_multimission_monthly_v2`
2. Fill in provider specifications using a tool like https://cmr.earthdata.nasa.gov/opensearch to find necessary information:
    harvester_type: cmr
    cmr_short_name: "OISSS_L4_multimission_monthly_v2"
    cmr_version: "2.0" -> required for 
    filename_date_regex: '\d{4}-\d{2}' -> filename date format as a regex pattern
    filename_date_fmt: '%Y-%m' 
    provider: 'archive.podaac' -> the base of the GET DATA url (not direct access!)
3. Fill in metadata about the dataset:
    aggregated: false # if data is available aggregated
    data_time_scale: "monthly" # daily or monthly
    regions: []
    fields:
    - name: sss
        long_name: multi-mission OISSS monthly average
        standard_name: sea_surface_salinity
        units: 1e-3
    original_dataset_title: Multi-Mission Optimally Interpolated Sea Surface Salinity Global Monthly Dataset V2
    original_dataset_short_name: OISSS_L4_multimission_monthly_v2
    original_dataset_url: https://podaac.jpl.nasa.gov/dataset/OISSS_L4_multimission_monthly_v2
    original_dataset_reference: http://smap.jpl.nasa.gov/
    original_dataset_doi: 10.1002/2015JC011343
4. Transformation can be filled in from hand inspection of data
5. Aggregation:
    `remove_nan_days_from_data` - data where entire days are nan are removed prior to computing the monthly mean
Add time_bounds_var if dataset has timebounds