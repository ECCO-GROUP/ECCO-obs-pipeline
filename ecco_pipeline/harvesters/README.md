# Harvesters

Harvesting typically consists of two stages: enumeration of desired data granules, and fetching of those granules.

Harvesting is supported for four kinds of datasets:
- Datasets available in NASA's CMR metadata repository
- Datasets available on OSISAF's FTP
- Datasets available on NSIDC's FTP
- Datasets available on iFremer's FTP

## Enumeration

Enumeration consists of determining which granules need to be downloaded using the parameters defined in a given dataset's config file.

For CMR datasets (typically those available on a NASA DAAC such as PODAAC), the pipeline queries CMR for relevant granules. It tracks some metadata for each granule: data url, time modified, etc., which is used during the fetch stage. The pipeline can also filter out unwanted results in this stage, as is the case with GRACE datasets that have drifted too far away from the target date.

For FTP datasets, the enumeration stage happens concurrently with the fetch stage, as the FTP must be traversed.

## Fetching

Fetching consists of downloading a granule, if needed. A granule only needs to be downloaded if the file modified time at the source is greater than the file modified time on disk.