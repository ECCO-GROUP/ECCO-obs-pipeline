# ECCO Pipeline Enumeration

For non-FTP data sources, an initial enumeration step builds up lists of possible granules to download for a given dataset. This is typically a download url along with a modification time from the source. The comparison between local data modification time and source data modification time does not happen during enumeration.

## CMR Enumeration

Datasets hosted at NASA DAACs in most instances are able to be queried via NASA's CMR metadata repository using a dataset's unique identifier. CMR metadata is parsed to find the appropriate download link.

## NSIDC Enumeration

Datasets hosted at NSIDC that are not in CMR are enumerated via web scraping as it is faster than navigating the FTP. 

## OSISAF Enumeration

Datasets hosted at OSISAF are enumerated via web scraping the thredds server as it is faster than navigating the FTP. Download urls are hosted at the thredds fileserver.