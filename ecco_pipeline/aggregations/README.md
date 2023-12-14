# Aggregation

Transformed data are aggregated into daily or monthly (or both) annual netCDFs. 

The aggregation step of the pipeline determines the required annual aggregation jobs needed to be performed for a given dataset across all target grids and all dataset fields. It constructs individual jobs for each required year, grid, field combination.

An aggregation job consists of:
1. Pulling the relevant transformation files for a given year, grid, and field
2. Opening and merging the transformation files (including merging hemispherical data for global coverage)
3. Perform monthly averaging if necessary
4. Saving the aggregated netCDFs with consistent metadata
5. Generating a provenance record which currently exists as a Solr dump of the relevant metadata for each harvested and transformed granule that the aggregation uses.