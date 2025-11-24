# ECCO Pipeline Tests

Unit tests for the ECCO observation pipeline.

## Structure

- `harvesters/` - Unit tests for data harvesting (enumeration and fetching)
- `transformations/` - Unit tests for grid transformation classes
- `aggregations/` - Unit tests for temporal aggregation classes
- `utils/` - Unit tests for utility modules (records, ds_functions, transformation_utils)
- `conftest.py` - Shared pytest fixtures

Visual validation notebooks are located in `ecco_pipeline/validation/`.

## Running Tests

Run all tests from `ecco_pipeline/`:
```bash
python -m pytest tests/ -v
```

Run harvester tests only:
```bash
python -m pytest tests/harvesters/ -v
```

Run aggregation tests only:
```bash
python -m pytest tests/aggregations/ -v
```

## Test Design

All unit tests mock external dependencies (Solr, HTTP requests) to run without network access or external services.

### Harvester Tests

Each harvester type has two test files:
- `test_<type>_enumerator.py` - URL enumeration and granule discovery
- `test_<type>_harvester.py` - Download logic and Solr updates

Base class tests are in `test_harvesterclasses.py`.

### Transformation Tests

- `test_transformation.py` - Tests for Transformation class initialization, factor generation, and Solr population
- `test_transformation_factory.py` - Tests for TxJobFactory job creation and execution

### Aggregation Tests

- `test_aggregation.py` - Tests for Aggregation class initialization, empty record creation, file operations, and monthly aggregation
- `test_aggregation_factory.py` - Tests for AgJobFactory job creation, execution, and pipeline cleanup

### Utility Tests

- `test_records.py` - Tests for TimeBound, make_empty_record, save_netcdf, and save_binary
- `test_ds_functions.py` - Tests for preprocessing, pre-transformation, and post-transformation functions
- `test_transformation_utils.py` - Tests for grid mapping utilities
