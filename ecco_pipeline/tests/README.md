# ECCO Pipeline Tests

Unit tests for the ECCO observation pipeline.

## Structure

- `harvesters/` - Unit tests for data harvesting (enumeration and fetching)
- `transformations/` - Unit tests for grid transformation classes
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

### Utility Tests

- `test_records.py` - Tests for TimeBound, make_empty_record, save_netcdf
- `test_ds_functions.py` - Tests for preprocessing, pre-transformation, and post-transformation functions
- `test_transformation_utils.py` - Tests for grid mapping utilities
