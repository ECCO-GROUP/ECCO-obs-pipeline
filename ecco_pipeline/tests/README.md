# ECCO Pipeline Tests

Unit tests for the ECCO observation pipeline.

## Structure

- `harvesters/` - Unit tests for data harvesting (enumeration and fetching)
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
