import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import yaml

# Anchored to this package rather than left relative to the cwd. The pipeline chdirs
# to ROOT_DIR when conf.global_settings is imported, which makes a relative path work
# in production but depend on import order anywhere else (tests, notebooks).
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset-to-grid coverage (see conf/grid_coverage.yaml). Resolved per dataset by
# Config.grids_for, which both the transformation and aggregation factories call.
GRID_COVERAGE_PATH = os.path.join(ROOT_DIR, "conf", "grid_coverage.yaml")

# Sentinel value of a grid block's ``datasets`` key meaning "every dataset".
ALL_DATASETS = "all"


class GridCoverageError(Exception):
    """
    conf/grid_coverage.yaml is missing, malformed, or does not cover an enabled grid.

    init_pipeline converts this into a PipelinePreconditionError so the run aborts with
    a recorded summary rather than silently processing the wrong set of grids.
    """


@dataclass(frozen=True)
class GridBlock:
    """One grid's coverage rule from conf/grid_coverage.yaml."""

    name: str
    # None means "all datasets"; otherwise the explicit list of dataset config names.
    datasets: Optional[List[str]]
    excluded_ecco_variables: frozenset

    def covers(self, ds_name: str, ecco_variable: Optional[str]) -> bool:
        if ecco_variable in self.excluded_ecco_variables:
            return False
        if self.datasets is None:
            return True
        return ds_name in self.datasets


class Config:
    user_cpus = 1
    grids_to_use = []
    # Parsed grid_coverage.yaml, keyed by grid name. Loaded lazily by grids_for and
    # cached here; set_grid_coverage overrides it (tests, and init_pipeline's
    # eager load so a malformed file aborts at startup rather than mid-run).
    grid_coverage: Optional[Dict[str, GridBlock]] = None
    # {grid_name: [ds_name, ...]} resolved at startup by
    # init_pipeline.resolve_grid_coverage, read back by the run summary. Once grids
    # vary by dataset, the CLI arguments no longer say what a run actually covered.
    resolved_grid_coverage: Optional[Dict[str, list]] = None

    @classmethod
    def grids_for(cls, config: dict) -> List[str]:
        """
        The grids a given dataset should be processed to: the grids enabled for this
        run intersected with the dataset's coverage in conf/grid_coverage.yaml.

        Order follows ``grids_to_use`` (i.e. GRIDS in global_settings.py) so output
        ordering stays stable and predictable.
        """
        coverage = cls.grid_coverage
        if coverage is None:
            coverage = load_grid_coverage()
            cls.grid_coverage = coverage

        # An empty grids_to_use means "everything this deployment knows about".
        # init_pipeline resolves it to GRIDS before the factories run; falling back to
        # the full declared set keeps direct in-process use (tests, notebooks) working
        # without letting a narrow grid fan out past its coverage block.
        enabled = list(cls.grids_to_use) or list(coverage)

        ds_name = config.get("ds_name")
        ecco_variable = config.get("ecco_variable")

        grids = []
        for grid_name in enabled:
            block = coverage.get(grid_name)
            if block is None:
                # validate_grid_coverage rejects this at startup; reaching it here
                # means a caller bypassed init_pipeline, so fail loudly rather than
                # quietly dropping or quietly including the grid.
                raise GridCoverageError(
                    f'Grid "{grid_name}" has no block in {GRID_COVERAGE_PATH}'
                )
            if block.covers(ds_name, ecco_variable):
                grids.append(grid_name)
        return grids


def load_grid_coverage(path: Optional[str] = None) -> Dict[str, GridBlock]:
    """
    Parse conf/grid_coverage.yaml into ``{grid_name: GridBlock}``.

    Raises GridCoverageError on a missing file or a structurally invalid block. Name
    existence (grids against the grid registry, datasets against conf/ds_configs) is
    checked separately by config_validator.validate_grid_coverage, which has access to
    both and can produce a better message.

    ``path`` defaults to the module-level GRID_COVERAGE_PATH, resolved at call time
    rather than captured as a default argument so tests can repoint it.
    """
    path = path or GRID_COVERAGE_PATH
    if not os.path.isfile(path):
        raise GridCoverageError(
            f"Missing grid coverage file: {path}. It declares which datasets are "
            "transformed to which grids; see conf/README.md."
        )

    with open(path, "r") as f:
        raw = yaml.load(f, Loader=yaml.Loader) or {}

    grids = raw.get("grids")
    if not isinstance(grids, dict) or not grids:
        raise GridCoverageError(f'{path} must contain a non-empty "grids" mapping.')

    coverage = {}
    for grid_name, block in grids.items():
        if not isinstance(block, dict):
            raise GridCoverageError(
                f'{path}: block for grid "{grid_name}" must be a mapping.'
            )

        if "datasets" not in block:
            raise GridCoverageError(
                f'{path}: grid "{grid_name}" is missing a "datasets" key '
                f'(use "{ALL_DATASETS}" or a list of dataset names).'
            )

        datasets = block["datasets"]
        if datasets == ALL_DATASETS:
            dataset_names = None
        elif isinstance(datasets, list):
            # An empty list is legitimate: a grid declared but not yet populated.
            dataset_names = [str(name) for name in datasets]
        else:
            raise GridCoverageError(
                f'{path}: grid "{grid_name}" has datasets={datasets!r}; expected '
                f'"{ALL_DATASETS}" or a list of dataset names.'
            )

        excluded = block.get("exclude_ecco_variables") or []
        if not isinstance(excluded, list):
            raise GridCoverageError(
                f'{path}: grid "{grid_name}" has exclude_ecco_variables='
                f"{excluded!r}; expected a list."
            )

        coverage[grid_name] = GridBlock(
            name=grid_name,
            datasets=dataset_names,
            excluded_ecco_variables=frozenset(str(v) for v in excluded),
        )

    return coverage


def set_grids(grids: Iterable[str]):
    Config.grids_to_use = grids


def set_grid_coverage(coverage: Optional[Dict[str, GridBlock]]):
    Config.grid_coverage = coverage


def set_resolved_grid_coverage(resolved: Optional[Dict[str, list]]):
    Config.resolved_grid_coverage = resolved


def set_cpus(cpus: int):
    Config.user_cpus = cpus


class Dataset:
    """
    Base class for working with a dataset's config values.
    """

    def __init__(self, config: dict) -> None:
        self.ds_name: str = config.get("ds_name")
        self.start: datetime = datetime.strptime(
            config.get("start"), "%Y%m%dT%H:%M:%SZ"
        )
        self.end: datetime = (
            datetime.now()
            if config.get("end") == "NOW"
            else datetime.strptime(config.get("end"), "%Y%m%dT%H:%M:%SZ")
        )
        self.harvester_type: str = config.get("harvester_type")
        self.ecco_variable: str = config.get("ecco_variable")
        self.filename_date_fmt: str = config.get("filename_date_fmt")
        self.filename_date_regex: str = config.get("filename_date_regex")
        self.data_time_scale: str = config.get("data_time_scale")
        self.hemi_pattern: dict = config.get("hemi_pattern")
        self.fields: Iterable[Field] = [
            Field(**config_field) for config_field in config.get("fields")
        ]
        self.og_ds_metadata: dict = {k: v for k, v in config.items() if "original" in k}
        self.preprocessing_function: str = config.get("preprocessing")
        self.t_version = config.get("t_version")
        self.a_version = config.get("a_version")
        self.note: str = config.get("notes")


@dataclass
class Field:
    name: str
    long_name: str
    standard_name: str
    units: str
    pre_transformations: Iterable[str]
    post_transformations: Iterable[str]
