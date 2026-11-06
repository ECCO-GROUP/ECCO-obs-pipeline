import difflib
from glob import glob
import logging
import os
from typing import Dict, Iterable

import yaml
from jsonschema import validate, ValidationError

import baseclasses

logger = logging.getLogger("pipeline")

DS_CONFIG_GLOB = "conf/ds_configs/*.yaml"
DEPRECATED_CONFIG_GLOB = "conf/ds_configs/deprecated/*.yaml"

# Absolute variants, for the coverage checks. The relative globs above work in
# production because conf.global_settings chdirs to ROOT_DIR on import, but that makes
# them depend on import order anywhere else.
_ABS_DS_CONFIG_GLOB = os.path.join(baseclasses.ROOT_DIR, DS_CONFIG_GLOB)
_ABS_DEPRECATED_CONFIG_GLOB = os.path.join(baseclasses.ROOT_DIR, DEPRECATED_CONFIG_GLOB)


def validate_configs():
    configs = glob(DS_CONFIG_GLOB)
    configs.sort()

    with open("conf/ds_configs/ds_schema.json", "r") as f:
        schema = yaml.load(f, Loader=yaml.Loader)

    for filepath in configs:
        logger.debug(f"Validating {os.path.basename(filepath)}")
        with open(filepath, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
            try:
                validate(config, schema)
            except ValidationError as e:
                logger.error(e)
                raise ValidationError(f"{os.path.basename(filepath)} is malformed.")


def _config_names(pattern: str) -> set:
    return {os.path.splitext(os.path.basename(f))[0] for f in glob(pattern)}


def _ecco_variables() -> set:
    """
    The ecco_variable values actually used by active dataset configs, so a typo in a
    coverage file's exclude_ecco_variables can be caught instead of silently excluding
    nothing.
    """
    variables = set()
    for filepath in glob(_ABS_DS_CONFIG_GLOB):
        with open(filepath, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader) or {}
        variable = config.get("ecco_variable")
        if variable:
            variables.add(variable)
    return variables


def validate_grid_coverage(
    grids_to_use: Iterable[str], grid_registry_entries: Dict[str, object]
) -> list:
    """
    Check conf/grid_coverage.yaml against the grids on disk and the active dataset
    configs. Returns a list of human-readable problems; empty means valid.

    Catching these at startup is what keeps the coverage file honest. The dangling
    dataset check in particular is load-bearing for version bumps: moving a config into
    conf/ds_configs/deprecated/ makes any coverage entry naming it dangle, in the very
    commit that adds the replacement, so a bump cannot quietly drop a grid.

    Returns problems rather than raising so the caller owns the exception type —
    init_pipeline raises PipelinePreconditionError, which is the only error
    run_pipeline.main catches and records in the run summary.
    """
    problems = []

    coverage = baseclasses.load_grid_coverage()
    active = _config_names(_ABS_DS_CONFIG_GLOB)
    deprecated = _config_names(_ABS_DEPRECATED_CONFIG_GLOB)
    known_variables = _ecco_variables()

    # Every enabled grid must declare its coverage. Without this a new grid would
    # silently inherit "every dataset", which is exactly wrong for a narrow grid.
    for grid_name in grids_to_use:
        if grid_name not in coverage:
            problems.append(
                f'grid "{grid_name}" is enabled for this run but has no block in '
                f"{baseclasses.GRID_COVERAGE_PATH}"
            )

    for grid_name, block in coverage.items():
        if grid_name not in grid_registry_entries:
            known = ", ".join(sorted(grid_registry_entries)) or "none"
            problems.append(
                f'{baseclasses.GRID_COVERAGE_PATH} names grid "{grid_name}", which has '
                f"no grid file in grids/. Known grids: {known}"
            )

        for ds_name in block.datasets or []:
            if ds_name in active:
                continue
            hint = ""
            if ds_name in deprecated:
                # The load-bearing case: a version bump moves the old config into
                # deprecated/ in the same commit that adds the new one, so this is
                # what tells the author to update the entry rather than lose the grid.
                hint = (
                    ". That config now lives in conf/ds_configs/deprecated/;"
                    " update this entry to the current version or remove it"
                )
            else:
                # Version suffixes vary in style across datasets (_V11, _v2.1, _V004),
                # so match on similarity rather than splitting on a fixed separator.
                near = difflib.get_close_matches(
                    ds_name, sorted(active), n=3, cutoff=0.7
                )
                if near:
                    hint = f" (did you mean {', '.join(near)}?)"
            problems.append(
                f'{baseclasses.GRID_COVERAGE_PATH}: grid "{grid_name}" names dataset '
                f'"{ds_name}", which has no config in conf/ds_configs/{hint}'
            )

        for variable in sorted(block.excluded_ecco_variables):
            if variable not in known_variables:
                known = ", ".join(sorted(known_variables)) or "none"
                problems.append(
                    f'{baseclasses.GRID_COVERAGE_PATH}: grid "{grid_name}" excludes '
                    f'ecco_variable "{variable}", which no dataset config uses. '
                    f"Known values: {known}"
                )

    return problems
