from argparse import Namespace
import os
from glob import glob
import logging
from typing import Iterable
import requests
import netrc
import yaml
from datetime import datetime
from utils.pipeline_utils import (
    solr_utils,
    config_validator,
    grid_registry,
    log_config,
    file_utils,
)
import baseclasses

try:
    from conf.global_settings import OUTPUT_DIR, SOLR_COLLECTION, SOLR_HOST, GRIDS
except ImportError:
    raise ImportError(
        "Missing global_settings.py file. See ecco_pipeline/conf/global_settings.py.example for more info."
    )


class PipelinePreconditionError(Exception):
    """
    A precondition for running the pipeline is not met — Solr unreachable, missing
    Earthdata credentials, missing output directory, grids absent from Solr.

    Raised instead of calling exit() so run_pipeline can record the aborted run and
    exit with a distinct non-zero code. A bare exit() returns 0, which made every one
    of these look like a successful run to cron.
    """


def setup_logger(args):
    logger = log_config.mp_logging("pipeline", args.log_level)
    # Set package logging level to WARNING
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger


def validate_output_dir():
    logger = logging.getLogger("pipeline")
    # Verify output directory is valid
    if not os.path.isdir(OUTPUT_DIR):
        logger.fatal("Missing output directory. Please fill in. Exiting.")
        raise PipelinePreconditionError(
            f"Output directory does not exist: {OUTPUT_DIR}"
        )
    logger.debug(f"Using output directory: {OUTPUT_DIR}")


def validate_solr():
    logger = logging.getLogger("pipeline")

    logger.debug(f"Using Solr collection: {SOLR_COLLECTION}")

    # Verify solr is running
    try:
        solr_utils.ping_solr()
    except requests.ConnectionError:
        logger.fatal("Solr is not currently running! Start Solr and try again.")
        raise PipelinePreconditionError(f"Solr unreachable at {SOLR_HOST}")

    if not solr_utils.collection_check():
        logger.fatal(
            f'Solr collection {SOLR_COLLECTION} does not exist. Add a collection using "bin/solr create -c {{collection_name}}".'
        )
        raise PipelinePreconditionError(
            f"Solr collection does not exist: {SOLR_COLLECTION}"
        )


def wipe_factors():
    logger = logging.getLogger("pipeline")

    logger.info("Removing all factors")
    all_factors = glob(f"{OUTPUT_DIR}/**/transformed_products/**/*_factors")
    for factors_file in all_factors:
        try:
            os.remove(factors_file)
        except Exception:
            logger.error(f"Error removing {factors_file}")
        logger.info("Successfully removed all factors")


def validate_netrc():
    logger = logging.getLogger("pipeline")

    try:
        nrc = netrc.netrc()
        if "urs.earthdata.nasa.gov" not in nrc.hosts.keys():
            logger.fatal("Earthdata login required in netrc file.")
            raise PipelinePreconditionError("netrc has no urs.earthdata.nasa.gov entry")
    except FileNotFoundError:
        logger.fatal(
            "No netrc found. Please create one and add Earthdata login credentials."
        )
        raise PipelinePreconditionError("No netrc file found")


def update_solr_grid(grid_name: str, grid_type: str, grid_file_path: str):
    """
    Update solr grid docs with latest versions of grids
    """
    logger = logging.getLogger("pipeline")

    fq = ["type_s:grid"]
    grid_docs = solr_utils.solr_query(fq)

    update_body = []

    grids_in_solr = [doc["grid_name_s"] for doc in grid_docs]

    # Add grid to solr
    if grid_name not in grids_in_solr:
        grid_meta = {
            "type_s": "grid",
            "grid_type_s": grid_type,
            "grid_name_s": grid_name,
            "grid_path_s": str(grid_file_path),
            "date_added_dt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid_checksum_s": file_utils.md5(grid_file_path),
        }
        update_body.append(grid_meta)
    # Verify grid in solr matches grid file
    else:
        current_checksum = file_utils.md5(grid_file_path)

        for doc in grid_docs:
            if doc["grid_name_s"] == grid_name:
                solr_checksum = doc["grid_checksum_s"]

        if current_checksum != solr_checksum:
            # Delete previous grid's transformations from Solr
            update_body = {
                "delete": {
                    "query": f"type_s:transformation AND grid_name_s:{grid_name}"
                }
            }

            r = solr_utils.solr_update(update_body, r=True)
            if r.status_code == 200:
                logger.debug(
                    f"Successfully deleted Solr transformation documents for {grid_name}"
                )
            else:
                logger.exception(
                    f"Failed to delete Solr transformation documents for {grid_name}"
                )

            # Delete previous grid's aggregations from Solr
            update_body = {
                "delete": {"query": f"type_s:aggregation AND grid_name_s:{grid_name}"}
            }

            r = solr_utils.solr_update(update_body, r=True)
            if r.status_code == 200:
                logger.debug(
                    f"Successfully deleted Solr aggregation documents for {grid_name}"
                )
            else:
                logger.exception(
                    f"Failed to delete Solr aggregation documents for {grid_name}"
                )

            # Update grid on Solr
            fq = [f"grid_name_s:{grid_name}", "type_s:grid"]
            grid_metadata = solr_utils.solr_query(fq, rows=1)[0]

            update_body = [
                {
                    "id": grid_metadata["id"],
                    "grid_type_s": {"set": grid_type},
                    "grid_name_s": {"set": grid_name},
                    "grid_path_s": {"set": str(grid_file_path)},
                    "grid_checksum_s": {"set": current_checksum},
                    "date_added_dt": {
                        "set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    },
                }
            ]
        else:
            # Contents unchanged, but the file may have been renamed or moved. Refresh
            # the stored path/type so they keep pointing at the real file: aggregation
            # opens the grid via grid_path_s (aggregations/aggregation.py:327), so a
            # stale path breaks it. Deliberately does NOT delete transformation or
            # aggregation docs the way a checksum change does: the grid's contents are
            # identical, so existing work remains valid.
            stale = {}
            for doc in grid_docs:
                if doc["grid_name_s"] != grid_name:
                    continue
                if doc.get("grid_path_s") != str(grid_file_path):
                    stale["grid_path_s"] = {"set": str(grid_file_path)}
                if doc.get("grid_type_s") != grid_type:
                    stale["grid_type_s"] = {"set": grid_type}
                if stale:
                    stale["id"] = doc["id"]
                break

            if not stale:
                return

            logger.info(
                f"Refreshing Solr grid document for {grid_name}: "
                f"{', '.join(k for k in stale if k != 'id')} changed"
            )
            update_body = [stale]

    r = solr_utils.solr_update(update_body, r=True)
    if r.status_code == 200:
        logger.debug(f"Successfully updated {grid_name} Solr grid document")
    else:
        logger.error(f"Failed to update Solr {grid_name} grid document")


def grids_to_solr(grids_to_use: Iterable[str] = []):
    logger = logging.getLogger("pipeline")

    # Grid identity (name/path/type) is owned by the registry, which validates the
    # metadata standard in grids/README.md and rejects duplicate names. This loop used
    # to subscript ds.attrs directly, so a grid file missing an attribute raised
    # KeyError into a caller that logged and continued, silently dropping the grid.
    registry = grid_registry.load_registry()

    # Create Solr grid-type document for each missing grid type
    for grid in registry.values():
        logger.debug(f"Uploading solr grid {grid.name} {grid.type} {grid.path}")
        update_solr_grid(grid.name, grid.type, grid.path)

    # Verify grid names supplied exist on Solr
    grids_not_in_solr = []
    logger.debug(f"Grids to use: {grids_to_use}")
    for grid_name in grids_to_use:
        fq = ["type_s:grid", f"grid_name_s:{grid_name}"]
        docs = solr_utils.solr_query(fq)
        if not docs:
            grids_not_in_solr.append(grid_name)

    return grids_not_in_solr


def resolve_grid_coverage(grids_to_use: Iterable[str]) -> dict:
    """
    Resolve the enabled grids against every active dataset config, returning
    ``{grid_name: [ds_name, ...]}``.

    Used for the startup log line and the run summary. Once grids vary by dataset,
    "which grids did this run use" is no longer answerable from the CLI arguments
    alone.
    """
    resolved = {grid_name: [] for grid_name in grids_to_use}

    for filepath in sorted(glob(config_validator._ABS_DS_CONFIG_GLOB)):
        with open(filepath, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader) or {}
        ds_name = (
            config.get("ds_name") or os.path.splitext(os.path.basename(filepath))[0]
        )
        for grid_name in baseclasses.Config.grids_for(config):
            resolved.setdefault(grid_name, []).append(ds_name)

    return resolved


def log_grid_coverage(grids_to_use: Iterable[str]):
    logger = logging.getLogger("pipeline")
    resolved = resolve_grid_coverage(grids_to_use)
    baseclasses.set_resolved_grid_coverage(resolved)
    summary = ", ".join(
        f"{grid_name} ({len(datasets)})" for grid_name, datasets in resolved.items()
    )
    logger.info(f"Grid coverage, datasets per grid: {summary or 'none'}")
    for grid_name, datasets in resolved.items():
        logger.debug(f"{grid_name}: {', '.join(datasets) or 'no datasets'}")


def init_pipeline(args: Namespace):
    logger = setup_logger(args)
    validate_output_dir()
    validate_solr()
    config_validator.validate_configs()
    validate_netrc()

    if args.harvested_entry_validation:
        solr_utils.validate_granules()

    if args.wipe_transformations:
        logger.info(
            "Removing transformations with out of sync version numbers from Solr and disk"
        )
        solr_utils.delete_mismatch_transformations()
        pass

    # A bare --grids_to_use (nargs="*") yields [], which must not be read as "every
    # grid": with per-grid dataset coverage that would fan a narrow grid across every
    # dataset. Fall back to the configured GRIDS in both the absent and empty cases.
    if isinstance(args.grids_to_use, list) and args.grids_to_use:
        grids_to_use = args.grids_to_use
    else:
        grids_to_use = GRIDS
    baseclasses.set_grids(grids_to_use)

    # Grid identity and dataset coverage, both validated before any stage runs.
    # Raised as PipelinePreconditionError so run_pipeline records the aborted run;
    # loaded here rather than inside the Solr block below, whose except clause logs
    # and continues.
    try:
        registry = grid_registry.load_registry()
        baseclasses.set_grid_coverage(baseclasses.load_grid_coverage())
    except (grid_registry.GridRegistryError, baseclasses.GridCoverageError) as e:
        logger.fatal(str(e))
        raise PipelinePreconditionError(str(e))

    coverage_problems = config_validator.validate_grid_coverage(grids_to_use, registry)
    if coverage_problems:
        for problem in coverage_problems:
            logger.error(problem)
        raise PipelinePreconditionError(
            f"Invalid grid coverage configuration: {'; '.join(coverage_problems)}"
        )

    log_grid_coverage(grids_to_use)

    if args.grids_to_solr or solr_utils.check_grids():
        grids_not_in_solr = []
        try:
            grids_not_in_solr = grids_to_solr(grids_to_use)
            if not grids_not_in_solr:
                logger.info("Successfully updated grids on Solr.")
        except Exception as e:
            logger.exception(e)
        # Raised outside the try: PipelinePreconditionError is an Exception, so raising
        # it inside would be swallowed by the handler above and the run would continue
        # without its grids — the bare exit() this replaced escaped as SystemExit.
        if grids_not_in_solr:
            for name in grids_not_in_solr:
                logger.error(
                    f'Grid "{name}" not in Solr. Ensure it\'s file name is present in grids_config.yaml and run pipeline with the --grids_to_solr argument'
                )
            raise PipelinePreconditionError(
                f"Grids missing from Solr: {', '.join(grids_not_in_solr)}"
            )

    if args.wipe_factors:
        wipe_factors()

    user_cpus = args.multiprocesses
    baseclasses.set_cpus(user_cpus)
    logger.debug(f"Using {user_cpus} processes for multiprocess transformations")

    return grids_to_use, user_cpus
