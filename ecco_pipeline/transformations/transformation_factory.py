import logging
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, current_process
from typing import Iterable

import xarray as xr
import baseclasses
from conf.global_settings import OUTPUT_DIR
from transformations.grid_transformation import Transformation, transform
from utils.pipeline_utils import file_utils, log_config, solr_utils

logger = logging.getLogger("pipeline")


def multiprocess_transformation(
    config: dict, granule: dict, tx_jobs: dict, log_level: str, log_dir: str
) -> tuple:
    """
    Callable function that performs the actual transformation on a granule.

    Returns a status marker ``(granule_name, status, detail)`` so the Pool driver
    can surface worker failures instead of silently dropping them. ``status`` is
    one of:
      - "ok":      transformation ran (per-grid failures are tracked in Solr).
      - "skipped": granule was not harvested properly and was intentionally skipped.
      - "error":   an unhandled exception escaped the transform path.

    The whole body is exception-safe: a granule that hard-crashes returns an
    "error" marker rather than propagating, so one bad granule cannot abort the
    rest of the batch (and the single-CPU loop stays robust for free).
    """
    try:
        logger = log_config.mp_logging(str(current_process().pid), log_level, log_dir)
    except Exception as e:
        print(e)

    granule_filepath = granule.get("pre_transformation_file_path_s")
    granule_name = granule.get("filename_s") or granule_filepath or "<unknown granule>"
    granule_date = granule.get("date_dt")

    try:
        # Skips granules that weren't harvested properly
        file_size = granule.get("file_size_l") or 0
        if not granule_filepath or file_size < 100:
            if not granule_filepath:
                error_msg = "Granule not harvested properly: no source file path recorded."
            else:
                error_msg = (
                    f"Granule not harvested properly: source file missing or too small "
                    f"({file_size} bytes)."
                )
            logger.error(f"{granule_name}: {error_msg} Skipping.")

            completed_dt = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            updates = []
            for grid_name, fields in tx_jobs.items():
                for field in fields:
                    docs = []
                    if granule_filepath:
                        fq = [
                            f"dataset_s:{config['ds_name']}",
                            "type_s:transformation",
                            f"grid_name_s:{grid_name}",
                            f"field_s:{field.name}",
                            f'pre_transformation_file_path_s:"{granule_filepath}"',
                        ]
                        docs = solr_utils.solr_query(fq)
                    if docs:
                        updates.append({
                            "id": docs[0]["id"],
                            "success_b": {"set": False},
                            "transformation_in_progress_b": {"set": False},
                            "transformation_completed_dt": {"set": completed_dt},
                            "error_message_s": {"set": error_msg},
                        })
                    else:
                        # No transformation doc exists yet — create one so the failure
                        # is visible on the dashboard instead of silently skipped.
                        updates.append({
                            "type_s": "transformation",
                            "dataset_s": config["ds_name"],
                            "date_dt": granule_date,
                            "grid_name_s": grid_name,
                            "field_s": field.name,
                            "pre_transformation_file_path_s": granule_filepath or "",
                            "transformation_started_dt": completed_dt,
                            "transformation_completed_dt": completed_dt,
                            "transformation_in_progress_b": False,
                            "success_b": False,
                            "error_message_s": error_msg,
                        })
            if updates:
                solr_utils.solr_update(updates)
            return (granule_name, "skipped", error_msg)

        # Perform remaining transformations
        logger.info(
            f'{sum([len(v) for v in tx_jobs.values()])} remaining transformations for {granule_filepath.split("/")[-1]}'
        )
        transform(granule_filepath, tx_jobs, config, granule_date)
        return (granule_name, "ok", "")
    except Exception as e:
        # Outer safety net: catches cases not already handled inside transform()
        # (Transformation construction, load_file, factor load, unexpected errors).
        logger.exception(f"Error transforming {granule_name}: {e}")
        return (granule_name, "error", str(e) or repr(e))


class TxJobFactory(baseclasses.Dataset):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.config = config
        self.user_cpus = baseclasses.Config.user_cpus
        # Count of transformation docs rebuilt from existing outputs during job
        # generation (get_tx_jobs); surfaced so a reconstruction-only run reports
        # its status instead of "No transformations performed".
        self.reconstructed_count = 0
        # fl scoped to the granule fields consumed downstream (get_tx_jobs,
        # need_to_update/need_to_transform, find_data_for_factors,
        # reconstruct_tx_solr_doc, and the file_size_l harvest-quality check in
        # multiprocess_transformation). Keep in sync if new granule fields are read.
        self.harvested_granules = solr_utils.solr_query(
            [f"dataset_s:{self.ds_name}", "type_s:granule", "harvest_success_b:true"],
            fl="filename_s,date_dt,pre_transformation_file_path_s,checksum_s,file_size_l",
        )

        if not baseclasses.Config.grids_to_use:
            fq = ["type_s:grid"]
            docs = solr_utils.solr_query(fq)
            self.grids = [doc["grid_name_s"] for doc in docs]
        else:
            self.grids = baseclasses.Config.grids_to_use
        if "hemi_pattern" in self.config:
            logger.info("Skipping job creation for TPOSE grid on sea ice data.")
            self.grids = [grid for grid in self.grids if "TPOSE" not in grid]

    def start_factory(self) -> str:
        if not self.harvested_granules:
            logger.info(f"No harvested granules found in solr for {self.ds_name}")
            return "No transformations performed"
        self.initialize_jobs()
        if self.job_params:
            self.execute_jobs()
        elif not self.reconstructed_count:
            # No new transformations and nothing reconstructed — genuinely nothing
            # changed, so skip the cleanup queries.
            return "No transformations performed"

        # Either new transformations ran or docs were reconstructed from existing
        # outputs during job generation; run cleanup so the dataset status and
        # counts reflect the actual Solr state (and refresh the dashboard).
        pipeline_status = self.pipeline_cleanup()
        return pipeline_status

    def initialize_jobs(self):
        self.pregenerate_factors()
        self.job_params = self.generate_jobs()
        logger.info(
            f"{len(self.job_params)} harvested granules with remaining transformations."
        )

    def execute_jobs(self):
        if self.job_params:
            if self.user_cpus == 1:
                logger.info("Not using multiprocessing to do transformation")
                results = [
                    multiprocess_transformation(*job_param)
                    for job_param in self.job_params
                ]
            else:
                # Honor the user's requested process count (already bounded to
                # [1, cpu_count()] by the --multiprocesses argparse choices), capped
                # only by the number of jobs actually available to run.
                user_cpus = min(self.user_cpus, len(self.job_params))
                logger.info(
                    f"Using {user_cpus} CPUs to do {len(self.job_params)} multiprocess transformation jobs"
                )

                # Synchronous starmap so worker return markers are collected (and any
                # unhandled exception is re-raised here). multiprocess_transformation
                # is exception-safe and returns a marker instead of propagating, so one
                # bad granule no longer aborts the remaining jobs.
                with Pool(processes=user_cpus) as pool:
                    results = pool.starmap(multiprocess_transformation, self.job_params)

            self.summarize_results(results)

            # Per-field completion writes use commitWithin (commit=False), so the
            # last granules' success_b=True updates may not be committed yet. Flush
            # them before pipeline_cleanup counts success_b:false, otherwise
            # just-finished transformations transiently read as failed.
            solr_utils.commit_solr()

    def summarize_results(self, results: Iterable[tuple]):
        """
        Log a summary of the per-granule status markers returned by
        multiprocess_transformation so worker-level failures are visible.
        """
        results = [r for r in results if r]
        total = len(results)
        errors = [r for r in results if r[1] == "error"]
        skipped = [r for r in results if r[1] == "skipped"]

        if errors:
            logger.error(
                f"{len(errors)} of {total} granules failed with an unhandled worker error:"
            )
            for granule_name, _, detail in errors:
                logger.error(f"  {granule_name}: {detail}")
        if skipped:
            logger.warning(
                f"{len(skipped)} of {total} granules were skipped (not harvested properly)."
            )
        logger.info(
            f"Transformation batch complete: {total - len(errors) - len(skipped)} of "
            f"{total} granules processed without a worker-level error."
        )

    def pipeline_cleanup(self) -> str:
        # Query Solr for dataset metadata
        fq = [f"dataset_s:{self.ds_name}", "type_s:dataset"]
        dataset_metadata = solr_utils.solr_query(fq, rows=1)[0]

        # Only counts are used below, so count instead of fetching every doc.
        fq = [f"dataset_s:{self.ds_name}", "type_s:transformation", "success_b:true"]
        successful_count = solr_utils.solr_count(fq)

        fq = [f"dataset_s:{self.ds_name}", "type_s:transformation", "success_b:false"]
        failed_count = solr_utils.solr_count(fq)

        transformation_status = "All transformations successful"

        if not successful_count and not failed_count:
            transformation_status = "No transformations performed"
        elif not successful_count:
            transformation_status = "No successful transformations"
        elif failed_count:
            transformation_status = f"{failed_count} transformations failed"

        # Update Solr dataset entry status to transformed
        update_body = [
            {
                "id": dataset_metadata["id"],
                "transformation_status_s": {"set": transformation_status},
                "last_transformation_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
            }
        ]

        r = solr_utils.solr_update(update_body, r=True)

        if r.status_code == 200:
            logger.debug(
                f"Successfully updated Solr with transformation information for {self.ds_name}"
            )
        else:
            logger.exception(
                f"Failed to update Solr with transformation information for {self.ds_name}"
            )

        return transformation_status

    def pregenerate_factors(self):
        """
        Generates mapping factors for all grids used for the given transformation version. Loads them into
        Factors object which is used to reduce I/O.
        """
        for grid in self.grids:
            for granule in self.find_data_for_factors():
                grid_ds = xr.open_dataset(f"grids/{grid}.nc")
                T = Transformation(
                    self.config, granule["pre_transformation_file_path_s"], "1972-01-01"
                )
                T.make_factors(grid_ds)

    def find_data_for_factors(self) -> Iterable[dict]:
        """
        Returns Solr granule entry (two in the case of hemispherical data) to be used
        to generate factors
        """
        data_for_factors = []
        nh_added = False
        sh_added = False
        hemi_pattern = self.config.get("hemi_pattern", "")
        # Find appropriate granule(s) to use for factor calculation
        for granule in self.harvested_granules:
            file_path = granule.get("pre_transformation_file_path_s")
            if file_path and hemi_pattern:
                # Get one of each
                if self.hemi_pattern["north"] in file_path and not nh_added:
                    data_for_factors.append(granule)
                    nh_added = True
                elif self.hemi_pattern["south"] in file_path and not sh_added:
                    data_for_factors.append(granule)
                    sh_added = True
                if nh_added and sh_added:
                    return data_for_factors
            elif file_path:
                data_for_factors.append(granule)
                return data_for_factors
        raise RuntimeError(
            "Unable to find sufficient data in order to pregenerate mapping factors."
        )

    def generate_jobs(self):
        logger.info("Generating jobs...")
        log_level = logging.getLevelName(logging.getLogger("pipeline").level)
        log_dir = os.path.dirname(
            logging.getLogger("pipeline").handlers[0].baseFilename
        )
        log_dir = os.path.join(log_dir[log_dir.find("logs/") :], f"tx_{self.ds_name}")

        all_jobs = self.get_tx_jobs()

        new_jobs = []
        for granule, grid_fields in all_jobs:
            job_params = (self.config, granule, grid_fields, log_level, log_dir)
            new_jobs.append(job_params)
        return new_jobs

    def get_tx_jobs(self):
        fq = [f"dataset_s:{self.ds_name}", "type_s:transformation"]
        # fl scoped to the tx fields used below + in need_to_update(). This is the
        # largest query in the stage (granules x grids x fields); keep in sync if new
        # tx fields are read here.
        solr_txs = solr_utils.solr_query(
            fq,
            fl="pre_transformation_file_path_s,field_s,grid_name_s,success_b,transformation_version_f,origin_checksum_s",
        )
        tx_dict = defaultdict(list)
        for tx in solr_txs:
            tx_dict[tx["pre_transformation_file_path_s"].split("/")[-1]].append(tx)

        all_jobs = []
        reconstructed = 0
        for granule in self.harvested_granules:
            grid_fields = {}
            for grid in self.grids:
                fields_for_grid = []
                for field in self.fields:
                    update = True
                    for tx in tx_dict[granule["filename_s"]]:
                        if tx["field_s"] == field.name and tx["grid_name_s"] == grid:
                            update = self.need_to_update(granule, tx)
                            break
                    else:
                        update = self.need_to_transform(granule, grid, field)
                        if not update:
                            self.reconstruct_tx_solr_doc(granule, grid, field)
                            reconstructed += 1
                            # Reconstruction hashes each output file, so a large
                            # backlog takes minutes — log progress so job generation
                            # doesn't look hung.
                            if reconstructed % 250 == 0:
                                logger.info(
                                    f"Reconstructed {reconstructed} missing transformation "
                                    f"docs from existing outputs so far..."
                                )
                    if update:
                        fields_for_grid.append(field)
                if fields_for_grid:
                    grid_fields[grid] = fields_for_grid
            if grid_fields:
                all_jobs.append((granule, grid_fields))

        self.reconstructed_count = reconstructed
        if reconstructed:
            logger.info(
                f"Reconstructed {reconstructed} missing transformation Solr doc(s) from "
                f"existing outputs (no re-transformation needed)."
            )
            # reconstruct_tx_solr_doc uses commitWithin; flush the batch now so the
            # docs are searchable before pipeline_cleanup counts them.
            solr_utils.commit_solr()
        return all_jobs

    def need_to_update(self, granule: dict, tx: dict) -> bool:
        """
        Triple if:
        1. do we have a version entry,
        2. compare transformation version number and current transformation version number
        3. compare checksum of harvested file (currently in solr) and checksum
        of the harvested file that was previously transformed (recorded in transformation entry)
        """
        if (
            tx.get("success_b")
            and tx.get("transformation_version_f") == self.t_version
            and tx["origin_checksum_s"] == granule["checksum_s"]
        ):
            return False
        return True

    def need_to_transform(self, granule: dict, grid_name: str, field) -> bool:
        """
        Filesystem fallback used when no Solr transformation doc exists for a
        granule/grid/field combination.  Skip reprocessing only if the transformed
        output file already exists, is newer than the source granule file, and was
        produced by the current transformation version — the same version contract
        need_to_update() applies to the Solr-doc path.
        """
        stem = os.path.splitext(granule["filename_s"])[0]
        output_path = os.path.join(
            OUTPUT_DIR,
            self.ds_name,
            "transformed_products",
            grid_name,
            "transformed",
            field.name,
            f"{grid_name}_{field.name}_{stem}.nc",
        )
        if not os.path.exists(output_path):
            return True
        source_path = granule.get("pre_transformation_file_path_s")
        if not source_path or not os.path.exists(source_path):
            return True
        if os.path.getmtime(output_path) <= os.path.getmtime(source_path):
            return True
        # mtime says the file is current, but only trust it if it was produced by
        # the current transformation version. Bumping t_version in config is the
        # single lever that forces re-transformation after any output-affecting
        # code change — keeps this path consistent with need_to_update().
        if self._file_transformation_version(output_path) != self.t_version:
            logger.info(
                f"Existing transformed file {output_path} was produced by a "
                f"different transformation version — will re-transform."
            )
            return True
        return False

    def _file_transformation_version(self, output_path: str):
        """
        Read the transformation_version global attribute baked into a transformed
        netCDF by grid_transformation.transform(). Returns None if the file can't
        be read or carries no version (treated as a version mismatch by callers).
        """
        try:
            with xr.open_dataset(output_path) as ds:
                version = ds.attrs.get("transformation_version")
        except Exception:
            return None
        if version is None:
            return None
        try:
            return float(version)
        except (TypeError, ValueError):
            return version

    def reconstruct_tx_solr_doc(self, granule: dict, grid_name: str, field) -> None:
        """
        Creates a Solr transformation doc from an existing output file when the
        doc is missing but processing was determined to be unnecessary.  Mirrors
        the fields written by prepopulate_solr() + the post-transform update in
        grid_transformation.transform().
        """
        stem = os.path.splitext(granule["filename_s"])[0]
        output_filename = f"{grid_name}_{field.name}_{stem}.nc"
        output_path = os.path.join(
            OUTPUT_DIR,
            self.ds_name,
            "transformed_products",
            grid_name,
            "transformed",
            field.name,
            output_filename,
        )

        hemi = ""
        if self.hemi_pattern:
            if self.hemi_pattern["north"] in granule["filename_s"]:
                hemi = "nh"
            elif self.hemi_pattern["south"] in granule["filename_s"]:
                hemi = "sh"

        completed_dt = datetime.utcfromtimestamp(
            os.path.getmtime(output_path)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        doc = {
            "type_s": "transformation",
            "date_dt": granule["date_dt"],
            "dataset_s": self.ds_name,
            "pre_transformation_file_path_s": granule.get("pre_transformation_file_path_s"),
            "hemisphere_s": hemi,
            "origin_checksum_s": granule.get("checksum_s"),
            "grid_name_s": grid_name,
            "field_s": field.name,
            "transformation_in_progress_b": False,
            "transformation_started_dt": completed_dt,
            "success_b": True,
            "filename_s": output_filename,
            "transformation_file_path_s": output_path,
            "transformation_completed_dt": completed_dt,
            "transformation_checksum_s": file_utils.md5(output_path),
            "transformation_version_f": self.t_version,
            "error_message_s": "",
        }

        # commit=False: these are written in bulk from get_tx_jobs (one per missing
        # grid/field/granule doc) and are not read back within the same run. A hard
        # commit per doc was a commit/searcher-warming storm that made job generation
        # hang for minutes on datasets with many existing outputs. get_tx_jobs issues
        # a single commit after the batch.
        r = solr_utils.solr_update([doc], r=True, commit=False)
        if r.status_code == 200:
            logger.debug(f"Reconstructed missing Solr transformation doc for {output_filename}")
        else:
            logger.warning(
                f"Failed to reconstruct Solr transformation doc for {output_filename}"
            )
