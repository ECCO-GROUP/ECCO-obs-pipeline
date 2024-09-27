import logging
import os
from multiprocessing import Pool, cpu_count, current_process
from typing import Iterable

from aggregations.aggregation import Aggregation
import baseclasses
from utils.pipeline_utils import log_config, solr_utils

logger = logging.getLogger("pipeline")


def multiprocess_aggregate(job: Aggregation, log_level: str, log_dir: str):
    """
    Function used to execute by multiprocessing to execute a single grid/year/field aggregation
    """
    logger = log_config.mp_logging(str(current_process().pid), log_level, log_dir)
    logger.info(
        f'Beginning aggregation for {job.grid["grid_name_s"]}, {job.year}, {job.field.name}'
    )
    try:
        job.aggregate()
    except Exception as e:
        logger.exception(f"JOB FAILED: {e}")


class AgJobFactory(baseclasses.Dataset):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.config = config
        self.user_cpus = baseclasses.Config.user_cpus
        self.grids = self.get_grids(baseclasses.Config.grids_to_use)
        self.agg_jobs = self.get_jobs()

    def start_factory(self) -> str:
        if not self.agg_jobs:
            logger.info("No new jobs to execute.")
        else:
            self.execute_jobs()
        pipeline_status = self.pipeline_cleanup()
        return pipeline_status

    def execute_jobs(self):
        log_level = logging.getLevelName(logging.getLogger("pipeline").level)
        log_dir = os.path.dirname(
            logging.getLogger("pipeline").handlers[0].baseFilename
        )
        log_dir = os.path.join(log_dir[log_dir.find("logs/") :], f"ag_{self.ds_name}")

        logger.info(
            f'Executing ({len(self.agg_jobs)}) jobs: {", ".join([str(job) for job in self.agg_jobs])}'
        )

        if self.user_cpus == 1:
            logger.info("Not using multiprocessing to do aggregations")
            for agg_job in self.agg_jobs:
                multiprocess_aggregate(agg_job, log_level, log_dir)
        else:
            user_cpus = min(self.user_cpus, int(cpu_count() / 2), 8, len(self.agg_jobs))
            logger.info(f"Using {user_cpus} CPUs to do multiprocess aggregation")
            with Pool(processes=user_cpus) as pool:
                pool.starmap_async(
                    multiprocess_aggregate,
                    [(job, log_level, log_dir) for job in self.agg_jobs],
                )
                pool.close()
                pool.join()

    def pipeline_cleanup(self) -> str:
        aggregation_status = self.get_agg_status()
        self.update_solr_ds(aggregation_status)
        return aggregation_status

    def get_agg_status(self) -> str:
        """
        Queries Solr for dataset aggregations.
        Returns overall status of aggregation, not specific to this particular execution.
        """
        # Query Solr for successful aggregation documents
        fq = [
            f"dataset_s:{self.ds_name}",
            "type_s:aggregation",
            "aggregation_success_b:true",
        ]
        successful_aggregations = solr_utils.solr_query(fq)

        # Query Solr for failed aggregation documents
        fq = [
            f"dataset_s:{self.ds_name}",
            "type_s:aggregation",
            "aggregation_success_b:false",
        ]
        failed_aggregations = solr_utils.solr_query(fq)

        aggregation_status = "All aggregations successful"

        if not successful_aggregations and not failed_aggregations:
            aggregation_status = "No aggregations performed"
        elif not successful_aggregations:
            aggregation_status = "No successful aggregations"
        elif failed_aggregations:
            aggregation_status = f"{len(failed_aggregations)} aggregations failed"
        return aggregation_status

    def update_solr_ds(self, aggregation_status: str):
        """
        Update Solr dataset entry with new aggregation status
        """
        ds_metadata = self.get_solr_ds_metadata()
        update_body = [
            {
                "id": ds_metadata["id"],
                "aggregation_version_s": {"set": str(self.a_version)},
                "aggregation_status_s": {"set": aggregation_status},
            }
        ]

        r = solr_utils.solr_update(update_body, r=True)

        if r.status_code == 200:
            logger.debug(
                f"Successfully updated Solr with aggregation information for {self.ds_name}"
            )
        else:
            logger.exception(
                f"Failed to update Solr dataset entry with aggregation information for {self.ds_name}"
            )

    def get_grids(self, grids_to_use: Iterable[str]) -> Iterable[dict]:
        """
        Queries for grids on Solr and filters based on grids to use
        """
        fq = ["type_s:grid"]
        grids = [grid for grid in solr_utils.solr_query(fq)]
        if grids_to_use:
            grids = [grid for grid in grids if grid["grid_name_s"] in grids_to_use]
        if "hemi_pattern" in self.config:
            logger.info("Skipping job creation for TPOSE grid on sea ice data.")
            grids = [grid for grid in grids if "TPOSE" not in grid["grid_name_s"]]
        return grids

    def get_jobs(self) -> Iterable[Aggregation]:
        """
        Gets list of AggJob objects for each annual aggregation to be performed for each grid / field combination
        """
        ds_metadata = self.get_solr_ds_metadata()
        existing_agg_version = ds_metadata.get("aggregation_version_s")

        if existing_agg_version and existing_agg_version != str(self.a_version):
            logger.debug("Making jobs for all years")
            agg_jobs = self.make_jobs_all_years(ds_metadata)
        else:
            logger.debug("Determining jobs")
            agg_jobs = self.make_jobs()
        return agg_jobs

    def get_solr_ds_metadata(self) -> dict:
        """
        Gets type_s:dataset Solr document for a given dataset
        """
        fq = [f"dataset_s:{self.ds_name}", "type_s:dataset"]
        ds_meta = solr_utils.solr_query(fq)[0]
        if "start_date_dt" not in ds_meta:
            logger.error("No transformed granules to aggregate.")
            raise Exception("No transformed granules to aggregate.")
        return ds_meta

    def make_jobs_all_years(self, ds_metadata: dict) -> Iterable[Aggregation]:
        """
        Makes AggJob objects for all years for all grids for all fields
        """
        start_year = int(ds_metadata.get("start_date_dt")[:4])
        end_year = int(ds_metadata.get("end_date_dt")[:4])
        years = [str(year) for year in range(start_year, end_year + 1)]
        jobs = []
        for grid in self.grids:
            for field in self.fields:
                jobs.extend(
                    [Aggregation(self.config, grid, year, field) for year in years]
                )
        return jobs

    def make_jobs(self) -> Iterable[Aggregation]:
        """
        Generates list of AggJob objects that define the grid/field/year aggregations to be performed.
        Checks if aggregation exists for a given grid/field/year combo and if so if it needs to be reprocessed.
        """
        all_jobs = []
        for grid in self.grids:
            for field in self.fields:
                grid_name = grid.get("grid_name_s")

                # Get grid / field transformation documents from Solr
                fq = [
                    f"dataset_s:{self.ds_name}",
                    f"field_s:{field.name}",
                    "type_s:transformation",
                    f"grid_name_s:{grid_name}",
                    "success_b:true",
                ]
                transformation_docs = solr_utils.solr_query(fq)
                transformation_years = list(
                    set([t["date_s"][:4] for t in transformation_docs])
                )
                transformation_years.sort()
                years_to_aggregate = []
                for year in transformation_years:
                    # Check for successful aggregation doc for this combo of grid / field / year
                    fq = [
                        f"dataset_s:{self.ds_name}",
                        "type_s:aggregation",
                        "aggregation_success_b:true",
                        f"field_s:{field.name}",
                        f"grid_name_s:{grid_name}",
                        f"year_s:{year}",
                    ]
                    aggregation_docs = solr_utils.solr_query(fq)

                    # If aggregation was previously done compare transformation time with aggregation time
                    if aggregation_docs:
                        agg_time = aggregation_docs[0]["aggregation_time_dt"]
                        for t in transformation_docs:
                            if t["date_s"][:4] != year:
                                continue
                            if t["transformation_completed_dt"] > agg_time:
                                years_to_aggregate.append(year)
                                break
                    else:
                        years_to_aggregate.append(year)
                all_jobs.extend(
                    [
                        Aggregation(self.config, grid, year, field)
                        for year in years_to_aggregate
                    ]
                )
        return all_jobs
