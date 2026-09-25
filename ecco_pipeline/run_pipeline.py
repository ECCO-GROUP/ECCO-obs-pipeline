import argparse
import importlib
import logging
import os
import sys
import time
import traceback
from glob import glob
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import yaml
from aggregations.aggregation_factory import AgJobFactory
from monitoring import run_summary
from monitoring.run_summary import StageResult
from transformations.transformation_factory import TxJobFactory
from utils.pipeline_utils import init_pipeline

logger = logging.getLogger("pipeline")


def create_parser() -> argparse.Namespace:
    """
    Creates command line argument parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--menu", default=False, action="store_true", help="Show interactive menu"
    )
    parser.add_argument(
        "--grids_to_solr",
        default=False,
        action="store_true",
        help="updates Solr with grids in grids_config",
    )
    parser.add_argument(
        "--multiprocesses",
        type=int,
        choices=range(1, cpu_count() + 1),
        default=int(cpu_count() / 2),
        metavar=f"[1, {cpu_count()}]",
        help=f"sets the number of multiprocesses used during transformation with a system max of {cpu_count()} \
                            with default set to half of system max (aggregation has a stricter cap due to high I/O volume)",
    )
    parser.add_argument(
        "--harvested_entry_validation",
        default=False,
        action="store_true",
        help="verifies each Solr granule entry points to a valid file.",
    )
    parser.add_argument(
        "--wipe_transformations",
        default=False,
        action="store_true",
        help="deletes transformations with version number different than what is currently in transformation_config",
    )
    parser.add_argument(
        "--grids_to_use",
        default=False,
        nargs="*",
        help="Names of grids to use during the pipeline",
    )
    parser.add_argument("--log_level", default="INFO", help="sets the log level")
    parser.add_argument(
        "--wipe_factors",
        default=False,
        action="store_true",
        help="removes all stored factors",
    )
    parser.add_argument(
        "--wipe_logs",
        default=False,
        action="store_true",
        help="removes all prior log files",
    )

    args, _ = parser.parse_known_args()

    if not args.menu:
        parser.add_argument("--dataset", help="Specify the dataset option", type=str)
        parser.add_argument(
            "--step",
            help="Specify the step option",
            type=str,
            default="all",
            choices=["harvest", "transform", "aggregate", "all"],
        )
        args = parser.parse_args()

        if args.dataset:
            # Check if dataset is valid option
            valid_datasets = [
                os.path.splitext(os.path.basename(f))[0]
                for f in glob("conf/ds_configs/*.yaml")
            ]
            if args.dataset not in valid_datasets:
                raise ValueError(
                    f"{args.dataset} is not valid dataset. Check spelling."
                )
    return args


def show_menu() -> List[StageResult]:
    results = []
    while True:
        print("\n===== ECCO PREPROCESSING PIPELINE =====")
        print("\n------------- OPTIONS -------------")
        print("1) Run all")
        print("2) Harvesters only")
        print("3) Up to aggregation")
        print("4) Dataset input")
        try:
            chosen_option = input("Enter option number: ")
        except KeyboardInterrupt:
            exit()

        if chosen_option in ["1", "2", "3", "4"]:
            break
        else:
            print(
                f'Unknown option entered, "{chosen_option}", please enter a valid option\n'
            )

    # Load all dataset configuration YAML names
    datasets = [
        os.path.splitext(os.path.basename(f))[0] for f in glob("conf/ds_configs/*.yaml")
    ]
    datasets.sort()

    # Run all
    if chosen_option == "1":
        for ds in datasets:
            results += run_harvester([ds])
            results += run_transformation([ds])
            results += run_aggregation([ds])

    # Run harvester
    elif chosen_option == "2":
        results += run_harvester(datasets)

    # Run up through transformation
    elif chosen_option == "3":
        for ds in datasets:
            results += run_harvester([ds])
            results += run_transformation([ds])

    # Manually enter dataset and pipeline step(s)
    elif chosen_option == "4":
        ds_dict = {i: ds for i, ds in enumerate(datasets, start=1)}
        while True:
            print("\nAvailable datasets:\n")
            for i, dataset in ds_dict.items():
                print(f"{i}) {dataset}")
            try:
                ds_index = input("\nEnter dataset number: ")
            except KeyboardInterrupt:
                exit()

            if not ds_index.isdigit() or int(ds_index) not in range(
                1, len(datasets) + 1
            ):
                print(f'Invalid dataset, "{ds_index}", please enter a valid selection')
            else:
                break

        wanted_ds = ds_dict[int(ds_index)]
        print(f"\nUsing {wanted_ds} dataset")

        steps = [
            "harvest",
            "transform",
            "aggregate",
            "harvest and transform",
            "transform and aggregate",
            "all",
        ]
        steps_dict = {i: step for i, step in enumerate(steps, start=1)}
        while True:
            print("\nAvailable steps:\n")
            for i, step in steps_dict.items():
                print(f"{i}) {step}")
            steps_index = input("\nEnter pipeline step(s) number: ")

            if not steps_index.isdigit() or int(steps_index) not in range(
                1, len(steps) + 1
            ):
                print(
                    f'Invalid step(s), "{steps_index}", please enter a valid selection'
                )
            else:
                break

        wanted_steps = steps_dict[int(steps_index)]

        if "harvest" in wanted_steps:
            results += run_harvester([wanted_ds])
        if "transform" in wanted_steps:
            results += run_transformation([wanted_ds])
        if "aggregate" in wanted_steps:
            results += run_aggregation([wanted_ds])
        if wanted_steps == "all":
            results += run_harvester([wanted_ds])
            results += run_transformation([wanted_ds])
            results += run_aggregation([wanted_ds])

    return results


def _failed_result(ds: str, stage: str, started: float, exc: Exception) -> StageResult:
    """
    Capture a stage exception for the run summary.

    The traceback is kept because it is what makes a failure triageable later — it
    distinguishes "the source returned 503" from "our parser broke on a new file
    layout". Until now it existed only in pipeline.log.
    """
    return StageResult(
        dataset=ds,
        stage=stage,
        duration_s=time.monotonic() - started,
        exc_type=type(exc).__name__,
        exc_message=str(exc) or repr(exc),
        traceback_tail=traceback.format_exc(),
    )


def run_harvester(datasets: List[str]) -> List[StageResult]:
    results = []
    for ds in datasets:
        started = time.monotonic()
        try:
            logger.info(f"Beginning harvesting {ds}")
            with open(Path(f"conf/ds_configs/{ds}.yaml"), "r") as stream:
                config = yaml.load(stream, yaml.Loader)
            harvester_type = config.get("harvester_type")
            if not harvester_type:
                raise ValueError(f"Harvester type missing from {ds} config. Exiting.")

            harvester = importlib.import_module(
                f"harvesters.{harvester_type}_harvester"
            )
            status = harvester.harvester(config)
            logger.info(f"{ds} harvesting complete. {status}")
            results.append(
                StageResult(
                    dataset=ds,
                    stage="harvest",
                    status_text=status,
                    duration_s=time.monotonic() - started,
                )
            )
        except Exception as e:
            logger.exception(f"{ds} harvesting failed. {e}")
            results.append(_failed_result(ds, "harvest", started, e))
    return results


def run_transformation(datasets: List[str]) -> List[StageResult]:
    results = []
    for ds in datasets:
        started = time.monotonic()
        try:
            logger.info(f"Beginning transformations on {ds}")
            with open(Path(f"conf/ds_configs/{ds}.yaml"), "r") as stream:
                config = yaml.load(stream, yaml.Loader)
            status = TxJobFactory(config).start_factory()
            logger.info(f"{ds} transformation complete. {status}")
            results.append(
                StageResult(
                    dataset=ds,
                    stage="transform",
                    status_text=status,
                    duration_s=time.monotonic() - started,
                )
            )
        except Exception as e:
            logger.exception(f"{ds} transformation failed.")
            results.append(_failed_result(ds, "transform", started, e))
    return results


def run_aggregation(datasets: List[str]) -> List[StageResult]:
    results = []
    for ds in datasets:
        started = time.monotonic()
        try:
            logger.info(f"Beginning aggregation on {ds}")
            with open(Path(f"conf/ds_configs/{ds}.yaml"), "r") as stream:
                config = yaml.load(stream, yaml.Loader)
            # Keep the factory so its collected worker-failure markers can be read
            # back; a worker can die before writing its Solr doc, in which case those
            # markers are the only record of the failure.
            factory = AgJobFactory(config)
            status = factory.start_factory()
            logger.info(f"{ds} aggregation complete. {status}")
            results.append(
                StageResult(
                    dataset=ds,
                    stage="aggregate",
                    status_text=status,
                    duration_s=time.monotonic() - started,
                    worker_errors=list(factory.worker_errors),
                )
            )
        except Exception as e:
            logger.exception(f"{ds} aggregation failed: {e}")
            results.append(_failed_result(ds, "aggregate", started, e))
    return results


def start_pipeline(args: argparse.Namespace) -> List[StageResult]:
    if args.dataset:
        datasets = [args.dataset]
    else:
        # Load all dataset configuration YAML names
        datasets = sorted(
            [
                os.path.splitext(os.path.basename(f))[0]
                for f in glob("conf/ds_configs/*.yaml")
            ]
        )

    results = []
    if args.step == "harvest":
        results += run_harvester(datasets)
    elif args.step == "transform":
        results += run_transformation(datasets)
    elif args.step == "aggregate":
        results += run_aggregation(datasets)
    else:
        results += run_harvester(datasets)
        results += run_transformation(datasets)
        results += run_aggregation(datasets)
    return results


def main() -> int:
    """
    Run the pipeline and return the process exit code.

    Every exit path records a run summary, so a run always leaves evidence of what it
    did — including an abort, which previously exited 0 with no trace at all and so
    looked identical to a clean run from cron's point of view.
    """
    args = create_parser()
    started_dt = run_summary.utc_now()

    try:
        init_pipeline.init_pipeline(args)
    except init_pipeline.PipelinePreconditionError as e:
        # The log dir may not exist yet and Solr may be the thing that is down, so the
        # JSON artifact is the only record this run will leave.
        summary = run_summary.record_run(
            [], args=args, started_dt=started_dt, outcome="aborted", abort_reason=str(e)
        )
        return summary["exit_code"]

    if args.menu:
        results = show_menu()
        outcome = None
    else:
        try:
            results = start_pipeline(args)
            outcome = None
        except Exception as e:
            logger.exception(f"Pipeline run failed: {e}")
            results = []
            outcome = "crashed"

    summary = run_summary.record_run(
        results or [],
        args=args,
        started_dt=started_dt,
        outcome=outcome,
        abort_reason=None,
    )
    totals = summary["totals"]
    logger.info(
        f"Run {summary['run_id']} finished: {summary['outcome']} "
        f"(exit {summary['exit_code']}); "
        f"{totals['datasets_with_failures']} of {totals['datasets']} datasets with failures"
    )
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
