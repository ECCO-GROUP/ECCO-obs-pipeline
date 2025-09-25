import argparse
import importlib
import logging
import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable

import yaml

from aggregations.aggregation_factory import AgJobFactory
from transformations.transformation_factory import TxJobFactory
from utils.pipeline_utils import init_pipeline


CONFIG_DIR = Path("conf/ds_configs")


def list_datasets(config_dir: Path = CONFIG_DIR) -> list[str]:
    """Return sorted list of dataset names (stems of YAML files)."""
    return sorted(p.stem for p in config_dir.glob("*.yaml"))


def load_config(dataset: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load a single dataset YAML config."""
    with (config_dir / f"{dataset}.yaml").open() as f:
        return yaml.safe_load(f)


def create_parser() -> argparse.Namespace:
    """Create and parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--menu", action="store_true", help="Show interactive menu")
    parser.add_argument("--grids_to_solr", action="store_true", help="Update Solr with grids in grids_config")
    parser.add_argument(
        "--multiprocesses",
        type=int,
        choices=range(1, cpu_count() + 1),
        default=max(1, cpu_count() // 2),
        metavar=f"[1,{cpu_count()}]",
        help="Number of processes for transformations",
    )
    parser.add_argument(
        "--harvested_entry_validation",
        action="store_true",
        help="Verify each Solr granule entry points to a valid file",
    )
    parser.add_argument(
        "--wipe_transformations", action="store_true", help="Delete transformations with out-of-sync version numbers"
    )
    parser.add_argument("--grids_to_use", nargs="*", default=False, help="Names of grids to use during the pipeline")
    parser.add_argument("--log_level", default="INFO", help="Set the log level")
    parser.add_argument("--wipe_factors", action="store_true", help="Remove all stored factors")
    parser.add_argument("--wipe_logs", action="store_true", help="Remove all prior log files")

    # dataset/step options only needed when not in menu mode
    args, _ = parser.parse_known_args()
    if not args.menu:
        parser.add_argument("--dataset", help="Dataset to process")
        parser.add_argument(
            "--step", choices=["harvest", "transform", "aggregate", "all"], default="all", help="Pipeline step to run"
        )
        args = parser.parse_args()

        if args.dataset and args.dataset not in list_datasets():
            raise ValueError(f"{args.dataset!r} is not a valid dataset name.")
    return args


def execute_step(name: str, datasets: list[str], action: Callable[[dict[str, Any]], Any]) -> None:
    """Run a named step (harvest/transform/aggregate) across datasets."""
    for ds in datasets:
        try:
            logger.info("Starting %s for %s", name, ds)
            cfg = load_config(ds)
            result = action(cfg)
            logger.info("%s %s complete. %s", ds, name, result)
        except Exception:
            logger.exception("%s %s failed", ds, name)


def run_harvester(datasets: list[str]) -> None:
    execute_step(
        "harvesting",
        datasets,
        lambda c: importlib.import_module(f"harvesters.{c['harvester_type']}_harvester").harvester(c),
    )


def run_transformation(datasets: list[str]) -> None:
    execute_step("transformation", datasets, lambda c: TxJobFactory(c).start_factory())


def run_aggregation(datasets: list[str]) -> None:
    execute_step("aggregation", datasets, lambda c: AgJobFactory(c).start_factory())


def start_pipeline(args: argparse.Namespace) -> None:
    """Run pipeline in non-interactive mode."""
    datasets = [args.dataset] if args.dataset else list_datasets()

    if args.step == "harvest":
        run_harvester(datasets)
    elif args.step == "transform":
        run_transformation(datasets)
    elif args.step == "aggregate":
        run_aggregation(datasets)
    else:
        run_harvester(datasets)
        run_transformation(datasets)
        run_aggregation(datasets)


def choose_from_menu(prompt: str, options: list[str]) -> str:
    """Utility to prompt user until a valid numbered choice is made."""
    while True:
        print(f"\n{prompt}\n")
        for i, opt in enumerate(options, 1):
            print(f"{i}) {opt}")
        sel = input("\nEnter number: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return options[int(sel) - 1]
        print(f'Invalid selection "{sel}", please try again.')


def show_menu() -> None:
    """Interactive menu mode."""
    print("\n===== ECCO PREPROCESSING PIPELINE =====")
    main_choice = choose_from_menu(
        "Main options",
        ["Run all", "Harvesters only", "Through transformation", "Dataset input"],
    )

    datasets = list_datasets()

    if main_choice == "Run all":
        run_harvester(datasets)
        run_transformation(datasets)
        run_aggregation(datasets)
    elif main_choice == "Harvesters only":
        run_harvester(datasets)
    elif main_choice == "Through transformation":
        run_harvester(datasets)
        run_transformation(datasets)
    else:
        dataset = choose_from_menu("Available datasets", datasets)
        step_choice = choose_from_menu(
            "Pipeline steps",
            ["harvest", "transform", "aggregate", "harvest, transform, aggregate"],
        )
        if "harvest" in step_choice:
            run_harvester([dataset])
        if "transform" in step_choice:
            run_transformation([dataset])
        if "aggregate" in step_choice:
            run_aggregation([dataset])


def main(args: argparse.Namespace) -> None:
    if args.menu:
        show_menu()
    else:
        start_pipeline(args)


if __name__ == "__main__":
    args = create_parser()

    # Perform pipeline-wide initialization
    init_pipeline.init_pipeline(args)
    logger = logging.getLogger("pipeline")
    try:
        main(args)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
