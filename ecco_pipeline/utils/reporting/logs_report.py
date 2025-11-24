"""
Pipeline Log Reporter

Parses pipeline log files to extract metrics on harvest/transform/aggregate
operations per dataset. Tracks files processed, duration, status, and errors.
"""
import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Optional, List


class PipelineLogReporter:
    """
    Parses pipeline logs to generate performance and status reports.

    Extracts information from pipeline.log files including:
    - Files processed per stage
    - Duration of each operation
    - Success/failure status
    - Error messages
    """

    # Log parsing patterns (compiled for performance)
    PATTERNS = {
        "harvest_start": re.compile(
            r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*Beginning harvesting (\S+)"
        ),
        "harvest_download": re.compile(
            r"\[INFO\].*Downloading .* to .*"
        ),
        "harvest_end": re.compile(
            r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*({}) harvesting complete.*"
        ),
        "transform_start": re.compile(
            r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*Beginning transformations on (\S+)"
        ),
        "transform_files": re.compile(
            r"(\d+) harvested granules with remaining transformations"
        ),
        "transform_end": re.compile(
            r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*({}) transformation complete.*"
        ),
        "aggregate_start": re.compile(
            r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*Beginning aggregation on (\S+)"
        ),
        "aggregate_jobs": re.compile(
            r"Executing \((\d+)\) jobs"
        ),
        "aggregate_end": re.compile(
            r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*({}) aggregation complete.*"
        ),
    }

    # Timestamp format used in logs
    TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

    def __init__(self, logs_dir: Path = None, config_dir: Path = None):
        """
        Initialize the log reporter.

        Parameters
        ----------
        logs_dir : Path
            Directory containing timestamped log subdirectories.
            Defaults to ecco_pipeline/logs
        config_dir : Path
            Directory containing dataset YAML configs.
            Defaults to conf/ds_configs
        """
        self.logs_dir = logs_dir or Path("logs")
        self.config_dir = config_dir or Path("conf/ds_configs")
        self.logger = logging.getLogger("pipeline.reporting.logs")

    def get_dataset_list(self) -> List[str]:
        """Return list of dataset names from config YAML files."""
        try:
            datasets = [p.stem for p in self.config_dir.glob("*.yaml")]
            self.logger.debug(f"Found {len(datasets)} datasets in config")
            return datasets
        except Exception as e:
            self.logger.error(f"Failed to read dataset configs: {e}")
            raise

    def parse_pipeline_logs(self, latest_only: bool = True) -> pd.DataFrame:
        """
        Parse pipeline logs and return a DataFrame with one row per dataset per stage.

        Parameters
        ----------
        latest_only : bool
            If True, parse only the latest log. Otherwise parse all logs.

        Returns
        -------
        pd.DataFrame
            Columns: run_timestamp, dataset, stage, files_processed,
                     duration_sec, status, errors
        """
        datasets = self.get_dataset_list()
        stages = ["Harvest", "Transform", "Aggregate"]

        # ---------------------------
        # Gather log files
        # ---------------------------
        log_dirs = sorted(self.logs_dir.glob("*"), reverse=True)
        if latest_only:
            log_dirs = [d for d in log_dirs if (d / "pipeline.log").exists()][:1]
        else:
            log_dirs = [d for d in log_dirs if (d / "pipeline.log").exists()]

        if not log_dirs:
            raise FileNotFoundError(
                f"No pipeline.log found in {self.logs_dir}"
            )

        self.logger.info(f"Parsing {len(log_dirs)} log file(s)")
        all_records = []

        for log_dir in log_dirs:
            log_file = log_dir / "pipeline.log"
            run_timestamp = log_dir.name

            # ---------------------------
            # Create default records for this run
            # ---------------------------
            records = []
            for ds in datasets:
                for stage in stages:
                    records.append({
                        "run_timestamp": run_timestamp,
                        "dataset": ds,
                        "stage": stage,
                        "files_processed": 0,
                        "duration_sec": 0.0,
                        "status": "Not Run",
                        "errors": None,
                    })
            df_defaults = pd.DataFrame(records)

            # ---------------------------
            # Parse log and override defaults
            # ---------------------------
            log_records = self._parse_log_file(log_file)

            # Merge log values into defaults for this run
            df_logs = pd.DataFrame(log_records)
            if not df_logs.empty:
                df_logs["dataset"] = df_logs["dataset"].astype(str)
                df_logs["stage"] = df_logs["stage"].astype(str)
                df_defaults = df_defaults.set_index(["dataset", "stage"])
                df_logs = df_logs.set_index(["dataset", "stage"])
                df_defaults.update(df_logs)
            all_records.append(df_defaults.reset_index())

        # Concatenate all runs
        df = pd.concat(all_records, ignore_index=True)

        # Reorder columns and sort
        cols_order = [
            "run_timestamp", "dataset", "stage", "files_processed",
            "duration_sec", "status", "errors"
        ]
        df = df[cols_order]

        stage_order = {"Harvest": 0, "Transform": 1, "Aggregate": 2}
        df["stage_order"] = df["stage"].map(stage_order)
        df = df.sort_values(
            ["run_timestamp", "dataset", "stage_order"]
        ).drop(columns="stage_order").reset_index(drop=True)

        self.logger.info(f"Parsed {len(df)} stage records from logs")
        return df

    def _parse_log_file(self, log_file: Path) -> List[dict]:
        """
        Parse a single log file and extract stage records.

        Parameters
        ----------
        log_file : Path
            Path to pipeline.log file

        Returns
        -------
        List[dict]
            List of stage records with dataset, stage, files_processed,
            duration_sec, status
        """
        log_records = []

        try:
            with open(log_file) as f:
                lines = f.readlines()
        except Exception as e:
            self.logger.error(f"Failed to read {log_file}: {e}")
            return []

        dataset = None
        start_time = None
        files_processed = 0
        stage = None

        def parse_time(ts_str: str) -> datetime:
            """Parse timestamp from log line."""
            try:
                return datetime.strptime(ts_str.strip(), self.TIMESTAMP_FORMAT)
            except ValueError as e:
                self.logger.warning(f"Failed to parse timestamp '{ts_str}': {e}")
                return None

        for line in lines:
            try:
                # HARVESTING
                m_h_start = self.PATTERNS["harvest_start"].search(line)
                if m_h_start:
                    start_time = parse_time(m_h_start.group(1))
                    dataset = m_h_start.group(2)
                    files_processed = 0
                    stage = "Harvest"
                    continue

                if stage == "Harvest" and dataset is not None:
                    # Count download operations
                    if self.PATTERNS["harvest_download"].match(line) and "complete" not in line:
                        files_processed += 1

                    # Check for harvest completion
                    pattern = self.PATTERNS["harvest_end"].pattern.format(
                        re.escape(dataset)
                    )
                    m_h_end = re.search(pattern, line)
                    if m_h_end and start_time is not None:
                        end_time = parse_time(m_h_end.group(1))
                        if end_time:
                            duration_sec = (end_time - start_time).total_seconds()
                            status = "Success" if "successfully harvested" in line else "Failed"
                            log_records.append({
                                "dataset": dataset,
                                "stage": stage,
                                "files_processed": files_processed,
                                "duration_sec": duration_sec,
                                "status": status,
                                "errors": None,
                            })
                        dataset = None
                        start_time = None
                        files_processed = 0
                        stage = None

                # TRANSFORMATION
                m_t_start = self.PATTERNS["transform_start"].search(line)
                if m_t_start:
                    start_time = parse_time(m_t_start.group(1))
                    dataset = m_t_start.group(2)
                    files_processed = 0
                    stage = "Transform"
                    continue

                if stage == "Transform" and dataset is not None:
                    # Count files to transform
                    m_files = self.PATTERNS["transform_files"].search(line)
                    if m_files:
                        files_processed = int(m_files.group(1))

                    # Check for transform completion
                    pattern = self.PATTERNS["transform_end"].pattern.format(
                        re.escape(dataset)
                    )
                    m_t_end = re.search(pattern, line)
                    if m_t_end and start_time is not None:
                        end_time = parse_time(m_t_end.group(1))
                        if end_time:
                            duration_sec = (end_time - start_time).total_seconds()
                            status = (
                                "Success"
                                if ("All transformations successful" in line or
                                    "No transformations performed" in line)
                                else "Failed"
                            )
                            log_records.append({
                                "dataset": dataset,
                                "stage": stage,
                                "files_processed": files_processed,
                                "duration_sec": duration_sec,
                                "status": status,
                                "errors": None,
                            })
                        dataset = None
                        start_time = None
                        files_processed = 0
                        stage = None

                # AGGREGATION
                m_a_start = self.PATTERNS["aggregate_start"].search(line)
                if m_a_start:
                    start_time = parse_time(m_a_start.group(1))
                    dataset = m_a_start.group(2)
                    files_processed = 0
                    stage = "Aggregate"
                    continue

                if stage == "Aggregate" and dataset is not None:
                    # Count aggregation jobs
                    m_jobs = self.PATTERNS["aggregate_jobs"].search(line)
                    if m_jobs:
                        files_processed = int(m_jobs.group(1))
                    elif "No new jobs to execute" in line:
                        files_processed = 0

                    # Check for aggregate completion
                    pattern = self.PATTERNS["aggregate_end"].pattern.format(
                        re.escape(dataset)
                    )
                    m_a_end = re.search(pattern, line)
                    if m_a_end and start_time is not None:
                        end_time = parse_time(m_a_end.group(1))
                        if end_time:
                            duration_sec = (end_time - start_time).total_seconds()
                            status = "Success" if "All aggregations successful" in line else "Failed"
                            log_records.append({
                                "dataset": dataset,
                                "stage": stage,
                                "files_processed": files_processed,
                                "duration_sec": duration_sec,
                                "status": status,
                                "errors": None,
                            })
                        dataset = None
                        start_time = None
                        files_processed = 0
                        stage = None

            except Exception as e:
                self.logger.warning(f"Error parsing line: {e}")
                continue

        return log_records

    def get_failed_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to only failed stages.

        Parameters
        ----------
        df : pd.DataFrame
            Output from parse_pipeline_logs()

        Returns
        -------
        pd.DataFrame
            Only rows where status is "Failed"
        """
        failed = df[df["status"] == "Failed"]
        self.logger.info(f"Found {len(failed)} failed stages")
        return failed

    def generate_plots(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None
    ):
        """
        Generate stacked bar chart visualization of files processed per stage.

        Parameters
        ----------
        df : pd.DataFrame
            Output from parse_pipeline_logs()
        output_path : Path, optional
            Path to save the plot. If None, returns without saving.
        """
        try:
            stages = ["Harvest", "Transform", "Aggregate"]
            colors = {"Harvest": "skyblue", "Transform": "orange", "Aggregate": "green"}

            datasets = sorted(df["dataset"].unique())
            n_datasets = len(datasets)
            stage_height = 0.8 / len(stages)  # fractional bar height

            # Map dataset to y positions
            y_pos = list(range(n_datasets))

            fig, ax = plt.subplots(figsize=(12, max(6, n_datasets / 2)))

            for i, stage in enumerate(stages):
                # Compute vertical offset for this stage within dataset
                offsets = [y - 0.4 + i * stage_height for y in y_pos]
                values = df[df["stage"] == stage].set_index("dataset").reindex(
                    datasets
                )["files_processed"].fillna(0)
                ax.barh(offsets, values, height=stage_height, color=colors[stage], label=stage)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(datasets)
            ax.set_xlabel("Files processed")
            ax.set_ylabel("Dataset")
            ax.set_title("Files processed per stage per dataset (latest run)")
            ax.legend()
            plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved plot to {output_path}")
                plt.close(fig)
            else:
                plt.show()

        except Exception as e:
            self.logger.error(f"Failed to generate plot: {e}")

    def generate_report(self, latest_only: bool = True) -> pd.DataFrame:
        """
        Main entry point - parse logs and return report DataFrame.

        Parameters
        ----------
        latest_only : bool
            If True, parse only the latest log. Otherwise parse all logs.

        Returns
        -------
        pd.DataFrame
            Log analysis with run_timestamp, dataset, stage, files_processed,
            duration_sec, status, errors
        """
        self.logger.info(
            f"Generating log-based report (latest_only={latest_only})..."
        )
        df = self.parse_pipeline_logs(latest_only=latest_only)
        return df


def main():
    """Command-line interface for manual report generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ECCO pipeline log report"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing log subdirectories"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Parse all logs instead of just latest"
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Generate plot and save to specified path"
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Show only failed stages"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s (%(name)s) - %(message)s'
    )

    # Create reporter and generate report
    reporter = PipelineLogReporter(logs_dir=args.logs_dir)

    try:
        df_pipeline = reporter.generate_report(latest_only=not args.all)

        if args.failures_only:
            df_display = reporter.get_failed_stages(df_pipeline)
            if df_display.empty:
                print("\nAll stages completed successfully in the latest run.")
            else:
                print(f"\n{len(df_display)} failed stage(s) in the latest run:")
                print(df_display[["dataset", "stage", "status", "files_processed"]].to_string(index=False))
        else:
            print("\n=== Pipeline Log Report ===")
            print(df_pipeline.to_string(index=False))

            # Show failures separately
            failed = reporter.get_failed_stages(df_pipeline)
            if not failed.empty:
                print(f"\n{len(failed)} failed stage(s) detected:")
                print(failed[["dataset", "stage", "status"]].to_string(index=False))

        # Generate plot if requested
        if args.plot:
            reporter.generate_plots(df_pipeline, output_path=args.plot)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
