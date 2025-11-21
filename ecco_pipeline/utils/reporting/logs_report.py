import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from pathlib import Path


def parse_pipeline_logs(latest_only=True):
    """
    Parse pipeline logs and return a DataFrame with one row per dataset per stage.

    Parameters
    ----------
    logs_root : Path
        Root directory containing timestamped pipeline log subdirectories.
    config_dir : Path
        Directory containing dataset configs (*.yaml).
    latest_only : bool
        If True, parse only the latest log. Otherwise parse all logs.

    Returns
    -------
    pd.DataFrame
    """
    # ---------------------------
    # Load datasets from configs
    # ---------------------------

    datasets = [p.stem for p in Path("conf/ds_configs").glob("*.yaml")]
    stages = ["Harvest", "Transform", "Aggregate"]

    # ---------------------------
    # Gather log files
    # ---------------------------
    log_dirs = sorted(Path("logs").glob("*"), reverse=True)
    if latest_only:
        log_dirs = [d for d in log_dirs if (d / "pipeline.log").exists()][:1]
    else:
        log_dirs = [d for d in log_dirs if (d / "pipeline.log").exists()]
    if not log_dirs:
        raise FileNotFoundError("No pipeline.log found in ecco_pipeline/logs")

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
                records.append(
                    {
                        "run_timestamp": run_timestamp,
                        "dataset": ds,
                        "stage": stage,
                        "files_processed": 0,
                        "duration_sec": 0.0,
                        "status": "Not Run",
                        "errors": None,
                    }
                )
        df_defaults = pd.DataFrame(records)

        # ---------------------------
        # Parse log and override defaults
        # ---------------------------
        log_records = []
        with open(log_file) as f:
            lines = f.readlines()

        dataset = None
        start_time = None
        files_processed = 0
        stage = None

        def parse_time(ts_str):
            return datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S,%f")

        for line in lines:
            # HARVESTING
            m_h_start = re.search(r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*Beginning harvesting (\S+)", line)
            if m_h_start:
                start_time = parse_time(m_h_start.group(1))
                dataset = m_h_start.group(2)
                files_processed = 0
                stage = "Harvest"
                continue

            if stage == "Harvest" and dataset is not None:
                if re.match(r"\[INFO\].*Downloading .* to .*", line) and "complete" not in line:
                    files_processed += 1

                m_h_end = re.search(
                    r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*{} harvesting complete.*".format(
                        re.escape(dataset)
                    ),
                    line,
                )
                if m_h_end and start_time is not None:
                    end_time = parse_time(m_h_end.group(1))
                    duration_sec = (end_time - start_time).total_seconds()
                    status = "Success" if "successfully harvested" in line else "Failed"
                    log_records.append(
                        {
                            "dataset": dataset,
                            "stage": stage,
                            "files_processed": files_processed,
                            "duration_sec": duration_sec,
                            "status": status,
                            "errors": None,
                        }
                    )
                    dataset = None
                    start_time = None
                    files_processed = 0
                    stage = None

            # TRANSFORMATION
            m_t_start = re.search(
                r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*Beginning transformations on (\S+)", line
            )
            if m_t_start:
                start_time = parse_time(m_t_start.group(1))
                dataset = m_t_start.group(2)
                files_processed = 0
                stage = "Transform"
                continue

            if stage == "Transform" and dataset is not None:
                m_files = re.search(r"(\d+) harvested granules with remaining transformations", line)
                if m_files:
                    files_processed = int(m_files.group(1))

                m_t_end = re.search(
                    r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*{} transformation complete.*".format(
                        re.escape(dataset)
                    ),
                    line,
                )
                if m_t_end and start_time is not None:
                    end_time = parse_time(m_t_end.group(1))
                    duration_sec = (end_time - start_time).total_seconds()
                    status = (
                        "Success"
                        if ("All transformations successful" in line or "No transformations performed" in line)
                        else "Failed"
                    )
                    log_records.append(
                        {
                            "dataset": dataset,
                            "stage": stage,
                            "files_processed": files_processed,
                            "duration_sec": duration_sec,
                            "status": status,
                            "errors": None,
                        }
                    )
                    dataset = None
                    start_time = None
                    files_processed = 0
                    stage = None

            # AGGREGATION
            m_a_start = re.search(r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*Beginning aggregation on (\S+)", line)
            if m_a_start:
                start_time = parse_time(m_a_start.group(1))
                dataset = m_a_start.group(2)
                files_processed = 0
                stage = "Aggregate"
                continue

            if stage == "Aggregate" and dataset is not None:
                m_jobs = re.search(r"Executing \((\d+)\) jobs", line)
                if m_jobs:
                    files_processed = int(m_jobs.group(1))
                elif "No new jobs to execute" in line:
                    files_processed = 0

                m_a_end = re.search(
                    r"\[INFO\]\s+([\d\-]+\s[\d:,]+)\s+\(pipeline\).*{} aggregation complete.*".format(
                        re.escape(dataset)
                    ),
                    line,
                )
                if m_a_end and start_time is not None:
                    end_time = parse_time(m_a_end.group(1))
                    duration_sec = (end_time - start_time).total_seconds()
                    status = "Success" if "All aggregations successful" in line else "Failed"
                    log_records.append(
                        {
                            "dataset": dataset,
                            "stage": stage,
                            "files_processed": files_processed,
                            "duration_sec": duration_sec,
                            "status": status,
                            "errors": None,
                        }
                    )
                    dataset = None
                    start_time = None
                    files_processed = 0
                    stage = None

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
    cols_order = ["run_timestamp", "dataset", "stage", "files_processed", "duration_sec", "status", "errors"]
    df = df[cols_order]

    stage_order = {"Harvest": 0, "Transform": 1, "Aggregate": 2}
    df["stage_order"] = df["stage"].map(stage_order)
    df = df.sort_values(["run_timestamp", "dataset", "stage_order"]).drop(columns="stage_order").reset_index(drop=True)

    return df


def generate_pipeline_report(df: pd.DataFrame):
    # ---------------------------
    # Status changes table (failed stages)
    # ---------------------------
    failed = df[df["status"] != "Success"]
    if not failed.empty:
        print("\n⚠ Some stages failed in the latest run:")
        print(failed[["dataset", "stage", "status", "files_processed"]].to_string(index=False))
    else:
        print("\nAll stages completed successfully in the latest run.")

    def plot_vertical_stacked_bars(df):
        """
        Plot each stage as its own horizontal bar, vertically stacked per dataset.
        """
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
            values = df[df["stage"] == stage].set_index("dataset").reindex(datasets)["files_processed"].fillna(0)
            ax.barh(offsets, values, height=stage_height, color=colors[stage], label=stage)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(datasets)
        ax.set_xlabel("Files processed")
        ax.set_ylabel("Dataset")
        ax.set_title("Files processed per stage per dataset (latest run)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    plot_vertical_stacked_bars(df)

    return df


df_pipeline = parse_pipeline_logs(latest_only=True)
report = generate_pipeline_report(df_pipeline)
print(report)
