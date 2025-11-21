from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import date

from conf.global_settings import OUTPUT_DIR


def get_dataset_list():
    """Return list of dataset names from config YAML files."""
    config_path = Path("conf/ds_configs")
    dataset_files = config_path.glob("*.yaml")
    return [f.stem for f in dataset_files]


def report_file_counts():
    """
    Walks the directory tree and counts harvested, transformed, and aggregated files
    for each dataset and grid. Prepopulates rows for all stages.
    """
    dataset_configs = get_dataset_list()

    records = []

    # --- Step 1: Determine full set of grids across all datasets ---
    all_grids = set()
    for ds_name in dataset_configs:
        transformed_path = OUTPUT_DIR / ds_name / "transformed_products"
        if transformed_path.exists():
            grids = [p.name for p in transformed_path.iterdir() if p.is_dir()]
            all_grids.update(grids)
    all_grids = sorted(all_grids)

    # --- Step 2: Prepopulate records for all datasets × stages × grids ---
    records = []
    for ds_name in dataset_configs:
        ds_path = OUTPUT_DIR / ds_name

        # Harvest stage
        harvested_path = ds_path / "harvested_granules"
        harvest_count = sum(1 for f in harvested_path.rglob("*.nc")) if harvested_path.exists() else 0
        records.append({"dataset": ds_name, "stage": "Harvest", "grid": "N/A", "files_count": harvest_count})

        # Transform & Aggregate stages
        transformed_path = ds_path / "transformed_products"
        for stage in ["Transform", "Aggregate"]:
            for grid in all_grids:
                grid_path = transformed_path / grid if transformed_path.exists() else None
                if stage == "Transform":
                    count = (
                        sum(1 for f in grid_path.rglob("transformed/*/*.nc")) if grid_path and grid_path.exists() else 0
                    )
                else:
                    count = (
                        sum(1 for f in grid_path.rglob("aggregated/*/netCDF/*.nc"))
                        if grid_path and grid_path.exists()
                        else 0
                    )
                records.append({"dataset": ds_name, "stage": stage, "grid": grid, "files_count": count})

    df = pd.DataFrame(records)

    # Sort nicely
    stage_order = {"Harvest": 0, "Transform": 1, "Aggregate": 2}
    df["stage_order"] = df["stage"].map(stage_order)
    df = df.sort_values(["dataset", "stage_order", "grid"]).drop(columns="stage_order").reset_index(drop=True)

    return df


def pivot_file_counts(df):
    """
    Pivot the prepopulated file counts dataframe into a pipeline-style report.

    Args:
        df (pd.DataFrame): DataFrame with columns ["dataset", "stage", "grid", "files_count"]

    Returns:
        pd.DataFrame: Pivoted report where grids are columns for Transform & Aggregate.
    """
    df_harvest = df[df["stage"] == "Harvest"].copy()
    df_harvest = df_harvest.drop(columns="grid").rename(columns={"files_count": "harvested_files"})

    df_tx_ag = df[df["stage"].isin(["Transform", "Aggregate"])].copy()
    # Pivot grids to columns
    df_tx_ag_pivot = df_tx_ag.pivot_table(
        index=["dataset", "stage"], columns="grid", values="files_count", fill_value=0
    ).reset_index()

    # Merge Harvest counts with Transform & Aggregate
    df_report = pd.merge(df_tx_ag_pivot, df_harvest, on="dataset", how="left")

    # Optional: sort stages nicely
    stage_order = {"Transform": 0, "Aggregate": 1}
    df_report["stage_order"] = df_report["stage_x"].map(stage_order)
    df_report = (
        df_report.sort_values(["dataset", "stage_order"])
        .drop(columns=["stage_order", "stage_y"])
        .rename(columns={"stage_x": "stage"})
        .reset_index(drop=True)
    )

    return df_report
    
    
def save_weekly_snapshot(df: pd.DataFrame, folder="reports"):
    folder_path = Path(folder)
    folder_path.mkdir(exist_ok=True)
    snapshot_file = folder_path / f"pipeline_report_{date.today()}.csv"
    df.to_csv(snapshot_file, index=False)
    return snapshot_file


def load_previous_snapshot(folder="reports") -> pd.DataFrame:
    folder_path = Path(folder)
    files = sorted(folder_path.glob("pipeline_report_*.csv"))
    if len(files) < 2:
        return None  # no previous week to compare
    return pd.read_csv(files[-2])  # second to last = previous week

def compute_weekly_delta_clean(current_df, previous_df):
    """
    Compute week-over-week delta of file counts per dataset, stage, and grid.
    Handles Harvest separately to avoid spurious extra rows.
    
    Args:
        current_df (pd.DataFrame): output of report_file_counts() for this week
        previous_df (pd.DataFrame): output of report_file_counts() for previous week
    
    Returns:
        pd.DataFrame: merged dataframe with columns:
                      ["dataset", "stage", "grid", "files_count_current",
                       "files_count_prev", "delta"]
    """
    if previous_df is None:
        print("No previous snapshot to compare.")
        return current_df.assign(
            files_count_current=current_df["files_count"],
            files_count_prev=np.nan,
            delta=np.nan
        ).drop(columns="files_count")
    
    # --- Step 1: Normalize Harvest stage ---
    current_df = current_df.copy()
    previous_df = previous_df.copy()
    current_df.loc[current_df["stage"] == "Harvest", "grid"] = "N/A"
    previous_df.loc[previous_df["stage"] == "Harvest", "grid"] = "N/A"

    # --- Step 2: Merge on dataset, stage, grid ---
    merged = pd.merge(
        current_df,
        previous_df,
        on=["dataset", "stage", "grid"],
        how="outer",
        suffixes=("_current", "_prev")
    ).fillna(0)

    # --- Step 3: Compute delta ---
    merged["delta"] = merged["files_count_current"] - merged["files_count_prev"]

    # Optional: sort nicely
    stage_order = {"Harvest": 0, "Transform": 1, "Aggregate": 2}
    merged["stage_order"] = merged["stage"].map(stage_order)
    merged = merged.sort_values(["dataset", "stage_order", "grid"]).drop(columns="stage_order").reset_index(drop=True)

    return merged

def plot_weekly_delta(delta_df):
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(
        data=delta_df,
        x="dataset",
        y="delta",
        hue="stage",
        dodge=True,
        ax=ax
    )
    ax.set_xticklabels(delta_df["dataset"].unique(), rotation=45, ha="right")
    ax.set_ylabel("Delta file count vs last week")
    ax.set_title("Weekly delta of pipeline files per dataset and stage")
    plt.tight_layout()
    plt.show()

df_pipeline = report_file_counts()
pivot_df = pivot_file_counts(df_pipeline)

# Save this week's snapshot
save_weekly_snapshot(df_pipeline)

# Compare with previous week
prev_snapshot = load_previous_snapshot()
delta_df = compute_weekly_delta_clean(df_pipeline, prev_snapshot)

# Plot delta
plot_weekly_delta(delta_df)
