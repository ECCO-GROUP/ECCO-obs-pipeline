from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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


def plot_harvest_transform(pivot_df):
    datasets = pivot_df["dataset"].unique()
    grids = [c for c in pivot_df.columns if c not in ("dataset", "stage", "harvested_files")]
    n_datasets = len(datasets)
    n_grids = len(grids)

    x = np.arange(n_datasets)  # dataset positions
    width = 0.8 / (1 + n_grids)  # total width split between Harvest + grids

    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 0.6), 6))

    # Harvest bars
    harvest_vals = pivot_df[pivot_df["stage"] == "Transform"].copy()
    harvest_vals = pivot_df[["dataset", "harvested_files"]].drop_duplicates()
    ax.bar(x - width / 2, harvest_vals["harvested_files"], width, label="Harvest")

    # Transform bars per grid
    for i, grid in enumerate(grids):
        transform_vals = []
        for ds in datasets:
            row = pivot_df[(pivot_df["dataset"] == ds) & (pivot_df["stage"] == "Transform")]
            transform_vals.append(row[grid].values[0])
        ax.bar(x + (i + 0.5) * width, transform_vals, width, label=f"Transform {grid}")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Number of files")
    ax.set_title("Harvested and Transformed file counts per dataset")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_aggregate(pivot_df):
    datasets = pivot_df["dataset"].unique()
    grids = [c for c in pivot_df.columns if c not in ("dataset", "stage", "harvested_files")]
    n_datasets = len(datasets)
    n_grids = len(grids)

    x = np.arange(n_datasets)
    width = 0.8 / n_grids  # split width among grids

    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 0.6), 6))

    for i, grid in enumerate(grids):
        agg_vals = []
        for ds in datasets:
            row = pivot_df[(pivot_df["dataset"] == ds) & (pivot_df["stage"] == "Aggregate")]
            agg_vals.append(row[grid].values[0])
        ax.bar(x + i * width - 0.4, agg_vals, width, label=f"Aggregate {grid}")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Number of files")
    ax.set_title("Aggregated file counts per dataset and grid")
    ax.legend()
    plt.tight_layout()
    plt.show()


df_pipeline = report_file_counts()
pivot_df = pivot_file_counts(df_pipeline)
plot_harvest_transform(pivot_df)
plot_aggregate(pivot_df)
