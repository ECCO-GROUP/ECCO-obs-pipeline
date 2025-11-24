"""
Pipeline Output Directory Reporter

Analyzes the filesystem structure of pipeline outputs to generate reports
on harvested, transformed, and aggregated file counts per dataset and grid.
Supports weekly snapshot comparison for delta reporting.
"""
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import date
import logging
from typing import Optional, List

from conf.global_settings import OUTPUT_DIR


class PipelineOutputReporter:
    """
    Generates reports by analyzing pipeline output directory structure.

    Counts files across harvest/transform/aggregate stages and compares
    with previous snapshots to detect weekly changes.
    """

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        config_dir: Path = None,
        reports_dir: Path = None
    ):
        """
        Initialize the output reporter.

        Parameters
        ----------
        output_dir : Path
            Root directory containing pipeline outputs (harvested_granules,
            transformed_products, aggregated_products)
        config_dir : Path
            Directory containing dataset YAML configs. Defaults to conf/ds_configs
        reports_dir : Path
            Directory for saving report snapshots. Defaults to output_dir/reports
        """
        self.output_dir = Path(output_dir)
        self.config_dir = config_dir or Path("conf/ds_configs")
        self.reports_dir = reports_dir or (self.output_dir / "reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("pipeline.reporting.output")

    def get_dataset_list(self) -> List[str]:
        """Return list of dataset names from config YAML files."""
        try:
            dataset_files = self.config_dir.glob("*.yaml")
            datasets = [f.stem for f in dataset_files]
            self.logger.debug(f"Found {len(datasets)} datasets in config")
            return datasets
        except Exception as e:
            self.logger.error(f"Failed to read dataset configs: {e}")
            raise

    def report_file_counts(self) -> pd.DataFrame:
        """
        Walk the directory tree and count harvested, transformed, and aggregated files
        for each dataset and grid. Prepopulates rows for all stages.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ["dataset", "stage", "grid", "files_count"]
        """
        dataset_configs = self.get_dataset_list()

        # --- Step 1: Determine full set of grids across all datasets ---
        all_grids = set()
        for ds_name in dataset_configs:
            transformed_path = self.output_dir / ds_name / "transformed_products"
            if transformed_path.exists():
                try:
                    grids = [p.name for p in transformed_path.iterdir() if p.is_dir()]
                    all_grids.update(grids)
                except Exception as e:
                    self.logger.warning(f"Error reading grids for {ds_name}: {e}")
        all_grids = sorted(all_grids)
        self.logger.debug(f"Found {len(all_grids)} grids: {all_grids}")

        # --- Step 2: Prepopulate records for all datasets × stages × grids ---
        records = []
        for ds_name in dataset_configs:
            ds_path = self.output_dir / ds_name

            # Harvest stage
            harvested_path = ds_path / "harvested_granules"
            try:
                harvest_count = sum(1 for f in harvested_path.rglob("*.nc")) if harvested_path.exists() else 0
            except Exception as e:
                self.logger.warning(f"Error counting harvested files for {ds_name}: {e}")
                harvest_count = 0
            records.append({
                "dataset": ds_name,
                "stage": "Harvest",
                "grid": "N/A",
                "files_count": harvest_count
            })

            # Transform & Aggregate stages
            transformed_path = ds_path / "transformed_products"
            for stage in ["Transform", "Aggregate"]:
                for grid in all_grids:
                    grid_path = transformed_path / grid if transformed_path.exists() else None

                    try:
                        if stage == "Transform":
                            count = (
                                sum(1 for f in grid_path.rglob("transformed/*/*.nc"))
                                if grid_path and grid_path.exists()
                                else 0
                            )
                        else:  # Aggregate
                            count = (
                                sum(1 for f in grid_path.rglob("aggregated/*/netCDF/*.nc"))
                                if grid_path and grid_path.exists()
                                else 0
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Error counting {stage} files for {ds_name}/{grid}: {e}"
                        )
                        count = 0

                    records.append({
                        "dataset": ds_name,
                        "stage": stage,
                        "grid": grid,
                        "files_count": count
                    })

        df = pd.DataFrame(records)

        # Sort nicely
        stage_order = {"Harvest": 0, "Transform": 1, "Aggregate": 2}
        df["stage_order"] = df["stage"].map(stage_order)
        df = df.sort_values(["dataset", "stage_order", "grid"]).drop(
            columns="stage_order"
        ).reset_index(drop=True)

        self.logger.info(f"Generated report with {len(df)} rows across {len(dataset_configs)} datasets")
        return df

    def pivot_file_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot the file counts dataframe into a pipeline-style report.

        Args:
            df (pd.DataFrame): DataFrame with columns ["dataset", "stage", "grid", "files_count"]

        Returns:
            pd.DataFrame: Pivoted report where grids are columns for Transform & Aggregate.
        """
        df_harvest = df[df["stage"] == "Harvest"].copy()
        df_harvest = df_harvest.drop(columns="grid").rename(
            columns={"files_count": "harvested_files"}
        )

        df_tx_ag = df[df["stage"].isin(["Transform", "Aggregate"])].copy()
        # Pivot grids to columns
        df_tx_ag_pivot = df_tx_ag.pivot_table(
            index=["dataset", "stage"],
            columns="grid",
            values="files_count",
            fill_value=0
        ).reset_index()

        # Merge Harvest counts with Transform & Aggregate
        df_report = pd.merge(df_tx_ag_pivot, df_harvest, on="dataset", how="left")

        # Sort stages nicely
        stage_order = {"Transform": 0, "Aggregate": 1}
        df_report["stage_order"] = df_report["stage_x"].map(stage_order)
        df_report = (
            df_report.sort_values(["dataset", "stage_order"])
            .drop(columns=["stage_order", "stage_y"])
            .rename(columns={"stage_x": "stage"})
            .reset_index(drop=True)
        )

        return df_report

    def save_weekly_snapshot(
        self,
        df: pd.DataFrame,
        custom_filename: Optional[str] = None
    ) -> Path:
        """
        Save a weekly snapshot of the file counts for delta comparison.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from report_file_counts()
        custom_filename : str, optional
            Custom filename. Defaults to pipeline_report_YYYY-MM-DD.csv

        Returns
        -------
        Path
            Path to the saved snapshot file
        """
        try:
            snapshot_file = self.reports_dir / (
                custom_filename or f"pipeline_report_{date.today()}.csv"
            )
            df.to_csv(snapshot_file, index=False)
            self.logger.info(f"Saved snapshot to {snapshot_file}")
            return snapshot_file
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")
            raise

    def load_previous_snapshot(self) -> Optional[pd.DataFrame]:
        """
        Load the previous week's snapshot for comparison.

        Returns
        -------
        pd.DataFrame or None
            Previous snapshot, or None if less than 2 snapshots exist
        """
        try:
            files = sorted(self.reports_dir.glob("pipeline_report_*.csv"))
            if len(files) < 2:
                self.logger.info("No previous snapshot found for comparison")
                return None

            # Second to last = previous week
            prev_file = files[-2]
            self.logger.info(f"Loading previous snapshot from {prev_file}")
            return pd.read_csv(prev_file)
        except Exception as e:
            self.logger.error(f"Failed to load previous snapshot: {e}")
            return None

    def compute_weekly_delta(
        self,
        current_df: pd.DataFrame,
        previous_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compute week-over-week delta of file counts per dataset, stage, and grid.
        Handles Harvest separately to avoid spurious extra rows.

        Parameters
        ----------
        current_df : pd.DataFrame
            Output of report_file_counts() for this week
        previous_df : pd.DataFrame or None
            Output of report_file_counts() for previous week

        Returns
        -------
        pd.DataFrame
            Merged dataframe with columns:
            ["dataset", "stage", "grid", "files_count_current",
             "files_count_prev", "delta"]
        """
        if previous_df is None:
            self.logger.info("No previous snapshot - returning current counts with null deltas")
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

        # Sort nicely
        stage_order = {"Harvest": 0, "Transform": 1, "Aggregate": 2}
        merged["stage_order"] = merged["stage"].map(stage_order)
        merged = merged.sort_values(
            ["dataset", "stage_order", "grid"]
        ).drop(columns="stage_order").reset_index(drop=True)

        # Log summary
        total_delta = merged["delta"].sum()
        self.logger.info(
            f"Delta computed: {total_delta:+.0f} total files "
            f"({merged[merged['delta'] > 0]['delta'].sum():+.0f} added, "
            f"{merged[merged['delta'] < 0]['delta'].sum():+.0f} removed)"
        )

        return merged

    def generate_plots(
        self,
        delta_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ):
        """
        Generate delta visualization bar chart.

        Parameters
        ----------
        delta_df : pd.DataFrame
            Output from compute_weekly_delta()
        output_path : Path, optional
            Path to save the plot. If None, defaults to reports_dir/plots/delta_latest.png
        """
        try:
            import seaborn as sns

            # Default output path
            if output_path is None:
                plots_dir = self.reports_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                output_path = plots_dir / "delta_latest.png"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create plot
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.barplot(
                data=delta_df,
                x="dataset",
                y="delta",
                hue="stage",
                dodge=True,
                ax=ax
            )
            ax.set_xticklabels(
                delta_df["dataset"].unique(),
                rotation=45,
                ha="right"
            )
            ax.set_ylabel("Delta file count vs last week")
            ax.set_title("Weekly delta of pipeline files per dataset and stage")
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()

            # Save instead of show
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Saved delta plot to {output_path}")

        except ImportError:
            self.logger.error("seaborn not installed - cannot generate plots")
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {e}")

    def generate_report(self, save_snapshot: bool = True) -> pd.DataFrame:
        """
        Main entry point - generate current report and optionally save snapshot.

        Parameters
        ----------
        save_snapshot : bool
            Whether to save this report as a weekly snapshot

        Returns
        -------
        pd.DataFrame
            Current file counts across all datasets/stages/grids
        """
        self.logger.info("Generating filesystem-based report...")
        df = self.report_file_counts()

        if save_snapshot:
            self.save_weekly_snapshot(df)

        return df

    def compare_with_previous(self) -> pd.DataFrame:
        """
        Generate delta comparison with the previous snapshot.

        Returns
        -------
        pd.DataFrame
            Delta comparison with previous week
        """
        current_df = self.report_file_counts()
        previous_df = self.load_previous_snapshot()
        delta_df = self.compute_weekly_delta(current_df, previous_df)
        return delta_df


def main():
    """Command-line interface for manual report generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ECCO pipeline output directory report"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Pipeline output directory"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Reports directory (default: OUTPUT_DIR/reports)"
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Don't save weekly snapshot"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate delta plot"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s (%(name)s) - %(message)s'
    )

    # Create reporter and generate report
    reporter = PipelineOutputReporter(
        output_dir=args.output_dir,
        reports_dir=args.reports_dir
    )

    # Generate current report
    df_pipeline = reporter.generate_report(save_snapshot=not args.no_snapshot)
    pivot_df = reporter.pivot_file_counts(df_pipeline)

    print("\n=== Pipeline Output Report ===")
    print(pivot_df.to_string(index=False))

    # Compare with previous
    prev_snapshot = reporter.load_previous_snapshot()
    delta_df = reporter.compute_weekly_delta(df_pipeline, prev_snapshot)

    if prev_snapshot is not None:
        print("\n=== Delta from Previous Week ===")
        # Show only rows with changes
        changes = delta_df[delta_df["delta"] != 0]
        if not changes.empty:
            print(changes[["dataset", "stage", "grid", "delta"]].to_string(index=False))
        else:
            print("No changes detected")

        # Generate plot if requested
        if args.plot:
            reporter.generate_plots(delta_df)


if __name__ == "__main__":
    main()
