"""
Pipeline Report Manager

Orchestrates both log-based and filesystem-based reporting.
Generates comprehensive weekly delta reports for cron job integration.
"""
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import pandas as pd
import logging

from conf.global_settings import OUTPUT_DIR
from utils.reporting.output_report import PipelineOutputReporter
from utils.reporting.logs_report import PipelineLogReporter


class PipelineReportManager:
    """
    Orchestrates both log-based and filesystem-based reporting.
    Generates weekly delta reports for cron job integration.
    """

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        logs_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
        include_log_report: bool = False
    ):
        """
        Initialize the report manager.

        Parameters
        ----------
        output_dir : Path
            Root directory containing pipeline outputs
        logs_dir : Path, optional
            Directory containing log subdirectories. Defaults to logs/
        reports_dir : Path, optional
            Directory for saving reports. Defaults to output_dir/reports
        include_log_report : bool
            Whether to include log-based reporting (default: False)
        """
        self.output_dir = Path(output_dir)
        self.logs_dir = logs_dir or Path("logs")
        self.reports_dir = reports_dir or (self.output_dir / "reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.include_log_report = include_log_report
        self.logger = logging.getLogger("pipeline.reporting")

        # Initialize output reporter (always enabled)
        self.output_reporter = PipelineOutputReporter(
            output_dir=self.output_dir,
            reports_dir=self.reports_dir
        )

        # Initialize log reporter (optional)
        self.log_reporter = None
        if self.include_log_report:
            self.log_reporter = PipelineLogReporter(
                logs_dir=self.logs_dir,
                config_dir=Path("conf/ds_configs")
            )

    def generate_weekly_report(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Generate comprehensive weekly report (filesystem analysis is primary).

        This is the main entry point for cron-based reporting. It:
        1. Counts files across harvest/transform/aggregate stages
        2. Compares with previous snapshot to compute deltas
        3. Optionally parses logs to extract performance metrics and failures
        4. Saves all reports to CSV files
        5. Generates visualization plots

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            (output_delta_df, log_summary_df or None)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting weekly report generation")
            self.logger.info("=" * 60)

            # Generate filesystem-based report (primary)
            self.logger.info("Generating filesystem-based report...")
            output_df = self.output_reporter.generate_report(save_snapshot=True)
            delta_df = self.output_reporter.compare_with_previous()

            # Generate log-based report (optional)
            log_df = None
            failed_stages = pd.DataFrame()
            if self.include_log_report and self.log_reporter:
                self.logger.info("Generating log-based report...")
                try:
                    log_df = self.log_reporter.generate_report(latest_only=True)
                    failed_stages = self.log_reporter.get_failed_stages(log_df)
                except Exception as e:
                    self.logger.warning(f"Log report generation failed: {e}")
                    log_df = None

            # Save combined report
            self._save_combined_report(
                output_df, delta_df, log_df, failed_stages, timestamp
            )

            # Generate plots
            self._generate_plots(delta_df, log_df, timestamp)

            # Log summary
            self._log_summary(delta_df, failed_stages)

            self.logger.info("=" * 60)
            self.logger.info(f"Weekly report complete. Files saved to {self.reports_dir}")
            self.logger.info("=" * 60)

            return delta_df, log_df

        except Exception as e:
            self.logger.exception(f"Failed to generate weekly report: {e}")
            raise

    def _save_combined_report(
        self,
        output_df: pd.DataFrame,
        delta_df: pd.DataFrame,
        log_df: Optional[pd.DataFrame],
        failed_stages: pd.DataFrame,
        timestamp: str
    ):
        """
        Save all report components to CSV files.

        Parameters
        ----------
        output_df : pd.DataFrame
            Current filesystem file counts (used for snapshot only)
        delta_df : pd.DataFrame
            Week-over-week delta (main output)
        log_df : pd.DataFrame or None
            Log analysis results (optional)
        failed_stages : pd.DataFrame
            Failed stages only
        timestamp : str
            Timestamp for filename
        """
        try:
            report_prefix = self.reports_dir / f"weekly_report_{timestamp}"

            # Save delta report (main output - contains current, previous, and delta)
            delta_df.to_csv(f"{report_prefix}_delta.csv", index=False)
            self.logger.info(f"Saved delta report: {report_prefix}_delta.csv")

            # Save log report components (optional)
            if log_df is not None:
                log_df.to_csv(f"{report_prefix}_logs.csv", index=False)
                self.logger.info(f"Saved log report: {report_prefix}_logs.csv")

                # Only save failures report if there are failures
                if not failed_stages.empty:
                    failed_stages.to_csv(f"{report_prefix}_FAILURES.csv", index=False)
                    self.logger.warning(
                        f"{len(failed_stages)} failed stages detected. "
                        f"See {report_prefix}_FAILURES.csv"
                    )

        except Exception as e:
            self.logger.error(f"Failed to save report CSVs: {e}")
            raise

    def _generate_plots(
        self,
        delta_df: pd.DataFrame,
        log_df: Optional[pd.DataFrame],
        timestamp: str
    ):
        """
        Generate and save visualization plots.

        Parameters
        ----------
        delta_df : pd.DataFrame
            Delta comparison data
        log_df : pd.DataFrame or None
            Log analysis data (optional)
        timestamp : str
            Timestamp for filename
        """
        try:
            plots_dir = self.reports_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Delta plot (always generated)
            delta_plot_path = plots_dir / f"delta_{timestamp}.png"
            self.output_reporter.generate_plots(delta_df, delta_plot_path)

            # Log summary plot (optional)
            if log_df is not None and self.log_reporter:
                log_plot_path = plots_dir / f"log_summary_{timestamp}.png"
                self.log_reporter.generate_plots(log_df, log_plot_path)

            self.logger.info(f"Plots saved to {plots_dir}")

        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")
            # Don't raise - plots are nice-to-have, not critical

    def _log_summary(self, delta_df: pd.DataFrame, failed_stages: pd.DataFrame):
        """
        Log a summary of the report to the console.

        Parameters
        ----------
        delta_df : pd.DataFrame
            Delta comparison data
        failed_stages : pd.DataFrame
            Failed stages (from log report, optional)
        """
        # Compute summary statistics
        total_delta = delta_df["delta"].sum()
        files_added = delta_df[delta_df["delta"] > 0]["delta"].sum()
        files_removed = delta_df[delta_df["delta"] < 0]["delta"].sum()

        # Log summary
        self.logger.info("")
        self.logger.info("REPORT SUMMARY")
        self.logger.info("-" * 60)
        self.logger.info(f"Total file delta: {total_delta:+.0f}")
        self.logger.info(f"  Files added:    {files_added:+.0f}")
        self.logger.info(f"  Files removed:  {files_removed:+.0f}")

        if not failed_stages.empty:
            self.logger.warning(f"  Failed stages: {len(failed_stages)}")
            for _, row in failed_stages.iterrows():
                self.logger.warning(
                    f"    - {row['dataset']}/{row['stage']}: {row['status']}"
                )
        elif self.include_log_report:
            self.logger.info("  All stages successful")

        self.logger.info("-" * 60)

    def cleanup_old_reports(self, keep_weeks: int = 12):
        """
        Remove old report files to prevent disk space issues.

        Parameters
        ----------
        keep_weeks : int
            Number of weeks of reports to keep (default: 12)
        """
        try:
            self.logger.info(f"Cleaning up reports older than {keep_weeks} weeks")

            # Find all weekly report files
            report_files = sorted(self.reports_dir.glob("weekly_report_*.csv"))

            # Keep only the most recent N weeks (assuming weekly runs)
            # Each week has 1-3 files: delta (always), logs (optional), failures (optional)
            # To be safe, estimate 3 files per week
            files_per_week = 3
            keep_count = keep_weeks * files_per_week

            if len(report_files) > keep_count:
                for old_file in report_files[:-keep_count]:
                    old_file.unlink()
                    self.logger.debug(f"Removed old report: {old_file.name}")

            # Also clean up old plot files
            plot_files = sorted((self.reports_dir / "plots").glob("*.png"))
            plots_per_week = 2  # delta (always) + log_summary (optional)
            keep_plot_count = keep_weeks * plots_per_week

            if len(plot_files) > keep_plot_count:
                for old_plot in plot_files[:-keep_plot_count]:
                    old_plot.unlink()
                    self.logger.debug(f"Removed old plot: {old_plot.name}")

            self.logger.info("Report cleanup complete")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old reports: {e}")
            # Don't raise - cleanup failure shouldn't break reporting


def main():
    """Command-line interface for manual report generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ECCO pipeline weekly report"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Pipeline output directory"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        help="Logs directory (default: logs/)"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Reports directory (default: OUTPUT_DIR/reports)"
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Include log-based reporting (optional)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old reports after generation"
    )
    parser.add_argument(
        "--keep-weeks",
        type=int,
        default=12,
        help="Number of weeks to keep when cleaning up (default: 12)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s (%(name)s) - %(message)s'
    )

    # Create manager and generate report
    manager = PipelineReportManager(
        output_dir=args.output_dir,
        logs_dir=args.logs_dir,
        reports_dir=args.reports_dir,
        include_log_report=args.include_logs
    )

    try:
        delta_df, log_df = manager.generate_weekly_report()

        # Optionally cleanup old reports
        if args.cleanup:
            manager.cleanup_old_reports(keep_weeks=args.keep_weeks)

        print("\nWeekly report generation complete")
        print(f"  Reports saved to: {manager.reports_dir}")

        # Show quick summary
        if log_df is not None:
            failed = log_df[log_df["status"] == "Failed"]
            if not failed.empty:
                print(f"  {len(failed)} failed stages detected - see FAILURES.csv")
            else:
                print("  All stages successful")

        return 0

    except Exception as e:
        print(f"\nReport generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
