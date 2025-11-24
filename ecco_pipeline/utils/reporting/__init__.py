"""
ECCO Pipeline Reporting Module

Provides log-based and filesystem-based reporting for pipeline runs.

Main Components:
- PipelineReportManager: Unified orchestration of all reporting
- PipelineOutputReporter: Filesystem-based file count reporting
- PipelineLogReporter: Log parsing and analysis

Usage:
    from utils.reporting import PipelineReportManager

    manager = PipelineReportManager()
    delta_df, log_df = manager.generate_weekly_report()

For cron integration, use run_pipeline.py with --generate_report flag.
"""
from utils.reporting.report_manager import PipelineReportManager
from utils.reporting.output_report import PipelineOutputReporter
from utils.reporting.logs_report import PipelineLogReporter

__all__ = [
    "PipelineReportManager",
    "PipelineOutputReporter",
    "PipelineLogReporter"
]

__version__ = "1.0.0"
