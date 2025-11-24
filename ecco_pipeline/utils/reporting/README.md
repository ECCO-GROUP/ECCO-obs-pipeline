# ECCO Pipeline Reporting

Generates reports on pipeline outputs by analyzing the filesystem structure and optionally parsing log files. Tracks week-over-week changes to detect new data, removed files, and pipeline issues.

## Overview

**Three modules:**

1. **`output_report.py`** - Counts files in OUTPUT_DIR per dataset/stage/grid (PRIMARY)
2. **`logs_report.py`** - Parses pipeline logs for performance metrics (OPTIONAL)
3. **`report_manager.py`** - Orchestrates both reporters, generates weekly reports

**Key feature:** Automatic delta tracking - compares current run with previous snapshot to show what changed.

## Quick Start

### Automatic Reporting (Recommended)

Add `--generate_report` flag to pipeline runs:

```bash
# Run pipeline with automatic reporting
python run_pipeline.py --step all --generate_report
```

### Manual Reporting

All commands run from `ecco_pipeline/` directory:

```bash
cd /path/to/ECCO-pipeline/ecco_pipeline

# Full report (filesystem only)
python -m utils.reporting.report_manager

# Full report with log analysis
python -m utils.reporting.report_manager --include-logs

# Filesystem report only
python -m utils.reporting.output_report

# Log report only
python -m utils.reporting.logs_report
```


## Output Files

Reports saved to `OUTPUT_DIR/reports/` (default: `/Users/marlis/Developer/ECCO/ecco_output/reports/`)

**Always generated:**
- `pipeline_report_YYYY-MM-DD.csv` - Snapshot for next week's delta comparison
- `weekly_report_TIMESTAMP_delta.csv` - Current, previous, and delta file counts
- `plots/delta_TIMESTAMP.png` - Visualization of changes

**Generated if `include_log_report=True`:**
- `weekly_report_TIMESTAMP_logs.csv` - Performance metrics from logs
- `weekly_report_TIMESTAMP_FAILURES.csv` - Failed stages (only if failures exist)
- `plots/log_summary_TIMESTAMP.png` - Log visualization

## Report Structure

### Delta Report (main output)
```csv
dataset,stage,grid,files_count_current,files_count_prev,delta
SMAP_RSS_L3_SSS_SMI_8DAY_V5,Harvest,N/A,1500,1450,+50
SMAP_RSS_L3_SSS_SMI_8DAY_V5,Transform,ECCO_llc90,1500,1450,+50
SMAP_RSS_L3_SSS_SMI_8DAY_V5,Aggregate,ECCO_llc90,12,10,+2
```

**Columns:**
- `files_count_current`: Files in OUTPUT_DIR right now
- `files_count_prev`: Files in last week's snapshot
- `delta`: Difference (current - previous)

**Interpreting delta:**
- Positive: New files added (expected for recent data)
- Negative: Files removed (investigate - may indicate re-processing or data loss)
- Zero: No changes (expected for stable/older data)

### Log Report (optional)
```csv
run_timestamp,dataset,stage,files_processed,duration_sec,status,errors
20250124_140530,SMAP_RSS_L3_SSS_SMI_8DAY_V5,Harvest,50,120.5,Success,
20250124_140530,SMAP_RSS_L3_SSS_SMI_8DAY_V5,Transform,50,450.2,Success,
20250124_140530,SMAP_RSS_L3_SSS_SMI_8DAY_V5,Aggregate,1,30.1,Success,
```

## Configuration

In `conf/global_settings.py`:

```python
OUTPUT_DIR: Path = Path("/Users/marlis/Developer/ECCO/ecco_output")
REPORTS_DIR: Path = OUTPUT_DIR / "reports"
REPORTS_RETENTION_WEEKS: int = 12
```

Override programmatically:

```python
from pathlib import Path
from utils.reporting import PipelineReportManager

manager = PipelineReportManager(
    output_dir=Path("/custom/output"),
    reports_dir=Path("/custom/reports"),
    logs_dir=Path("/custom/logs")
)
```

## Command-Line Reference

### report_manager.py

```bash
python -m utils.reporting.report_manager [OPTIONS]

Options:
  --output-dir PATH      Pipeline output directory
  --logs-dir PATH        Logs directory (default: logs/)
  --reports-dir PATH     Reports directory (default: OUTPUT_DIR/reports)
  --include-logs         Include log-based reporting
  --cleanup              Clean up old reports after generation
  --keep-weeks INT       Weeks to keep during cleanup (default: 12)
```

### output_report.py

```bash
python -m utils.reporting.output_report [OPTIONS]

Options:
  --output-dir PATH      Pipeline output directory
  --reports-dir PATH     Reports directory
  --no-snapshot          Don't save weekly snapshot
  --plot                 Generate and save delta plot
```

### logs_report.py

```bash
python -m utils.reporting.logs_report [OPTIONS]

Options:
  --logs-dir PATH        Logs directory (default: logs/)
  --all                  Parse all logs instead of just latest
  --plot PATH            Save plot to specified path
  --failures-only        Show only failed stages
```

## Cleanup

Reports accumulate over time. Remove old reports periodically:

```bash
# Keep last 12 weeks (default)
python -m utils.reporting.report_manager --cleanup

# Keep last 4 weeks
python -m utils.reporting.report_manager --cleanup --keep-weeks 4
```

Or programmatically:

```python
manager = PipelineReportManager()
manager.cleanup_old_reports(keep_weeks=8)
```

## Common Issues

**"No previous snapshot found"**
This is the first report - delta will be available after the second run.

**"No pipeline.log found"**
Pipeline hasn't run yet or logs directory path is incorrect. Verify `logs/` exists.

**Report generation fails but pipeline succeeds**
By design - reporting errors don't fail the pipeline. Check pipeline logs for exceptions.

**Negative delta for stable data**
Check if config versions (`t_version`, `a_version`) changed, triggering re-processing.
