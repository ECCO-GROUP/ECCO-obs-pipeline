#!/usr/bin/env bash
#
# Cron entrypoint for the weekly pipeline run.
#
# Use this instead of invoking run_pipeline.py directly, so the pipeline's exit code is
# what cron sees even though the notifier runs afterwards.
#
#   0 10 * * 6  cd /path/to/ECCO-pipeline && ./scripts/run_pipeline_cron.sh >> cron.log 2>&1
#
# Exit codes come from run_pipeline.py: 0 clean, 1 dataset/stage failures,
# 2 precondition abort (Solr down, missing credentials), 3 unexpected fatal error.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 3

uv run python ecco_pipeline/run_pipeline.py "$@"
rc=$?

# The digest is advisory: a broken SMTP relay must not change the run's verdict, and a
# failed send is already logged by notify.py.
uv run python ecco_pipeline/monitoring/notify.py --run-id latest || true

exit $rc
