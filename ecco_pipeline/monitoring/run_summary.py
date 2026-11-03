"""
Per-run record of what a pipeline run actually did.

Each run produces two artifacts:

* ``logs/<run_id>/run_summary.json`` — canonical. Written unconditionally, including
  when the run aborts because Solr is unreachable, so "Solr was down" leaves a trace
  instead of silence.
* a ``type_s:pipeline_run`` Solr doc — a mirror for the dashboard and run-history
  queries. Written best-effort; a Solr failure never changes the run's exit code.

Why this exists: the per-stage status strings the pipeline already produces
(``harvester_status``, the transformation ``pipeline_cleanup``, ``get_agg_status``) are
*cumulative* — they count every doc ever written for a dataset, so one historical
failure makes them read "N failed" on every subsequent run forever. They are preserved
here as ``status_text`` but cannot be used to decide whether *this* run went well.

Time: ``run_id`` is local time (it is the ``logs/<run_id>/`` directory name, fixed by
log_config at import) while every ``*_dt`` field is UTC, matching the convention used
throughout the pipeline. Do not compare the two directly.

Counting per-run activity uses two independent sources, deliberately:

* **timestamp windows** (primary) — every stage already writes an activity timestamp on
  every doc it writes, so this source cannot miss a write site. It can over-attribute
  when two runs overlap.
* **run_id_s stamps** (cross-check) — exact attribution, but silently under-counts if a
  write site is ever added without a stamp.

Under-reporting failures is the unacceptable direction for a monitoring system, so the
timestamp window is primary. See ``_divergence`` for why the comparison between them is
one-directional.
"""

import json
import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Iterable, Optional

from utils.pipeline_utils import log_config, solr_utils

logger = logging.getLogger("pipeline")

SCHEMA_VERSION = 1

STAGES = ("harvest", "transform", "aggregate")

# stage -> (solr type_s, activity timestamp field, boolean success field)
_STAGE_FIELDS = {
    "harvest": ("granule", "last_attempt_dt", "harvest_success_b"),
    "transform": ("transformation", "transformation_started_dt", "success_b"),
    "aggregate": ("aggregation", "aggregation_time_dt", "aggregation_success_b"),
}

_ERROR_SAMPLE_LIMIT = 5
_LOG_SAMPLE_LIMIT = 5
_TRACEBACK_TAIL_CHARS = 2000

# Exit codes. 0 is the only value meaning "nothing to look at".
EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_ABORTED = 2
EXIT_CRASHED = 3


def utc_now() -> str:
    """UTC timestamp in the Solr format used throughout the pipeline."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class StageResult:
    """
    What one (dataset, stage) did during this run, as observed by run_pipeline.

    Covers only what the parent process can see directly: the stage's own status
    string, how long it took, and any exception that escaped it. Per-doc counts are
    filled in later from Solr by :func:`build_summary`.
    """

    dataset: str
    stage: str
    status_text: str = ""
    duration_s: float = 0.0
    exc_type: Optional[str] = None
    exc_message: str = ""
    traceback_tail: str = ""
    # Worker-level failure markers, currently only produced by aggregation. These
    # matter because a worker can die before it writes its Solr doc, in which case
    # the Solr counts below cannot see the failure at all.
    worker_errors: list = field(default_factory=list)

    @property
    def raised(self) -> bool:
        return self.exc_type is not None


def _git_commit() -> str:
    """Short HEAD sha, or "" if git isn't available or this isn't a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _fq_dataset(ds_name: str) -> str:
    # Quoted because dataset names contain '.' and '-' (e.g. AVHRR_OI-NCEI-L4-GLOB-v2.1).
    return f'dataset_s:"{ds_name}"'


def _count_window(ds_name: str, stage: str, since: str) -> tuple:
    """(attempted, failed) for docs whose activity timestamp falls at or after `since`."""
    type_s, ts_field, success_field = _STAGE_FIELDS[stage]
    base = [f"type_s:{type_s}", _fq_dataset(ds_name), f"{ts_field}:[{since} TO *]"]
    attempted = solr_utils.solr_count(base)
    failed = solr_utils.solr_count(base + [f"{success_field}:false"])
    return attempted, failed


def _count_by_run_id(ds_name: str, stage: str, run_id: str) -> tuple:
    """(attempted, failed) for docs stamped with this run's id."""
    type_s, _, success_field = _STAGE_FIELDS[stage]
    base = [f"type_s:{type_s}", _fq_dataset(ds_name), f'run_id_s:"{run_id}"']
    attempted = solr_utils.solr_count(base)
    failed = solr_utils.solr_count(base + [f"{success_field}:false"])
    return attempted, failed


def _error_samples(ds_name: str, stage: str, since: str) -> list:
    """A bounded sample of this run's failure docs, for the digest and for triage."""
    type_s, ts_field, success_field = _STAGE_FIELDS[stage]
    fq = [
        f"type_s:{type_s}",
        _fq_dataset(ds_name),
        f"{ts_field}:[{since} TO *]",
        f"{success_field}:false",
    ]
    fl = "error_message_s,filename_s,date_dt,grid_name_s,field_s,year_i"
    try:
        docs = solr_utils.solr_query(fq, fl=fl, rows=_ERROR_SAMPLE_LIMIT)
    except Exception as e:
        logger.warning(f"Could not fetch {stage} error samples for {ds_name}: {e}")
        return []
    samples = []
    for doc in docs:
        detail = doc.get("filename_s") or ""
        if not detail:
            parts = [doc.get("grid_name_s"), doc.get("field_s"), doc.get("year_i")]
            detail = " / ".join(str(p) for p in parts if p not in (None, ""))
        samples.append(
            {
                "detail": detail,
                "date": str(doc.get("date_dt", ""))[:10],
                "error": doc.get("error_message_s", ""),
            }
        )
    return samples


def scan_log_errors(log_dir: str, subdirs: Iterable[str] = ()) -> dict:
    """
    Count ERROR/CRITICAL lines across a run's log files, keeping a few samples.

    A backstop for failures that never reach Solr: a worker killed outright (OOM) can
    log and die without writing its result doc, so the Solr counts miss it entirely.
    Pass the per-stage log subdirectories (``tx_<ds>``, ``ag_<ds>``) to scope the scan
    to one stage; with none, only the top-level ``pipeline.log`` is read.
    """
    paths = [os.path.join(log_dir, "pipeline.log")]
    for subdir in subdirs:
        paths.extend(sorted(glob(os.path.join(log_dir, subdir, "*.log"))))

    count = 0
    samples = []
    for path in paths:
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    if line.startswith("[ERROR]") or line.startswith("[CRITICAL]"):
                        count += 1
                        if len(samples) < _LOG_SAMPLE_LIMIT:
                            samples.append(line.rstrip()[:500])
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"Could not scan log file {path}: {e}")
    return {"count": count, "samples": samples}


def _log_subdirs(stage: str, ds_name: str) -> tuple:
    """Per-dataset log subdirectory names used by each stage's worker logging."""
    if stage == "transform":
        return (f"tx_{ds_name}",)
    if stage == "aggregate":
        return (f"ag_{ds_name}",)
    return ()  # harvest logs to pipeline.log only


def _stage_failed(stage: dict) -> bool:
    """
    Whether a stage entry represents a failure for exit-code and streak purposes.

    Log error lines are deliberately excluded: recoverable conditions are logged at
    ERROR level in places (e.g. "Failed to update Solr dataset document"), so letting
    them drive the exit code would make a red run the normal state. They are reported,
    not adjudicated.
    """
    if not stage.get("ran"):
        return False
    return bool(
        stage.get("exception") or stage.get("failed", 0) or stage.get("worker_errors")
    )


def _divergence(stage: dict) -> bool:
    """
    Whether the two count sources disagree in the direction that indicates a problem.

    Only ``timestamp > run_id`` is flagged. That direction means docs were written in
    this run's window without this run's stamp — either a write site that was added
    without a stamp (which would silently under-report failures) or a second run
    overlapping this one.

    The reverse (``run_id > timestamp``) is expected and benign: the two
    doc-reconstruction paths (``reconstruct_tx_solr_doc``,
    ``reconstruct_agg_solr_doc``) write a doc *now* but backdate its activity
    timestamp to the output file's mtime, so a reconstructed doc carries this run's id
    while falling outside this run's time window. Both numbers are always recorded, so
    that case stays visible without being flagged.
    """
    return stage.get("attempted", 0) > stage.get("attempted_by_run_id", 0)


def _previous_streaks(summary: Optional[dict]) -> dict:
    """{(dataset, stage): (consecutive_failures, first_failed_run_id)} from a summary."""
    streaks = {}
    if not summary:
        return streaks
    for entry in summary.get("datasets", []):
        ds_name = entry.get("dataset")
        for stage_name, stage in entry.get("stages", {}).items():
            streaks[(ds_name, stage_name)] = (
                stage.get("consecutive_failures", 0),
                stage.get("first_failed_run_id"),
            )
    return streaks


def build_summary(
    results: Iterable[StageResult],
    run_id: str,
    started_dt: str,
    args=None,
    outcome: Optional[str] = None,
    abort_reason: Optional[str] = None,
) -> dict:
    """
    Assemble the run summary from the parent's stage results plus per-run Solr counts.

    `outcome` forces a terminal state ("aborted"/"crashed"); otherwise it is derived
    from whether anything failed.
    """
    log_dir = f"logs/{run_id}"
    results = list(results)

    # Harvest flushes granule docs with commitWithin, so a pending batch would read as
    # absent here. One hard commit before counting; the same ordering hazard is handled
    # this way in transformation_factory before its own cleanup counts.
    if results:
        try:
            solr_utils.commit_solr()
        except Exception as e:
            logger.warning(f"Could not commit Solr before building run summary: {e}")

    prev = None if outcome == "aborted" else previous_summary(run_id)
    streaks = _previous_streaks(prev)

    by_dataset = {}
    for r in results:
        stage = {
            "ran": True,
            "status_text": r.status_text,
            "duration_s": round(r.duration_s, 1),
            "exception": None,
            "worker_errors": list(r.worker_errors),
        }
        if r.raised:
            stage["exception"] = {
                "type": r.exc_type,
                "message": r.exc_message,
                "traceback_tail": r.traceback_tail[-_TRACEBACK_TAIL_CHARS:],
            }

        try:
            attempted, failed = _count_window(r.dataset, r.stage, started_dt)
            by_run_id = _count_by_run_id(r.dataset, r.stage, run_id)
            samples = _error_samples(r.dataset, r.stage, started_dt) if failed else []
        except Exception as e:
            # Counting is best-effort: a Solr problem here must not lose the exception
            # and worker-error detail already captured above.
            logger.warning(f"Could not count {r.stage} activity for {r.dataset}: {e}")
            attempted = failed = 0
            by_run_id = (0, 0)
            samples = []

        stage["attempted"] = attempted
        stage["failed"] = failed
        stage["attempted_by_run_id"] = by_run_id[0]
        stage["failed_by_run_id"] = by_run_id[1]
        stage["error_samples"] = samples
        stage["log_errors"] = scan_log_errors(log_dir, _log_subdirs(r.stage, r.dataset))
        stage["outcome"] = "failures" if _stage_failed(stage) else "ok"

        by_dataset.setdefault(r.dataset, {})[r.stage] = stage

    # Carry forward every (dataset, stage) this run did not touch, so a partial run
    # (--dataset X --step harvest) does not erase the other datasets' streaks — the
    # next run reads its previous state from this file.
    for (ds_name, stage_name), (streak, first_run) in streaks.items():
        if stage_name in by_dataset.get(ds_name, {}):
            continue
        by_dataset.setdefault(ds_name, {})[stage_name] = {
            "ran": False,
            "consecutive_failures": streak,
            "first_failed_run_id": first_run,
        }

    # Streaks for what did run: increment on failure, reset on success.
    for ds_name, stages in by_dataset.items():
        for stage_name, stage in stages.items():
            if not stage.get("ran"):
                continue
            prev_streak, prev_first = streaks.get((ds_name, stage_name), (0, None))
            if _stage_failed(stage):
                stage["consecutive_failures"] = prev_streak + 1
                stage["first_failed_run_id"] = prev_first or run_id
            else:
                stage["consecutive_failures"] = 0
                stage["first_failed_run_id"] = None

    datasets = [
        {"dataset": ds_name, "stages": stages}
        for ds_name, stages in sorted(by_dataset.items())
    ]

    ran_stages = [s for d in datasets for s in d["stages"].values() if s.get("ran")]

    def total(key: str, stage_name: str) -> int:
        return sum(
            s.get(key, 0)
            for d in datasets
            for name, s in d["stages"].items()
            if name == stage_name and s.get("ran")
        )

    divergence = [
        f"{d['dataset']}/{name}"
        for d in datasets
        for name, s in d["stages"].items()
        if s.get("ran") and _divergence(s)
    ]

    failed_datasets = sorted(
        {
            d["dataset"]
            for d in datasets
            for s in d["stages"].values()
            if _stage_failed(s)
        }
    )

    if outcome is None:
        outcome = "failures" if failed_datasets else "ok"

    ended = utc_now()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "log_dir": log_dir,
        "started_dt": started_dt,
        "ended_dt": ended,
        "duration_s": _elapsed(started_dt, ended),
        "outcome": outcome,
        "abort_reason": abort_reason,
        "invocation": {
            "argv": sys.argv[1:],
            "step": getattr(args, "step", None),
            "dataset": getattr(args, "dataset", None),
            "grids": getattr(args, "grids_to_use", None) or None,
            "multiprocesses": getattr(args, "multiprocesses", None),
            "git_commit": _git_commit(),
            "hostname": socket.gethostname(),
        },
        "totals": {
            "datasets": len({d["dataset"] for d in datasets if d["stages"]}),
            "datasets_with_failures": len(failed_datasets),
            "failed_datasets": failed_datasets,
            "granules_attempted": total("attempted", "harvest"),
            "granules_failed": total("failed", "harvest"),
            "transformations_attempted": total("attempted", "transform"),
            "transformations_failed": total("failed", "transform"),
            "aggregations_attempted": total("attempted", "aggregate"),
            "aggregations_failed": total("failed", "aggregate"),
            "log_error_lines": sum(
                s.get("log_errors", {}).get("count", 0) for s in ran_stages
            ),
            "count_divergence": divergence,
        },
        "datasets": datasets,
    }
    summary["exit_code"] = exit_code_for(summary)
    return summary


def _elapsed(start: str, end: str) -> int:
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        return int(
            (
                datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
            ).total_seconds()
        )
    except Exception:
        return 0


def exit_code_for(summary: dict) -> int:
    outcome = summary.get("outcome")
    if outcome == "aborted":
        return EXIT_ABORTED
    if outcome == "crashed":
        return EXIT_CRASHED
    if outcome == "failures":
        return EXIT_FAILURES
    return EXIT_OK


def write_summary(summary: dict) -> Path:
    """Write the canonical JSON artifact. Returns its path."""
    log_dir = Path(summary["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "run_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


def mirror_to_solr(summary: dict) -> bool:
    """
    Mirror the summary to a ``type_s:pipeline_run`` doc.

    Best-effort by design: the JSON artifact is canonical, and a Solr failure here must
    not change the run's exit code — not least because "Solr is broken" is one of the
    conditions this whole mechanism exists to report.
    """
    totals = summary.get("totals", {})
    doc = {
        "id": f"run-{summary['run_id']}",
        "type_s": "pipeline_run",
        "run_id_s": summary["run_id"],
        "started_dt": summary["started_dt"],
        "ended_dt": summary["ended_dt"],
        "duration_i": summary.get("duration_s", 0),
        "exit_code_i": summary.get("exit_code", 0),
        "outcome_s": summary.get("outcome", ""),
        "abort_reason_s": summary.get("abort_reason") or "",
        "n_datasets_i": totals.get("datasets", 0),
        "n_datasets_failed_i": totals.get("datasets_with_failures", 0),
        "granules_attempted_i": totals.get("granules_attempted", 0),
        "granules_failed_i": totals.get("granules_failed", 0),
        "transformations_attempted_i": totals.get("transformations_attempted", 0),
        "transformations_failed_i": totals.get("transformations_failed", 0),
        "aggregations_attempted_i": totals.get("aggregations_attempted", 0),
        "aggregations_failed_i": totals.get("aggregations_failed", 0),
        "log_error_lines_i": totals.get("log_error_lines", 0),
        # The full summary travels with the doc so the dashboard and triage tooling can
        # read everything from one query instead of reaching for the log directory.
        "summary_json_s": json.dumps(summary, default=str),
    }
    try:
        solr_utils.solr_update([doc])
        return True
    except Exception as e:
        logger.warning(f"Could not mirror run summary to Solr: {e}")
        return False


def load_summary(run_id: str) -> Optional[dict]:
    """Read a run's summary from its log directory."""
    path = Path(f"logs/{run_id}/run_summary.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None


def _run_ids_from_disk() -> list:
    """Run ids that have a summary on disk, oldest first."""
    found = []
    for path in glob("logs/*/run_summary.json"):
        found.append(os.path.basename(os.path.dirname(path)))
    return sorted(found)


def latest_run_id() -> Optional[str]:
    """Most recent run id with a summary on disk."""
    ids = _run_ids_from_disk()
    return ids[-1] if ids else None


def previous_summary(before_run_id: str) -> Optional[dict]:
    """
    The most recent summary from a run other than `before_run_id`.

    Prefers Solr (indexed, and it holds the blob) and falls back to the log directories
    when Solr is unreachable — which is exactly when this lookup still needs to work.
    """
    try:
        docs = solr_utils.solr_query(
            ["type_s:pipeline_run"], fl="id,run_id_s,started_dt"
        )
        candidates = sorted(
            (d for d in docs if d.get("run_id_s") != before_run_id),
            key=lambda d: d.get("started_dt") or "",
        )
        if candidates:
            newest = candidates[-1]
            full = solr_utils.solr_query(
                [f'id:"{newest["id"]}"'], fl="summary_json_s", rows=1
            )
            if full and full[0].get("summary_json_s"):
                return json.loads(full[0]["summary_json_s"])
    except Exception as e:
        logger.debug(f"Falling back to log dirs for previous run summary: {e}")

    for run_id in reversed(_run_ids_from_disk()):
        if run_id == before_run_id:
            continue
        summary = load_summary(run_id)
        if summary:
            return summary
    return None


def record_run(
    results: Iterable[StageResult],
    args=None,
    started_dt: str = "",
    outcome: Optional[str] = None,
    abort_reason: Optional[str] = None,
) -> dict:
    """
    Build, persist, and mirror the run summary. Returns the summary.

    The single entry point run_pipeline calls, on every exit path.
    """
    run_id = log_config.run_id()
    summary = build_summary(
        results,
        run_id=run_id,
        started_dt=started_dt or utc_now(),
        args=args,
        outcome=outcome,
        abort_reason=abort_reason,
    )
    try:
        path = write_summary(summary)
        logger.info(f"Wrote run summary to {path}")
    except Exception as e:
        # Nothing left to fall back on, but the exit code still carries the verdict.
        logger.error(f"Could not write run summary: {e}")
    mirror_to_solr(summary)
    return summary
