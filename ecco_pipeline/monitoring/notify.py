"""
Email digest for a pipeline run.

Run by cron *after* run_pipeline.py rather than from inside it, for two reasons: a hung
SMTP relay can then never affect a pipeline run, and a digest can be re-rendered for any
past run without re-running anything.

    python ecco_pipeline/monitoring/notify.py --run-id latest --dry-run

Configuration lives in conf/global_settings.py (gitignored, so credentials are not
committed). Every setting is read with getattr and a default: existing settings files —
and the template tests/conftest.py copies on a fresh checkout — predate these names, and
a hard import would break collection everywhere.
"""

import argparse
import html
import logging
import smtplib
import sys
import os
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from monitoring import run_summary  # noqa: E402

logger = logging.getLogger("pipeline")

# Expected gap between runs. Only used to decide whether to call out a long gap in the
# digest; the weekly cron cadence is the reason for the default.
EXPECTED_CADENCE_DAYS = 7

_DEFAULTS = {
    "SMTP_HOST": None,
    "SMTP_PORT": 25,
    "SMTP_USE_TLS": False,
    "SMTP_USER": None,
    "SMTP_PASSWORD": None,
    "NOTIFY_FROM": None,
    "NOTIFY_TO": [],
    # "always" while the plumbing is new: a digest that is expected every week makes its
    # absence a signal, which is the only cover for a run that never starts. Switch to
    # "failures_only" once normal is well understood.
    "NOTIFY_ON": "always",
}


def setting(name: str):
    try:
        from conf import global_settings
    except ImportError:
        return _DEFAULTS[name]
    return getattr(global_settings, name, _DEFAULTS[name])


def is_configured() -> bool:
    return bool(
        setting("SMTP_HOST") and setting("NOTIFY_FROM") and setting("NOTIFY_TO")
    )


def _fmt_duration(seconds: int) -> str:
    if not seconds:
        return "0s"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _fmt_local(utc_str: Optional[str]) -> str:
    """
    A summary's UTC timestamp in the host's local time, with the zone named.

    The run id is local time (it is the log directory name), so showing the start in UTC
    next to it reads as two times that disagree.
    """
    try:
        dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return str(utc_str)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# (stage, column label) in the order the per-dataset table shows them.
_ACTIVITY_COLUMNS = (
    ("harvest", "granules"),
    ("transform", "transformations"),
    ("aggregate", "aggregations"),
)


def _activity_lines(summary: dict) -> list:
    """
    One line per dataset that did work this run, plus a count of those that ran idle.

    Datasets the run never touched (a partial --dataset run carries the rest forward
    with ran=False) appear in neither.
    """
    active, idle = [], 0
    for entry in summary.get("datasets", []):
        ran = {
            name: stage
            for name, stage in entry.get("stages", {}).items()
            if stage.get("ran")
        }
        if not ran:
            continue
        counts = [
            ran.get(stage, {}).get("attempted", 0) for stage, _ in _ACTIVITY_COLUMNS
        ]
        if any(counts):
            active.append((entry["dataset"], counts))
        else:
            idle += 1

    if not active and not idle:
        return []

    lines = ["", "Datasets with new data", "----------------------"]
    if active:
        name_width = max(len(name) for name, _ in active)
        num_widths = [
            max(len(str(counts[i])) for _, counts in active)
            for i in range(len(_ACTIVITY_COLUMNS))
        ]
        for name, counts in active:
            cells = [
                f"{n:>{w}} {label}"
                for n, w, (_, label) in zip(counts, num_widths, _ACTIVITY_COLUMNS)
            ]
            lines.append(f"  {name:<{name_width}}  " + "  ".join(cells))
    else:
        lines.append("  None.")
    if idle:
        plural = "dataset" if idle == 1 else "datasets"
        lines.append(f"  ({idle} {plural} had nothing new)")
    return lines


def _gap_note(summary: dict, previous: Optional[dict]) -> Optional[str]:
    """
    How long since the previous run, flagged when it exceeds the expected cadence.

    This is the partial cover for having no dead-man's switch: a run that never happened
    cannot report itself, but the run *after* it can report the hole it left.
    """
    if not previous:
        return "No previous run on record."
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        this_start = datetime.strptime(summary["started_dt"], fmt)
        prev_start = datetime.strptime(previous["started_dt"], fmt)
    except Exception:
        return None
    days = (this_start - prev_start).total_seconds() / 86400
    note = (
        f"Previous run {previous.get('run_id')} was {days:.1f} days ago "
        f"({previous.get('outcome')})."
    )
    if days > EXPECTED_CADENCE_DAYS * 1.5:
        note += (
            f"  <-- longer than the expected {EXPECTED_CADENCE_DAYS}-day cadence; "
            "a scheduled run may have been missed."
        )
    return note


def subject_for(summary: dict) -> str:
    run_id = summary.get("run_id", "?")
    outcome = summary.get("outcome")
    if outcome == "aborted":
        return f"[ECCO pipeline] {run_id} — ABORTED: {summary.get('abort_reason', 'unknown')}"
    if outcome == "crashed":
        return f"[ECCO pipeline] {run_id} — CRASHED"
    n = summary.get("totals", {}).get("datasets_with_failures", 0)
    if n:
        plural = "dataset" if n == 1 else "datasets"
        return f"[ECCO pipeline] {run_id} — {n} {plural} with failures"
    return f"[ECCO pipeline] {run_id} — OK"


def _failed_stages(summary: dict) -> list:
    """
    (dataset, stage_name, stage) for every stage that failed, worst streak first, so
    persistent breakage outranks one-off noise in the digest.
    """
    failed = []
    for entry in summary.get("datasets", []):
        for stage_name, stage in entry.get("stages", {}).items():
            if not stage.get("ran"):
                continue
            if (
                stage.get("exception")
                or stage.get("failed", 0)
                or stage.get("worker_errors")
            ):
                failed.append((entry["dataset"], stage_name, stage))
    failed.sort(key=lambda t: t[2].get("consecutive_failures", 0), reverse=True)
    return failed


def body_for(summary: dict, previous: Optional[dict] = None) -> str:
    totals = summary.get("totals", {})
    outcome = summary.get("outcome")
    outcome_line = f"Outcome:  {outcome}"
    if summary.get("exit_code"):
        outcome_line += f" (exit {summary.get('exit_code')})"
    lines = [
        f"Run:      {summary.get('run_id')}",
        outcome_line,
        f"Started:  {_fmt_local(summary.get('started_dt'))}",
        f"Duration: {_fmt_duration(summary.get('duration_s', 0))}",
    ]

    invocation = summary.get("invocation", {})
    if invocation.get("hostname") or invocation.get("git_commit"):
        lines.append(
            f"Host:     {invocation.get('hostname', '?')} @ "
            f"{invocation.get('git_commit') or 'unknown commit'}"
        )
    if invocation.get("dataset") or invocation.get("step") not in (None, "all"):
        lines.append(
            f"Scope:    dataset={invocation.get('dataset') or 'all'} "
            f"step={invocation.get('step') or 'all'}"
        )

    if summary.get("outcome") == "aborted":
        lines += [
            "",
            "The run aborted before doing any work:",
            f"  {summary.get('abort_reason')}",
        ]

    gap = _gap_note(summary, previous)
    if gap:
        lines += ["", gap]

    # Zero failure counts are left off: on a clean run they are noise, and their absence
    # is what makes a non-zero one stand out.
    def counted(label: str, key: str) -> str:
        line = f"  {label:<17}{totals.get(f'{key}_attempted', 0)} attempted"
        failed = totals.get(f"{key}_failed", 0)
        return line + (f", {failed} failed" if failed else "")

    datasets_line = f"  {'Datasets:':<17}{totals.get('datasets', 0)}"
    if totals.get("datasets_with_failures"):
        datasets_line += f" ({totals['datasets_with_failures']} with failures)"
    lines += [
        "",
        "Activity this run",
        "-----------------",
        datasets_line,
        counted("Granules:", "granules"),
        counted("Transformations:", "transformations"),
        counted("Aggregations:", "aggregations"),
    ]
    if totals.get("log_error_lines"):
        lines.append(f"  {'ERROR log lines:':<17}{totals['log_error_lines']}")

    lines += _activity_lines(summary)

    divergence = totals.get("count_divergence") or []
    if divergence:
        lines += [
            "",
            "Count mismatch (docs written in this run's window without its run id):",
            f"  {', '.join(divergence)}",
            "  Either a Solr write path is missing its run_id stamp, or another run",
            "  overlapped this one. Worth investigating — it means the failure counts",
            "  above may be attributing another run's work to this one.",
        ]

    failed = _failed_stages(summary)
    if failed:
        lines += ["", "Failures", "--------"]
        for ds_name, stage_name, stage in failed:
            streak = stage.get("consecutive_failures", 0)
            streak_note = ""
            if streak > 1:
                streak_note = (
                    f"  [failing for {streak} consecutive runs, "
                    f"since {stage.get('first_failed_run_id')}]"
                )
            lines.append(f"\n  {ds_name} / {stage_name}{streak_note}")
            lines.append(
                f"    {stage.get('failed', 0)} of {stage.get('attempted', 0)} failed"
                f"  ({_fmt_duration(stage.get('duration_s', 0))})"
            )
            lines.append(f"    cumulative status: {stage.get('status_text', '')}")

            exc = stage.get("exception")
            if exc:
                lines.append(f"    exception: {exc.get('type')}: {exc.get('message')}")
                tail = (exc.get("traceback_tail") or "").strip()
                if tail:
                    for tb_line in tail.splitlines()[-12:]:
                        lines.append(f"      | {tb_line}")

            for sample in stage.get("error_samples", [])[:5]:
                detail = sample.get("detail") or sample.get("date") or ""
                lines.append(f"    - {detail}: {sample.get('error', '')}")

            for worker_error in stage.get("worker_errors", [])[:5]:
                lines.append(
                    f"    - worker {worker_error.get('job')}: {worker_error.get('error')}"
                )

            log_errors = stage.get("log_errors") or {}
            if log_errors.get("count"):
                lines.append(
                    f"    {log_errors['count']} ERROR lines in this stage's logs"
                )

    lines += ["", f"Logs: {_abs_log_dir(summary.get('log_dir'))}", ""]
    return "\n".join(lines)


def _abs_log_dir(log_dir: Optional[str]) -> Optional[str]:
    """
    Summaries written before log_dir was stored absolute hold "logs/<run_id>", relative
    to the ecco_pipeline directory; resolve those so re-rendered old digests match.
    """
    if not log_dir or os.path.isabs(log_dir):
        return log_dir
    return str(PIPELINE_DIR / log_dir)


def build_message(summary: dict, previous: Optional[dict] = None) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject_for(summary)
    msg["From"] = setting("NOTIFY_FROM")
    recipients = setting("NOTIFY_TO")
    if isinstance(recipients, str):
        recipients = [recipients]
    msg["To"] = ", ".join(recipients)
    body = body_for(summary, previous)
    msg.set_content(body)
    # The digest is aligned with spaces, and Outlook renders text/plain in a
    # proportional font, which breaks the columns. An HTML alternative of the same text
    # in <pre> forces monospace; clients that prefer plain text still get that part.
    msg.add_alternative(
        '<html><body><pre style="font-family: Consolas, Menlo, monospace; '
        f'font-size: 13px;">{html.escape(body)}</pre></body></html>',
        subtype="html",
    )
    return msg


def should_send(summary: dict) -> bool:
    if setting("NOTIFY_ON") == "failures_only":
        return summary.get("outcome") != "ok"
    return True


def send(msg: EmailMessage) -> None:
    host = setting("SMTP_HOST")
    port = setting("SMTP_PORT")
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if setting("SMTP_USE_TLS"):
            smtp.starttls()
        user, password = setting("SMTP_USER"), setting("SMTP_PASSWORD")
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)


def notify(run_id: str = "latest", dry_run: bool = False) -> int:
    if run_id == "latest":
        resolved = run_summary.latest_run_id()
        if not resolved:
            logger.error("No run summaries found under logs/; nothing to report.")
            return 1
        run_id = resolved

    summary = run_summary.load_summary(run_id)
    if not summary:
        logger.error(f"No run summary found for run {run_id}.")
        return 1

    previous = run_summary.previous_summary(run_id)
    msg = build_message(summary, previous)

    if dry_run:
        print(f"Subject: {msg['Subject']}")
        print(f"To: {msg['To']}")
        print()
        print(msg.get_body(preferencelist=("plain",)).get_content())
        return 0

    if not is_configured():
        logger.warning(
            "Email notification is not configured (need SMTP_HOST, NOTIFY_FROM and "
            "NOTIFY_TO in conf/global_settings.py); skipping send."
        )
        return 0

    if not should_send(summary):
        logger.info(
            f"Run {run_id} finished {summary.get('outcome')} and NOTIFY_ON is "
            "'failures_only'; not sending."
        )
        return 0

    try:
        send(msg)
    except Exception as e:
        logger.error(f"Could not send run digest for {run_id}: {e}")
        return 1
    logger.info(f"Sent run digest for {run_id} to {msg['To']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Run id (the logs/<run_id> directory name), or 'latest'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the digest instead of sending it",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s (%(name)s) - %(message)s",
    )
    return notify(args.run_id, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
