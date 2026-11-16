"""
Unit tests for monitoring.notify. SMTP and settings are mocked.
"""

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from monitoring import notify

RUN_ID = "20260901T100000"

CONFIGURED = {
    "SMTP_HOST": "smtp.example.org",
    "SMTP_PORT": 25,
    "SMTP_USE_TLS": False,
    "SMTP_USER": None,
    "SMTP_PASSWORD": None,
    "NOTIFY_FROM": "pipeline@example.org",
    "NOTIFY_TO": ["you@example.org"],
    "NOTIFY_ON": "always",
}


def fake_setting(overrides=None):
    values = dict(CONFIGURED)
    values.update(overrides or {})
    return lambda name: values[name]


def summary(
    outcome="ok",
    failures=0,
    datasets=None,
    abort_reason=None,
    run_id=RUN_ID,
    started="2026-09-01T10:00:00Z",
):
    return {
        "schema_version": 1,
        "run_id": run_id,
        "log_dir": f"logs/{run_id}",
        "started_dt": started,
        "ended_dt": "2026-09-01T11:00:00Z",
        "duration_s": 3600,
        "exit_code": 0 if outcome == "ok" else 1,
        "outcome": outcome,
        "abort_reason": abort_reason,
        "invocation": {"hostname": "box", "git_commit": "abc1234", "step": "all"},
        "totals": {
            "datasets": 3,
            "datasets_with_failures": failures,
            "granules_attempted": 42,
            "granules_failed": 2 if failures else 0,
            "transformations_attempted": 100,
            "transformations_failed": 0,
            "aggregations_attempted": 10,
            "aggregations_failed": 0,
            "log_error_lines": 0,
            "count_divergence": [],
        },
        "datasets": datasets or [],
    }


def failing_dataset(streak=1, first_run=RUN_ID):
    return [
        {
            "dataset": "DS_A",
            "stages": {
                "harvest": {
                    "ran": True,
                    "outcome": "failures",
                    "status_text": "2 harvested granules failed",
                    "attempted": 10,
                    "failed": 2,
                    "duration_s": 30,
                    "consecutive_failures": streak,
                    "first_failed_run_id": first_run,
                    "exception": None,
                    "worker_errors": [],
                    "error_samples": [
                        {"detail": "file.nc", "date": "2026-08-30", "error": "503"}
                    ],
                    "log_errors": {"count": 0, "samples": []},
                }
            },
        }
    ]


class SubjectTestCase(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(
            notify.subject_for(summary()), f"[ECCO pipeline] {RUN_ID} — OK"
        )

    def test_failures_are_counted_and_pluralised(self):
        self.assertIn(
            "3 datasets with failures", notify.subject_for(summary("failures", 3))
        )
        self.assertIn(
            "1 dataset with failures", notify.subject_for(summary("failures", 1))
        )

    def test_aborted_carries_the_reason(self):
        subject = notify.subject_for(
            summary("aborted", abort_reason="Solr unreachable at localhost")
        )
        self.assertIn("ABORTED", subject)
        self.assertIn("Solr unreachable", subject)

    def test_crashed(self):
        self.assertIn("CRASHED", notify.subject_for(summary("crashed")))


class BodyTestCase(unittest.TestCase):
    def test_ok_body_reports_activity(self):
        body = notify.body_for(summary())

        self.assertIn("Outcome:  ok", body)
        self.assertIn("42 attempted", body)
        self.assertIn(f"logs/{RUN_ID}", body)

    def test_failure_body_includes_samples_and_status(self):
        body = notify.body_for(summary("failures", 1, failing_dataset()))

        self.assertIn("DS_A / harvest", body)
        self.assertIn("2 of 10 failed", body)
        self.assertIn("file.nc: 503", body)
        self.assertIn("cumulative status: 2 harvested granules failed", body)

    def test_streak_is_called_out_when_repeating(self):
        body = notify.body_for(
            summary(
                "failures", 1, failing_dataset(streak=3, first_run="20260811T100000")
            )
        )

        self.assertIn("failing for 3 consecutive runs", body)
        self.assertIn("20260811T100000", body)

    def test_single_failure_has_no_streak_note(self):
        body = notify.body_for(summary("failures", 1, failing_dataset(streak=1)))

        self.assertNotIn("consecutive runs", body)

    def test_failures_are_ordered_by_streak_length(self):
        """Persistent breakage should outrank one-off noise in the digest."""
        datasets = [
            {
                "dataset": "DS_NEW",
                "stages": {
                    "harvest": {
                        "ran": True,
                        "failed": 1,
                        "attempted": 2,
                        "consecutive_failures": 1,
                        "exception": None,
                        "worker_errors": [],
                        "error_samples": [],
                    }
                },
            },
            {
                "dataset": "DS_OLD",
                "stages": {
                    "harvest": {
                        "ran": True,
                        "failed": 1,
                        "attempted": 2,
                        "consecutive_failures": 9,
                        "exception": None,
                        "worker_errors": [],
                        "error_samples": [],
                    }
                },
            },
        ]
        body = notify.body_for(summary("failures", 2, datasets))
        failures = body[body.index("Failures\n--------") :]

        self.assertLess(failures.index("DS_OLD"), failures.index("DS_NEW"))

    def test_traceback_is_included(self):
        datasets = [
            {
                "dataset": "DS_A",
                "stages": {
                    "harvest": {
                        "ran": True,
                        "failed": 0,
                        "attempted": 0,
                        "consecutive_failures": 1,
                        "exception": {
                            "type": "HTTPError",
                            "message": "503 Server Error",
                            "traceback_tail": "Traceback:\n  line one\nHTTPError: 503\n",
                        },
                        "worker_errors": [],
                        "error_samples": [],
                    }
                },
            }
        ]
        body = notify.body_for(summary("failures", 1, datasets))

        self.assertIn("HTTPError: 503 Server Error", body)
        self.assertIn("| HTTPError: 503", body)

    def test_worker_errors_are_shown(self):
        datasets = [
            {
                "dataset": "DS_A",
                "stages": {
                    "aggregate": {
                        "ran": True,
                        "failed": 0,
                        "attempted": 5,
                        "consecutive_failures": 1,
                        "exception": None,
                        "error_samples": [],
                        "worker_errors": [
                            {"job": "GRID1, 2020, ssha", "error": "killed"}
                        ],
                    }
                },
            }
        ]
        body = notify.body_for(summary("failures", 1, datasets))

        self.assertIn("worker GRID1, 2020, ssha: killed", body)

    def test_aborted_body_explains_the_abort(self):
        body = notify.body_for(summary("aborted", abort_reason="No netrc file found"))

        self.assertIn("aborted before doing any work", body)
        self.assertIn("No netrc file found", body)

    def test_divergence_is_explained(self):
        s = summary()
        s["totals"]["count_divergence"] = ["DS_A/harvest"]

        body = notify.body_for(s)

        self.assertIn("Count mismatch", body)
        self.assertIn("DS_A/harvest", body)

    def test_long_gap_since_previous_run_is_flagged(self):
        """A run that never happened cannot report itself; the next one reports the hole."""
        previous = summary(run_id="20260801T100000", started="2026-08-01T10:00:00Z")

        body = notify.body_for(summary(), previous)

        self.assertIn("31.0 days ago", body)
        self.assertIn("may have been missed", body)

    def test_normal_cadence_is_not_flagged(self):
        previous = summary(run_id="20260825T100000", started="2026-08-25T10:00:00Z")

        body = notify.body_for(summary(), previous)

        self.assertIn("7.0 days ago", body)
        self.assertNotIn("may have been missed", body)

    def test_no_previous_run_is_stated(self):
        self.assertIn("No previous run on record", notify.body_for(summary()))

    def test_clean_run_omits_zero_failure_counts(self):
        body = notify.body_for(summary())

        self.assertNotIn("failed", body)
        self.assertNotIn("with failures", body)
        self.assertNotIn("ERROR log lines", body)
        self.assertNotIn("(exit 0)", body)

    def test_nonzero_failure_counts_are_shown(self):
        s = summary("failures", 1)
        s["totals"]["log_error_lines"] = 4

        body = notify.body_for(s)

        self.assertIn("(exit 1)", body)
        self.assertIn("(1 with failures)", body)
        self.assertIn("42 attempted, 2 failed", body)
        self.assertIn("ERROR log lines: 4", body)

    def test_start_time_is_shown_in_local_time(self):
        try:
            with patch.dict(os.environ, {"TZ": "America/Los_Angeles"}):
                time.tzset()
                body = notify.body_for(summary(started="2026-11-16T17:00:40Z"))
        finally:
            time.tzset()  # patch.dict has restored TZ by now

        self.assertIn("Started:  2026-11-16 09:00:40 PST", body)

    def test_relative_log_dir_from_older_summaries_is_made_absolute(self):
        body = notify.body_for(summary())

        expected = notify.PIPELINE_DIR / "logs" / RUN_ID
        self.assertIn(f"Logs: {expected}", body)

    def test_absolute_log_dir_is_left_alone(self):
        s = summary()
        s["log_dir"] = f"/data/ecco/logs/{RUN_ID}"

        self.assertIn(f"Logs: /data/ecco/logs/{RUN_ID}", notify.body_for(s))

    def test_per_dataset_activity_lists_only_datasets_with_work(self):
        def stage(attempted):
            return {"ran": True, "attempted": attempted, "failed": 0}

        datasets = [
            {
                "dataset": "BUSY_DS",
                "stages": {
                    "harvest": stage(12),
                    "transform": stage(80),
                    "aggregate": stage(6),
                },
            },
            {
                "dataset": "IDLE_DS",
                "stages": {"harvest": stage(0), "transform": stage(0)},
            },
            {
                "dataset": "UNTOUCHED_DS",
                "stages": {"harvest": {"ran": False, "consecutive_failures": 0}},
            },
        ]

        body = notify.body_for(summary(datasets=datasets))

        self.assertIn("Datasets with new data", body)
        self.assertIn("BUSY_DS  12 granules  80 transformations  6 aggregations", body)
        self.assertNotIn("IDLE_DS", body)
        self.assertNotIn("UNTOUCHED_DS", body)
        self.assertIn("(1 dataset had nothing new)", body)


class MessageTestCase(unittest.TestCase):
    def test_html_part_keeps_alignment_in_monospace(self):
        """Outlook shows text/plain in a proportional font, which breaks the columns."""
        with patch.object(notify, "setting", fake_setting()):
            msg = notify.build_message(summary())

        plain = msg.get_body(preferencelist=("plain",)).get_content()
        html_part = msg.get_body(preferencelist=("html",)).get_content()

        self.assertIn("Activity this run", plain)
        self.assertIn("<pre", html_part)
        self.assertIn("Activity this run", html_part)

    def test_html_part_escapes_the_body(self):
        s = summary("aborted", abort_reason="bad <tag> & more")
        with patch.object(notify, "setting", fake_setting()):
            msg = notify.build_message(s)

        html_part = msg.get_body(preferencelist=("html",)).get_content()
        self.assertIn("bad &lt;tag&gt; &amp; more", html_part)


class GatingTestCase(unittest.TestCase):
    def test_always_sends_on_success(self):
        with patch.object(notify, "setting", fake_setting()):
            self.assertTrue(notify.should_send(summary()))

    def test_failures_only_skips_success(self):
        with patch.object(
            notify, "setting", fake_setting({"NOTIFY_ON": "failures_only"})
        ):
            self.assertFalse(notify.should_send(summary()))

    def test_failures_only_sends_on_failure(self):
        with patch.object(
            notify, "setting", fake_setting({"NOTIFY_ON": "failures_only"})
        ):
            self.assertTrue(notify.should_send(summary("failures", 1)))
            self.assertTrue(notify.should_send(summary("aborted")))

    def test_is_configured_requires_host_from_and_to(self):
        with patch.object(notify, "setting", fake_setting()):
            self.assertTrue(notify.is_configured())
        with patch.object(notify, "setting", fake_setting({"SMTP_HOST": None})):
            self.assertFalse(notify.is_configured())
        with patch.object(notify, "setting", fake_setting({"NOTIFY_TO": []})):
            self.assertFalse(notify.is_configured())


class NotifyTestCase(unittest.TestCase):
    def test_unconfigured_is_a_logged_no_op(self):
        with (
            patch.object(notify, "setting", fake_setting({"SMTP_HOST": None})),
            patch.object(notify.run_summary, "load_summary", return_value=summary()),
            patch.object(notify.run_summary, "previous_summary", return_value=None),
            patch("smtplib.SMTP") as mock_smtp,
        ):
            self.assertEqual(notify.notify(RUN_ID), 0)

        mock_smtp.assert_not_called()

    def test_sends_when_configured(self):
        with (
            patch.object(notify, "setting", fake_setting()),
            patch.object(
                notify.run_summary, "load_summary", return_value=summary("failures", 1)
            ),
            patch.object(notify.run_summary, "previous_summary", return_value=None),
            patch("smtplib.SMTP") as mock_smtp,
        ):
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            self.assertEqual(notify.notify(RUN_ID), 0)

        mock_server.send_message.assert_called_once()
        sent = mock_server.send_message.call_args[0][0]
        self.assertIn("1 dataset with failures", sent["Subject"])
        self.assertEqual(sent["To"], "you@example.org")

    def test_starttls_and_login_when_configured(self):
        overrides = {"SMTP_USE_TLS": True, "SMTP_USER": "u", "SMTP_PASSWORD": "p"}
        with (
            patch.object(notify, "setting", fake_setting(overrides)),
            patch.object(notify.run_summary, "load_summary", return_value=summary()),
            patch.object(notify.run_summary, "previous_summary", return_value=None),
            patch("smtplib.SMTP") as mock_smtp,
        ):
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            notify.notify(RUN_ID)

        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("u", "p")

    def test_send_failure_returns_nonzero(self):
        with (
            patch.object(notify, "setting", fake_setting()),
            patch.object(notify.run_summary, "load_summary", return_value=summary()),
            patch.object(notify.run_summary, "previous_summary", return_value=None),
            patch("smtplib.SMTP", side_effect=OSError("relay down")),
        ):
            self.assertEqual(notify.notify(RUN_ID), 1)

    def test_dry_run_prints_and_never_sends(self):
        with (
            patch.object(notify, "setting", fake_setting()),
            patch.object(notify.run_summary, "load_summary", return_value=summary()),
            patch.object(notify.run_summary, "previous_summary", return_value=None),
            patch("smtplib.SMTP") as mock_smtp,
        ):
            self.assertEqual(notify.notify(RUN_ID, dry_run=True), 0)

        mock_smtp.assert_not_called()

    def test_missing_summary_returns_nonzero(self):
        with patch.object(notify.run_summary, "load_summary", return_value=None):
            self.assertEqual(notify.notify(RUN_ID), 1)

    def test_latest_resolves_the_newest_run(self):
        with (
            patch.object(notify, "setting", fake_setting()),
            patch.object(
                notify.run_summary, "latest_run_id", return_value=RUN_ID
            ) as mock_latest,
            patch.object(
                notify.run_summary, "load_summary", return_value=summary()
            ) as mock_load,
            patch.object(notify.run_summary, "previous_summary", return_value=None),
            patch("smtplib.SMTP"),
        ):
            notify.notify("latest")

        mock_latest.assert_called_once()
        mock_load.assert_called_once_with(RUN_ID)

    def test_no_runs_at_all_returns_nonzero(self):
        with patch.object(notify.run_summary, "latest_run_id", return_value=None):
            self.assertEqual(notify.notify("latest"), 1)


if __name__ == "__main__":
    unittest.main()
