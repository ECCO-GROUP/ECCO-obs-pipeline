"""
Unit tests for monitoring.run_summary. All Solr calls are mocked.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from monitoring import run_summary
from monitoring.run_summary import StageResult

STARTED = "2026-09-01T10:00:00Z"
RUN_ID = "20260901T100000"


def ok_result(dataset="DS_A", stage="harvest"):
    return StageResult(
        dataset=dataset,
        stage=stage,
        status_text="All granules successfully harvested",
        duration_s=12.3,
    )


def raised_result(dataset="DS_A", stage="harvest"):
    return StageResult(
        dataset=dataset,
        stage=stage,
        exc_type="HTTPError",
        exc_message="503 Server Error",
        traceback_tail="Traceback...\nHTTPError: 503 Server Error\n",
    )


class BuildSummaryTestCase(unittest.TestCase):
    """Summary construction from stage results plus mocked Solr counts."""

    def build(self, results, counts=(0, 0), by_run_id=None, **kwargs):
        """
        Build a summary with Solr counting stubbed.

        `counts` is the (attempted, failed) the timestamp window reports; `by_run_id`
        defaults to the same, which is the no-divergence case.
        """
        if by_run_id is None:
            by_run_id = counts
        with (
            patch.object(run_summary.solr_utils, "commit_solr"),
            patch.object(
                run_summary,
                "previous_summary",
                return_value=kwargs.pop("previous", None),
            ),
            patch.object(run_summary, "_count_window", return_value=counts),
            patch.object(run_summary, "_count_by_run_id", return_value=by_run_id),
            patch.object(run_summary, "_error_samples", return_value=[]),
            patch.object(
                run_summary, "scan_log_errors", return_value={"count": 0, "samples": []}
            ),
            patch.object(run_summary, "_git_commit", return_value="abc1234"),
        ):
            return run_summary.build_summary(
                results, run_id=RUN_ID, started_dt=STARTED, **kwargs
            )

    def test_clean_run_is_ok_and_exits_zero(self):
        summary = self.build([ok_result()], counts=(10, 0))

        self.assertEqual(summary["outcome"], "ok")
        self.assertEqual(summary["exit_code"], run_summary.EXIT_OK)
        self.assertEqual(summary["totals"]["granules_attempted"], 10)
        self.assertEqual(summary["totals"]["datasets_with_failures"], 0)

    def test_failed_docs_make_the_run_fail(self):
        summary = self.build([ok_result()], counts=(10, 2))

        self.assertEqual(summary["outcome"], "failures")
        self.assertEqual(summary["exit_code"], run_summary.EXIT_FAILURES)
        self.assertEqual(summary["totals"]["granules_failed"], 2)
        self.assertEqual(summary["totals"]["failed_datasets"], ["DS_A"])

    def test_exception_is_captured_with_traceback(self):
        summary = self.build([raised_result()], counts=(0, 0))

        stage = summary["datasets"][0]["stages"]["harvest"]
        self.assertEqual(summary["exit_code"], run_summary.EXIT_FAILURES)
        self.assertEqual(stage["exception"]["type"], "HTTPError")
        self.assertIn("503", stage["exception"]["traceback_tail"])

    def test_worker_errors_alone_make_a_stage_fail(self):
        """
        An aggregation worker can die before writing its Solr doc, so the Solr counts
        read clean while the job did not run. The marker has to be enough on its own.
        """
        result = StageResult(
            dataset="DS_A",
            stage="aggregate",
            worker_errors=[{"job": "GRID1, 2020, ssha", "error": "killed"}],
        )
        summary = self.build([result], counts=(5, 0))

        self.assertEqual(summary["outcome"], "failures")
        self.assertEqual(
            summary["datasets"][0]["stages"]["aggregate"]["outcome"], "failures"
        )

    def test_log_errors_alone_do_not_fail_the_run(self):
        """
        Recoverable conditions are logged at ERROR level in places, so letting log lines
        drive the exit code would make red the normal state. They are reported only.
        """
        with (
            patch.object(run_summary.solr_utils, "commit_solr"),
            patch.object(run_summary, "previous_summary", return_value=None),
            patch.object(run_summary, "_count_window", return_value=(4, 0)),
            patch.object(run_summary, "_count_by_run_id", return_value=(4, 0)),
            patch.object(run_summary, "_error_samples", return_value=[]),
            patch.object(
                run_summary,
                "scan_log_errors",
                return_value={"count": 7, "samples": ["[ERROR] x"]},
            ),
            patch.object(run_summary, "_git_commit", return_value=""),
        ):
            summary = run_summary.build_summary(
                [ok_result()], run_id=RUN_ID, started_dt=STARTED
            )

        self.assertEqual(summary["outcome"], "ok")
        self.assertEqual(summary["totals"]["log_error_lines"], 7)

    def test_aborted_run_exits_two(self):
        summary = self.build(
            [], outcome="aborted", abort_reason="Solr unreachable at localhost"
        )

        self.assertEqual(summary["exit_code"], run_summary.EXIT_ABORTED)
        self.assertEqual(summary["abort_reason"], "Solr unreachable at localhost")
        self.assertEqual(summary["datasets"], [])

    def test_crashed_run_exits_three(self):
        summary = self.build([], outcome="crashed")

        self.assertEqual(summary["exit_code"], run_summary.EXIT_CRASHED)

    def test_invocation_is_recorded(self):
        class Args:
            step = "harvest"
            dataset = "DS_A"
            grids_to_use = ["ECCO_llc90"]
            multiprocesses = 8

        summary = self.build([ok_result()], counts=(1, 0), args=Args())

        self.assertEqual(summary["invocation"]["step"], "harvest")
        self.assertEqual(summary["invocation"]["dataset"], "DS_A")
        self.assertEqual(summary["invocation"]["grids"], ["ECCO_llc90"])
        self.assertEqual(summary["invocation"]["git_commit"], "abc1234")

    def test_counting_failure_preserves_exception_detail(self):
        """A Solr problem while counting must not lose the exception already captured."""
        with (
            patch.object(run_summary.solr_utils, "commit_solr"),
            patch.object(run_summary, "previous_summary", return_value=None),
            patch.object(
                run_summary, "_count_window", side_effect=RuntimeError("solr down")
            ),
            patch.object(
                run_summary, "scan_log_errors", return_value={"count": 0, "samples": []}
            ),
            patch.object(run_summary, "_git_commit", return_value=""),
        ):
            summary = run_summary.build_summary(
                [raised_result()], run_id=RUN_ID, started_dt=STARTED
            )

        stage = summary["datasets"][0]["stages"]["harvest"]
        self.assertEqual(stage["exception"]["type"], "HTTPError")
        self.assertEqual(summary["exit_code"], run_summary.EXIT_FAILURES)


class DivergenceTestCase(unittest.TestCase):
    """The two count sources are compared in one direction only."""

    def test_timestamp_exceeding_run_id_is_flagged(self):
        """
        Docs written in this run's window without its stamp means either a write path
        with no stamp (which would under-report failures) or an overlapping run.
        """
        self.assertTrue(
            run_summary._divergence({"attempted": 10, "attempted_by_run_id": 4})
        )

    def test_run_id_exceeding_timestamp_is_not_flagged(self):
        """
        The reconstruction paths write a doc now but backdate its activity timestamp to
        the output file's mtime, so this direction is expected and benign.
        """
        self.assertFalse(
            run_summary._divergence({"attempted": 4, "attempted_by_run_id": 10})
        )

    def test_agreement_is_not_flagged(self):
        self.assertFalse(
            run_summary._divergence({"attempted": 7, "attempted_by_run_id": 7})
        )


class StreakTestCase(unittest.TestCase):
    """Consecutive-failure tracking across runs."""

    def build(self, results, counts, previous, run_id=RUN_ID):
        with (
            patch.object(run_summary.solr_utils, "commit_solr"),
            patch.object(run_summary, "previous_summary", return_value=previous),
            patch.object(run_summary, "_count_window", return_value=counts),
            patch.object(run_summary, "_count_by_run_id", return_value=counts),
            patch.object(run_summary, "_error_samples", return_value=[]),
            patch.object(
                run_summary, "scan_log_errors", return_value={"count": 0, "samples": []}
            ),
            patch.object(run_summary, "_git_commit", return_value=""),
        ):
            return run_summary.build_summary(results, run_id=run_id, started_dt=STARTED)

    def summary_with_streak(self, dataset, stage, streak, first_run):
        return {
            "run_id": "20260825T100000",
            "started_dt": "2026-08-25T10:00:00Z",
            "datasets": [
                {
                    "dataset": dataset,
                    "stages": {
                        stage: {
                            "ran": True,
                            "consecutive_failures": streak,
                            "first_failed_run_id": first_run,
                        }
                    },
                }
            ],
        }

    def test_first_failure_starts_the_streak(self):
        summary = self.build([ok_result()], (5, 1), previous=None)

        stage = summary["datasets"][0]["stages"]["harvest"]
        self.assertEqual(stage["consecutive_failures"], 1)
        self.assertEqual(stage["first_failed_run_id"], RUN_ID)

    def test_repeat_failure_increments_and_keeps_first_run(self):
        previous = self.summary_with_streak("DS_A", "harvest", 2, "20260811T100000")

        summary = self.build([ok_result()], (5, 1), previous=previous)

        stage = summary["datasets"][0]["stages"]["harvest"]
        self.assertEqual(stage["consecutive_failures"], 3)
        self.assertEqual(stage["first_failed_run_id"], "20260811T100000")

    def test_success_resets_the_streak(self):
        previous = self.summary_with_streak("DS_A", "harvest", 4, "20260804T100000")

        summary = self.build([ok_result()], (5, 0), previous=previous)

        stage = summary["datasets"][0]["stages"]["harvest"]
        self.assertEqual(stage["consecutive_failures"], 0)
        self.assertIsNone(stage["first_failed_run_id"])

    def test_partial_run_carries_untouched_streaks_forward(self):
        """
        A --dataset X --step harvest run must not zero out every other dataset's streak:
        the next run reads its previous state from this summary, so a dropped entry
        silently forgets that something has been broken for weeks.
        """
        previous = self.summary_with_streak(
            "DS_OTHER", "transform", 3, "20260804T100000"
        )

        summary = self.build([ok_result("DS_A", "harvest")], (1, 0), previous=previous)

        carried = {d["dataset"]: d["stages"] for d in summary["datasets"]}
        self.assertIn("DS_OTHER", carried)
        self.assertEqual(carried["DS_OTHER"]["transform"]["consecutive_failures"], 3)
        self.assertFalse(carried["DS_OTHER"]["transform"]["ran"])
        # A carried-forward failure is not this run's failure.
        self.assertEqual(summary["outcome"], "ok")


class ScanLogErrorsTestCase(unittest.TestCase):
    """Log scanning is the backstop for failures that never reach Solr."""

    def test_counts_error_and_critical_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pipeline.log"), "w") as f:
                f.write("[INFO] 2026-09-01 (pipeline) - starting\n")
                f.write("[ERROR] 2026-09-01 (pipeline) - something broke\n")
                f.write("[CRITICAL] 2026-09-01 (pipeline) - worse\n")

            result = run_summary.scan_log_errors(tmp)

        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["samples"]), 2)

    def test_scans_stage_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "pipeline.log"), "w").close()
            stage_dir = os.path.join(tmp, "ag_DS_A")
            os.makedirs(stage_dir)
            with open(os.path.join(stage_dir, "1234.log"), "w") as f:
                f.write("[ERROR] JOB FAILED: killed\n")

            result = run_summary.scan_log_errors(tmp, ["ag_DS_A"])

        self.assertEqual(result["count"], 1)
        self.assertIn("JOB FAILED", result["samples"][0])

    def test_missing_log_dir_is_not_an_error(self):
        result = run_summary.scan_log_errors("/nonexistent/logs/dir")

        self.assertEqual(result, {"count": 0, "samples": []})


class PersistenceTestCase(unittest.TestCase):
    """JSON artifact and Solr mirror."""

    def make_summary(self, run_id=RUN_ID, log_dir=None):
        return {
            "schema_version": 1,
            "run_id": run_id,
            "log_dir": log_dir or f"logs/{run_id}",
            "started_dt": STARTED,
            "ended_dt": "2026-09-01T11:00:00Z",
            "duration_s": 3600,
            "exit_code": 1,
            "outcome": "failures",
            "abort_reason": None,
            "totals": {
                "datasets": 2,
                "datasets_with_failures": 1,
                "granules_failed": 3,
            },
            "datasets": [],
        }

    def test_write_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.make_summary(log_dir=os.path.join(tmp, "logs", RUN_ID))

            path = run_summary.write_summary(summary)

            with open(path) as f:
                self.assertEqual(json.load(f)["run_id"], RUN_ID)

    def test_mirror_builds_a_pipeline_run_doc(self):
        with patch.object(run_summary.solr_utils, "solr_update") as mock_update:
            self.assertTrue(run_summary.mirror_to_solr(self.make_summary()))

        doc = mock_update.call_args[0][0][0]
        self.assertEqual(doc["type_s"], "pipeline_run")
        self.assertEqual(doc["id"], f"run-{RUN_ID}")
        self.assertEqual(doc["outcome_s"], "failures")
        self.assertEqual(doc["exit_code_i"], 1)
        self.assertEqual(json.loads(doc["summary_json_s"])["run_id"], RUN_ID)

    def test_mirror_failure_is_swallowed(self):
        """
        "Solr is broken" is one of the things this mechanism exists to report, so a
        failed mirror must not propagate and change the run's exit code.
        """
        with patch.object(
            run_summary.solr_utils, "solr_update", side_effect=RuntimeError("down")
        ):
            self.assertFalse(run_summary.mirror_to_solr(self.make_summary()))

    def test_load_missing_summary_returns_none(self):
        self.assertIsNone(run_summary.load_summary("20200101T000000"))


class PreviousSummaryTestCase(unittest.TestCase):
    """Previous-run lookup, including the Solr-unreachable fallback."""

    def test_prefers_newest_solr_doc_excluding_this_run(self):
        blob = json.dumps({"run_id": "20260825T100000", "outcome": "ok"})

        def fake_query(fq, fl="", rows=None):
            if "type_s:pipeline_run" in fq:
                return [
                    {
                        "id": "run-20260818T100000",
                        "run_id_s": "20260818T100000",
                        "started_dt": "2026-08-18T10:00:00Z",
                    },
                    {
                        "id": "run-20260825T100000",
                        "run_id_s": "20260825T100000",
                        "started_dt": "2026-08-25T10:00:00Z",
                    },
                    {"id": f"run-{RUN_ID}", "run_id_s": RUN_ID, "started_dt": STARTED},
                ]
            return [{"summary_json_s": blob}]

        with patch.object(run_summary.solr_utils, "solr_query", side_effect=fake_query):
            result = run_summary.previous_summary(RUN_ID)

        self.assertEqual(result["run_id"], "20260825T100000")

    def test_falls_back_to_log_dirs_when_solr_is_down(self):
        with (
            patch.object(
                run_summary.solr_utils, "solr_query", side_effect=RuntimeError("down")
            ),
            patch.object(
                run_summary,
                "_run_ids_from_disk",
                return_value=["20260825T100000", RUN_ID],
            ),
            patch.object(
                run_summary, "load_summary", return_value={"run_id": "20260825T100000"}
            ),
        ):
            result = run_summary.previous_summary(RUN_ID)

        self.assertEqual(result["run_id"], "20260825T100000")

    def test_no_prior_run_returns_none(self):
        with (
            patch.object(run_summary.solr_utils, "solr_query", return_value=[]),
            patch.object(run_summary, "_run_ids_from_disk", return_value=[RUN_ID]),
        ):
            self.assertIsNone(run_summary.previous_summary(RUN_ID))


if __name__ == "__main__":
    unittest.main()
