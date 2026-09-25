"""
Unit tests for run_pipeline's exit codes and stage-result capture.

The point of these: every run used to exit 0, so cron could not tell a clean run from
one where every dataset failed, or from one that aborted before doing anything.
"""

import unittest
from unittest.mock import MagicMock, patch

import run_pipeline
from monitoring import run_summary
from utils.pipeline_utils.init_pipeline import PipelinePreconditionError


class Args:
    menu = False
    dataset = "DS_A"
    step = "harvest"
    grids_to_use = False
    multiprocesses = 4
    log_level = "INFO"


class ExitCodeTestCase(unittest.TestCase):
    """main() maps the run's outcome onto a distinct process exit code."""

    def run_main(self, summary=None, start_side_effect=None, init_side_effect=None):
        """
        Drive main() with init, the stages, and summary recording stubbed.

        record_run is stubbed to echo a supplied summary so these tests pin the
        exit-code plumbing rather than re-testing summary construction.
        """
        recorded = {}

        def fake_record_run(
            results, args=None, started_dt="", outcome=None, abort_reason=None
        ):
            recorded["results"] = results
            recorded["outcome"] = outcome
            recorded["abort_reason"] = abort_reason
            if summary is not None:
                return summary
            code = {
                "aborted": run_summary.EXIT_ABORTED,
                "crashed": run_summary.EXIT_CRASHED,
            }.get(outcome, run_summary.EXIT_OK)
            return {
                "run_id": "20260901T100000",
                "outcome": outcome or "ok",
                "exit_code": code,
                "totals": {"datasets": 1, "datasets_with_failures": 0},
            }

        with (
            patch.object(run_pipeline, "create_parser", return_value=Args()),
            patch.object(
                run_pipeline.init_pipeline,
                "init_pipeline",
                side_effect=init_side_effect,
            ),
            patch.object(
                run_pipeline,
                "start_pipeline",
                side_effect=start_side_effect,
                return_value=[],
            ),
            patch.object(
                run_pipeline.run_summary, "record_run", side_effect=fake_record_run
            ),
        ):
            code = run_pipeline.main()
        return code, recorded

    def test_clean_run_exits_zero(self):
        code, recorded = self.run_main()

        self.assertEqual(code, run_summary.EXIT_OK)
        self.assertIsNone(recorded["outcome"])

    def test_dataset_failures_exit_one(self):
        summary = {
            "run_id": "R",
            "outcome": "failures",
            "exit_code": run_summary.EXIT_FAILURES,
            "totals": {"datasets": 3, "datasets_with_failures": 2},
        }
        code, _ = self.run_main(summary=summary)

        self.assertEqual(code, run_summary.EXIT_FAILURES)

    def test_precondition_abort_exits_two_and_is_recorded(self):
        """
        Solr down / missing credentials used to exit 0 with no trace at all. The summary
        is the only record such a run leaves, so it has to be written even here.
        """
        code, recorded = self.run_main(
            init_side_effect=PipelinePreconditionError("Solr unreachable at localhost")
        )

        self.assertEqual(code, run_summary.EXIT_ABORTED)
        self.assertEqual(recorded["outcome"], "aborted")
        self.assertIn("Solr unreachable", recorded["abort_reason"])

    def test_unexpected_error_exits_three(self):
        code, recorded = self.run_main(start_side_effect=RuntimeError("boom"))

        self.assertEqual(code, run_summary.EXIT_CRASHED)
        self.assertEqual(recorded["outcome"], "crashed")


class StageResultTestCase(unittest.TestCase):
    """The run_* functions report what happened instead of only logging it."""

    def setUp(self):
        self.config = {"harvester_type": "cmr", "ds_name": "DS_A"}

    def test_successful_harvest_returns_status(self):
        harvester_module = MagicMock()
        harvester_module.harvester.return_value = "All granules successfully harvested"

        with (
            patch("builtins.open", MagicMock()),
            patch.object(run_pipeline.yaml, "load", return_value=self.config),
            patch.object(
                run_pipeline.importlib, "import_module", return_value=harvester_module
            ),
        ):
            results = run_pipeline.run_harvester(["DS_A"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].dataset, "DS_A")
        self.assertEqual(results[0].stage, "harvest")
        self.assertEqual(results[0].status_text, "All granules successfully harvested")
        self.assertFalse(results[0].raised)

    def test_harvest_exception_is_captured_with_traceback(self):
        """
        The traceback is what distinguishes "the source returned 503" from "our parser
        broke", and it previously existed only in pipeline.log.
        """
        harvester_module = MagicMock()
        harvester_module.harvester.side_effect = ValueError("bad response")

        with (
            patch("builtins.open", MagicMock()),
            patch.object(run_pipeline.yaml, "load", return_value=self.config),
            patch.object(
                run_pipeline.importlib, "import_module", return_value=harvester_module
            ),
        ):
            results = run_pipeline.run_harvester(["DS_A"])

        self.assertTrue(results[0].raised)
        self.assertEqual(results[0].exc_type, "ValueError")
        self.assertEqual(results[0].exc_message, "bad response")
        self.assertIn("ValueError", results[0].traceback_tail)

    def test_one_failing_dataset_does_not_stop_the_others(self):
        harvester_module = MagicMock()
        harvester_module.harvester.side_effect = [ValueError("bad"), "All good"]

        with (
            patch("builtins.open", MagicMock()),
            patch.object(run_pipeline.yaml, "load", return_value=self.config),
            patch.object(
                run_pipeline.importlib, "import_module", return_value=harvester_module
            ),
        ):
            results = run_pipeline.run_harvester(["DS_A", "DS_B"])

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].raised)
        self.assertFalse(results[1].raised)

    def test_missing_harvester_type_is_a_captured_failure(self):
        with (
            patch("builtins.open", MagicMock()),
            patch.object(run_pipeline.yaml, "load", return_value={"ds_name": "DS_A"}),
        ):
            results = run_pipeline.run_harvester(["DS_A"])

        self.assertEqual(results[0].exc_type, "ValueError")

    def test_aggregation_collects_worker_errors(self):
        """
        Worker markers travel to the summary: an aggregation worker that dies before
        writing its Solr doc leaves no other trace of the failure.
        """
        factory = MagicMock()
        factory.start_factory.return_value = "All aggregations successful"
        factory.worker_errors = [{"job": "GRID1, 2020, ssha", "error": "killed"}]

        with (
            patch("builtins.open", MagicMock()),
            patch.object(run_pipeline.yaml, "load", return_value=self.config),
            patch.object(run_pipeline, "AgJobFactory", return_value=factory),
        ):
            results = run_pipeline.run_aggregation(["DS_A"])

        self.assertEqual(results[0].stage, "aggregate")
        self.assertEqual(len(results[0].worker_errors), 1)

    def test_transformation_returns_status(self):
        with (
            patch("builtins.open", MagicMock()),
            patch.object(run_pipeline.yaml, "load", return_value=self.config),
            patch.object(run_pipeline, "TxJobFactory") as mock_factory,
        ):
            mock_factory.return_value.start_factory.return_value = (
                "All transformations successful"
            )
            results = run_pipeline.run_transformation(["DS_A"])

        self.assertEqual(results[0].stage, "transform")
        self.assertEqual(results[0].status_text, "All transformations successful")


class StartPipelineTestCase(unittest.TestCase):
    """start_pipeline aggregates results across the stages it runs."""

    def make_args(self, step):
        args = Args()
        args.step = step
        return args

    def test_single_step_runs_only_that_stage(self):
        with (
            patch.object(run_pipeline, "run_harvester", return_value=["h"]) as mock_h,
            patch.object(
                run_pipeline, "run_transformation", return_value=["t"]
            ) as mock_t,
            patch.object(run_pipeline, "run_aggregation", return_value=["a"]) as mock_a,
        ):
            results = run_pipeline.start_pipeline(self.make_args("harvest"))

        self.assertEqual(results, ["h"])
        mock_h.assert_called_once()
        mock_t.assert_not_called()
        mock_a.assert_not_called()

    def test_all_runs_every_stage_and_concatenates(self):
        with (
            patch.object(run_pipeline, "run_harvester", return_value=["h"]),
            patch.object(run_pipeline, "run_transformation", return_value=["t"]),
            patch.object(run_pipeline, "run_aggregation", return_value=["a"]),
        ):
            results = run_pipeline.start_pipeline(self.make_args("all"))

        self.assertEqual(results, ["h", "t", "a"])


if __name__ == "__main__":
    unittest.main()
