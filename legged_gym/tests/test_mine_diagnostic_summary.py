"""CPU/stdlib-only coverage for old A and two-phase B diagnostic reports."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legged_gym.scripts.summarize_training_diagnostics import summarize


def diagnostic_row(iteration, phase_b=False):
    metrics = {
        "Loss/learning_rate": 3e-5,
        "Diagnostics/pre_update_kl": 4e-5,
        "Diagnostics/pre_update_value_max_abs_error": 0.,
        "Diagnostics/pre_update_reset_count": 2,
        "Diagnostics/pre_update_non_reset_count": 6,
        "Diagnostics/pre_update_kl_reset": 4e-5,
        "Diagnostics/pre_update_kl_non_reset": 4e-5,
        "Diagnostics/post_update_kl": 0.02,
        "Latent/latent_cos_to_mean_median": 0.999,
    }
    if phase_b:
        metrics.update({
            "Diagnostics/update_ppo_count": 4,
            "Diagnostics/update_estimator_count": 4,
        })
        for phase, kl in (("post_ppo", 0.3), ("estimator_phase", 0.04)):
            metrics.update({
                f"Diagnostics/{phase}_kl": kl,
                f"Diagnostics/{phase}_kl_sample_max": kl * 2,
                f"Diagnostics/{phase}_reset_count": 2,
                f"Diagnostics/{phase}_non_reset_count": 6,
                f"Diagnostics/{phase}_kl_reset": kl,
                f"Diagnostics/{phase}_kl_non_reset": kl,
            })
    return {
        "iteration": iteration,
        "metrics": metrics,
        "minibatches": [{"kl": kl, "lr_at_floor": 0} for kl in (0.01, 0.05, 0.1, 0.2)],
    }


class TestMINEDiagnosticSummary(unittest.TestCase):
    def report(self, rows, **kwargs):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "training_diagnostics.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            return summarize(path, **kwargs)

    def test_legacy_a_quantiles_strict_tail_counts_and_original_fields(self):
        row = diagnostic_row(1)
        report = self.report([row])
        self.assertAlmostEqual(report["kl_minibatch_mean"], 0.09)
        self.assertAlmostEqual(report["kl_minibatch_p95"], 0.185)
        self.assertAlmostEqual(report["kl_minibatch_p99"], 0.197)
        self.assertEqual(report["kl_minibatch_max"], 0.2)
        self.assertEqual(report["kl_minibatch_gt_0_05_count"], 2)
        self.assertEqual(report["kl_minibatch_gt_0_1_count"], 1)
        self.assertEqual(report["post_update_kl_p95"], 0.02)
        self.assertTrue(report["integrity_pass"])
        self.assertTrue(report["lr_target_pass"])
        self.assertNotIn("stage1_pass", report)
        self.assertNotIn("post_ppo_kl_mean", report)
        self.assertNotIn("estimator_phase_kl_mean", report)

    def test_post_update_statistics_use_iteration_not_minibatch_samples(self):
        rows = [diagnostic_row(1), diagnostic_row(2)]
        rows[0]["metrics"]["Diagnostics/post_update_kl"] = 0.01
        rows[1]["metrics"]["Diagnostics/post_update_kl"] = 0.11
        report = self.report(rows)
        self.assertAlmostEqual(report["post_update_kl_mean"], 0.06)
        self.assertAlmostEqual(report["post_update_kl_p95"], 0.105)
        self.assertAlmostEqual(report["post_update_kl_p99"], 0.109)
        self.assertEqual(report["post_update_kl_max"], 0.11)
        self.assertEqual(report["post_update_kl_gt_0_05_count"], 1)
        self.assertEqual(report["post_update_kl_gt_0_1_count"], 1)

    def test_b_reports_direct_estimator_distribution_not_kl_difference(self):
        report = self.report([diagnostic_row(1, phase_b=True)])
        self.assertEqual(report["post_ppo_kl_mean"], 0.3)
        self.assertEqual(report["estimator_phase_kl_mean"], 0.04)
        self.assertNotAlmostEqual(
            report["estimator_phase_kl_mean"],
            report["post_update_kl_mean"] - report["post_ppo_kl_mean"],
        )
        for phase, kl in (("post_ppo", 0.3), ("estimator_phase", 0.04)):
            for stat in ("mean", "p95", "p99", "max"):
                self.assertEqual(report[f"{phase}_kl_{stat}"], kl)
                self.assertEqual(report[f"{phase}_kl_reset_{stat}"], kl)
                self.assertEqual(report[f"{phase}_kl_non_reset_{stat}"], kl)
            self.assertEqual(report[f"{phase}_reset_samples"], 2)
            self.assertEqual(report[f"{phase}_non_reset_samples"], 6)
            self.assertEqual(report[f"{phase}_kl_sample_max"], kl * 2)

    def test_b_missing_group_is_not_reported_as_zero(self):
        row = diagnostic_row(1, phase_b=True)
        for phase in ("pre_update", "post_ppo", "estimator_phase"):
            row["metrics"][f"Diagnostics/{phase}_reset_count"] = 0
            del row["metrics"][f"Diagnostics/{phase}_kl_reset"]
        report = self.report([row])
        self.assertFalse(report["integrity_pass"])
        self.assertEqual(report["estimator_phase_reset_samples"], 0)
        self.assertIsNone(report["estimator_phase_kl_reset_mean"])
        self.assertIsNone(report["post_ppo_kl_reset_p99"])
        json.dumps(report, allow_nan=False)

    def test_b_incomplete_phase_metrics_are_rejected(self):
        complete = diagnostic_row(1, phase_b=True)
        phase_fields = [key for key in complete["metrics"] if key.startswith(
            ("Diagnostics/post_ppo_", "Diagnostics/estimator_phase_")
        )]
        for field in phase_fields:
            with self.subTest(field=field):
                row = copy.deepcopy(complete)
                del row["metrics"][field]
                with self.assertRaisesRegex(ValueError, "Incomplete B"):
                    self.report([row])

    def test_b_phase_counts_must_match_rollout(self):
        for count in (-1, 0.5, 3):
            with self.subTest(count=count):
                row = diagnostic_row(1, phase_b=True)
                row["metrics"]["Diagnostics/estimator_phase_reset_count"] = count
                with self.assertRaisesRegex(ValueError, "count"):
                    self.report([row])

    def test_update_counts_match_each_iterations_minibatches(self):
        for phase_b in (False, True):
            for field in ("Diagnostics/update_ppo_count", "Diagnostics/update_estimator_count"):
                with self.subTest(phase_b=phase_b, field=field):
                    row = diagnostic_row(1, phase_b=phase_b)
                    row["metrics"].update({
                        "Diagnostics/update_ppo_count": 4,
                        "Diagnostics/update_estimator_count": 4,
                    })
                    self.report([row])
                    row["metrics"][field] = 3
                    with self.assertRaisesRegex(ValueError, "mini-batch count"):
                        self.report([row])
                    del row["metrics"][field]
                    with self.assertRaisesRegex(ValueError, "Incomplete update-count"):
                        self.report([row])

    def test_mixed_a_b_window_rejected_but_selected_homogeneous_window_works(self):
        rows = [diagnostic_row(1), diagnostic_row(2, phase_b=True)]
        with self.assertRaisesRegex(ValueError, "Mixed A/B"):
            self.report(rows)
        self.assertEqual(self.report(rows, start=2, end=2)["iterations"], 1)

    def test_one_empty_iteration_is_rejected_even_if_others_have_batches(self):
        rows = [diagnostic_row(1), diagnostic_row(2)]
        rows[1]["minibatches"] = []
        with self.assertRaisesRegex(ValueError, "No mini-batch diagnostics at iteration 2"):
            self.report(rows)

    def test_nonfinite_metrics_or_batches_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for destination in ("metrics", "minibatches"):
                with self.subTest(value=value, destination=destination):
                    row = diagnostic_row(1)
                    if destination == "metrics":
                        row["metrics"]["Diagnostics/post_update_kl"] = value
                    else:
                        row["minibatches"][0]["kl"] = value
                    with self.assertRaisesRegex(ValueError, "Nonfinite"):
                        self.report([row])

    def test_iteration_window_boundaries(self):
        rows = [diagnostic_row(1), diagnostic_row(2)]
        for kwargs in ({"start": 0}, {"start": 2, "end": 1}, {"start": 3}, {"end": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.report(rows, **kwargs)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.report([diagnostic_row(1), diagnostic_row(1)])
        with self.assertRaisesRegex(ValueError, "Missing iterations"):
            self.report([diagnostic_row(1), diagnostic_row(3)])


if __name__ == "__main__":
    unittest.main()
