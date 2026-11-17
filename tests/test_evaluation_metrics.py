# Adapted from wheel_gym_CQ/tests/test_evaluation_metrics.py
"""CPU checks that raw captures reconstruct formulas offline; no simulator."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from legged_gym.evaluation.metrics import (
    CsvEvaluationWriter, VectorEpisodeMetrics, summarize_episodes,
    EPISODE_VALUE_FIELDS, SUMMARY_FIELDS, physics_fields,
)


def start(metrics, ids=None, positions=None, velocities=None, masses=None, metadata=None):
    ids = list(range(metrics.num_envs)) if ids is None else ids
    n = len(ids)
    metrics.start(ids, position_world=torch.zeros(n, 3) if positions is None else positions,
                  velocity_world=torch.zeros(n, 3) if velocities is None else velocities,
                  goal_direction_xy=torch.tensor([[1., 0.]] * n),
                  mass_kg=torch.ones(n) if masses is None else masses,
                  gravity_m_s2=10., metadata=metadata)


def step(metrics, position, velocity=None, body=None, command=None, angular=None, dt=1., **kwargs):
    zeros = torch.zeros(metrics.num_envs, 3)
    metrics.control_step(position, zeros if velocity is None else velocity,
                         zeros if body is None else body, zeros if angular is None else angular,
                         zeros if command is None else command, dt, **kwargs)


def offline_work(rows, num_joints):
    """These reductions exist only in tests, never in the raw collector."""
    work = [0., 0., 0.]
    for row in rows:
        for j in range(num_joints):
            torque = row["torque_j%02d_nm" % j]
            p0 = torque * row["joint_vel_before_j%02d_radps" % j]
            p1 = torque * row["joint_vel_after_j%02d_radps" % j]
            for k, transform in enumerate((lambda p: p, abs, lambda p: max(p, 0.))):
                work[k] += .5 * row["dt_s"] * (transform(p0) + transform(p1))
    return work


class EvaluationMetricsTests(unittest.TestCase):
    def assert_raw_only(self, row):
        for key in row:
            self.assertFalse(key.startswith(("energy_", "power_", "mean_", "cot_", "tracking_", "vir_")), key)
            self.assertNotIn(key, ("success_rate", "success_rate_started", "velocity_instability_rate",
                                   "path_xy_m", "path_3d_m", "progress_m", "warmup_s"))

    def test_joint_endpoints_reconstruct_net_absolute_positive_work(self):
        physics = []
        metrics = VectorEpisodeMetrics(2, "cpu", physics_sink=physics.extend, num_joints=2)
        start(metrics, ids=[0])
        # Powers [2,-6] -> [6,-6], dt=.5: net=-1, absolute=5, positive=2 J.
        metrics.physics_step([[2., -3.], [100., 100.]], [[1., 2.], [100., 100.]],
                             [[3., 2.], [100., 100.]], .5)
        step(metrics, [[1., 0., 0.], [50., 0., 0.]], dt=.5)
        row = metrics.finish([0], "success")[0]
        self.assertEqual(len(physics), 1)
        self.assertEqual(offline_work(physics, 2), [-1., 5., 2.])
        self.assertEqual(physics[0]["torque_j01_nm"], -3.)
        self.assertEqual(row["physics_sample_count"], 1)
        self.assertTrue(row["has_physics_samples"])
        self.assertEqual(metrics.physics_elapsed_s[1], 0.)
        self.assertFalse(hasattr(metrics, "energy_j"))
        self.assert_raw_only(row)
        self.assert_raw_only(physics[0])
        start(metrics, ids=[0])
        physics.clear()
        metrics.physics_step([[2., 0.], [0., 0.]], [[-1., 0.], [0., 0.]], [[1., 0.], [0., 0.]], 1.)
        self.assertEqual(offline_work(physics, 2), [0., 2., 1.])
        self.assertEqual(physics[0]["time_start_s"], 0.)
        self.assertEqual(physics[0]["physics_step"], 1)

    def test_raw_path_progress_mass_and_gravity_reconstruct_cot_offline(self):
        physics, traces = [], []
        metrics = VectorEpisodeMetrics(1, "cpu", physics_sink=physics.extend, trajectory_sink=traces.extend)
        start(metrics, masses=[2.])
        for x in (2., 1.):
            metrics.physics_step([[4.]], [[2.]], [[2.]], 1.)
            step(metrics, [[x, 0., 0.]])
        row = metrics.finish([0], "failure", "fall")[0]
        path = sum(math.hypot(sample["world_x_m"] - sample["previous_world_x_m"],
                              sample["world_y_m"] - sample["previous_world_y_m"]) for sample in traces)
        progress = ((row["end_world_x_m"] - row["start_world_x_m"]) * row["goal_direction_x"]
                    + (row["end_world_y_m"] - row["start_world_y_m"]) * row["goal_direction_y"])
        energy = offline_work(physics, 1)[0]
        self.assertEqual((path, progress, energy), (3., 1., 16.))
        self.assertAlmostEqual(energy / (row["mass_kg"] * row["gravity_m_s2"] * path), 16. / 60.)
        self.assertAlmostEqual(energy / (row["mass_kg"] * row["gravity_m_s2"] * progress), .8)
        self.assertEqual(row["reason"], "fall")
        for sample in traces + [row]:
            self.assert_raw_only(sample)

    def test_world_differences_full_state_and_offline_tracking_and_thresholds(self):
        traces = []
        metrics = VectorEpisodeMetrics(1, "cpu", trajectory_sink=traces.extend)
        start(metrics, velocities=[[100., 0., 0.]])
        step(metrics, [[1., 2., 3.]], velocity=[[100.5, 1., 2.]], body=[[4., 6., 8.]],
             command=[[1., 2., 1.]], angular=[[5., 6., 3.]], dt=.25,
             orientation_world=[[0., 0., 0., 1.]], angular_velocity_world=[[7., 8., 9.]])
        step(metrics, [[2., 3., 4.]], velocity=[[100.5, 1., 2.]], body=[[1., 2., 0.]],
             command=[[1., 2., 1.]], angular=[[0., 0., 1.]], dt=.75)
        row = metrics.finish([0], "timeout")[0]
        first, second = traces
        self.assertEqual([first["previous_world_" + axis + "_m"] for axis in "xyz"], [0., 0., 0.])
        self.assertEqual([first["previous_world_v" + axis + "_mps"] for axis in "xyz"], [100., 0., 0.])
        self.assertEqual([first["world_a" + axis + "_mps2"] for axis in "xyz"], [2., 4., 8.])
        self.assertAlmostEqual(first["acceleration_xy_mps2"], math.sqrt(20.))
        self.assertAlmostEqual(first["acceleration_xyz_mps2"], math.sqrt(84.))
        self.assertEqual(first["body_vz_mps"], 8.)
        self.assertEqual([first["body_w" + axis + "_radps"] for axis in "xyz"], [5., 6., 3.])
        self.assertEqual([first["world_w" + axis + "_radps"] for axis in "xyz"], [7., 8., 9.])
        self.assertEqual(first["world_qw"], 1.)
        self.assertIsNone(second["world_qw"])
        self.assertIsNone(second["world_wx_radps"])
        self.assertEqual(second["acceleration_xyz_mps2"], 0.)
        rates = [sum(sample["acceleration_xy_mps2"] >= threshold for sample in traces) / len(traces)
                 for threshold in (2., 5.)]
        self.assertEqual(rates, [.5, 0.])
        duration = sum(sample["control_dt_s"] for sample in traces)
        squared = [sum(sample["control_dt_s"] * (sample[actual] - sample[desired]) ** 2 for sample in traces) / duration
                   for actual, desired in (("body_vx_mps", "command_vx_mps"), ("body_vy_mps", "command_vy_mps"),
                                            ("body_wz_radps", "command_wz_radps"))]
        self.assertEqual([math.sqrt(value) for value in squared], [1.5, 2., 1.])
        self.assertEqual(math.sqrt(squared[0] + squared[1]), 2.5)
        self.assertEqual(row["control_sample_count"], 2)
        self.assert_raw_only(row)

    def test_substep_indices_and_dual_clocks_align_with_the_window(self):
        physics, traces = [], []
        metrics = VectorEpisodeMetrics(1, "cpu", physics_sink=physics.extend, trajectory_sink=traces.extend)
        start(metrics, metadata=[{"entry_time_s": 4., "trial_elapsed_s": 4., "terrain": "flat",
                                 "entry_wheel_bounds_xy": [[0., 1., 2., 3.]] * 4}])
        for _ in range(2):
            for _ in range(4):
                metrics.physics_step([[2.]], [[1.]], [[3.]], .005)
            step(metrics, [[.1, 0., 0.]], dt=.02)
        for i, sample in enumerate(physics):
            self.assertEqual((sample["physics_step"], sample["control_step"], sample["substep_index"]),
                             (i + 1, i // 4 + 1, i % 4 + 1))
            self.assertAlmostEqual(sample["time_start_s"], i * .005)
            self.assertAlmostEqual(sample["time_end_s"], (i + 1) * .005)
            self.assertAlmostEqual(sample["trial_time_start_s"], 4. + i * .005)
            self.assertAlmostEqual(sample["trial_time_end_s"], 4. + (i + 1) * .005)
            self.assertNotIn("trial_elapsed_s", sample)
            self.assertNotIn("entry_wheel_bounds_xy", sample)
        self.assertEqual(traces[0]["time_start_s"], 0.)
        self.assertEqual(traces[1]["time_start_s"], .02)
        row = metrics.finish([0], "success")[0]
        self.assertAlmostEqual(row["elapsed_s"], row["physics_elapsed_s"])
        self.assertEqual(row["physics_sample_count"], 8)

    def test_vector_masks_preserve_baselines_without_recording_excluded_intervals(self):
        physics, traces = [], []
        metrics = VectorEpisodeMetrics(2, "cpu", physics_sink=physics.extend, trajectory_sink=traces.extend)
        start(metrics)
        metrics.physics_step([[2.], [2.]], [[1.], [1.]], [[1.], [1.]], 1., sample_mask=[False, True])
        step(metrics, [[100., 0., 0.], [1., 0., 0.]], velocity=[[100., 0., 0.], [0., 0., 0.]],
             sample_mask=[False, True])
        metrics.physics_step([[2.], [2.]], [[1.], [1.]], [[1.], [1.]], 1., sample_mask=[True, False])
        step(metrics, [[101., 0., 0.], [500., 0., 0.]], velocity=[[100., 0., 0.], [900., 0., 0.]],
             sample_mask=[True, False])
        rows = metrics.finish([0, 1], "success")
        self.assertEqual([sample["env_id"] for sample in traces], [1, 0])
        self.assertEqual([sample["env_id"] for sample in physics], [1, 0])
        self.assertEqual(traces[1]["previous_world_x_m"], 100.)
        self.assertEqual(traces[1]["world_x_m"], 101.)
        self.assertEqual(traces[1]["acceleration_xy_mps2"], 0.)
        self.assertEqual(rows[1]["end_world_x_m"], 1.)
        for row in rows:
            self.assertEqual(row["elapsed_s"], 1.)
            self.assertEqual(row["physics_elapsed_s"], 1.)
            self.assertEqual(row["control_sample_count"], 1)
            self.assertEqual(row["physics_sample_count"], 1)

    def test_restart_isolates_reset_jumps_and_independent_environment_slots(self):
        traces = []
        metrics = VectorEpisodeMetrics(2, "cpu", trajectory_sink=traces.extend)
        start(metrics)
        step(metrics, [[10., 0., 0.], [1., 0., 0.]], velocity=[[10., 0., 0.], [0., 0., 0.]])
        metrics.finish([0], "success")
        step(metrics, [[999., 0., 0.], [2., 0., 0.]], velocity=[[999., 0., 0.], [0., 0., 0.]])
        start(metrics, ids=[0], positions=[[100., 0., 0.]], velocities=[[3., 0., 0.]])
        step(metrics, [[101., 0., 0.], [3., 0., 0.]], velocity=[[3., 0., 0.], [0., 0., 0.]])
        rows = metrics.finish([0, 1], ["success", "interrupted"])
        reset_sample = [row for row in traces if row["env_id"] == 0][-1]
        self.assertEqual(reset_sample["previous_world_x_m"], 100.)
        self.assertEqual(reset_sample["previous_world_vx_mps"], 3.)
        self.assertEqual(reset_sample["acceleration_xyz_mps2"], 0.)
        self.assertEqual([row["episode_index"] for row in rows], [1, 0])
        self.assertEqual([row["control_sample_count"] for row in rows], [1, 3])
        self.assertFalse(bool(metrics.active.any()))

    def test_unstarted_attempts_have_null_windows_and_summary_is_counts_only(self):
        metrics = VectorEpisodeMetrics(4, "cpu")
        start(metrics, ids=[0, 1], metadata=[{"terrain": "flat", "travel_direction": 1}] * 2)
        metrics.physics_step(torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 1), 1.)
        step(metrics, torch.ones(4, 3))
        rows = metrics.finish([0, 1], ["success", "failure"])
        missing = metrics.finish_unstarted([2, 3], ["timeout", "interrupted"],
                                          ["approach_timeout", "user_interrupt"],
                                          metadata=[{"terrain": "flat", "travel_direction": 1, "trial_elapsed_s": 2.,
                                                     "mass_kg": 10., "gravity_m_s2": 9.81, "goal_direction_xy": [1., 0.],
                                                     "terminal_base_position": [4., 5., 6.]}] * 2)
        for row in missing:
            self.assertFalse(row["window_started"])
            self.assertFalse(row["has_physics_samples"])
            self.assertEqual(row["mass_kg"], 10.)
            self.assertEqual(row["trial_elapsed_s"], 2.)
            self.assertEqual(row["terminal_base_position"], [4., 5., 6.])
            self.assertEqual(row["physics_sample_count"], 0)
            self.assertEqual(row["control_sample_count"], 0)
            static = {"mass_kg", "gravity_m_s2", "goal_direction_x", "goal_direction_y",
                      "physics_sample_count", "control_sample_count"}
            for key in set(EPISODE_VALUE_FIELDS) - static:
                self.assertIsNone(row[key], key)
        summary = summarize_episodes(rows + missing)[-1]
        self.assertEqual(set(summary), set(SUMMARY_FIELDS))
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["completed_episodes"], 3)
        self.assertEqual(summary["started_tests"], 2)
        self.assertEqual(summary["started_success"], 1)
        self.assertEqual(summary["not_started_timeout"], 1)
        self.assertEqual(summary["not_started_interrupted"], 1)
        self.assert_raw_only(summary)
        with self.assertRaises(ValueError):
            metrics.finish_unstarted([3], "success")

    def test_summary_separates_directions_and_handles_empty_interrupted_cases(self):
        summaries = summarize_episodes([
            {"terrain": "flat", "travel_direction": 1, "window_started": True, "status": "success"},
            {"terrain": "flat", "travel_direction": -1, "window_started": True, "status": "failure"},
            {"terrain": "flat", "travel_direction": "backward", "window_started": False, "status": "interrupted"},
        ])
        overall = {row["travel_direction"]: row for row in summaries if row["scope"] == "overall"}
        self.assertEqual(set(overall), {"forward", "backward"})
        self.assertEqual(overall["forward"]["success"], 1)
        self.assertEqual(overall["forward"]["completed_episodes"], 1)
        self.assertEqual(overall["backward"]["failure"], 1)
        self.assertEqual(overall["backward"]["completed_episodes"], 1)
        self.assertEqual(overall["backward"]["not_started_interrupted"], 1)
        self.assertEqual(summarize_episodes([])[0]["episodes"], 0)
        interrupted = summarize_episodes([{"status": "interrupted", "terrain": "flat"}])[-1]
        self.assertEqual(interrupted["completed_episodes"], 0)
        self.assertEqual(interrupted["started_interrupted"], 1)

    def test_zero_duration_started_window_has_endpoints_but_no_samples(self):
        metrics = VectorEpisodeMetrics(1, "cpu")
        start(metrics, positions=[[2., 3., 4.]], metadata=[{"entry_time_s": .02, "exit_time_s": .02}])
        row = metrics.finish([0], "success")[0]
        self.assertEqual(row["elapsed_s"], 0.)
        self.assertEqual(row["physics_elapsed_s"], 0.)
        self.assertTrue(row["window_started"])
        self.assertFalse(row["has_physics_samples"])
        self.assertEqual(row["physics_sample_count"], 0)
        self.assertEqual(row["control_sample_count"], 0)
        self.assertEqual(row["start_world_x_m"], row["end_world_x_m"])
        self.assert_raw_only(row)

    def test_float32_dt_is_preserved_and_all_samples_remain(self):
        traces, physics = [], []
        physics_dt = float(torch.tensor(.005, dtype=torch.float32))
        control_dt = physics_dt * 4
        self.assertLess(control_dt, .02)
        metrics = VectorEpisodeMetrics(1, "cpu", trajectory_sink=traces.extend, physics_sink=physics.extend)
        start(metrics)
        for _ in range(100):
            for _ in range(4):
                metrics.physics_step([[0.]], [[0.]], [[0.]], physics_dt)
            step(metrics, [[0., 0., 0.]], dt=control_dt)
        row = metrics.finish([0], "timeout")[0]
        self.assertEqual((len(traces), len(physics)), (100, 400))
        self.assertEqual((row["control_sample_count"], row["physics_sample_count"]), (100, 400))
        self.assertEqual(traces[0]["control_dt_s"], control_dt)
        self.assertEqual(physics[0]["dt_s"], physics_dt)
        self.assertAlmostEqual(row["elapsed_s"], 100 * control_dt)
        self.assertAlmostEqual(row["physics_elapsed_s"], 400 * physics_dt)

    def test_stream_csv_raw_schema_metadata_filter_and_recursive_nulls(self):
        with tempfile.TemporaryDirectory(prefix="evaluation_raw_") as temp:
            with CsvEvaluationWriter(temp, num_joints=2) as writer:
                metrics = VectorEpisodeMetrics(1, "cpu", num_joints=2, trajectory_sink=writer.write_trajectories,
                                               physics_sink=writer.write_physics)
                start(metrics, metadata=[{"terrain": "slope", "eval_seed": 7, "task": "mine", "checkpoint": "model.pt",
                                         "dof_names": ["hip", "wheel"], "gravity_vector_m_s2": [0., 0., -10.],
                                         "entry_wheel_bounds_xy": [[0., float("nan"), float("inf"), 1.]],
                                         "entry_velocity_body": [1., 2., 3.], "energy_net_j": 99., "warmup_s": 10.,
                                         "metadata_json": '{"cot_net_path": 1, "unapproved": 3}'}])
                metrics.update_metadata([0], [{"terminal_orientation_xyzw": [0., 0., 0., 1.],
                                               "terminal_velocity_world": [1., 2., float("nan")],
                                               "velocity_instability_rate": .5}])
                self.assertNotIn("energy_net_j", metrics.metadata[0])
                self.assertNotIn("velocity_instability_rate", metrics.metadata[0])
                metrics.physics_step([[2., -3.]], [[1., 2.]], [[3., 2.]], .5)
                step(metrics, [[1., 0., 0.]], dt=.5)
                rows = metrics.finish([0], "failure", "fall")
                rows[0]["cot_net_path"] = 4.  # Writer also rejects stale metrics from other callers.
                writer.write_episodes(rows)
                writer.write_summary(rows)
                saved = {}
                for name in ("episodes", "physics", "trajectory", "summary", "trial_index"):
                    with (Path(temp) / (name + ".csv")).open() as handle:
                        saved[name] = list(csv.DictReader(handle))
                self.assertEqual(len(saved["episodes"]), 1)  # flushed before closing
                self.assertEqual(len(saved["trajectory"]), 1)
                trial = saved["episodes"][0]["trial_id"]
                self.assertTrue(trial.endswith('/mine/slope/env0000/ep0000'))
                for kind in ("episodes", "physics", "trajectory", "trial_index"):
                    self.assertTrue(all(row["trial_id"] == trial and row["run_id"] == writer.run_id
                                        for row in saved[kind]))
                self.assertEqual(saved["trial_index"][0]["status"], "failure")
                self.assertEqual(saved["trial_index"][0]["control_sample_count"], "1")
                self.assertEqual(set(saved["physics"][0]), set(physics_fields(2)))
                metadata = json.loads(saved["episodes"][0]["metadata_json"])
                self.assertEqual(metadata["dof_names"], ["hip", "wheel"])
                self.assertEqual(metadata["gravity_vector_m_s2"], [0., 0., -10.])
                self.assertEqual(metadata["entry_wheel_bounds_xy"], [[0., None, None, 1.]])
                self.assertEqual(metadata["terminal_velocity_world"], [1., 2., None])
                self.assertEqual(metadata["entry_velocity_body"], [1., 2., 3.])
                self.assertEqual(metadata["terminal_orientation_xyzw"], [0., 0., 0., 1.])
                self.assertNotIn("unapproved", metadata)
                for item in [rows[0] for rows in saved.values()] + [metadata]:
                    self.assert_raw_only(item)
                self.assertEqual(saved["trajectory"][0]["world_qw"], "")
                numeric_physics = [{key: float(value) for key, value in sample.items()
                                    if key.startswith(("torque_", "joint_vel_")) or key == "dt_s"}
                                   for sample in saved["physics"]]
                self.assertEqual(offline_work(numeric_physics, 2), [-1., 5., 2.])
                self.assertEqual(saved["summary"][-1]["failure"], "1")
                with self.assertRaises(ValueError):
                    writer.write_physics([{"torque_j00_nm": 1.}])
            with self.assertRaises(FileExistsError):
                CsvEvaluationWriter(temp, num_joints=2)
            with self.assertRaises(ValueError):
                writer.write_episodes([])

    def test_joint_shape_inference_validation_and_disabled_trajectory_file(self):
        metrics = VectorEpisodeMetrics(1, "cpu", trajectory_stride=0)
        start(metrics)
        metrics.physics_step([[1., 2.]], [[0., 0.]], [[0., 0.]], .1)
        self.assertEqual(metrics.num_joints, 2)
        with self.assertRaisesRegex(ValueError, "joint count"):
            metrics.physics_step([[1.]], [[0.]], [[0.]], .1)
        with self.assertRaises(ValueError):
            metrics.physics_step([[1., 2.]], [[0.]], [[0., 0.]], .1)
        with self.assertRaises(ValueError):
            step(metrics, [[0., 0., 0.]], dt=0.)
        with self.assertRaises(ValueError):
            step(metrics, [[0., 0., 0.]], orientation_world=[[0., 0., 1.]])
        with self.assertRaises(ValueError):
            start(metrics)
        metrics.finish([0], "interrupted")
        with self.assertRaises(ValueError):
            start(metrics, masses=[0.])
        with self.assertRaises(ValueError):
            metrics.finish([0], "success")
        with self.assertRaises(ValueError):
            metrics.finish_unstarted([0, 0], "failure")
        with self.assertRaises(TypeError):
            VectorEpisodeMetrics(1, "cpu", warmup_s=1.)
        with self.assertRaises(TypeError):
            VectorEpisodeMetrics(1, "cpu", vir_threshold_mps2=2.)
        with tempfile.TemporaryDirectory(prefix="evaluation_raw_disabled_") as temp:
            with CsvEvaluationWriter(temp, num_joints=2, save_trajectories=False) as writer:
                writer.write_trajectories([{"world_x_m": 1.}])
                self.assertTrue((Path(temp) / "physics.csv").exists())
                self.assertFalse((Path(temp) / "trajectory.csv").exists())

    def test_config_import_does_not_preload_torch_or_isaacgym(self):
        subprocess.run([sys.executable, "-B", "-c", "import sys; import legged_gym.evaluation.config; "
                        "assert 'torch' not in sys.modules; assert 'isaacgym' not in sys.modules"],
                       check=True, cwd=str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    unittest.main()
