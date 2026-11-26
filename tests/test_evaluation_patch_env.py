# Adapted from wheel_gym_CQ/tests/test_evaluation_patch_env.py
"""Local-patch measurement timing, real wheel projections, and reset isolation."""
import json
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import isaacgym
import torch

from legged_gym.envs.nezha.nezha_moe_cts_env import NezhaMoECTSEnv
from legged_gym.evaluation.crossing import WheelWindowCrossing
from legged_gym.evaluation.geometry import wheel_geometry_from_cfg
from legged_gym.evaluation.patch_env import PatchEvaluationEnv, project_wheel_cylinders
from legged_gym.evaluation.patch_terrain import PatchEvaluationTerrain
from legged_gym.evaluation.env import EvaluationEnv
from legged_gym.evaluation.metrics import VectorEpisodeMetrics

def bare_env(n=4, quota=1):
    env = EvaluationEnv.__new__(EvaluationEnv)
    env.num_envs, env.num_dof, env.device = n, 2, "cpu"
    env.evaluation = NS(max_tilt_deg=60., command_vx=1., task="test_task", checkpoint="model_1.pt")
    env.cfg = NS(env=NS(fail_to_terminal_time_s=.6), terrain=NS(mesh_type="heightfield"))
    env.dt, env.max_episode_length = .02, 10
    env.sim_params = NS(dt=.005, gravity=NS(x=0., y=0., z=-9.81))
    env._evaluation_started, env._seed, env._episode_quota = True, 42, quota
    env.episode_counts = torch.zeros(n, dtype=torch.long)
    env.attempt_indices = torch.full((n,), -1, dtype=torch.long)
    env._episode_status, env._episode_reason = [None] * n, [""] * n
    env.completed_episodes, env.episode_sink = [], Mock()
    env.measurement_joint_names = ["joint_a", "joint_b"]
    env.control_rows, env.physics_rows = [], []
    env.metrics = VectorEpisodeMetrics(n, "cpu", trajectory_sink=env.control_rows.extend,
                                       physics_sink=env.physics_rows.extend, num_joints=2)
    env.root_states = torch.zeros(n, 13)
    env.root_states[:, 0], env.root_states[:, 1], env.root_states[:, 6] = 1., 5., 1.
    env.base_pos = env.root_states[:, :3]
    env.base_lin_vel, env.base_ang_vel = torch.zeros(n, 3), torch.zeros(n, 3)
    env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(n, 1)
    env.contact_forces, env.termination_contact_indices = torch.zeros(n, 1, 3), torch.tensor([0])
    env.fail_buf = torch.zeros(n)
    env.episode_length_buf = torch.ones(n, dtype=torch.long)
    env.traversal_bounds = torch.tensor([[0., 10., 0., 10.]]).repeat(n, 1)
    env.goal_positions = torch.tensor([[9., 5., 0.]]).repeat(n, 1)
    env.goal_directions = torch.tensor([[1., 0.]]).repeat(n, 1)
    env.spawn_positions = env.root_states[:, :3].clone()
    env.commands = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    env._step_commands = env.commands[:, :3].clone()
    env.total_mass_kg = torch.full((n,), 15.)
    env.terrain_types = torch.zeros(n, dtype=torch.long)
    env.terrain = NS(type_names=("flat",))
    env.terrain_x_min, env.terrain_y_min = 0., 0.
    env.terrain_x_max, env.terrain_y_max = 10., 10.
    return env


def offline_energy(rows, num_joints):
    """Test-only reconstruction; the recorder must not perform this reduction."""
    return sum(row["dt_s"] * sum(row["torque_j%02d_nm" % j] *
               (row["joint_vel_before_j%02d_radps" % j] + row["joint_vel_after_j%02d_radps" % j]) * .5
               for j in range(num_joints)) for row in rows)





def geometry_cfg():
    angles = {"%s_%s_joint" % (leg, joint): value
              for leg in ("FL", "FR", "RL", "RR")
              for joint, value in (("hip", -.1 if leg.endswith("L") else .1),
                                   ("thigh", .925), ("calf", -1.85), ("foot", 0.))}
    return NS(asset=NS(file="{LEGGED_GYM_ROOT_DIR}/resources/robots/urdf2.1/urdf/nezha3_description.urdf"),
              init_state=NS(default_joint_angles=angles))


def patch_env(sign=1, quota=1, all_wheels=False):
    base = bare_env(1, quota)
    env = PatchEvaluationEnv.__new__(PatchEvaluationEnv)
    env.__dict__.update(base.__dict__)
    env.metrics.active[:] = False
    env.metrics.episode_index[:] = -1
    env.attempt_indices[:] = -1
    env._attempt_metadata = [{}]
    env.crossing = WheelWindowCrossing(1, "cpu", exit_rule="opposite_or_sides")
    env._init_obstacle_traversal()
    env.attempt_active = torch.zeros(1, dtype=torch.bool)
    env.evaluation.command_vx = float(sign)
    env.evaluation.wheel_side = "all" if all_wheels else "both"
    env.root_states[:, 0] = -.8 if sign > 0 else (7. if all_wheels else 1.2)
    env.root_states[:, 1] = 0. if all_wheels else -.25
    env.patch_bounds = torch.tensor([[0., 6.2, -3.1, 3.1]] if all_wheels else [[0., .4, -.2, .2]])
    env.traversal_bounds = torch.tensor([[-1.5, 7.7, -4.1, 4.1]] if all_wheels else [[-2., 3., -1., 1.]])
    env.terrain_x_min, env.terrain_x_max, env.terrain_y_min, env.terrain_y_max = env.traversal_bounds[0].tolist()
    env.selected_sides = torch.full((1,), 2 if all_wheels else 0, dtype=torch.long)
    env.travel_signs = torch.tensor([float(sign)])
    env.goal_directions = torch.tensor([[float(sign), 0.]])
    env.spawn_positions = env.root_states[:, :3].clone()
    env._wheel_center_offsets = torch.tensor([[.43, .25, -.4], [.43, -.25, -.4],
                                               [-.43, .25, -.4], [-.43, -.25, -.4]])
    env._wheel_default_half_extents = torch.tensor([[.11, .02, .11]]).repeat(4, 1)
    env._wheel_collision_offsets = torch.zeros(4, 3)
    env._wheel_collision_axes = torch.tensor([[0., 1., 0.]]).repeat(4, 1)
    env._wheel_radii = torch.full((4,), .11)
    env._wheel_widths = torch.full((4,), .04)
    env.measurement_wheel_indices = torch.arange(4)
    env.wheel_body_indices = torch.full((4,), -1)  # Legacy wrong mapping must not be used.
    env.num_bodies = 5
    env.rigid_body_states = torch.zeros(1, 5, 13)
    env.rigid_state = env.rigid_body_states
    env.rigid_state[:, :, 6] = 1.
    env.rigid_state[:, 4, :3] = 999.
    env.contact_forces = torch.zeros(1, 5, 3)
    env.termination_contact_indices = torch.tensor([4])
    env.episode_length_buf[:] = 0
    env.max_episode_length = 100
    env._resample_commands(torch.tensor([0]))
    env._step_commands = env.commands[:, :3].clone()
    env._start(torch.tensor([0]))
    env.dof_state = torch.zeros(2, 2)
    env.dof_state[:, 1] = 2.
    env.torques = torch.ones(1, 2)
    return env


def advance(env, x, y=None):
    # This interval belongs to the old active mask, before entry detection.
    for _ in range(4):
        env._before_physics_substep()
        env._after_physics_substep()
    env.root_states[0, 0] = x
    if y is not None:
        env.root_states[0, 1] = y
    env.rigid_state[:, :4, :3] = env.root_states[:, None, :3] + env._wheel_center_offsets[None]
    env.root_states[:, 7] = env.travel_signs
    env.base_lin_vel[:, 0] = env.travel_signs
    env.episode_length_buf += 1
    env.check_termination()


class PatchEvaluationEnvTests(unittest.TestCase):
    def test_angle_trace_records_approach_and_terminal_state_before_reset(self):
        env = patch_env()
        ids = torch.tensor([0])
        events = []
        env.joint_angle_trace = Mock()
        env.joint_angle_trace.start.side_effect = lambda ids: events.append(("start", ids.tolist()))
        env.joint_angle_trace.sample.side_effect = lambda ids, reasons: events.append(
            ("step", ids.tolist(), int(env.episode_length_buf[0]), reasons))
        env._start(ids)
        env.max_episode_length = 2
        advance(env, -.8)
        self.assertFalse(env.metrics.active[0])
        advance(env, -.8)
        self.assertTrue(env.reset_buf[0])
        self.assertEqual(events, [("start", [0]), ("step", [0], 1, {}),
                                  ("step", [0], 2, {0: "time_limit"})])

    def test_platform_requires_support_on_top_then_sustained_landing_in_both_directions(self):
        for sign in (1, -1):
            with self.subTest(sign=sign):
                env = patch_env(sign=sign, all_wheels=True)
                env.terrain = NS(type_names=("platform_50cm",),
                                 metadata={"tiles": [dict(platform_size_xy_m=[5., 5.], platform_height_m=.5)]})
                env._init_obstacle_traversal()
                env.contact_forces[0, :4, 2] = 50.
                # Test FK places wheel bottoms 0.4+0.11 m below the base.
                env.root_states[0, 2] = .51
                advance(env, .5 if sign > 0 else 5.7)
                self.assertTrue(env.metrics.active[0])
                env.root_states[0, 2] += .5
                advance(env, 3.1)
                self.assertTrue(env.obstacle_traversal.reached_middle[0])
                self.assertFalse(env.reset_buf[0])
                far = 6.75 if sign > 0 else -.55
                advance(env, far)
                self.assertFalse(env.reset_buf[0])
                self.assertIsNone(env._attempt_metadata[0]["exit_time_s"])
                env.root_states[0, 2] = .51
                for _ in range(9):
                    advance(env, far)
                    self.assertFalse(env.reset_buf[0])
                advance(env, far)
                self.assertEqual(env._episode_reason[0], "stable_landing")
                self.assertEqual(env._attempt_metadata[0]["exit_edge"], "xmax" if sign > 0 else "xmin")
                self.assertAlmostEqual(env._attempt_metadata[0]["exit_time_s"], env.episode_length_buf.item() * env.dt)

    def test_skipping_platform_or_landing_off_to_the_side_cannot_report_success(self):
        env = patch_env(all_wheels=True)
        env.terrain = NS(type_names=("platform_50cm",),
                         metadata={"tiles": [dict(platform_size_xy_m=[5., 5.], platform_height_m=.5)]})
        env._init_obstacle_traversal()
        env.contact_forces[0, :4, 2] = 50.
        env.root_states[0, 2] = .51
        advance(env, .5)
        for _ in range(11):
            advance(env, 6.75)
        self.assertFalse(env.reset_buf[0])
        self.assertIsNone(env._attempt_metadata[0]["exit_time_s"])
        env.root_states[0, 2] += .5
        advance(env, 3.1)
        self.assertTrue(env.obstacle_traversal.reached_middle[0])
        env.root_states[0, 2] = .51
        advance(env, 3.1, y=4.)
        for _ in range(11):
            advance(env, 6.75, y=0.)
        self.assertFalse(env.obstacle_traversal.reached_middle[0])
        self.assertFalse(env.reset_buf[0])
        self.assertIsNone(env._attempt_metadata[0]["exit_time_s"])

    def test_concave_stairs_require_supported_valley_before_climbing_to_the_far_ground(self):
        env = patch_env(all_wheels=True)
        env.terrain = NS(type_names=("stairs_concave_25cm",),
                         metadata={"tiles": [dict(central_platform_size_m=3., patch_length_m=6.2,
                                                  tread_width_m=.3, central_platform_height_m=-2.)]})
        env._init_obstacle_traversal()
        env.contact_forces[0, :4, 2] = 50.
        env.root_states[0, 2] = .51
        advance(env, .5)
        advance(env, 3.1)
        self.assertFalse(env.obstacle_traversal.reached_middle[0])
        env.root_states[0, 2] -= 2.
        advance(env, 3.1)
        self.assertTrue(env.obstacle_traversal.reached_middle[0])
        env.root_states[0, 2] = .51
        for _ in range(10):
            advance(env, 6.75)
        self.assertEqual(env._episode_reason[0], "stable_landing")

    def test_standard_all_wheels_window_keeps_raw_sampling_until_last_exit(self):
        env = patch_env(all_wheels=True)
        advance(env, -.5)
        self.assertEqual(env.crossing.wheels_entered_mask[0].tolist(), [True, True, False, False])
        self.assertEqual(env._attempt_metadata[0]["selected_side"], "all")
        self.assertEqual(env.physics_rows, [])
        advance(env, .5)
        self.assertTrue(env.crossing.wheels_entered_mask.all())
        advance(env, 6.1)
        self.assertFalse(env.reset_buf[0])
        advance(env, 6.7)  # The last row still overlaps the exit plane.
        self.assertFalse(env.reset_buf[0])
        advance(env, 6.75)
        self.assertEqual(env._episode_status[0], "success")
        env._finish_attempt(torch.tensor([0]), "success", env._episode_reason[0])
        self.assertEqual(len(env.control_rows), 4)
        self.assertEqual(len(env.physics_rows), 16)
        self.assertEqual(env.completed_episodes[0]["wheels_entered_mask"], [True] * 4)
        self.assertEqual(env.completed_episodes[0]["selected_side"], "all")
        self.assertTrue(all(bounds[0] >= 6.2 for bounds in env.completed_episodes[0]["terminal_wheel_bounds_xy"]))

    def test_standard_window_does_not_succeed_or_fail_if_one_wheel_never_entered(self):
        env = patch_env(all_wheels=True)
        env._wheel_center_offsets[3, 1] = 4.
        advance(env, -.5)
        advance(env, .5)
        advance(env, 6.75)
        self.assertEqual(env.crossing.wheels_entered_mask[0].tolist(), [True, True, True, False])
        self.assertFalse(env.reset_buf[0])
        self.assertTrue(env.crossing.active[0])
        env.max_episode_length = 4
        advance(env, 6.8)
        self.assertEqual(env._episode_status[0], "timeout")

    def test_urdf_fk_and_collision_geometry_are_json_safe_and_match_robot(self):
        cfg = geometry_cfg()
        geometry = wheel_geometry_from_cfg(cfg)
        json.dumps(geometry)
        np.testing.assert_allclose(geometry["center_offsets"],
                                   [[.42596576598, .23621531514, -.40252671222],
                                    [.42594638235, -.26037980227, -.40254179047],
                                    [-.43603423402, .23621531514, -.40252671222],
                                    [-.43605361765, -.26037980227, -.40254179047]], atol=1e-9)
        self.assertEqual(geometry["radii"], [.11] * 4)
        self.assertEqual(geometry["widths"], [.04] * 4)
        self.assertAlmostEqual(geometry["wheel_x_max"] - geometry["wheel_x_min"], 1.08201938363)
        self.assertAlmostEqual(geometry["default_half_extents"][0][1], .03088175914)
        cfg.init_state.default_joint_angles["FL_thigh_joint"] += .1
        changed = wheel_geometry_from_cfg(cfg)
        self.assertNotEqual(changed["center_offsets"][0], geometry["center_offsets"][0])
        self.assertEqual(changed["center_offsets"][1:], geometry["center_offsets"][1:])

    def test_oriented_cylinder_projection_uses_width_axis_and_collision_offset(self):
        states = torch.zeros(1, 1, 13)
        states[0, 0, :3] = torch.tensor([3., 5., 20.])  # Height does not prevent projected entry.
        states[0, 0, 3:7] = torch.tensor([0., 0., 2 ** -.5, 2 ** -.5])
        bounds = project_wheel_cylinders(states, torch.tensor([[.1, 0., 0.]]),
                                         torch.tensor([[0., 1., 0.]]), torch.tensor([.11]), torch.tensor([.04]))
        torch.testing.assert_close(bounds, torch.tensor([[[2.98, 3.02, 4.99, 5.21]]]), atol=1e-5, rtol=1e-5)

    def test_entry_step_excluded_gap_kept_last_wheel_exit_recorded_once(self):
        env = patch_env()
        advance(env, -.7)
        self.assertFalse(env.metrics.active[0])
        advance(env, -.5)
        self.assertTrue(env.metrics.active[0])
        self.assertEqual(env.physics_rows, [])
        self.assertEqual(env.control_rows, [])
        self.assertEqual(env.metrics.elapsed_s[0], 0.)
        self.assertEqual(env.crossing.wheels_entered_mask[0].tolist(), [True, False, False, False])
        advance(env, .2)  # Front axle is beyond the patch; rear has not entered yet.
        self.assertFalse(env.reset_buf[0])
        self.assertTrue(env.metrics.active[0])
        self.assertAlmostEqual(env.metrics.elapsed_s[0].item(), .02)
        advance(env, .4)
        self.assertTrue(env.crossing.wheels_entered_mask[0, 2])
        advance(env, .95)
        self.assertTrue(env.reset_buf[0])
        self.assertEqual(env._episode_status[0], "success")
        env._finish_attempt(torch.tensor([0]), "success", env._episode_reason[0])
        row = env.completed_episodes[0]
        self.assertEqual(len(env.control_rows), 3)
        self.assertEqual(len(env.physics_rows), 12)
        self.assertAlmostEqual(offline_energy(env.physics_rows, 2), .24)
        for index, sample in enumerate(env.physics_rows):
            self.assertEqual(sample["control_step"], index // 4 + 1)
            self.assertEqual(sample["substep_index"], index % 4 + 1)
            self.assertGreater(sample["time_end_s"], 0.)
            self.assertLessEqual(sample["time_end_s"], .06 + 1e-12)
        self.assertAlmostEqual(row["elapsed_s"], .06)
        self.assertAlmostEqual(row["physics_elapsed_s"], .06)
        self.assertAlmostEqual(row["exit_time_s"] - row["entry_time_s"], row["elapsed_s"])
        path = sum(math.hypot(sample["world_x_m"] - sample["previous_world_x_m"],
                              sample["world_y_m"] - sample["previous_world_y_m"])
                   for sample in env.control_rows)
        self.assertAlmostEqual(path, 1.45, places=6)
        for calculated in ("path_xy_m", "energy_net_j", "mean_power_net_w", "cot_net_path",
                           "velocity_instability_rate", "tracking_rmse_xy_mps"):
            self.assertNotIn(calculated, row)
        self.assertTrue(row["window_started"])
        self.assertAlmostEqual(row["entry_base_position"][0], -.5)
        self.assertAlmostEqual(row["terminal_base_position"][0], .95)
        self.assertTrue(all(bounds[0] >= .4 for bounds in row["terminal_wheel_bounds_xy"]))
        self.assertAlmostEqual(row["window_end_time_s"], row["exit_time_s"])
        self.assertEqual(row["wheel_contact_mask"], [False] * 4)  # Jumping/projection need not touch.
        self.assertEqual(env.episode_counts[0], 1)

    def test_reverse_keeps_yaw_zero_and_rear_axle_enters_first(self):
        env = patch_env(sign=-1)
        self.assertEqual(env.commands[0, 0], -1.)
        advance(env, .9)
        self.assertTrue(env.metrics.active[0])
        self.assertEqual(env.crossing.wheels_entered_mask[0].tolist(), [False, False, True, False])
        advance(env, .2)
        self.assertFalse(env.reset_buf[0])
        advance(env, -.1)
        advance(env, -.6)
        self.assertEqual(env._episode_status[0], "success")
        torch.testing.assert_close(env.root_states[0, 3:7], torch.tensor([0., 0., 0., 1.]))
        self.assertEqual(env._attempt_metadata[0]["leading_axle"], "rear")

    def test_before_entry_timeout_has_attempt_but_no_physical_measurements(self):
        env = patch_env()
        env.max_episode_length = 1
        advance(env, -.7)
        self.assertEqual(env._episode_status[0], "timeout")
        env._finish_attempt(torch.tensor([0]), "timeout", "time_limit")
        row = env.completed_episodes[0]
        self.assertFalse(row["window_started"])
        self.assertEqual(env.physics_rows, [])
        self.assertEqual(env.control_rows, [])
        self.assertIsNone(row["elapsed_s"])
        self.assertEqual(row["mass_kg"], 15.)
        self.assertEqual(row["gravity_m_s2"], 9.81)
        self.assertNotIn("success_rate", row)

    def test_only_fall_terminates_early_bypass_retreat_and_corridor_exit_can_continue(self):
        with self.subTest("bypass"):
            env = patch_env()
            advance(env, -.7, y=.5)
            advance(env, 1., y=.5)
            self.assertFalse(env.reset_buf[0])
            self.assertFalse(env.metrics.active[0])
        with self.subTest("retreat"):
            env = patch_env()
            advance(env, -.5)
            advance(env, -.8)
            self.assertFalse(env.reset_buf[0])
            self.assertTrue(env.metrics.active[0])
        with self.subTest("fall"):
            env = patch_env()
            env.projected_gravity[0, 2] = 0.
            advance(env, -.7)
            self.assertEqual(env._episode_reason[0], "fall")
        with self.subTest("out_of_corridor"):
            env = patch_env()
            advance(env, -.7, y=1.1)
            self.assertFalse(env.reset_buf[0])

    def test_contact_alone_does_not_count_as_a_fall(self):
        env = patch_env(all_wheels=True)
        env.contact_forces[0, 4, 2] = 100.
        env.fail_buf[:] = 1000
        advance(env, .5)
        self.assertFalse(env.reset_buf[0])

    def test_lateral_exit_is_success_and_records_the_actual_edge(self):
        for y, edge in ((4., "ymax"), (-4., "ymin")):
            env = patch_env(all_wheels=True)
            advance(env, .5)
            advance(env, 3., y=y)
            self.assertEqual(env._episode_status[0], "success")
            self.assertEqual(env._attempt_metadata[0]["exit_edge"], edge)
            env._finish_attempt(torch.tensor([0]), "success", "all_wheels_cleared")
            self.assertEqual(len(env.control_rows), 1)
            self.assertEqual(len(env.physics_rows), 4)

    def test_return_through_entrance_is_not_success_and_reentry_can_succeed(self):
        env = patch_env(all_wheels=True)
        advance(env, .5)
        advance(env, -.8)
        self.assertFalse(env.reset_buf[0])
        self.assertTrue(env.metrics.active[0])
        advance(env, .5)
        advance(env, 6.75)
        self.assertEqual(env._episode_status[0], "success")

    def test_fall_wins_over_simultaneous_exit_and_success_wins_over_timeout(self):
        env = patch_env(all_wheels=True)
        advance(env, .5)
        env.projected_gravity[0, 2] = 0.
        advance(env, 6.75)
        self.assertEqual(env._episode_status[0], "failure")
        self.assertEqual(env._episode_reason[0], "fall")
        env = patch_env(all_wheels=True)
        advance(env, .5)
        env.max_episode_length = 2
        advance(env, 6.75)
        self.assertEqual(env._episode_status[0], "success")

    def test_invalid_velocity_aborts_instead_of_counting_a_fall(self):
        env = patch_env()
        env.root_states[0, 0] = -.5
        env.root_states[0, 7] = float("nan")
        env.rigid_state[:, :4, :3] = env.root_states[:, None, :3] + env._wheel_center_offsets[None]
        env.episode_length_buf += 1
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            env.check_termination()
        self.assertFalse(env.metrics.active[0])
        env.interrupt("evaluation_error")
        self.assertEqual(env.completed_episodes[0]["status"], "interrupted")
        self.assertFalse(env.completed_episodes[0]["window_started"])
        self.assertIsNone(env.completed_episodes[0]["window_end_time_s"])

    def test_crossing_in_one_control_step_is_marked_without_fake_samples(self):
        env = patch_env()
        advance(env, 1.)
        self.assertEqual(env._episode_status[0], "success")
        env._finish_attempt(torch.tensor([0]), "success", "all_wheels_cleared")
        row = env.completed_episodes[0]
        self.assertTrue(row["window_started"])
        self.assertTrue(row["window_zero_duration"])
        self.assertEqual(row["elapsed_s"], 0.)
        self.assertEqual(env.physics_rows, [])
        self.assertEqual(env.control_rows, [])
        self.assertNotIn("energy_net_j", row)

    def test_new_attempt_is_reset_before_measuring_and_side_selection_is_balanced(self):
        env = patch_env(quota=2)
        env.max_episode_length = 1
        advance(env, -.7)

        def reset(_self, ids):
            self.assertEqual(len(env.completed_episodes), 1)
            self.assertFalse(env.attempt_active[0])
            env.root_states[ids, 0] = -.8
            env.episode_length_buf[ids] = 0

        with patch.object(NezhaMoECTSEnv, "reset_idx", side_effect=reset), patch.object(env, "_sync_reset_state"):
            env.reset_idx(torch.tensor([0]))
        self.assertTrue(env.attempt_active[0])
        self.assertFalse(env.metrics.active[0])
        self.assertFalse(env.crossing.window_started[0])
        self.assertEqual(env._attempt_metadata[0]["episode_index"], 1)
        env.interrupt("test")
        self.assertEqual([row["status"] for row in env.completed_episodes], ["timeout", "interrupted"])
        self.assertFalse(env.evaluation_complete)

    def test_continuous_preview_restarts_the_clock_and_stays_active_after_each_timeout(self):
        env = patch_env(quota=math.inf)
        env.max_episode_length = 1
        ids = torch.tensor([0])

        def physical_reset(_self, env_ids):
            env.root_states[env_ids, 0], env.episode_length_buf[env_ids] = -.8, 0

        with patch.object(NezhaMoECTSEnv, "reset_idx", side_effect=physical_reset), \
                patch.object(env, "_sync_reset_state"):
            for index in range(3):
                advance(env, -.7)
                env.reset_idx(ids)
                row = env.completed_episodes[-1]
                self.assertEqual((row["episode_index"], row["status"], row["trial_elapsed_s"]),
                                 (index, "timeout", env.dt))
                self.assertTrue(env.attempt_active[0])
                self.assertFalse(env.evaluation_complete)

    def test_explicit_window_resets_keep_raw_attempt_keys_unique_and_quota_intact(self):
        env = patch_env(quota=1)

        def physical_reset(_self, ids):
            env.root_states[ids, 0], env.episode_length_buf[ids] = -.8, 0

        with patch.object(NezhaMoECTSEnv, "reset_idx", side_effect=physical_reset), \
                patch.object(env, "_sync_reset_state"):
            for _ in range(2):
                advance(env, -.5)
                advance(env, .2)
                env.reset_idx(torch.tensor([0]))
                self.assertEqual(env.episode_counts[0], 0)
                self.assertFalse(env.evaluation_complete)
            env.max_episode_length = 1
            advance(env, -.7)  # The third attempt ends without entering the window.
            env.reset_idx(torch.tensor([0]))

        self.assertEqual([row["episode_index"] for row in env.completed_episodes], [0, 1, 2])
        self.assertEqual([row["status"] for row in env.completed_episodes],
                         ["interrupted", "interrupted", "timeout"])
        self.assertEqual([(row["episode_index"], row["control_step"]) for row in env.control_rows],
                         [(0, 1), (1, 1)])
        self.assertEqual([(row["episode_index"], row["physics_step"]) for row in env.physics_rows],
                         [(attempt, step) for attempt in (0, 1) for step in range(1, 5)])
        self.assertFalse(env.completed_episodes[-1]["window_started"])
        self.assertEqual(env.episode_counts[0], 1)
        self.assertTrue(env.evaluation_complete)

    def test_standard_default_spawns_center_full_footprint_and_use_actual_ground_height(self):
        geometry = wheel_geometry_from_cfg(geometry_cfg())
        for sign in (1, -1):
            with self.subTest(direction=sign):
                env = patch_env(sign=sign, all_wheels=True)
                del env.evaluation.wheel_side  # Production fallback must be all.
                cfg = NS(mesh_type="heightfield", type_names=("slope",),
                         terrain_length=7.8, terrain_width=7.8, horizontal_scale=.02,
                         vertical_scale=.005, border_size=2., corridor_width=9.8,
                         **{key: geometry[key] for key in ("left_wheel_y", "right_wheel_y", "front_wheel_x",
                             "rear_wheel_x", "wheel_radius", "wheel_x_min", "wheel_x_max", "wheel_y_min", "wheel_y_max")})
                env.terrain = PatchEvaluationTerrain(cfg, 1)
                env.base_init_state = torch.tensor([0., 0., .6, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
                env._spawn_rngs = [np.random.default_rng(123)]
                env.gym, env.sim = NS(set_actor_root_state_tensor_indexed=Mock()), object()
                with patch("legged_gym.evaluation.patch_env.gymtorch.unwrap_tensor", side_effect=lambda x: x), \
                        patch.object(env.terrain, "sample_spawns", wraps=env.terrain.sample_spawns) as sample:
                    env._reset_root_states(torch.tensor([0]))
                self.assertEqual(sample.call_args.kwargs["side"], "center")
                self.assertEqual(env.selected_sides[0], 2)
                bounds = env.patch_bounds[0]
                x, y, z = env.root_states[0, :3].tolist()
                self.assertLessEqual(abs(y - float((bounds[2] + bounds[3]) * .5)), .040001)
                self.assertGreater(y + geometry["wheel_y_min"], bounds[2])
                self.assertLess(y + geometry["wheel_y_max"], bounds[3])
                if sign > 0:
                    self.assertLess(x + geometry["wheel_x_max"], bounds[0])
                else:
                    self.assertGreater(x + geometry["wheel_x_min"], bounds[1])
                    self.assertGreater(z, .6)
                height = env.terrain.ground_heights(np.asarray([[x, y]]))[0]
                self.assertAlmostEqual(z, height + .6, places=6)
                self.assertEqual(env.commands[0, 0], float(sign))
                torch.testing.assert_close(env.root_states[0, 3:7], torch.tensor([0., 0., 0., 1.]))


if __name__ == "__main__":
    unittest.main()


def test_deterministic_torque_override_preserves_training_default():
    from legged_gym.envs.nezha.nezha_robot import NezhaRobot
    env = NezhaRobot.__new__(NezhaRobot)
    env.num_envs, env.num_dof, env.device = 1, 4, 'cpu'
    env.cfg = NS(control=NS(action_scale=.25, vel_scale=10., control_type='T'),
                 domain_rand=NS(randomize_motor_strength=False))
    env.default_dof_pos = env.dof_pos = torch.zeros(1, 4)
    env.wheel_indices = torch.tensor([3])
    env.joint_pos_target = torch.zeros(1, 4)
    env.torque_limits = torch.ones(4)
    with patch('legged_gym.envs.nezha.nezha_robot.torch_rand_float', return_value=torch.full((1, 4), 1.1)) as random:
        training = env._compute_torques(torch.ones(1, 4))
        random.assert_called_once()
        env.cfg.domain_rand.randomize_torque_scale = False
        random.reset_mock()
        evaluation = env._compute_torques(torch.ones(1, 4))
        random.assert_not_called()
    torch.testing.assert_close(training, torch.tensor([[.275, .275, .275, 0.]]))
    torch.testing.assert_close(evaluation, torch.tensor([[.25, .25, .25, 0.]]))
