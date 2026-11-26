"""Exercise real terrain reward dispatch on CPU without the Isaac Gym extension."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legged_gym.evaluation.presets import PATCH_HIGH_PLATFORM_HEIGHTS_M
from legged_gym.terrains.platform_rewards import (
    platform_reward_scales, platform_v1_reward_scales, platform_v2_reward_scales,
)
from legged_gym.terrains.wf import PLATFORM_TYPES, column_types, high_platform_wf_sub_terrains
from test_wf_curriculum import make_env


@pytest.fixture
def robot():
    # Load the Python classes without importing their native-simulator modules.
    root = Path(__file__).parents[1] / "envs"
    def rotate_vector(quaternion, vector):
        xyz = quaternion[:, :3]
        cross = 2. * torch.cross(xyz, vector, dim=-1)
        return vector + quaternion[:, 3:] * cross + torch.cross(xyz, cross, dim=-1)

    namespace = {"BaseTask": object, "torch": torch, "np": np, "PLATFORM_TYPES": PLATFORM_TYPES,
                 "quat_apply": rotate_vector, "wrap_to_pi": lambda a: (a + torch.pi) % (2 * torch.pi) - torch.pi,
                 "PATCH_HIGH_PLATFORM_HEIGHTS_M": PATCH_HIGH_PLATFORM_HEIGHTS_M}
    for path in (root / "base/legged_robot.py", root / "nezha/nezha_robot.py"):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        exec(compile(tree, str(path), "exec"), namespace)
    cls = namespace["NezhaRobot"]
    robot = cls.__new__(cls)
    robot.device, robot.num_envs, robot.dt = "cpu", 3, 0.02
    robot.wf_curriculum = None
    robot.cfg = SimpleNamespace(
        terrain=SimpleNamespace(num_rows=2),
        rewards=SimpleNamespace(clip_single_reward=10., only_positive_rewards=False, tracking_sigma=.25),
    )
    robot.terrain = SimpleNamespace(
        tile_types=np.array([["flat", "platform_pit"], ["platform_block", "stairs_up"]])
    )
    robot.terrain_types = torch.tensor([1, 1, 0])
    robot.terrain_levels = torch.tensor([1, 0, 1])
    robot._init_platform_terrain_mask()
    robot.root_states = torch.zeros(3, 13)
    robot.root_states[:, 6] = 1.
    robot.base_quat = robot.root_states[:, 3:7]
    robot.forward_vec = torch.tensor([[1., 0., 0.]]).repeat(3, 1)
    return robot


@pytest.mark.parametrize("scales_fn,clip_ordinary", [
    (platform_reward_scales, False), (platform_v1_reward_scales, False),
    (platform_v2_reward_scales, False), (platform_v2_reward_scales, True),
])
def test_reward_dispatch_and_clipping(robot, scales_fn, clip_ordinary):
    assert robot.high_platform_mask.tolist() == [False, True, True]
    scales = scales_fn()
    v2 = scales_fn is platform_v2_reward_scales
    robot.platform_scales = {k: v * robot.dt for k, v in scales.items()}
    robot.reward_scales = {"tracking_lin_vel": 3., "ang_vel_xy": -.2, "lin_vel_z": -1.,
                           "collision": -.1, "termination": -.8}
    robot.rew_buf = torch.zeros(3)
    robot.commands = torch.zeros(3, 4)
    robot.base_lin_vel = torch.tensor([[0., 0., 3.], [0., 0., .5], [0., 0., 4.]])
    robot.base_ang_vel = torch.tensor([[0., 5., 0.], [0., 5., .5], [0., 5., .5]])
    robot.base_height = torch.tensor([.5, .5, .54 if v2 else .52])
    robot.cfg.rewards.base_height_target = .54 if v2 else .5
    robot.cfg.rewards.only_positive_rewards = clip_ordinary
    robot.reward_scales["base_height"] = -1.
    if scales_fn is platform_v1_reward_scales:
        robot.cfg.rewards.platform_base_height_target = .52
    if v2:
        robot.cfg.rewards.platform_reward_profile = "dreamwaq_highplatform"
        robot.cfg.rewards.platform_heading_target = 0.
        robot.cfg.rewards.platform_base_height_target = .54
        # Ordinary training has a different velocity limit and uses only leg rates.
        robot.cfg.rewards.soft_dof_vel_limit = .9
        robot.dof_vel_limits = torch.full((16,), 20.)
    robot.projected_gravity = torch.tensor([[0., 0., -1.], [0., .6, -.8], [0., 0., -1.]])
    robot.dof_pos = torch.zeros(3, 16)
    robot.default_dof_pos = torch.zeros(1, 16)
    robot.wheel_indices = [3, 7, 11, 15]
    robot.dof_pos_limits = torch.tensor([[-1., 1.]] * 16)
    for name in ("torques", "actions", "last_actions", "dof_vel", "last_dof_vel"):
        setattr(robot, name, torch.zeros(3, 16))
    if v2:
        robot.commands[:, 0] = robot.base_lin_vel[:, 0] = .5
        robot.dof_pos[:, 1:3] = torch.tensor([.3, .4])
        robot.dof_vel[:, 1] = 3.
        robot.dof_vel[:, 3] = 19.5
        robot.last_dof_vel.copy_(robot.dof_vel)
        robot.reward_scales.update(run_still=-.05, dof_vel=-1e-6)
    robot.contact_forces = torch.zeros(3, 2, 3)
    robot.penalised_contact_indices = [0]
    robot.trunk_body_index = 1
    robot.contact_forces[:, 1, 0] = 1.0
    robot.reset_buf = torch.tensor([True, False, True])
    robot.time_out_buf = torch.tensor([True, False, False])
    robot._prepare_reward_function()
    robot.compute_reward()
    # G12 floors the platform sum and adds its failure penalty afterwards.
    expected = torch.tensor([
        .06 - .2 * .02 * 25 - 1. * .02 * 9,
        .06 - .05 * .02 * .25 - 2. * .02 * .25 - .02,
        -.8 * .02,
    ])
    if v2:
        # The same pitch rate contributes only on ordinary terrain. Both platform
        # tiles contribute the yaw-rate penalty and retain the signed sum.
        torch.testing.assert_close(robot.episode_sums["ang_vel_xy"],
                                   .02 * torch.tensor([-.2 * 25., 0., 0.]))
        torch.testing.assert_close(robot.episode_sums["ang_vel_xz"],
                                   .02 * torch.tensor([0., -.05 * .5**2, -.05 * .5**2]))
        torch.testing.assert_close(robot.episode_sums["base_height"],
                                   .02 * torch.tensor([-.04**2, -5. * .04**2, 0.]))
        torch.testing.assert_close(robot.episode_sums["orientation"],
                                   torch.tensor([0., -.2 * .02 * .6**2, 0.]))
        assert torch.count_nonzero(robot.episode_sums["collision"]) == 0
        assert torch.count_nonzero(robot.episode_sums["termination"]) == 0
        torch.testing.assert_close(robot.episode_sums["run_still"],
                                   .02 * torch.tensor([-.05 * .7, -.15 * .5, -.15 * .5]))
        platform_velocity_cost = -1e-3 * (3.**2 + (0.1 * 19.5)**2)
        torch.testing.assert_close(robot.episode_sums["dof_vel"],
                                   .02 * torch.tensor([-1e-6 * 3.**2, platform_velocity_cost, platform_velocity_cost]))
        torch.testing.assert_close(robot.episode_sums["dof_vel_limits"],
                                   .02 * torch.tensor([0., -10. * .5, -10. * .5]))
        expected[0] += .02 * (-.05 * .7 - 1e-6 * 3.**2 - .04**2)
        common = 3. + 1.5 * np.exp(-1.) - .05 * .5**2 - .15 * .5 + platform_velocity_cost - 10. * .5
        expected[1] = .02 * (common - 2. * .25 - .2 * .6**2 - 5. * .04**2)
        expected[2] = .02 * (common - 2. * 16.)
        if clip_ordinary:
            expected[0] = 0.
        assert "platform_positive_clip" not in robot.episode_sums
    else:
        assert robot.episode_sums["platform_positive_clip"][2] > .2
        if scales_fn is platform_v1_reward_scales:
            torch.testing.assert_close(robot.episode_sums["base_height"],
                                       torch.tensor([0., -10. * .02 * .02**2, 0.]))
            torch.testing.assert_close(robot.episode_sums["orientation"],
                                       torch.tensor([0., -.5 * .02 * .6**2, 0.]))
            expected[1] += .02 * (1.5 * np.exp(-1.) - .5 * .6**2 - 10. * .02**2)
    torch.testing.assert_close(robot.rew_buf, expected)
    assert robot.episode_sums["lin_vel_z"][2].item() == pytest.approx(-.64)
    # The task logs ordinary term contributions before optional total clipping.
    torch.testing.assert_close(sum(robot.episode_sums.values())[1:], robot.rew_buf[1:])


def test_configured_rewards_match_m20_and_mujica_axes(robot):
    assert platform_reward_scales() == {
        "tracking_lin_vel": 3., "lin_vel_z": -2., "ang_vel_xz": -.05,
        "torques": -5e-6, "dof_vel": -1e-6, "dof_acc": -1e-7,
        "collision": -1., "action_rate": -.011, "stand_still": -.5,
        "dof_pos_limits": -5., "hip_default": -.5, "run_still": -.05,
        "termination": -.8,
    }
    assert platform_v1_reward_scales() == dict(
        platform_reward_scales(), tracking_ang_vel=1.5, orientation=-.5, base_height=-10.,
    )
    assert platform_v2_reward_scales() == {
        "tracking_lin_vel": 3., "tracking_ang_vel": 1.5, "lin_vel_z": -2.,
        "ang_vel_xz": -.05, "orientation": -.2, "base_height": -5.,
        "torques": -4e-5, "dof_vel": -1e-3, "dof_acc": -2.5e-7,
        "collision": -1., "action_rate": -.01, "stand_still": -.5,
        "dof_pos_limits": -5., "dof_vel_limits": -10.,
        "hip_default": -.25, "run_still": -.15, "highplatform_yaw": -3.5,
    }
    # Pure roll, pitch and yaw, respectively, at 2 rad/s.
    robot.base_ang_vel = 2. * torch.eye(3)
    torch.testing.assert_close(robot._reward_ang_vel_xz(), torch.tensor([4., 0., 4.]))
    torch.testing.assert_close(robot._reward_ang_vel_xy(), torch.tensor([4., 4., 0.]))


def test_platform_heading_reward_and_heading_feedback_are_independent(robot):
    platform_commands = {}
    robot.cfg.commands = SimpleNamespace(
        curriculum=True, wf_settings={"terrain_max_command_ranges": {"high_platform": platform_commands}},
    )
    robot.cfg.rewards.platform_heading_target = 0.
    robot.platform_scales = {"highplatform_yaw": -3.5 * robot.dt}
    robot.commands = torch.tensor([[.8, .2, .3, .4], [.7, 0., 0., 0.], [.6, 0., 0., 0.]])
    initial = robot.commands.clone()
    # The second platform quaternion stores a world yaw of -0.4 via its equivalent 2π-0.4.
    angles = torch.tensor([-.6, 2.8, 2 * torch.pi - .4])
    robot.root_states[:, 5] = (angles / 2).sin()
    robot.root_states[:, 6] = (angles / 2).cos()
    robot._update_platform_heading_command()
    torch.testing.assert_close(robot.commands[:, :2], initial[:, :2])
    torch.testing.assert_close(robot.commands[:, 2], torch.tensor([.3, -1., .2]))
    torch.testing.assert_close(robot._reward_highplatform_yaw()[robot.high_platform_mask],
                               torch.tensor([2.8**2, .4**2]))
    robot.commands.copy_(initial)
    platform_commands["heading_feedback"] = False
    robot._update_platform_heading_command()
    torch.testing.assert_close(robot.commands, initial)
    torch.testing.assert_close(robot._reward_highplatform_yaw()[robot.high_platform_mask],
                               torch.tensor([2.8**2, .4**2]))
    platform_commands["heading_feedback"] = True
    robot.cfg.commands.curriculum = False
    robot._update_platform_heading_command()
    torch.testing.assert_close(robot.commands, initial)


def test_task_mask_with_evaluation_columns_and_plane(robot):
    platforms = [f"platform_{cm}cm" for cm in range(45, 111, 5)]
    robot.terrain = SimpleNamespace(type_names=["stairs_convex_35cm"] + platforms)
    robot.num_envs = len(robot.terrain.type_names)
    robot.terrain_types = torch.arange(robot.num_envs)
    robot.terrain_levels = torch.zeros_like(robot.terrain_types)
    robot._init_platform_terrain_mask()
    assert robot.high_platform_mask.tolist() == [False] + [True] * len(platforms)
    del robot.terrain
    robot._init_platform_terrain_mask()
    assert not robot.high_platform_mask.any()


@pytest.mark.parametrize("count", [2, 3, 8, 16])
def test_small_g12_batches_visit_ordinary_and_platform_terrain(robot, monkeypatch, count):
    monkeypatch.setattr(type(robot).__base__, "_get_env_origins", lambda self: None)
    robot.cfg.rewards.platform_scales = platform_reward_scales()
    robot.cfg.terrain = SimpleNamespace(curriculum=True, num_cols=20, wf_sub_terrains=high_platform_wf_sub_terrains())
    robot.terrain.column_types = column_types(robot.cfg.terrain)
    robot.num_envs = count
    robot.terrain_levels = torch.zeros(count, dtype=torch.long)
    robot.terrain_types = torch.arange(count) * 20 // count
    robot.terrain_origins = torch.arange(60.).reshape(1, 20, 3)
    robot.env_origins = torch.zeros(count, 3)
    robot._get_env_origins()
    kinds = {robot.terrain.column_types[i] for i in robot.terrain_types.tolist()}
    assert kinds.difference(PLATFORM_TYPES)
    assert len(kinds.intersection(PLATFORM_TYPES)) == min(2, count - 1)
    torch.testing.assert_close(robot.env_origins, robot.terrain_origins[0, robot.terrain_types])


def test_reset_faces_raised_platform_from_approach_and_keeps_other_headings(robot, monkeypatch):
    robot.block_tiles = torch.as_tensor(robot.terrain.tile_types == "platform_block")
    robot.base_init_state = torch.tensor([0., 0., .6] + [0.] * 10)
    robot.env_origins = torch.tensor([[13.5, 13.5, 0.], [4.5, 13.5, -.7], [10.45, 4.5, 0.]])
    robot.custom_origins = True
    robot.wf_curriculum = object()
    robot.cfg.terrain.spawn_xy_range = .5
    robot.cfg.terrain.spawn_z_range = [0., .05]
    robot.cfg.terrain.yaw_init_range = 3.14
    robot.root_states = torch.zeros(3, 13)
    robot.sim = None
    robot.gym = SimpleNamespace(set_actor_root_state_tensor_indexed=lambda *args: None)
    runtime = robot._reset_root_states.__globals__
    monkeypatch.setitem(runtime, "gymtorch", SimpleNamespace(unwrap_tensor=lambda t: t))
    monkeypatch.setitem(runtime, "torch_rand_float", lambda lo, hi, shape, device: torch.full(shape, hi))

    def yaw_quaternion(yaw, axis):
        zeros = torch.zeros_like(yaw)
        return torch.stack((zeros, zeros, (yaw / 2).sin(), (yaw / 2).cos()), dim=-1)

    monkeypatch.setitem(runtime, "quat_from_angle_axis", yaw_quaternion)
    robot._reset_root_states(torch.tensor([2, 0, 1]))
    torch.testing.assert_close(robot.root_states[2, :7], torch.tensor([10.95, 5., .65, 0., 0., 0., 1.]))
    assert robot.root_states[1, 2].item() == pytest.approx(-.05)
    assert torch.all(robot.root_states[:2, 5] > .99)


def test_high_platform_samples_do_not_expand_g7_command_curriculum(robot):
    calls = []
    robot.wf_curriculum = SimpleNamespace(
        uses_reward_command_curriculum=True, update_reward_command_ranges=calls.append
    )
    robot.update_command_curriculum(torch.tensor([1, 2]))
    robot.update_command_curriculum(torch.tensor([0, 1, 2]))
    assert len(calls) == 1 and calls[0].tolist() == [0]


def test_v1_uses_final_xy_distance_and_command_for_all_terrains(robot):
    # All three robots reached 6 m earlier. Only the first is still far enough away.
    robot.wf_curriculum = make_env(("stairs_up", "platform_pit", "platform_block")).wf
    robot.wf_curriculum.terrain_length = 8.0  # WF excludes the two 0.5 m borders
    robot.wf_curriculum.max_move_distance.fill_(6.0)
    robot.terrain.env_length = 9.0
    robot.max_episode_length_s = 25.0
    robot.episode_length_buf = torch.full((3,), 200, dtype=torch.long)
    robot.env_origins = torch.tensor([[13.5, 13.5, 0.], [4.5, 13.5, -.7], [10.45, 4.5, 0.]])
    robot.root_states = robot.env_origins + torch.tensor([[3., -4., 0.], [0., 0., 1.], [0., 4.5, .5]])
    robot.commands = torch.tensor([[.8, 0., 0.], [0., -.8, 0.], [0., .36, 1.]])
    ids = torch.tensor([2, 0, 1])
    robot.cfg.terrain.curriculum_rule = "dreamwaq"
    up, down = robot._terrain_move_masks(ids)
    # Env 2 equals both thresholds (4.5 m). Env 0 advances even though 5 m < .8 * 25 / 2.
    assert up.tolist() == [False, True, False]
    assert down.tolist() == [False, False, True]
    robot.commands[1] = torch.tensor([0., 0., 1.])
    assert not robot._terrain_move_masks(torch.tensor([1]))[1].item()

    # A second reset before the first physics step must neither advance nor demote.
    robot.episode_length_buf.zero_()
    robot.commands[1, 0] = 1.0
    up, down = robot._terrain_move_masks(ids)
    assert not up.any() and not down.any()

    robot.cfg.terrain.curriculum_rule = "wf"
    assert robot._terrain_move_masks(ids)[0].tolist() == [True, True, True]


@pytest.mark.parametrize("mixed_training,rule,near_edge_timeout", [
    (False, "wf", False), (True, "wf", True), (True, "dreamwaq", False),
])
@pytest.mark.parametrize("direction", [-1., 1.])
def test_tile_exit_and_physical_failure_have_distinct_outcomes(
        robot, mixed_training, rule, near_edge_timeout, direction):
    robot.platform_scales = platform_reward_scales() if mixed_training else {}
    robot.cfg.terrain.curriculum = True
    robot.cfg.terrain.curriculum_rule = rule
    robot.cfg.terrain.terrain_length, robot.cfg.terrain.terrain_width = 9., 9.
    robot.cfg.env = SimpleNamespace(fail_to_terminal_time_s=.6)
    robot.wf_curriculum = object()
    centers = torch.stack((robot.terrain_levels + .5, robot.terrain_types + .5), dim=1) * 9.
    robot.env_origins = torch.column_stack((centers, torch.zeros(3)))
    robot.env_origins[2, 0] -= 3.05  # lower approach of the raised platform
    robot.root_states = centers + direction * torch.tensor([[4.4, 0.], [0., -4.6], [3., 0.]])
    robot.contact_forces = torch.zeros(3, 1, 3)
    robot.termination_contact_indices = [0]
    robot.projected_gravity = torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., -1.]])
    robot.fail_buf = torch.tensor([0, 30, 0])
    robot.episode_length_buf = torch.full((3,), 200, dtype=torch.long)
    robot.max_episode_length = 1250
    robot.reset_buf = torch.zeros(3, dtype=torch.long)
    robot.time_out_buf = torch.zeros(3, dtype=torch.bool)
    robot.check_termination()
    assert robot.reset_buf.tolist() == [int(near_edge_timeout), 1, 0]
    assert robot.time_out_buf.tolist() == [near_edge_timeout, False, False]
    if rule == "dreamwaq":
        robot.root_states[0, 0] = centers[0, 0] + direction * 4.5
        robot.check_termination()
        assert not robot.reset_buf[0]
        robot.root_states[0, 0] += direction * .01
        robot.check_termination()
        assert robot.time_out_buf[0]
        robot.terrain.env_length = 9.0
        robot.max_episode_length_s = 25.0
        robot.commands = torch.ones(3, 3)
        assert robot._terrain_move_masks(torch.tensor([0]))[0].item()
