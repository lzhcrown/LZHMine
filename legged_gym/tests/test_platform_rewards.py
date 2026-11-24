"""Exercise real terrain reward dispatch on CPU without the Isaac Gym extension."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legged_gym.evaluation.presets import PATCH_HIGH_PLATFORM_HEIGHTS_M
from legged_gym.terrains.platform_rewards import platform_reward_scales
from legged_gym.terrains.wf import PLATFORM_TYPES, column_types, high_platform_wf_sub_terrains


@pytest.fixture
def robot():
    # Load the Python classes without importing their native-simulator modules.
    root = Path(__file__).parents[1] / "envs"
    namespace = {"BaseTask": object, "torch": torch, "np": np, "PLATFORM_TYPES": PLATFORM_TYPES,
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
    return robot


def test_reward_dispatch_and_clipping(robot):
    assert robot.high_platform_mask.tolist() == [False, True, True]
    robot.platform_scales = {k: v * robot.dt for k, v in platform_reward_scales().items()}
    robot.reward_scales = {"tracking_lin_vel": 3., "ang_vel_xy": -.2, "lin_vel_z": -1.,
                           "collision": -.1, "termination": -.8}
    robot.rew_buf = torch.zeros(3)
    robot.commands = torch.zeros(3, 3)
    robot.base_lin_vel = torch.tensor([[0., 0., 3.], [0., 0., .5], [0., 0., 4.]])
    robot.base_ang_vel = torch.tensor([[0., 5., 0.], [0., 5., 2.], [0., 5., 2.]])
    robot.dof_pos = torch.zeros(3, 16)
    robot.default_dof_pos = torch.zeros(1, 16)
    robot.wheel_indices = [3, 7, 11, 15]
    robot.dof_pos_limits = torch.tensor([[-1., 1.]] * 16)
    for name in ("torques", "actions", "last_actions", "dof_vel", "last_dof_vel"):
        setattr(robot, name, torch.zeros(3, 16))
    robot.contact_forces = torch.zeros(3, 2, 3)
    robot.penalised_contact_indices = [0]
    robot.trunk_body_index = 1
    robot.contact_forces[:, 1, 0] = 1.0
    robot.reset_buf = torch.tensor([True, False, True])
    robot.time_out_buf = torch.tensor([True, False, False])
    robot._prepare_reward_function()
    robot.compute_reward()
    # Ordinary G7 uses pitch rate, signed rewards, no trunk cost. The
    # platform ignores pitch, counts trunk contact and floors the combined
    # nonterminal reward at zero before the -0.8 * dt failure penalty.
    torch.testing.assert_close(robot.rew_buf, torch.tensor([
        .06 - .2 * .02 * 25 - 1. * .02 * 9,
        .06 - .05 * .02 * 4 - 2. * .02 * .25 - .02,
        -.8 * .02,
    ]))
    assert robot.episode_sums["lin_vel_z"][2].item() == pytest.approx(-.64)
    assert robot.episode_sums["platform_positive_clip"][2] > .2
    torch.testing.assert_close(sum(robot.episode_sums.values()), robot.rew_buf)


def test_configured_rewards_match_m20_and_mujica_axes():
    assert platform_reward_scales() == {
        "tracking_lin_vel": 3., "lin_vel_z": -2., "ang_vel_xz": -.05,
        "torques": -5e-6, "dof_vel": -1e-6, "dof_acc": -1e-7,
        "collision": -1., "action_rate": -.011, "stand_still": -.5,
        "dof_pos_limits": -5., "hip_default": -.5, "run_still": -.05,
        "termination": -.8,
    }


def test_task_mask_with_evaluation_columns_and_plane(robot):
    platforms = [f"platform_{cm}cm" for cm in range(45, 81, 5)]
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


@pytest.mark.parametrize("mixed_training", [False, True])
def test_tile_exit_and_physical_failure_have_distinct_outcomes(robot, mixed_training):
    robot.platform_scales = platform_reward_scales() if mixed_training else {}
    robot.cfg.terrain.curriculum = True
    robot.cfg.terrain.terrain_length, robot.cfg.terrain.terrain_width = 9., 9.
    robot.cfg.env = SimpleNamespace(fail_to_terminal_time_s=.6)
    robot.wf_curriculum = object()
    centers = torch.stack((robot.terrain_levels + .5, robot.terrain_types + .5), dim=1) * 9.
    robot.env_origins = torch.column_stack((centers, torch.zeros(3)))
    robot.env_origins[2, 0] -= 3.05  # lower approach of the raised platform
    robot.root_states = centers + torch.tensor([[4.4, 0.], [-4.4, 0.], [3., 0.]])
    robot.contact_forces = torch.zeros(3, 1, 3)
    robot.termination_contact_indices = [0]
    robot.projected_gravity = torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., -1.]])
    robot.fail_buf = torch.tensor([0, 30, 0])
    robot.episode_length_buf = torch.zeros(3, dtype=torch.long)
    robot.max_episode_length = 1250
    robot.reset_buf = torch.zeros(3, dtype=torch.long)
    robot.time_out_buf = torch.zeros(3, dtype=torch.bool)
    robot.check_termination()
    assert robot.reset_buf.tolist() == [int(mixed_training), 1, 0]
    assert robot.time_out_buf.tolist() == [mixed_training, False, False]
