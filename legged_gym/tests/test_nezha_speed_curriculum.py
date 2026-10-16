"""Nezha reward-driven speed ranges with the retained WF terrain/sampler."""

import ast
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from rsl_rl.utils.wf_curriculum import WFCurriculum, default_wf_command_config


def make_env(kinds=("flat", "stairs_up", "wave"), curriculum=True):
    settings = default_wf_command_config()
    settings["range_curriculum"] = "nezha_reward"
    count = len(kinds)
    env = SimpleNamespace(
        num_envs=count, device="cpu", dt=0.02,
        common_step_counter=0, max_episode_length=1250, max_episode_length_s=25.0,
        cfg=SimpleNamespace(
            commands=SimpleNamespace(
                curriculum=curriculum, wf_settings=settings,
                max_curriculum=1.5, max_curriculum_yaw=1.5,
            ),
            terrain=SimpleNamespace(
                mesh_type="trimesh", terrain_length=9.0,
                sub_terrain_border_width=0.5, curriculum=True,
            ),
        ),
        command_ranges={
            "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.6, 0.6],
            "ang_vel_yaw": [-1.0, 1.0],
        },
        terrain=SimpleNamespace(column_types=list(kinds)),
        terrain_types=torch.arange(count),
        root_states=torch.zeros(count, 13), env_origins=torch.zeros(count, 3),
        commands=torch.zeros(count, 4),
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        reward_scales={
            "tracking_lin_vel": 3.0 * 0.02, "tracking_ang_vel": 1.5 * 0.02,
            "lin_vel_z": -2.0 * 0.02, "base_height": -1.0 * 0.02,
        },
        episode_sums={
            "tracking_lin_vel": torch.zeros(count),
            "tracking_ang_vel": torch.zeros(count),
        },
    )
    env.wf_curriculum = WFCurriculum(env)
    return env


def set_tracking_scores(env, linear, angular):
    for key, value in (("tracking_lin_vel", linear), ("tracking_ang_vel", angular)):
        env.episode_sums[key][:] = (
            value * env.reward_scales[key] * env.max_episode_length
        )


@pytest.mark.parametrize("score", [0.0, 0.79, 0.8])
def test_rewards_must_strictly_exceed_eighty_percent(score):
    env = make_env()
    original = deepcopy(env.command_ranges)
    set_tracking_scores(env, score, score)
    env.wf_curriculum.update_reward_command_ranges(torch.arange(env.num_envs))
    assert env.command_ranges == original


@pytest.mark.parametrize("linear,angular,expected_x,expected_yaw", [
    (0.81, 0.79, 1.2, 1.0),
    (0.79, 0.81, 1.0, 1.1),
    (0.81, 0.81, 1.2, 1.1),
])
def test_linear_and_yaw_thresholds_expand_independently(linear, angular, expected_x, expected_yaw):
    env = make_env()
    set_tracking_scores(env, linear, angular)
    env.wf_curriculum.update_reward_command_ranges(torch.arange(env.num_envs))
    assert env.command_ranges["lin_vel_x"] == pytest.approx([-expected_x, expected_x])
    assert env.command_ranges["ang_vel_yaw"] == pytest.approx([-expected_yaw, expected_yaw])
    assert env.command_ranges["lin_vel_y"] == [-0.6, 0.6]


def test_reward_mean_uses_only_reset_environments_and_maximum_episode_length():
    env = make_env()
    set_tracking_scores(env, 1.0, 1.0)
    # A short, perfectly tracked episode must not count as a full good episode.
    env.episode_length_buf[0] = 125
    for values in env.episode_sums.values():
        values[0] *= 0.1
    env.wf_curriculum.update_reward_command_ranges(torch.tensor([0]))
    assert env.command_ranges["lin_vel_x"] == [-1.0, 1.0]
    assert env.command_ranges["ang_vel_yaw"] == [-1.0, 1.0]
    env.wf_curriculum.update_reward_command_ranges(torch.tensor([1, 2]))
    assert env.command_ranges["lin_vel_x"] == pytest.approx([-1.2, 1.2])
    assert env.command_ranges["ang_vel_yaw"] == pytest.approx([-1.1, 1.1])


def test_expansion_stops_at_nezha_caps_and_preserves_lateral_range():
    env = make_env()
    set_tracking_scores(env, 1.0, 1.0)
    for _ in range(12):
        env.wf_curriculum.update_reward_command_ranges(torch.arange(env.num_envs))
    assert env.command_ranges == {
        "lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-0.6, 0.6],
        "ang_vel_yaw": [-1.5, 1.5],
    }


def test_iteration_stages_cannot_expand_or_overwrite_reward_driven_ranges():
    env = make_env()
    env.common_step_counter = 75000 * 24
    env.wf_curriculum.update_command_ranges()
    assert env.command_ranges["lin_vel_x"] == [-1.0, 1.0]
    assert env.command_ranges["ang_vel_yaw"] == [-1.0, 1.0]
    set_tracking_scores(env, 1.0, 1.0)
    env.wf_curriculum.update_reward_command_ranges(torch.arange(env.num_envs))
    expected = deepcopy(env.command_ranges)
    env.wf_curriculum.resample(torch.arange(env.num_envs))
    env.common_step_counter = 0
    env.wf_curriculum.update_command_ranges()
    assert env.command_ranges == expected


def test_disabled_curriculum_keeps_fixed_play_commands_even_after_good_rewards():
    env = make_env(("stairs_up",), curriculum=False)
    env.command_ranges = {
        "lin_vel_x": [2.0, 2.0], "lin_vel_y": [0.1, 0.1],
        "ang_vel_yaw": [0.0, 0.0],
    }
    expected = deepcopy(env.command_ranges)
    set_tracking_scores(env, 1.0, 1.0)
    env.wf_curriculum.update_reward_command_ranges(torch.tensor([0]))
    env.wf_curriculum.resample(torch.tensor([0]))
    assert env.command_ranges == expected
    assert env.commands[0, :3].tolist() == pytest.approx([2.0, 0.1, 0.0])


def test_expanded_ranges_still_obey_wf_terrain_limits_and_limiting_samples():
    env = make_env()
    set_tracking_scores(env, 1.0, 1.0)
    for _ in range(5):
        env.wf_curriculum.update_reward_command_ranges(torch.arange(env.num_envs))
    env.wf_curriculum.cfg["limit_vel_prob"] = 1.0
    env.wf_curriculum.resample(torch.arange(env.num_envs))
    assert env.wf_curriculum.env_command_ranges["lin_vel_x"][:, 1].tolist() == [1.5, 1.0, 1.5]
    assert env.commands[:, 0].abs().tolist() == [1.5, 1.0, 1.5]
    assert env.commands[:, 1].abs().tolist() == pytest.approx([0.6] * 3)
    first = env.commands[:, :3].clone()
    env.wf_curriculum.resample(torch.arange(env.num_envs))
    assert torch.equal(env.commands[:, :3], -first)


def test_wf_timer_dynamic_sampling_standing_and_reward_schedules_are_retained():
    env = make_env(("flat",) * 32)
    wf = env.wf_curriculum
    assert wf.resampling_time == 5.0
    assert wf.cfg["dynamic_resample_commands"]
    wf.cfg["limit_vel_prob"] = 0.0
    env.episode_length_buf.fill_(env.max_episode_length)
    wf.resample(torch.arange(env.num_envs))
    # WF excludes small XY speeds and saturates at positive limits near timeout.
    assert torch.all(env.commands[:, 0] >= 1.0)
    assert torch.all(env.commands[:, 0] <= 1.0 + 1e-6)
    assert torch.all(env.commands[:, 1] >= 0.6)
    # 0.6 itself rounds upward in float32; allow that extra endpoint ULP.
    assert torch.all(env.commands[:, 1] <= 0.6 + 1.1e-6)
    for iteration, probability, vertical, height in [
        (0, 0.0, -2.0, -1.0), (750, 0.075, -1.5, -2.35),
        (1500, 0.15, -1.0, -3.7), (5000, 0.15, -1.0, -10.0),
    ]:
        env.common_step_counter = iteration * 24
        wf.resample(torch.arange(env.num_envs))
        wf.update_reward_curriculum()
        assert wf.zero_command_prob == pytest.approx(probability)
        assert env.reward_scales["lin_vel_z"] == pytest.approx(vertical * env.dt)
        assert env.reward_scales["base_height"] == pytest.approx(height * env.dt)


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_older_wf_settings_keep_iteration_curriculum(explicit_mode):
    env = make_env()
    settings = default_wf_command_config()
    if not explicit_mode:
        settings.pop("range_curriculum", None)
    env.cfg.commands.wf_settings = settings
    env.wf_curriculum = WFCurriculum(env)
    assert not env.wf_curriculum.uses_reward_command_curriculum
    env.common_step_counter = 75000 * 24
    env.wf_curriculum.update_command_ranges()
    assert env.command_ranges["lin_vel_x"] == [-3.0, 3.0]
    assert env.command_ranges["ang_vel_yaw"] == [-3.14, 3.14]


def load_nezha_method(name):
    """Compile the actual method body without importing simulator extensions."""
    source = Path(__file__).parents[1] / "envs" / "nezha" / "nezha_robot.py"
    module = ast.parse(source.read_text())
    robot = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NezhaRobot")
    method = next(node for node in robot.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("step,expected_x", [(1249, 1.0), (1250, 1.2), (2500, 1.2)])
def test_actual_reset_entry_checks_rewards_only_on_original_global_cadence(step, expected_x):
    env = make_env()
    env.cfg.env = SimpleNamespace(send_timeouts=False)
    env.common_step_counter = step
    env.episode_length_buf.fill_(env.max_episode_length)
    env.terrain_levels = torch.zeros(env.num_envs, dtype=torch.long)
    env.extras = {}
    env.obs_buf = torch.zeros(env.num_envs, 57)
    env.privileged_obs_buf = None
    for name in (
        "last_last_actions", "actions", "last_actions", "last_joint_pos_target",
        "last_last_joint_pos_target", "last_dof_vel", "feet_air_time",
        "reset_buf", "fail_buf", "envs_steps_buf",
    ):
        setattr(env, name, torch.zeros(env.num_envs))
    for name in (
        "_update_terrain_curriculum", "_randomize_dof_props",
        "_refresh_actor_dof_props", "_reset_dofs", "_reset_root_states",
    ):
        setattr(env, name, lambda ids: None)
    env.update_command_curriculum = MethodType(load_nezha_method("update_command_curriculum"), env)
    set_tracking_scores(env, 1.0, 1.0)
    load_nezha_method("reset_idx")(env, torch.arange(env.num_envs))
    assert env.command_ranges["lin_vel_x"] == pytest.approx([-expected_x, expected_x])
    assert all(torch.count_nonzero(values) == 0 for values in env.episode_sums.values())
    assert torch.count_nonzero(env.episode_length_buf) == 0
