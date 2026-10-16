"""WF command and terrain curriculum checks without either simulator."""

from types import SimpleNamespace
import ctypes
import math

import pytest
import torch

from rsl_rl.utils.wf_curriculum import (
    WFCurriculum,
    configure_wf_timing,
    default_wf_command_config,
    sample_disjoint_intervals,
)


@pytest.mark.parametrize("terrain_profile,expected_steps", [("legbot_wf", 1250), ("wheel_gym_cq", 1251)])
def test_gym_c_float_timestep_does_not_add_a_wf_episode_or_randomization_step(terrain_profile, expected_steps):
    simulated_dt = ctypes.c_float(0.005).value * 4
    env = SimpleNamespace(
        dt=simulated_dt, max_episode_length_s=25.0,
        max_episode_length=math.ceil(25.0 / simulated_dt),
        cfg=SimpleNamespace(
            commands=SimpleNamespace(curriculum_profile="legbot_wf"),
            terrain=SimpleNamespace(generator_profile=terrain_profile),
            control=SimpleNamespace(decimation=4), sim=SimpleNamespace(dt=0.005),
            domain_rand=SimpleNamespace(
                push_interval_s=10.0, rand_interval_s=4.0,
                push_interval=math.ceil(10.0 / simulated_dt),
                rand_interval=math.ceil(4.0 / simulated_dt),
            ),
        ),
    )
    configure_wf_timing(env)
    assert env.max_episode_length == expected_steps
    if terrain_profile == "legbot_wf":
        assert env.dt == 0.02
        assert env.cfg.domain_rand.push_interval == 500
        assert env.cfg.domain_rand.rand_interval == 200
    else:
        assert env.dt == simulated_dt
        assert env.cfg.domain_rand.push_interval == 501
        assert env.cfg.domain_rand.rand_interval == 201


def make_env(kinds=("flat", "stairs_up", "wave"), curriculum=True):
    num_envs = len(kinds)
    settings = default_wf_command_config()
    env = SimpleNamespace(
        num_envs=num_envs, device="cpu", dt=0.02,
        common_step_counter=0, max_episode_length=1250, max_episode_length_s=25.0,
        cfg=SimpleNamespace(
            commands=SimpleNamespace(curriculum=curriculum, wf_settings=settings),
            terrain=SimpleNamespace(mesh_type="trimesh", terrain_length=10.0, sub_terrain_border_width=0.5),
        ),
        command_ranges={"lin_vel_x": [-0.5, 0.5], "lin_vel_y": [-0.5, 0.5], "ang_vel_yaw": [-1.0, 1.0]},
        terrain=SimpleNamespace(column_types=list(kinds)),
        terrain_types=torch.arange(num_envs),
        root_states=torch.zeros(num_envs, 13), env_origins=torch.zeros(num_envs, 3),
        commands=torch.zeros(num_envs, 4), episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        reward_scales={"lin_vel_z": -4.0 * 0.02, "base_height": -1.0 * 0.02},
    )
    env.wf = WFCurriculum(env)
    return env


def test_default_config_returns_independent_legbot_values():
    first = default_wf_command_config()
    first["terrain_max_command_ranges"]["flat"]["lin_vel_x"][1] = 999
    cfg = default_wf_command_config()
    assert cfg["terrain_max_command_ranges"]["flat"]["lin_vel_x"] == [-3.0, 3.0]
    assert cfg["zero_command_curriculum"]["end_value"] == 0.15
    assert cfg["num_steps_per_iter"] == 24
    assert cfg["resampling_time"] == 5.0


@pytest.mark.parametrize("iteration,expected", [
    (0, [0.5, 0.5, 1.0]),
    (19999, [0.5, 0.5, 1.0]),
    (20000, [1.0, 1.0, 1.5]),
    (49999, [1.0, 1.0, 1.5]),
    (50000, [2.0, 1.0, 2.0]),
    (75000, [3.0, 1.0, 3.14]),
])
def test_iteration_stages_use_24_policy_steps_and_terrain_limits(iteration, expected):
    env = make_env()
    env.common_step_counter = iteration * 24
    env.wf.update_command_ranges()
    assert [env.command_ranges[key][1] for key in env.wf.env_command_ranges] == expected
    flat = [env.wf.env_command_ranges[key][0, 1].item() for key in env.wf.env_command_ranges]
    assert flat == pytest.approx(expected)
    assert env.wf.env_command_ranges["lin_vel_x"][1, 1] <= 1.0
    assert env.wf.env_command_ranges["ang_vel_yaw"][1, 1] <= 1.5
    assert env.wf.env_command_ranges["lin_vel_x"][2, 1] <= 1.5
    # Recompute by absolute training clock, including when restoring a run.
    env.common_step_counter = 0
    env.wf.update_command_ranges()
    assert env.command_ranges["lin_vel_x"] == [-0.5, 0.5]


def test_stage_does_not_advance_one_policy_step_early():
    env = make_env()
    env.common_step_counter = 20000 * 24 - 1
    env.wf.update_command_ranges()
    assert env.command_ranges["lin_vel_x"] == [-0.5, 0.5]
    env.common_step_counter += 1
    env.wf.update_command_ranges()
    assert env.command_ranges["lin_vel_x"] == [-1.0, 1.0]


def test_disjoint_sampling_avoids_deadzone_and_matches_reference_saturation():
    torch.manual_seed(17)
    lower = torch.full((4096,), 0.3)
    values = sample_disjoint_intervals(lower, torch.full_like(lower, -0.5), torch.full_like(lower, 0.5))
    assert values.abs().min() >= 0.3 - 1e-6
    assert values.abs().max() <= 0.5 + 1e-6
    assert (values > 0).float().mean() == pytest.approx(0.5, abs=0.03)
    saturated = sample_disjoint_intervals(torch.ones(100), torch.full((100,), -0.5), torch.full((100,), 0.5))
    assert torch.all(saturated >= 0.5)
    assert torch.all(saturated <= 0.5 + 1e-6)


def test_terrain_uses_maximum_displacement_strict_thresholds_and_command_vector_sum():
    env = make_env(("flat",) * 5)
    ids = torch.arange(5)
    env.wf.max_move_distance[:] = torch.tensor([4.51, 4.50, 2.0, 1.99, 0.0])
    env.wf.commands_xy_accumulation[:] = torch.tensor([[100.0, 0.0], [0.0, 0.0], [0.8, 0.0], [0.8, 0.0], [0.0, 0.0]])
    up, down = env.wf.terrain_move_masks(ids)
    assert up.tolist() == [True, False, False, False, False]
    assert down.tolist() == [False, False, False, True, False]
    # Returning to the origin does not discard the furthest episode position.
    env.wf.time_left.fill_(5.0)
    env.root_states[:, 0] = 6.0
    env.wf.step()
    env.root_states[:, 0] = 0.0
    env.wf.step()
    assert torch.equal(env.wf.max_move_distance, torch.full((5,), 6.0))


def test_terrain_expected_distance_accounts_for_current_standing_probability():
    env = make_env(("flat",))
    env.wf.commands_xy_accumulation[0, 0] = 1.0
    env.wf.max_move_distance[0] = 2.2
    env.wf.zero_command_prob = 0.0
    assert env.wf.terrain_move_masks(torch.tensor([0]))[1].item()
    env.wf.zero_command_prob = 0.15
    assert not env.wf.terrain_move_masks(torch.tensor([0]))[1].item()


def test_continuous_limiting_commands_invert_and_accumulate_as_vectors():
    env = make_env(("flat",) * 32)
    env.wf.cfg["limit_vel_prob"] = 1.0
    ids = torch.arange(env.num_envs)
    env.wf.resample(ids)
    first = env.commands[:, :3].clone()
    assert torch.all(first[:, :2].abs() == 0.5)
    env.wf.resample(ids)
    assert torch.equal(env.commands[:, :3], -first)
    assert torch.equal(env.wf.commands_xy_accumulation, torch.zeros(32, 2))
    env.wf.max_move_distance.fill_(5.0)
    env.wf.reset(ids)
    assert torch.count_nonzero(env.wf.max_move_distance) == 0
    assert torch.equal(env.wf.commands_xy_accumulation, env.commands[:, :2])


def test_standing_zeroes_commands_shortens_timer_and_can_request_yaw():
    env = make_env(("flat",) * 128)
    ids = torch.arange(env.num_envs)
    env.wf.cfg["limit_vel_prob"] = 0.0
    env.wf.cfg["zero_command_curriculum"]["start_value"] = 1.0
    env.wf.cfg["zero_command_curriculum"]["end_value"] = 1.0
    env.wf.cfg["limit_ang_vel_at_zero_command_prob"] = 0.0
    env.episode_length_buf[:] = 500  # remaining standing budget is 0.9375 s.
    env.wf.resample(ids)
    assert torch.count_nonzero(env.commands) == 0
    assert torch.allclose(env.wf.time_left, torch.full((128,), 0.9375))
    env.wf.cfg["limit_ang_vel_at_zero_command_prob"] = 1.0
    env.wf.resample(ids)
    assert torch.count_nonzero(env.commands[:, :2]) == 0
    assert torch.all(env.commands[:, 2].abs() == 1.0)
    # Near episode end standing is disallowed if crossing the tile needs time.
    env.episode_length_buf.fill_(1249)
    env.wf.resample(ids)
    assert torch.all(env.commands[:, :2].abs() >= 0.5)


def test_standing_probability_ramps_and_reward_scales_use_policy_dt():
    env = make_env()
    ids = torch.arange(env.num_envs)
    for iteration, probability, vertical, height in [
        (0, 0.0, -2.0, -1.0), (750, 0.075, -1.5, -2.35),
        (1500, 0.15, -1.0, -3.7), (5000, 0.15, -1.0, -10.0),
    ]:
        env.common_step_counter = iteration * 24
        env.wf.resample(ids)
        env.wf.update_reward_curriculum()
        assert env.wf.zero_command_prob == pytest.approx(probability)
        assert env.reward_scales["lin_vel_z"] == pytest.approx(vertical * env.dt)
        assert env.reward_scales["base_height"] == pytest.approx(height * env.dt)


def test_time_left_matches_reference_float_clock_at_nominal_250_step_boundary():
    env = make_env(("flat",), curriculum=False)
    env.command_ranges = {"lin_vel_x": [0.3, 0.3], "lin_vel_y": [0.0, 0.0], "ang_vel_yaw": [0.0, 0.0]}
    env.wf.reset(torch.tensor([0]))
    reference_time = torch.tensor([5.0])
    expected_samples = 1
    for _ in range(251):
        reference_time -= 0.02
        if reference_time.item() <= 0.0:
            reference_time[:] = 5.0
            expected_samples += 1
        env.wf.step()
        assert torch.equal(env.wf.time_left, reference_time)
    assert env.wf.commands_xy_accumulation[0, 0].item() == pytest.approx(0.3 * expected_samples)
    assert expected_samples == 2


def test_timeout_reset_uses_completed_episode_clock_before_clearing_it():
    env = make_env(("flat",) * 32)
    env.wf.cfg["limit_vel_prob"] = 0.0
    env.episode_length_buf.fill_(env.max_episode_length)
    env.wf.reset(torch.arange(env.num_envs))
    # The reference manager resets commands before episode_length_buf. With
    # only one nominal step remaining, its interval sampler saturates at +max.
    assert torch.all(env.commands[:, :2] >= 0.5)
    assert torch.all(env.commands[:, :2] <= 0.5 + 1e-6)
    env.episode_length_buf.zero_()
    env.wf.step()
    assert torch.allclose(env.wf.time_left, torch.full((32,), 4.98))


def test_disabled_curriculum_preserves_fixed_play_commands_and_inference_buffers():
    env = make_env(("stairs_up",), curriculum=False)
    env.command_ranges = {"lin_vel_x": [2.0, 2.0], "lin_vel_y": [0.1, 0.1], "ang_vel_yaw": [0.0, 0.0]}
    ids = torch.tensor([0])
    names = ("time_left", "max_move_distance", "commands_xy_accumulation", "last_is_limit_vel")
    buffers = [getattr(env.wf, name) for name in names]
    with torch.inference_mode():
        env.wf.reset(ids)
        env.wf.step()
    # External reset must be legal after an inference_mode rollout.
    env.wf.reset(ids)
    assert all(getattr(env.wf, name) is buffer for name, buffer in zip(names, buffers))
    assert env.commands[0, :3].tolist() == pytest.approx([2.0, 0.1, 0.0])
    assert env.wf.zero_command_prob == 0.0
