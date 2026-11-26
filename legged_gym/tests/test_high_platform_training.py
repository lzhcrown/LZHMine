"""G7-to-platform initialization and mixed WF command integration without Isaac Gym."""

import sys

import pytest
import torch

from experiments.registry import (
    PLATFORM_EXPERIMENT, PLATFORM_EXPERIMENT_KEYS, PLATFORM_V1_EXPERIMENT, PLATFORM_V2_EXPERIMENT,
    PLATFORM_V3_EXPERIMENT, PLATFORM_V4_EXPERIMENT, clone_config, configure_experiment, get_experiment,
)
from legged_gym.terrains.platform_rewards import (
    platform_reward_scales, platform_v1_reward_scales, platform_v2_reward_scales,
)
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.wf_curriculum import WFCurriculum
from test_cts_checkpoint_restoration import restore_args
from test_cts_experiments import actual_config_module, config_dict
from test_cts_initialization import all_moe_config
from test_cts_runner import FakeVectorEnv, cpu_task_registry
from test_platform_rewards import robot
from test_wf_curriculum import make_env


class CpuPlatformEnv(FakeVectorEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg.sim.dt * cfg.control.decimation
        self.high_platform_mask = torch.arange(self.num_envs) % 2 == 0


@pytest.mark.parametrize("target_key", PLATFORM_EXPERIMENT_KEYS)
def test_g7_warm_start_retains_target_terrain_commands_rewards_and_two_values(
        tmp_path, monkeypatch, actual_config_module, cpu_task_registry, target_key):
    source_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    source_cfg.rewards.scales.stand_still = -3.1  # distinguish the restored source settings
    source_settings = config_dict(cpu_task_registry, configure_experiment(
        actual_config_module.NezhaMoECTSCfgPPO(), "g7_all_moe",
    ))
    for section, overrides in all_moe_config().items():
        if isinstance(overrides, dict):
            source_settings[section].update(overrides)
    source = MoECTSOnPolicyRunner(CpuPlatformEnv(source_cfg), source_settings)
    path = tmp_path / "g7.pt"
    source.save(path)
    # Exercise the on-disk schema from before the extension and platform split.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_terrain = checkpoint["environment_config"]["terrain"]
    saved_terrain["num_rows"] = 10
    saved_terrain.pop("difficulty_reference_rows", None)
    for kind in ("platform_pit", "platform_block"):
        saved_terrain["wf_sub_terrains"].pop(kind)
        checkpoint["environment_config"]["commands"]["wf_settings"]["terrain_max_command_ranges"].pop(kind)
    saved_terrain["wf_sub_terrains"]["obstacles"] = dict(
        proportion=.20, slope_threshold=.25, obstacle_height_mode="choice",
        obstacle_width_range=[1., 2.], obstacle_height_range=[.05, .275],
        num_obstacles=20, platform_width=3.,
    )
    torch.save(checkpoint, path)

    group = get_experiment(target_key)
    env_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group.configure_environment(env_cfg)
    train_cfg = configure_experiment(actual_config_module.NezhaMoECTSCfgPPO(), target_key)
    cpu_task_registry.register(group.TASK_NAME, CpuPlatformEnv, env_cfg, train_cfg)
    registry_module = sys.modules[type(cpu_task_registry).__module__]
    monkeypatch.setattr(registry_module, "parse_sim_params", lambda *args: None)
    args = restore_args("init_from", path)
    env, restored = cpu_task_registry.make_env(group.TASK_NAME, args=args)
    assert restored.rewards.scales.stand_still == -3.1
    expected_scales = platform_reward_scales()
    if target_key in (PLATFORM_V1_EXPERIMENT, PLATFORM_V3_EXPERIMENT, PLATFORM_V4_EXPERIMENT):
        expected_scales = platform_v1_reward_scales()
        if target_key in (PLATFORM_V3_EXPERIMENT, PLATFORM_V4_EXPERIMENT):
            expected_scales["run_still"] = -.10
        assert restored.rewards.platform_base_height_target == (.54 if target_key == PLATFORM_V4_EXPERIMENT else .52)
        assert getattr(restored.rewards, "platform_reward_profile", None) is None
    elif target_key == PLATFORM_V2_EXPERIMENT:
        expected_scales = platform_v2_reward_scales()
        assert restored.rewards.platform_reward_profile == "dreamwaq_highplatform"
        assert restored.rewards.platform_heading_target == 0.
        assert restored.rewards.platform_base_height_target == .54
    assert restored.rewards.platform_scales == expected_scales
    expected_height = .54 if target_key in (PLATFORM_V2_EXPERIMENT, PLATFORM_V4_EXPERIMENT) else source_cfg.rewards.base_height_target
    assert restored.rewards.base_height_target == expected_height
    assert not getattr(restored.rewards, "platform_climb", False)
    assert restored.terrain.curriculum_rule == ("wf" if target_key == PLATFORM_EXPERIMENT else "dreamwaq")
    assert (restored.terrain.num_rows, restored.terrain.difficulty_reference_rows) == (20, 10)
    for kind in ("platform_pit", "platform_block"):
        spec = restored.terrain.wf_sub_terrains[kind]
        assert spec["proportion"] == .10
        assert spec["platform_height_range"] == [.20, .70]
    assert "obstacles" not in restored.terrain.wf_sub_terrains
    expected_commands = {
        "lin_vel_x": [0.5, 1.0], "lin_vel_y": [0.0, 0.0], "ang_vel_yaw": [0.0, 0.0],
        "heading_feedback": False,
    }
    assert restored.commands.wf_settings["terrain_max_command_ranges"]["high_platform"] == expected_commands
    assert "high_platform" not in source_cfg.terrain.wf_sub_terrains
    assert "high_platform" not in source_cfg.commands.wf_settings["terrain_max_command_ranges"]

    target, _ = cpu_task_registry.make_alg_runner(env, group.TASK_NAME, args=args, log_root=None)
    assert target.policy_config["critic_task_count"] == 2
    probe = torch.randn(5, source.env.num_obs)
    torch.testing.assert_close(target.alg.actor_critic.act_inference(probe),
                               source.alg.actor_critic.act_inference(probe))
    target.learn(1)
    assert target.current_learning_iteration == source.current_learning_iteration + 1


def test_resume_preserves_pre_alignment_g12_v1_rewards_and_commands(actual_config_module, cpu_task_registry):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(PLATFORM_V1_EXPERIMENT).configure_environment(cfg)
    previous = dict(platform_reward_scales(), tracking_ang_vel=1.5, orientation=-.5, base_height=-10.)
    saved_rewards = {"platform_scales": previous, "platform_base_height_target": .52}
    legacy = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(PLATFORM_EXPERIMENT).configure_environment(legacy)
    saved = {"rewards": saved_rewards, "commands": {"wf_settings": legacy.commands.wf_settings}}
    cpu_task_registry.restore_environment_config(cfg, saved)
    assert cfg.rewards.platform_scales == previous
    assert cfg.rewards.platform_base_height_target == .52
    assert getattr(cfg.rewards, "platform_reward_profile", None) != "dreamwaq_highplatform"
    assert getattr(cfg.rewards, "platform_heading_target", None) is None
    platform = cfg.commands.wf_settings["terrain_max_command_ranges"]["high_platform"]
    assert platform["lin_vel_x"] == [.5, 1.]
    assert "resampling_time" not in platform


def test_intermediate_highplatform_pitch_rate_snapshot_restores_until_v2_warm_start(actual_config_module, cpu_task_registry):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group = get_experiment(PLATFORM_V2_EXPERIMENT)
    group.configure_environment(cfg)
    previous = dict(cfg.rewards.platform_scales)
    previous["ang_vel_xy"] = previous.pop("ang_vel_xz")
    saved = {"rewards": {"platform_scales": previous, "platform_reward_profile": "dreamwaq_highplatform"}}
    cpu_task_registry.restore_environment_config(cfg, saved)
    assert cfg.rewards.platform_scales["ang_vel_xy"] == -.05
    assert "ang_vel_xz" not in cfg.rewards.platform_scales
    group.configure_environment(cfg)
    assert cfg.rewards.platform_scales["ang_vel_xz"] == -.05
    assert "ang_vel_xy" not in cfg.rewards.platform_scales
    assert cfg.rewards.scales.ang_vel_xy == -.2


@pytest.mark.parametrize("source_key,target_key", [
    (PLATFORM_EXPERIMENT, PLATFORM_V1_EXPERIMENT),
    (PLATFORM_V1_EXPERIMENT, PLATFORM_V2_EXPERIMENT),
    (PLATFORM_V2_EXPERIMENT, PLATFORM_V1_EXPERIMENT),
    (PLATFORM_V1_EXPERIMENT, PLATFORM_V3_EXPERIMENT),
    (PLATFORM_V2_EXPERIMENT, PLATFORM_V3_EXPERIMENT),
    (PLATFORM_EXPERIMENT, PLATFORM_V3_EXPERIMENT),
    (PLATFORM_V3_EXPERIMENT, PLATFORM_V4_EXPERIMENT),
])
def test_platform_group_transfer_requires_warm_start_and_copies_both_value_heads(
        tmp_path, monkeypatch, actual_config_module, cpu_task_registry, source_key, target_key):
    from test_cts_checkpoint_restoration import train_namespace

    source_config = all_moe_config()
    source_config["policy"]["critic_task_count"] = 2
    source_config["runner"].update(get_experiment(source_key).RUNNER)
    source_env_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(source_key).configure_environment(source_env_cfg)
    source = MoECTSOnPolicyRunner(CpuPlatformEnv(source_env_cfg), source_config)
    source.learn(1)
    path = tmp_path / (source_key + ".pt")
    source.save(path)
    train_cfg = configure_experiment(train_namespace(source_config, path), target_key)
    group = get_experiment(target_key)
    task = group.TASK_NAME
    target_env_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group.configure_environment(target_env_cfg)
    cpu_task_registry.register(task, CpuPlatformEnv, target_env_cfg, train_cfg)
    registry_module = sys.modules[type(cpu_task_registry).__module__]
    monkeypatch.setattr(registry_module, "parse_sim_params", lambda *args: None)
    with pytest.raises(ValueError, match="Use --init_from"):
        cpu_task_registry.make_env(task, args=restore_args("resume", path))
    with pytest.raises(ValueError, match="Use --init_from"):
        cpu_task_registry.make_alg_runner(source.env, task, args=restore_args("resume", path), log_root=None)
    args = restore_args("init_from", path)
    env, restored = cpu_task_registry.make_env(task, args=args)
    assert restored.rewards.platform_scales == target_env_cfg.rewards.platform_scales
    assert restored.rewards.base_height_target == target_env_cfg.rewards.base_height_target
    assert restored.rewards.platform_base_height_target == target_env_cfg.rewards.platform_base_height_target
    for attr in ("platform_heading_target", "platform_reward_profile"):
        assert getattr(restored.rewards, attr, None) == getattr(target_env_cfg.rewards, attr, None)
    assert restored.commands.wf_settings == target_env_cfg.commands.wf_settings
    target, _ = cpu_task_registry.make_alg_runner(env, task, args=args, log_root=None)
    assert target.current_learning_iteration == 1
    assert not target.alg.optimizer.state and not target.alg.optimizer_stu_enc.state
    for name, tensor in source.alg.actor_critic.state_dict().items():
        torch.testing.assert_close(target.alg.actor_critic.state_dict()[name], tensor, rtol=0, atol=0)
    target_path = tmp_path / (target_key + ".pt")
    target.save(target_path)
    train_cfg.runner.resume_path = str(target_path)
    cpu_task_registry.register(task, CpuPlatformEnv, target_env_cfg, train_cfg)
    resume = restore_args("resume", target_path)
    resumed_env, restored_cfg = cpu_task_registry.make_env(task, args=resume)
    resumed, _ = cpu_task_registry.make_alg_runner(resumed_env, task, args=resume, log_root=None)
    assert resumed.cfg["experiment_key"] == target_key
    assert restored_cfg.rewards.platform_scales == target_env_cfg.rewards.platform_scales
    assert restored_cfg.rewards.base_height_target == target_env_cfg.rewards.base_height_target
    assert restored_cfg.rewards.platform_base_height_target == target_env_cfg.rewards.platform_base_height_target


@pytest.mark.parametrize("target_key", PLATFORM_EXPERIMENT_KEYS)
def test_high_platform_and_locomotion_commands_in_the_same_resample(actual_config_module, target_key):
    env = make_env(("flat",) * 32)
    group_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(target_key).configure_environment(group_cfg)
    env.cfg.commands = group_cfg.commands
    env.terrain.column_types = ["platform_pit", "platform_block", "flat", "stairs_up"]
    env.terrain_types = torch.arange(env.num_envs) % len(env.terrain.column_types)
    env.high_platform_mask = env.terrain_types < 2
    # Force ordinary WF samples to its endpoint commands. This exercises its
    # signed lateral commands and makes accidental plain WF sampling detectable.
    env.cfg.commands.wf_settings["limit_vel_prob"] = 1.0
    wf = WFCurriculum(env)
    ids = torch.arange(env.num_envs)
    for iteration in (0, 5000):
        env.common_step_counter = iteration * wf.steps_per_iteration
        wf.reset(ids)
        platform = env.commands[env.high_platform_mask, :3]
        assert torch.all((platform[:, 0] >= .5) & (platform[:, 0] <= 1.0))
        assert torch.count_nonzero(platform[:, 1:]) == 0
        assert torch.all(wf.time_left[env.high_platform_mask] == 5.)
        assert torch.all(wf.time_left[~env.high_platform_mask] == 5.)
        locomotion = env.commands[~env.high_platform_mask, :2]
        assert torch.all(locomotion.abs() == 0.5)
        torch.testing.assert_close(wf.commands_xy_accumulation, env.commands[:, :2])


def test_v3_moving_posture_penalizes_leg_error_on_both_platform_shapes(actual_config_module, robot):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(PLATFORM_V3_EXPERIMENT).configure_environment(cfg)
    robot.cfg.rewards = cfg.rewards
    robot.wheel_indices = [3, 7, 11, 15]
    robot.commands = torch.tensor([[.8, 0., 0.]] * robot.num_envs)
    joint_names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                   for joint in ("hip", "thigh", "calf", "foot")]
    robot.default_dof_pos = torch.tensor([[cfg.init_state.default_joint_angles[name] for name in joint_names]])
    torch.testing.assert_close(robot.default_dof_pos, torch.tensor([[0., .8, -1.6, 0.] * 4]))
    robot.dof_pos = robot.default_dof_pos.repeat(robot.num_envs, 1)
    robot.dof_pos[:, robot.wheel_indices] = 7.0
    torch.testing.assert_close(robot._reward_run_still(), torch.zeros(robot.num_envs))
    robot.dof_pos[:, 1] += .25
    robot.dof_pos[:, 10] -= .50
    platform_cost = robot._reward_run_still()[robot.high_platform_mask] * cfg.rewards.platform_scales["run_still"]
    torch.testing.assert_close(platform_cost, torch.tensor([-.075, -.075]))


def test_legacy_v2_resume_keeps_saved_command_sampler_and_episode_timing(actual_config_module, cpu_task_registry, robot):
    group_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(PLATFORM_V2_EXPERIMENT).configure_environment(group_cfg)
    saved = config_dict(cpu_task_registry, group_cfg)
    platform = saved["commands"]["wf_settings"]["terrain_max_command_ranges"]["high_platform"]
    platform.pop("heading_feedback")
    platform.update(lin_vel_x=[0., 1.], resampling_time=10.)
    saved["rewards"].update(base_height_target=.50, platform_base_height_target=.556)
    cpu_task_registry.restore_environment_config(group_cfg, saved)
    assert group_cfg.rewards.base_height_target == .50
    assert group_cfg.rewards.platform_base_height_target == .556
    assert "heading_feedback" not in group_cfg.commands.wf_settings["terrain_max_command_ranges"]["high_platform"]
    env = make_env(("flat", "platform_pit", "platform_block"))
    env.cfg.commands = group_cfg.commands
    env.high_platform_mask = robot.high_platform_mask
    env.cfg.commands.wf_settings["limit_vel_prob"] = 1.0
    wf = WFCurriculum(env)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        wf.reset(torch.arange(env.num_envs))
    vx = env.commands[:, 0]
    assert 0 < vx[1] < .1 < vx[2] <= 1
    robot.commands = env.commands
    robot.cfg.rewards = group_cfg.rewards
    robot.platform_scales = group_cfg.rewards.platform_scales
    robot.default_dof_pos = torch.zeros(1, 16)
    robot.dof_pos = torch.zeros(3, 16)
    robot.dof_pos[:, 1:3] = torch.tensor([.3, .4])
    robot.wheel_indices = [3, 7, 11, 15]
    torch.testing.assert_close(robot._reward_stand_still()[1:], torch.tensor([.7, 0.]))
    torch.testing.assert_close(robot._reward_run_still()[1:], torch.tensor([0., .5]))

    # Stagger the two episode clocks by one step. Each platform must resample
    # exactly once at its own step 500, before the post-reward WF update.
    initial = env.commands[:, :2].clone()
    env.episode_length_buf[2] = 1
    for step in range(1, 501):
        env.episode_length_buf += 1
        wf.resample_platform_commands_if_due()
        if step == 499:
            torch.testing.assert_close(wf.commands_xy_accumulation[1], initial[1])
            torch.testing.assert_close(wf.commands_xy_accumulation[2], initial[2] + env.commands[2, :2])
        wf.step()
        if step == 260:
            # Ordinary WF has already resampled once at its five-second timer.
            torch.testing.assert_close(env.commands[0, :2], -initial[0])
        if step == 498:
            torch.testing.assert_close(wf.commands_xy_accumulation[1:], initial[1:])
    torch.testing.assert_close(wf.commands_xy_accumulation[1:], initial[1:] + env.commands[1:, :2])
