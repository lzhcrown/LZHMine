"""G7-to-platform initialization and mixed WF command integration without Isaac Gym."""

import sys

import torch

from experiments.registry import PLATFORM_EXPERIMENT, clone_config, configure_experiment, get_experiment
from legged_gym.terrains.platform_rewards import platform_reward_scales
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.wf_curriculum import WFCurriculum
from test_cts_checkpoint_restoration import restore_args
from test_cts_experiments import actual_config_module, config_dict
from test_cts_initialization import all_moe_config
from test_cts_runner import FakeVectorEnv, cpu_task_registry
from test_wf_curriculum import make_env


def test_g7_warm_start_retains_target_terrain_commands_rewards_and_two_values(
        tmp_path, monkeypatch, actual_config_module, cpu_task_registry):
    class CpuEnv(FakeVectorEnv):
        def __init__(self, cfg, **kwargs):
            super().__init__()
            self.cfg = cfg
            self.dt = cfg.sim.dt * cfg.control.decimation
            self.high_platform_mask = torch.arange(self.num_envs) % 2 == 0

    source_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    source_cfg.rewards.scales.stand_still = -3.1  # distinguish the restored source settings
    source_settings = config_dict(cpu_task_registry, configure_experiment(
        actual_config_module.NezhaMoECTSCfgPPO(), "g7_all_moe",
    ))
    for section, overrides in all_moe_config().items():
        if isinstance(overrides, dict):
            source_settings[section].update(overrides)
    source = MoECTSOnPolicyRunner(CpuEnv(source_cfg), source_settings)
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

    group = get_experiment(PLATFORM_EXPERIMENT)
    env_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group.configure_environment(env_cfg)
    train_cfg = configure_experiment(actual_config_module.NezhaMoECTSCfgPPO(), PLATFORM_EXPERIMENT)
    cpu_task_registry.register(group.TASK_NAME, CpuEnv, env_cfg, train_cfg)
    registry_module = sys.modules[type(cpu_task_registry).__module__]
    monkeypatch.setattr(registry_module, "parse_sim_params", lambda *args: None)
    args = restore_args("init_from", path)
    env, restored = cpu_task_registry.make_env(group.TASK_NAME, args=args)
    assert restored.rewards.scales.stand_still == -3.1
    assert restored.rewards.platform_scales == platform_reward_scales()
    assert (restored.terrain.num_rows, restored.terrain.difficulty_reference_rows) == (20, 10)
    for kind in ("platform_pit", "platform_block"):
        spec = restored.terrain.wf_sub_terrains[kind]
        assert spec["proportion"] == .10
        assert spec["platform_height_range"] == [.20, .70]
    assert "obstacles" not in restored.terrain.wf_sub_terrains
    assert restored.commands.wf_settings["terrain_max_command_ranges"]["high_platform"] == {
        "lin_vel_x": [0.5, 1.0], "lin_vel_y": [0.0, 0.0], "ang_vel_yaw": [0.0, 0.0],
    }
    assert "high_platform" not in source_cfg.terrain.wf_sub_terrains
    assert "high_platform" not in source_cfg.commands.wf_settings["terrain_max_command_ranges"]

    target, _ = cpu_task_registry.make_alg_runner(env, group.TASK_NAME, args=args, log_root=None)
    assert target.policy_config["critic_task_count"] == 2
    probe = torch.randn(5, source.env.num_obs)
    torch.testing.assert_close(target.alg.actor_critic.act_inference(probe),
                               source.alg.actor_critic.act_inference(probe))
    target.learn(1)
    assert target.current_learning_iteration == source.current_learning_iteration + 1


def test_high_platform_and_locomotion_commands_in_the_same_resample(actual_config_module):
    env = make_env(("flat",) * 32)
    group_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(PLATFORM_EXPERIMENT).configure_environment(group_cfg)
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
        assert torch.all((platform[:, 0] >= 0.5) & (platform[:, 0] <= 1.0))
        assert torch.count_nonzero(platform[:, 1:]) == 0
        locomotion = env.commands[~env.high_platform_mask, :2]
        assert torch.all(locomotion.abs() == 0.5)
        torch.testing.assert_close(wf.commands_xy_accumulation, env.commands[:, :2])
