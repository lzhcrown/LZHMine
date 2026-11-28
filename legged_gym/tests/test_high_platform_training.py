"""G7-to-platform initialization and mixed WF command integration without Isaac Gym."""

import sys

import pytest
import torch

from deploy.deploy_mujoco.runtime import NezhaCTSPolicyRuntime
from experiments.common import export_main
from experiments.registry import (
    PLATFORM_EXPERIMENT, PLATFORM_EXPERIMENT_KEYS, PLATFORM_V1_EXPERIMENT, PLATFORM_V2_EXPERIMENT,
    PLATFORM_V3_EXPERIMENT, PLATFORM_V4_EXPERIMENT, PLATFORM_V5_EXPERIMENT, PLATFORM_V6_EXPERIMENT,
    SKILL_EXPERIMENT_KEYS, SKILL_V1_EXPERIMENT, SKILL_V2_EXPERIMENT, clone_config, configure_experiment, get_experiment,
)
from legged_gym.terrains.platform_rewards import (
    platform_reward_scales, platform_v1_reward_scales, platform_v2_reward_scales,
)
from rsl_rl.modules.frozen_terrain_encoder import TerrainEncoder, terrain_encoder_checkpoint
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import validate_checkpoint_policy
from rsl_rl.utils.cts_observations import observation_dimensions
from rsl_rl.utils.wf_curriculum import WFCurriculum
from test_cts_checkpoint_restoration import restore_args
from test_cts_experiments import actual_config_module, config_dict
from test_cts_initialization import all_moe_config
from test_cts_runner import FakeVectorEnv, cpu_task_registry, runner_config
from test_platform_rewards import robot
from test_wf_curriculum import make_env


class CpuPlatformEnv(FakeVectorEnv):
    def __init__(self, cfg, **kwargs):
        schema = cfg.env.observation_config
        if schema is not None:
            self.num_one_step_obs, self.num_obs, self.num_privileged_obs = observation_dimensions(schema)
        super().__init__()
        if schema is not None:
            self.observation_config = schema
        self.cfg = cfg
        self.dt = cfg.sim.dt * cfg.control.decimation
        self.high_platform_mask = torch.arange(self.num_envs) % 2 == 0


def stage_one_artifact(tmp_path):
    path = tmp_path / "terrain_encoder.pt"
    torch.save(terrain_encoder_checkpoint(TerrainEncoder()), path)
    return str(path)


@pytest.mark.parametrize("field,change", [
    ("skill_id", {"noise": .1}),
    ("height_scan", {"name": "wrong_map"}),
])
def test_v6_rejects_same_width_checkpoint_with_wrong_teacher_inputs(actual_config_module, field, change):
    group = get_experiment(PLATFORM_V6_EXPERIMENT)
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group.configure_environment(cfg)
    schema = cfg.env.observation_config
    term = next(term for term in schema["privileged_terms"] if term["name"] == field)
    term.update(change)
    assert observation_dimensions(schema) == (57, 228, 296)
    with pytest.raises(ValueError, match=field):
        validate_checkpoint_policy({"policy_config": group.POLICY, "observation_config": schema}, group.POLICY)


def test_v6_trains_with_teacher_skill_and_replays_from_proprioception(
        tmp_path, monkeypatch, actual_config_module):
    torch.manual_seed(17)
    group = get_experiment(PLATFORM_V6_EXPERIMENT)
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group.configure_environment(cfg)
    env = CpuPlatformEnv(cfg)
    settings = runner_config()
    settings["policy"].update(group.POLICY)
    settings["runner"].update(group.RUNNER)
    runner = MoECTSOnPolicyRunner(env, settings)
    env.max_episode_length = runner.num_steps_per_env + 1
    env.privileged[:, -1] = env.high_platform_mask.float()
    runner.learn(1)
    policy = runner.alg.actor_critic
    assert not (policy.student_frozen_terrain or policy.teacher_frozen_terrain or policy.teacher_map_attention)
    # Identical body history and map, different task ID, including in the teaching latent.
    history = torch.randn(1, 228).repeat(2, 1)
    privileged = torch.randn(1, 296).repeat(2, 1)
    privileged[:, -1] = torch.tensor([0., 1.])
    with torch.inference_mode():
        z = policy.teacher_encoder(privileged)
        assert not torch.allclose(z[0], z[1])
        actions = runner.get_inference_policy()(history)
        torch.testing.assert_close(actions[0], actions[1])
        student = policy.action_mean_with_diagnostics(history, privileged, is_teacher=False)[0]
        torch.testing.assert_close(student, actions)
        assert actions.shape == (2, 16)
    snapshot = tmp_path / "model_1.pt"
    runner.save(snapshot)
    output = tmp_path / "v6_policy.pt"
    monkeypatch.setattr(sys, "argv", ["export.py", "--checkpoint", str(snapshot), "--output", str(output)])
    export_main(PLATFORM_V6_EXPERIMENT)
    runtime = NezhaCTSPolicyRuntime(str(output), default_dof_pos=[0.] * 16,
                                    p_gains=[10.] * 16, d_gains=[2.] * 16, expected_policy=group.POLICY)
    assert runtime.policy.get_critic_type() == "task_mlp"
    assert runtime.policy.get_critic_task_count() == 2
    with torch.inference_mode():
        torch.testing.assert_close(runtime.policy(history), actions)
    applied, torques = runtime.step([0.] * 3, [0., 0., -1.], [.8, 0., 0.], [0.] * 16, [0.] * 16)
    assert applied.shape == torques.shape == (16,)
    assert torch.isfinite(applied).all() and torch.isfinite(torques).all()


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
    if target_key == SKILL_V1_EXPERIMENT:
        with pytest.raises(ValueError, match="--init_from is unsupported for frozen terrain policies"):
            cpu_task_registry.make_env(group.TASK_NAME, args=args)
        return
    if target_key == SKILL_V2_EXPERIMENT:
        args.map_encoder_checkpoint = None
        with pytest.raises(ValueError, match="requires --map_encoder_checkpoint"):
            cpu_task_registry.make_env(group.TASK_NAME, args=args)
        args.map_encoder_checkpoint = stage_one_artifact(tmp_path)
    env, restored = cpu_task_registry.make_env(group.TASK_NAME, args=args)
    assert restored.rewards.scales.stand_still == -3.1
    expected_scales = platform_reward_scales()
    if target_key in (PLATFORM_V1_EXPERIMENT, PLATFORM_V3_EXPERIMENT, PLATFORM_V4_EXPERIMENT,
                      PLATFORM_V5_EXPERIMENT, PLATFORM_V6_EXPERIMENT, *SKILL_EXPERIMENT_KEYS):
        expected_scales = platform_v1_reward_scales()
        if target_key in (PLATFORM_V3_EXPERIMENT, PLATFORM_V4_EXPERIMENT):
            expected_scales["run_still"] = -.10
        assert restored.rewards.platform_base_height_target == (.54 if target_key == PLATFORM_V4_EXPERIMENT else .52)
        assert getattr(restored.rewards, "platform_reward_profile", None) is None
    elif target_key == PLATFORM_V2_EXPERIMENT:
        expected_scales = dict(platform_v1_reward_scales(), run_still=-.10, hip_default=-6.,
                               stand_still=-2.25, orientation=-1.5, ang_vel_xz=-.15)
        assert restored.rewards.platform_reward_profile == "shared_platform"
        assert restored.rewards.platform_base_height_target == .55
    assert restored.rewards.platform_scales == expected_scales
    expected_height = {PLATFORM_V2_EXPERIMENT: .55, PLATFORM_V4_EXPERIMENT: .54}.get(
        target_key, source_cfg.rewards.base_height_target,
    )
    assert restored.rewards.base_height_target == expected_height
    assert not getattr(restored.rewards, "platform_climb", False)
    assert restored.terrain.curriculum_rule == ("wf" if target_key == PLATFORM_EXPERIMENT else "dreamwaq")
    expected_cap = 9 if target_key in (PLATFORM_V5_EXPERIMENT, PLATFORM_V6_EXPERIMENT, *SKILL_EXPERIMENT_KEYS) else None
    assert getattr(restored.terrain, "ordinary_max_level", None) == expected_cap
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
    scan = None
    if target_key in (PLATFORM_V6_EXPERIMENT, SKILL_V2_EXPERIMENT):
        skills = (torch.arange(5) % 2).float()[:, None]
        assert (env.num_one_step_obs, env.num_obs, env.num_privileged_obs) == (57, 228, 296)
        privileged = torch.randn(5, 295)
        scan = target.alg.actor_critic.student_height_scan(torch.cat((privileged, skills), dim=1))
        assert target.initialization["critic"] == "reinitialized_task_heads"
        for teacher in (False, True):
            actual = target.alg.actor_critic.action_mean_with_diagnostics(
                probe, torch.cat((privileged, skills), dim=1), teacher)
            expected = source.alg.actor_critic.action_mean_with_diagnostics(probe, privileged, teacher)
            torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(target.alg.actor_critic.act_inference(probe, scan),
                               source.alg.actor_critic.act_inference(probe))
    target.learn(1)
    assert target.current_learning_iteration == source.current_learning_iteration + 1


@pytest.mark.parametrize("warm_start", [False, True])
def test_restore_g13_map_from_before_the_ordinary_cap(actual_config_module, cpu_task_registry, warm_start):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group = get_experiment(SKILL_V2_EXPERIMENT)
    group.configure_environment(cfg)
    saved = config_dict(cpu_task_registry, cfg)
    assert saved["terrain"].pop("ordinary_max_level") == 9
    cpu_task_registry.restore_environment_config(cfg, saved, warm_start=warm_start)
    if warm_start:
        group.configure_environment(cfg)
    assert getattr(cfg.terrain, "ordinary_max_level", None) == (9 if warm_start else None)
    assert (cfg.terrain.num_rows, cfg.terrain.difficulty_reference_rows) == (20, 10)


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
    previous = platform_v2_reward_scales()
    previous["ang_vel_xy"] = previous.pop("ang_vel_xz")
    saved = {"rewards": {"platform_scales": previous, "platform_reward_profile": "dreamwaq_highplatform"}}
    cpu_task_registry.restore_environment_config(cfg, saved)
    assert cfg.rewards.platform_reward_profile == "dreamwaq_highplatform"
    assert cfg.rewards.platform_scales["ang_vel_xy"] == -.05
    assert "ang_vel_xz" not in cfg.rewards.platform_scales
    group.configure_environment(cfg)
    assert cfg.rewards.platform_reward_profile == "shared_platform"
    assert cfg.rewards.platform_scales["ang_vel_xz"] == -.15
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
    (PLATFORM_V4_EXPERIMENT, PLATFORM_V2_EXPERIMENT),
    (PLATFORM_V1_EXPERIMENT, PLATFORM_V5_EXPERIMENT),
    (PLATFORM_V5_EXPERIMENT, PLATFORM_V1_EXPERIMENT),
    (PLATFORM_V5_EXPERIMENT, PLATFORM_V6_EXPERIMENT),
    (PLATFORM_V1_EXPERIMENT, SKILL_V2_EXPERIMENT),
])
def test_platform_group_transfer_requires_warm_start_and_preserves_compatible_weights(
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
    if target_key == SKILL_V2_EXPERIMENT:
        args.map_encoder_checkpoint = stage_one_artifact(tmp_path)
    env, restored = cpu_task_registry.make_env(task, args=args)
    expected_cap = 9 if target_key in (PLATFORM_V5_EXPERIMENT, PLATFORM_V6_EXPERIMENT, *SKILL_EXPERIMENT_KEYS) else None
    assert getattr(restored.terrain, "ordinary_max_level", None) == expected_cap
    assert restored.rewards.platform_scales == target_env_cfg.rewards.platform_scales
    assert restored.rewards.base_height_target == target_env_cfg.rewards.base_height_target
    assert restored.rewards.platform_base_height_target == target_env_cfg.rewards.platform_base_height_target
    for attr in ("platform_heading_target", "platform_reward_profile"):
        assert getattr(restored.rewards, attr, None) == getattr(target_env_cfg.rewards, attr, None)
    assert restored.commands.wf_settings == target_env_cfg.commands.wf_settings
    target, _ = cpu_task_registry.make_alg_runner(env, task, args=args, log_root=None)
    assert target.current_learning_iteration == 1
    assert not target.alg.optimizer.state and not target.alg.optimizer_stu_enc.state
    independent_critic = target_key in (PLATFORM_V6_EXPERIMENT, SKILL_V2_EXPERIMENT)
    if independent_critic:
        assert target.initialization["critic"] == "reinitialized_task_heads"
    for name, tensor in source.alg.actor_critic.state_dict().items():
        if independent_critic and name.startswith("critic."):
            assert name not in target.alg.actor_critic.state_dict()
            continue
        transferred = target.alg.actor_critic.state_dict()[name]
        if independent_critic and transferred.shape != tensor.shape:
            assert transferred.ndim == tensor.ndim == 2 and transferred.shape[1] > tensor.shape[1]
            assert torch.count_nonzero(transferred[:, tensor.shape[1]:]) == 0
            transferred = transferred[:, :tensor.shape[1]]
        torch.testing.assert_close(transferred, tensor, rtol=0, atol=0)
    target_path = tmp_path / (target_key + ".pt")
    target.save(target_path)
    train_cfg.runner.resume_path = str(target_path)
    cpu_task_registry.register(task, CpuPlatformEnv, target_env_cfg, train_cfg)
    resume = restore_args("resume", target_path)
    resumed_env, restored_cfg = cpu_task_registry.make_env(task, args=resume)
    resumed, _ = cpu_task_registry.make_alg_runner(resumed_env, task, args=resume, log_root=None)
    assert resumed.cfg["experiment_key"] == target_key
    assert getattr(restored_cfg.terrain, "ordinary_max_level", None) == expected_cap
    assert restored_cfg.rewards.platform_scales == target_env_cfg.rewards.platform_scales
    assert restored_cfg.rewards.base_height_target == target_env_cfg.rewards.base_height_target
    assert restored_cfg.rewards.platform_base_height_target == target_env_cfg.rewards.platform_base_height_target
    if target_key == SKILL_V2_EXPERIMENT:
        assert resumed_env.num_obs == 228
        with pytest.raises(ValueError, match="cannot discard trained teacher map attention"):
            cpu_task_registry.make_alg_runner(
                source.env, args=restore_args("init_from", target_path),
                train_cfg=configure_experiment(train_namespace(source_config, path), source_key), log_root=None)


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


@pytest.mark.parametrize("command_x,posture_weight", [(.8, -.10), (0., -2.25)])
def test_v2_scores_identical_states_equally_on_stairs_pit_and_block(
        actual_config_module, cpu_task_registry, robot, command_x, posture_weight):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(PLATFORM_V2_EXPERIMENT).configure_environment(cfg)
    robot.cfg = cfg
    robot.platform_scales = {key: value * robot.dt for key, value in cfg.rewards.platform_scales.items()}
    robot.reward_scales = config_dict(cpu_task_registry, cfg.rewards.scales)
    robot.wf_curriculum = make_env().wf
    robot.wf_curriculum.env = robot
    robot.common_step_counter = 0
    robot.rew_buf = torch.zeros(3)
    robot._prepare_reward_function()
    robot.common_step_counter = 5000 * robot.wf_curriculum.steps_per_iteration
    robot.wf_curriculum.update_reward_curriculum()
    assert robot.reward_scales == robot.platform_scales
    assert set(robot.reward_names) == set(robot.platform_scales) - {"termination"}
    assert robot.high_platform_mask.tolist() == [False, True, True]

    robot.commands = torch.tensor([[command_x, 0., 0.]] * 3)
    robot.base_lin_vel = robot.commands.clone()
    robot.commands[:, 2] = .2
    robot.base_ang_vel = torch.tensor([[.4, 2., .2]] * 3)
    robot.projected_gravity = torch.tensor([[.6, 0., -.8]] * 3)
    robot.base_height = torch.full((3,), .50)
    robot.default_dof_pos = torch.zeros(1, 16)
    for field in ("dof_pos", "dof_vel", "last_dof_vel", "torques", "actions", "last_actions"):
        setattr(robot, field, torch.zeros(3, 16))
    robot.dof_pos[:, :3] = torch.tensor([.2, .3, .4])
    robot.dof_pos_limits = torch.tensor([[-1., 1.]] * 16)
    robot.wheel_indices = [3, 7, 11, 15]
    robot.contact_forces = torch.zeros(3, 2, 3)
    robot.penalised_contact_indices = [0]
    robot.trunk_body_index = 1
    robot.contact_forces[:, 1, 2] = 2.0
    robot.reset_buf = torch.zeros(3, dtype=torch.bool)
    robot.time_out_buf = torch.zeros_like(robot.reset_buf)
    robot.compute_reward()
    # Apply hip and tilt costs at both command speeds. Only the L1 leg posture weight changes.
    angular_rate_cost = -.15 * (.4**2 + .2**2)
    torch.testing.assert_close(robot.episode_sums["ang_vel_xz"], torch.full((3,), angular_rate_cost * robot.dt))
    expected_rate = 3. + 1.5 + angular_rate_cost + posture_weight * .9 - 6. * .2**2 - 1.5 * .6**2 - 10. * .05**2 - 1.
    torch.testing.assert_close(robot.rew_buf, torch.full((3,), expected_rate * robot.dt))

    robot.base_lin_vel[:, 2] = 4.
    robot.reset_buf.fill_(True)
    robot.compute_reward()
    # The vertical cost exceeds the ordinary single-term clip. All tiles apply
    # the full cost, floor the sum, then add the failure penalty.
    torch.testing.assert_close(robot.episode_sums["lin_vel_z"], torch.full((3,), -32. * robot.dt))
    torch.testing.assert_close(robot.episode_sums["platform_positive_clip"],
                               torch.full((3,), (32. - expected_rate) * robot.dt))
    torch.testing.assert_close(robot.rew_buf, torch.full((3,), -.8 * robot.dt))


def test_legacy_v2_resume_keeps_saved_command_sampler_and_episode_timing(actual_config_module, cpu_task_registry, robot):
    group_cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment(PLATFORM_V2_EXPERIMENT).configure_environment(group_cfg)
    saved = config_dict(cpu_task_registry, group_cfg)
    platform = saved["commands"]["wf_settings"]["terrain_max_command_ranges"]["high_platform"]
    platform.pop("heading_feedback")
    platform.update(lin_vel_x=[0., 1.], resampling_time=10.)
    saved["rewards"].update(
        base_height_target=.50, platform_base_height_target=.556, platform_scales=platform_v2_reward_scales(),
        platform_reward_profile="dreamwaq_highplatform", platform_heading_target=0.,
    )
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
