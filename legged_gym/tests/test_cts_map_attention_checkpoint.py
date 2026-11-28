"""Warm-start a geometry-conditioned student from archived labelled G13 checkpoints."""

import pytest
import torch

from experiments.registry import clone_config, configure_experiment, get_experiment
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import (
    load_cts_checkpoint, policy_from_checkpoint, validate_checkpoint_policy, validate_cts_checkpoint,
)
from rsl_rl.utils.cts_initialization import validate_initialization_source
from rsl_rl.utils.cts_observations import add_skill_id, default_observation_config, observation_dimensions
from test_cts_checkpoint_restoration import restore_args, train_namespace
from test_cts_experiments import actual_config_module
from test_cts_initialization import all_moe_config
from test_cts_runner import assert_nested_equal, cpu_task_registry
from test_high_platform_training import CpuPlatformEnv, stage_one_artifact


def archived_labelled_runner(actual_config_module, attention):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    base = get_experiment("g12_v5")
    base.configure_environment(cfg)
    schema = add_skill_id(default_observation_config())
    cfg.env.observation_config = schema
    current, history, privileged = observation_dimensions(schema)
    cfg.env.num_one_step_observations, cfg.env.num_observations = current, history
    cfg.env.num_one_step_privileged_obs = cfg.env.num_privileged_obs = privileged
    train = all_moe_config()
    train["policy"].update(base.POLICY, observation_dims=[3, 3, 3, 16, 16, 16, 1], teacher_map_attention=attention)
    train["runner"]["experiment_key"] = "g13_v2" if attention else "g13_v1"
    return MoECTSOnPolicyRunner(CpuPlatformEnv(cfg), train)


@pytest.mark.parametrize("legacy_attention", [False, True], ids=["archived_g13_v1", "archived_g13_v2"])
def test_legacy_g13_to_geometry_student_round_trip(tmp_path, actual_config_module, cpu_task_registry, legacy_attention):
    def joint_runner(**checkpoint_args):
        group = get_experiment("g13_v2")
        cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
        group.configure_environment(cfg)
        train = all_moe_config()
        train["policy"].update(group.POLICY)
        train["runner"].update(group.RUNNER)
        return MoECTSOnPolicyRunner(CpuPlatformEnv(cfg), train, **checkpoint_args)

    torch.manual_seed(40)
    source = archived_labelled_runner(actual_config_module, legacy_attention)
    source.learn(1)
    path = tmp_path / "archived.pt"
    source.save(path)
    legacy = torch.load(path, map_location="cpu", weights_only=False)
    if not legacy_attention:
        del legacy["policy_config"]["teacher_map_attention"]
        del legacy["policy_config"]["teacher_frozen_terrain"]
    with pytest.raises(ValueError, match="observation_dims"):
        validate_checkpoint_policy(legacy, get_experiment("g13_v2").POLICY)
    torch.save(legacy, path)

    if not legacy_attention:
        # The archived run has the same key as the frozen experiment. Resume must
        # reject its labelled student before restoring the saved environment.
        frozen = get_experiment("g13_v1")
        cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
        frozen.configure_environment(cfg)
        train = configure_experiment(train_namespace(all_moe_config(), path), frozen.KEY)
        cpu_task_registry.register(frozen.TASK_NAME, CpuPlatformEnv, cfg, train)
        with pytest.raises(ValueError, match="observation_dims"):
            cpu_task_registry.make_env(frozen.TASK_NAME, args=restore_args("resume", path))
        with pytest.raises(ValueError, match="--init_from is unsupported for frozen terrain policies"):
            cpu_task_registry.make_env(frozen.TASK_NAME, args=restore_args("init_from", path))

    artifact_path = stage_one_artifact(tmp_path)
    target = joint_runner(map_encoder_checkpoint=artifact_path)
    with pytest.raises(ValueError, match="Policy configuration differs"):
        target.load(path)
    target.initialize_from(path)
    stage_one_weights = torch.load(artifact_path, map_location="cpu", weights_only=True)["encoder_state_dict"]
    assert_nested_equal(target.alg.actor_critic.student_moe_encoder.terrain_encoder.state_dict(), stage_one_weights)
    assert target.current_learning_iteration == source.current_learning_iteration
    assert not target.alg.optimizer.state and not target.alg.optimizer_stu_enc.state
    history = torch.randn(8, target.env.num_obs)
    assert history.size(1) == 228
    # Removing trained ID columns is equivalent to the old student's ID=0,
    # even when the privileged teacher and critic still receive platform ID=1.
    labelled_history = torch.cat((history, torch.zeros(8, 4)), dim=1)
    privileged = torch.randn(8, target.env.num_privileged_obs)
    privileged[:, -1] = torch.arange(8) % 2
    platform = privileged[:, -1].bool()
    for teacher in (True, False):
        previous = source.alg.actor_critic.action_mean_with_diagnostics(labelled_history, privileged, teacher)
        transferred = target.alg.actor_critic.action_mean_with_diagnostics(history, privileged, teacher)
        torch.testing.assert_close(previous, transferred)
        torch.testing.assert_close(
            source.alg.actor_critic.evaluate(labelled_history, privileged, teacher, platform),
            target.alg.actor_critic.evaluate(history, privileged, teacher, platform),
        )
    torch.testing.assert_close(source.alg.actor_critic.teacher_encoder(privileged),
                               target.alg.actor_critic.teacher_encoder(privileged))

    target.learn(1)
    assert torch.count_nonzero(target.alg.actor_critic.teacher_encoder.map_attention.to_latent.weight) > 0
    updated = tmp_path / "v2.pt"
    target.save(updated)
    resumed = joint_runner(resume_path=updated)
    assert_nested_equal(resumed.alg.optimizer.state_dict(), target.alg.optimizer.state_dict())
    assert_nested_equal(resumed.alg.actor_critic.state_dict(), target.alg.actor_critic.state_dict())
    checkpoint = load_cts_checkpoint(updated)
    validate_checkpoint_policy(checkpoint, target.policy_config)
    restored = policy_from_checkpoint(checkpoint)
    scan = target.alg.actor_critic.student_height_scan(privileged)
    assert torch.equal(restored.act_inference(history, scan), target.alg.actor_critic.act_inference(history, scan))
    assert torch.equal(restored.teacher_encoder(privileged), target.alg.actor_critic.teacher_encoder(privileged))
    with pytest.raises(ValueError, match="discard trained teacher map attention"):
        validate_initialization_source(checkpoint, get_experiment("g12_v5").POLICY)
    del checkpoint["policy_config"]["teacher_map_attention"]
    with pytest.raises(ValueError, match="teacher_map_attention disagrees"):
        validate_cts_checkpoint(checkpoint)
