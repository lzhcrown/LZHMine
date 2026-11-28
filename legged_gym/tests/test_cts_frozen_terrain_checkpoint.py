"""Exercise stage-one import, locomotion training, resume and map-conditioned export on CPU."""

import pytest
import torch

from experiments.registry import clone_config, get_experiment
from rsl_rl.modules.frozen_terrain_encoder import TerrainEncoder, terrain_encoder_checkpoint
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import (
    load_cts_checkpoint, policy_from_checkpoint, validate_checkpoint_policy, validate_cts_checkpoint,
)
from rsl_rl.utils.cts_export import CTSStudentExporter
from rsl_rl.utils.cts_initialization import validate_initialization_source
from test_cts_checkpoint_restoration import restore_args, train_namespace
from test_cts_experiments import actual_config_module
from test_cts_initialization import all_moe_config
from test_cts_runner import assert_nested_equal, cpu_task_registry
from test_high_platform_training import CpuPlatformEnv


@pytest.mark.parametrize("saved_key", ["g13_v1", "g13_v3"], ids=["current_key", "compatible_alias"])
def test_frozen_terrain_training_resume_and_export(tmp_path, actual_config_module, cpu_task_registry, saved_key):
    group = get_experiment("g13_v1")
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    group.configure_environment(cfg)
    train = all_moe_config()
    train["policy"].update(group.POLICY)
    train["policy"].pop("residual_hidden_dims")
    train["runner"].update(group.RUNNER)
    checkpoint_path = tmp_path / "locomotion.pt"
    cpu_task_registry.register(group.TASK_NAME, CpuPlatformEnv, cfg, train_namespace(train, checkpoint_path))
    args = restore_args("fresh", None)
    args.map_encoder_checkpoint = None
    env = CpuPlatformEnv(cfg)

    with pytest.raises(ValueError, match="requires --map_encoder_checkpoint"):
        cpu_task_registry.make_env(group.TASK_NAME, args=args)
    with pytest.raises(ValueError, match="requires --map_encoder_checkpoint"):
        MoECTSOnPolicyRunner(env, train)
    artifact_path = tmp_path / "stage1.pt"
    args.map_encoder_checkpoint = str(artifact_path)
    pretrained = TerrainEncoder()
    torch.save(terrain_encoder_checkpoint(pretrained), artifact_path)
    runner, _ = cpu_task_registry.make_alg_runner(env, group.TASK_NAME, args=args, log_root=None)
    model = runner.alg.actor_critic
    reader = model.teacher_encoder.terrain_encoder
    assert_nested_equal(pretrained.state_dict(), reader.state_dict())
    student_reader = model.student_moe_encoder.terrain_encoder
    assert_nested_equal(pretrained.state_dict(), student_reader.state_dict())
    for opt in (runner.alg.optimizer, runner.alg.optimizer_stu_enc):
        parameters = {id(p) for group_params in opt.param_groups for p in group_params["params"]}
        for frozen in (reader, student_reader):
            assert parameters.isdisjoint(id(p) for p in frozen.parameters())
    runner.learn(1)
    for frozen in (reader, student_reader):
        assert not frozen.training
        assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
        assert_nested_equal(pretrained.state_dict(), frozen.state_dict())
    runner.save(checkpoint_path)
    artifact_path.unlink()
    if saved_key != group.KEY:
        checkpoint = load_cts_checkpoint(checkpoint_path)
        checkpoint["train_config"]["runner"].update(
            experiment_key=saved_key, run_name=saved_key, wandb_group=saved_key,
            experiment_name="nezha_moe_cts/" + saved_key, wandb_tags=["nezha", "cts", saved_key],
        )
        torch.save(checkpoint, checkpoint_path)

    resume_args = restore_args("resume", checkpoint_path)
    restored, _ = cpu_task_registry.make_alg_runner(
        CpuPlatformEnv(cfg), group.TASK_NAME, args=resume_args, log_root=None,
    )
    assert restored.current_learning_iteration == 1
    assert restored.cfg["experiment_key"] == restored.cfg["wandb_group"] == group.KEY
    assert restored.initialization == {"map_encoder_checkpoint": str(artifact_path)}
    assert_nested_equal(model.state_dict(), restored.alg.actor_critic.state_dict())
    assert_nested_equal(runner.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert_nested_equal(runner.alg.optimizer_stu_enc.state_dict(), restored.alg.optimizer_stu_enc.state_dict())
    restored.learn(1)
    assert_nested_equal(pretrained.state_dict(), restored.alg.actor_critic.teacher_encoder.terrain_encoder.state_dict())

    checkpoint = load_cts_checkpoint(checkpoint_path)
    validate_checkpoint_policy(checkpoint, group.POLICY)
    with pytest.raises(ValueError, match="teacher_map_attention|teacher_frozen_terrain"):
        validate_checkpoint_policy(checkpoint, get_experiment("g13_v2").POLICY)
    with pytest.raises(ValueError, match="--init_from is unsupported for frozen terrain policies"):
        validate_initialization_source(checkpoint, get_experiment("g13_v2").POLICY)
    exported = tmp_path / "student.pt"
    inference = policy_from_checkpoint(checkpoint)
    CTSStudentExporter(inference).export(str(exported))
    student = torch.jit.load(str(exported))
    assert student.get_teacher_frozen_terrain() and not student.get_teacher_map_attention()
    assert not any("teacher" in name for name in student.state_dict())
    history = torch.randn(3, 228)
    scan = torch.randn(3, 187)
    assert student.get_student_frozen_terrain()
    torch.testing.assert_close(student(history, scan), model.act_inference(history, scan))
    del checkpoint["policy_config"]["teacher_frozen_terrain"]
    with pytest.raises(ValueError, match="teacher_frozen_terrain disagrees"):
        validate_cts_checkpoint(checkpoint)


def test_stage_one_flag_cannot_replace_resume_or_change_another_policy(tmp_path, actual_config_module):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment("g13_v1").configure_environment(cfg)
    train = all_moe_config()
    train["policy"].update(get_experiment("g13_v1").POLICY)
    env = CpuPlatformEnv(cfg)
    artifact = tmp_path / "bad.pt"
    torch.save({"architecture": "other"}, artifact)
    with pytest.raises(ValueError, match="--resume restores its own terrain encoder"):
        MoECTSOnPolicyRunner(env, train, map_encoder_checkpoint=artifact, resume_path=tmp_path / "rl.pt")
    with pytest.raises(ValueError):
        MoECTSOnPolicyRunner(env, train, map_encoder_checkpoint=artifact)
    with pytest.raises(ValueError, match="requires a frozen terrain policy"):
        MoECTSOnPolicyRunner(env, all_moe_config(), map_encoder_checkpoint=artifact)
