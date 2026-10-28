"""Keep the original CTS baseline and actor-MoE experiment independently usable."""

import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from experiments.registry import EXPERIMENT_KEYS, configure_experiment, export_path, get_experiment
from experiments.common import export_main
from legged_gym.scripts import export_nezha_moe_cts as export_script
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint
from test_cts_runner import (
    FakeVectorEnv, assert_nested_equal, cpu_task_registry, runner_config,
)


@pytest.fixture
def actual_config_module(monkeypatch):
    """Load actual configuration files without importing simulator classes."""
    env_dir = Path(__file__).resolve().parents[1] / "envs"
    for suffix, directory in (("", env_dir), (".base", env_dir / "base"),
                              (".nezha", env_dir / "nezha")):
        name = "legged_gym.envs" + suffix
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, package)
    for relative_name in (
        "base.base_config", "base.legged_robot_config", "nezha.nezha_robot_config",
        "nezha.nezha_moe_cts_config",
    ):
        name = "legged_gym.envs." + relative_name
        path = env_dir / (relative_name.replace(".", "/") + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def config_dict(cpu_task_registry, value):
    helpers_name = type(cpu_task_registry).__module__.rsplit(".", 1)[0] + ".helpers"
    return copy.deepcopy(sys.modules[helpers_name].class_to_dict(value))


def test_experiments_only_change_actor_and_run_metadata(actual_config_module, cpu_task_registry):
    base = actual_config_module.NezhaMoECTSCfgPPO()
    initial = config_dict(cpu_task_registry, base)
    configs = [configure_experiment(base, key) for key in EXPERIMENT_KEYS]
    baseline, actor_moe = [config_dict(cpu_task_registry, cfg) for cfg in configs]
    assert baseline["policy"]["actor_type"] == "mlp"
    assert baseline["policy"]["actor_expert_num"] == 0
    assert actor_moe["policy"]["actor_type"] == "moe"
    assert actor_moe["policy"]["actor_expert_num"] == 8
    for value, key in zip((baseline, actor_moe), EXPERIMENT_KEYS):
        experiment = get_experiment(key)
        assert value["runner"]["experiment_key"] == key
        assert value["runner"]["experiment_name"] == "nezha_moe_cts/" + key
        assert export_path(key).endswith("/nezha/policy/" + key + "/policy.pt")
        for field in experiment.POLICY:
            value["policy"].pop(field)
        for field in experiment.RUNNER:
            value["runner"].pop(field)
    assert baseline == actor_moe
    assert config_dict(cpu_task_registry, base) == initial
    assert baseline["runner"]["save_interval"] == 1000
    assert baseline["policy"]["expert_num"] == 8  # Both keep the student MoE.


def test_mutating_one_experiment_cannot_change_other_or_shared_defaults(actual_config_module):
    base = actual_config_module.NezhaMoECTSCfgPPO()
    baseline = configure_experiment(base, "exp01_baseline")
    actor_moe = configure_experiment(base, "exp02_actor_moe")
    baseline.policy.actor_hidden_dims.append(17)
    baseline.runner.wandb_tags.append("temporary-test-tag")
    baseline.algorithm.learning_rate = 0.2
    assert actor_moe.policy.actor_hidden_dims == [512, 256, 128]
    assert base.policy.actor_hidden_dims == [512, 256, 128]
    assert actual_config_module.NezhaMoECTSCfgPPO().policy.actor_hidden_dims == [512, 256, 128]
    assert "temporary-test-tag" not in actor_moe.runner.wandb_tags
    assert "temporary-test-tag" not in get_experiment("exp01_baseline").RUNNER["wandb_tags"]
    assert actor_moe.algorithm.learning_rate == base.algorithm.learning_rate == 1e-3


def test_registered_environment_and_training_configs_remain_independent(
        actual_config_module, cpu_task_registry):
    source_env = actual_config_module.NezhaMoECTSCfg()
    source_train = actual_config_module.NezhaMoECTSCfgPPO()
    original_env = config_dict(cpu_task_registry, source_env)
    for key in EXPERIMENT_KEYS:
        cpu_task_registry.register(
            key, FakeVectorEnv, source_env, configure_experiment(source_train, key),
        )
    baseline_env, baseline_train = cpu_task_registry.get_cfgs("exp01_baseline")
    actor_env, actor_train = cpu_task_registry.get_cfgs("exp02_actor_moe")
    assert config_dict(cpu_task_registry, baseline_env) == config_dict(cpu_task_registry, actor_env)
    baseline_env.commands.ranges.lin_vel_x[0] = -99.0
    baseline_env.terrain.wf_sub_terrains.clear()
    baseline_train.policy.student_encoder_hidden_dims.append(17)
    for key in EXPERIMENT_KEYS:
        env, train = cpu_task_registry.get_cfgs(key)
        assert env.commands.ranges.lin_vel_x == [-1.0, 1.0]
        assert env.terrain.wf_sub_terrains
        assert train.policy.student_encoder_hidden_dims == [512, 256, 256]
    assert actor_env.commands.ranges.lin_vel_x == [-1.0, 1.0]
    assert actor_train.policy.student_encoder_hidden_dims == [512, 256, 256]
    assert config_dict(cpu_task_registry, source_env) == original_env


def training_config(key, checkpoint=None):
    settings = runner_config()
    settings["runner"].update(resume=bool(checkpoint), resume_path=checkpoint,
                              load_run=-1, checkpoint=-1)
    cfg = SimpleNamespace(
        seed=10, runner_class_name="MoECTSOnPolicyRunner",
        **{section: SimpleNamespace(**values) for section, values in settings.items()},
    )
    return configure_experiment(cfg, key)


def training_args():
    return SimpleNamespace(
        seed=None, max_iterations=None, resume=True, experiment_name=None,
        run_name=None, load_run=None, checkpoint=None, rl_device="cpu",
    )


@pytest.mark.parametrize("key", EXPERIMENT_KEYS)
def test_latest_resolution_stays_inside_selected_experiment(
        tmp_path, monkeypatch, cpu_task_registry, key):
    registry_module = sys.modules[type(cpu_task_registry).__module__]
    monkeypatch.setattr(registry_module, "LEGGED_GYM_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(export_script, "REPO_ROOT", str(tmp_path))
    expected = {}
    for i, candidate_key in enumerate(EXPERIMENT_KEYS):
        root = tmp_path / "logs" / get_experiment(candidate_key).RUNNER["experiment_name"]
        older = root / "20260101_000000_run"
        newest = root / "20260201_000000_run"
        incomplete = root / "20260301_000000_incomplete"
        for directory in (older, newest, incomplete):
            directory.mkdir(parents=True)
        (older / "model_9000.pt").touch()
        (newest / "model_2.pt").touch()
        (newest / "model_10.pt").touch()
        # Other experiment and unfinished run are newer, but must never win.
        for offset, directory in enumerate((older, newest, incomplete)):
            os.utime(directory, (1000 + i * 100 + offset, 1000 + i * 100 + offset))
        expected[candidate_key] = str(newest / "model_10.pt")
    cfg = training_config(key)
    cfg.runner.resume = True
    assert cpu_task_registry._resume_path(cfg) == expected[key]
    assert export_script.resolve_checkpoint(
        experiment=get_experiment(key).RUNNER["experiment_name"],
    ) == expected[key]


@pytest.fixture(scope="module")
def experiment_checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("cts_experiment_checkpoints")
    result = {}
    for label, actor_type, count in (("legacy_mlp", "mlp", 0), ("moe8", "moe", 8),
                                     ("moe4", "moe", 4)):
        settings = runner_config(actor_type)
        settings["policy"]["actor_expert_num"] = count
        settings["runner"]["run_name"] = "original_" + label
        runner = MoECTSOnPolicyRunner(FakeVectorEnv(), settings)
        runner.learn(1)
        path = root / (label + ".pt")
        runner.save(path)
        if label == "legacy_mlp":
            checkpoint = load_cts_checkpoint(path)
            for section in (checkpoint["policy_config"], checkpoint["train_config"]["policy"]):
                section.pop("actor_type")
                section.pop("actor_expert_num")
            torch.save(checkpoint, path)
        result[label] = (path, runner)
    return result


MISMATCHES = (
    ("exp01_baseline", "moe8", "actor_type"),
    ("exp02_actor_moe", "legacy_mlp", "actor_type"),
    ("exp02_actor_moe", "moe4", "actor_expert_num"),
)


@pytest.mark.parametrize("key,checkpoint_name,error", MISMATCHES)
@pytest.mark.parametrize("entry", ["make_env", "make_alg_runner"])
def test_explicit_resume_rejects_other_experiment_before_allocating_or_restoring(
        tmp_path, cpu_task_registry, experiment_checkpoints, key, checkpoint_name, error, entry):
    path, _ = experiment_checkpoints[checkpoint_name]
    config = training_config(key, str(path))
    original = config_dict(cpu_task_registry, config)
    cpu_task_registry.register("test", FakeVectorEnv, SimpleNamespace(), config)
    args = training_args()
    output_root = tmp_path / "new_run"
    with pytest.raises(ValueError, match=error):
        if entry == "make_env":
            # No simulator API is stubbed: a missing check would reach real
            # environment initialization and fail outside this expected error.
            cpu_task_registry.make_env("test", args=args)
        else:
            cpu_task_registry.make_alg_runner(
                FakeVectorEnv(), "test", args=args, log_root=str(output_root),
            )
    assert config_dict(cpu_task_registry, config) == original
    assert not output_root.exists()
    assert not hasattr(args, "cts_resume_path")


@pytest.mark.parametrize("key,checkpoint_name", [
    ("exp01_baseline", "legacy_mlp"), ("exp02_actor_moe", "moe8"),
])
def test_matching_experiment_resumes_legacy_weights_and_optimizers(
        tmp_path, cpu_task_registry, experiment_checkpoints, key, checkpoint_name):
    path, original = experiment_checkpoints[checkpoint_name]
    config = training_config(key, str(path))
    config.policy.actor_hidden_dims = [64]  # Checkpoint dimensions are authoritative.
    cpu_task_registry.register("test", FakeVectorEnv, SimpleNamespace(), config)
    restored, restored_config = cpu_task_registry.make_alg_runner(
        FakeVectorEnv(), "test", args=training_args(), log_root=str(tmp_path),
    )
    assert restored_config.runner.experiment_key == key
    assert restored_config.runner.experiment_name == get_experiment(key).RUNNER["experiment_name"]
    assert restored.current_learning_iteration == original.current_learning_iteration == 1
    assert restored.alg.actor_critic.actor_type == get_experiment(key).POLICY["actor_type"]
    assert restored.alg.actor_critic.actor_expert_num == get_experiment(key).POLICY["actor_expert_num"]
    assert_nested_equal(original.alg.actor_critic.state_dict(), restored.alg.actor_critic.state_dict())
    assert_nested_equal(original.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert_nested_equal(original.alg.optimizer_stu_enc.state_dict(), restored.alg.optimizer_stu_enc.state_dict())
    restored.learn(1)
    assert restored.current_learning_iteration == 2


@pytest.mark.parametrize("key,checkpoint_name,error", MISMATCHES)
@pytest.mark.parametrize("existing_output", [False, True])
def test_wrong_experiment_export_cannot_create_or_overwrite_policy(
        tmp_path, monkeypatch, experiment_checkpoints, key, checkpoint_name, error, existing_output):
    checkpoint_path, _ = experiment_checkpoints[checkpoint_name]
    output = tmp_path / "exported" / "policy.pt"
    original = b"existing deployable policy must survive failed export"
    if existing_output:
        output.parent.mkdir()
        output.write_bytes(original)
    monkeypatch.setattr(sys, "argv", ["export.py", "--checkpoint", str(checkpoint_path),
                                     "--output", str(output)])
    with pytest.raises(ValueError, match=error):
        export_main(key)
    if existing_output:
        assert output.read_bytes() == original
        assert sorted(path.name for path in output.parent.iterdir()) == ["policy.pt"]
    else:
        assert not output.parent.exists()


@pytest.mark.parametrize("key,checkpoint_name", [
    ("exp01_baseline", "legacy_mlp"), ("exp02_actor_moe", "moe8"),
])
def test_matching_experiment_export_preserves_actions_and_actor_metadata(
        tmp_path, monkeypatch, experiment_checkpoints, key, checkpoint_name):
    checkpoint_path, runner = experiment_checkpoints[checkpoint_name]
    output = tmp_path / key / "policy.pt"
    monkeypatch.setattr(sys, "argv", ["export.py", "--checkpoint", str(checkpoint_path),
                                     "--output", str(output)])
    expected = get_experiment(key).POLICY
    assert export_main(key) == str(output)
    policy = torch.jit.load(str(output))
    history = torch.randn(3, FakeVectorEnv.num_obs)
    assert policy.get_actor_type() == expected["actor_type"]
    assert policy.get_num_actor_experts() == expected["actor_expert_num"]
    with torch.no_grad():
        assert torch.allclose(policy(history), runner.get_inference_policy()(history), atol=1e-6, rtol=1e-5)
