"""Portable checkpoint assets and unambiguous comparison-group resumes."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.registry import get_experiment
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint
from test_cts_initialization import all_moe_config
from test_cts_runner import FakeVectorEnv, cpu_task_registry


def train_namespace(config, checkpoint):
    config = copy.deepcopy(config)
    config["runner"].update(
        resume=False, resume_path=str(checkpoint), experiment_name="test",
        run_name="test", load_run=-1, checkpoint=-1,
    )
    return SimpleNamespace(
        seed=config["seed"], runner_class_name="MoECTSOnPolicyRunner",
        **{key: SimpleNamespace(**value) for key, value in config.items()
           if isinstance(value, dict)},
    )


def restore_args(mode, checkpoint):
    return SimpleNamespace(
        init_from=str(checkpoint) if mode == "init_from" else None,
        resume=mode == "resume", from_scratch=False, seed=None, num_envs=None,
        max_iterations=None, experiment_name=None, run_name=None, load_run=None,
        checkpoint=None, rl_device="cpu", sim_device="cpu", physics_engine=0,
        headless=True,
    )


@pytest.mark.parametrize("mode", ["init_from", "resume"])
@pytest.mark.parametrize("asset_kind", ["stale", "existing", "template", "missing", "external_missing"])
def test_checkpoint_asset_is_resolved_before_environment_creation(
        tmp_path, monkeypatch, cpu_task_registry, mode, asset_kind):
    root = tmp_path / "current_checkout"
    suffix = "robots/robot/robot.urdf"
    local_asset = root / "resources" / suffix
    local_asset.parent.mkdir(parents=True)
    if asset_kind not in ("missing", "external_missing"):
        local_asset.write_text("<robot name='local'/>")
    configured_asset = str(tmp_path / "old_checkout" / "resources" / suffix)
    if asset_kind == "existing":
        Path(configured_asset).parent.mkdir(parents=True)
        Path(configured_asset).write_text("<robot name='original'/>")
    elif asset_kind == "template":
        configured_asset = "{LEGGED_GYM_ROOT_DIR}/resources/" + suffix
    elif asset_kind == "external_missing":
        configured_asset = str(tmp_path / "external" / "missing.urdf")

    config = all_moe_config()
    env_cfg = SimpleNamespace(
        seed=42, asset=SimpleNamespace(file=configured_asset),
        env=SimpleNamespace(num_envs=8), sim=SimpleNamespace(),
    )
    source_env = FakeVectorEnv()
    source_env.cfg = env_cfg
    checkpoint = tmp_path / "source.pt"
    MoECTSOnPolicyRunner(source_env, config).save(checkpoint)
    original_checkpoint = checkpoint.read_bytes()

    created = []

    class AssetCheckingEnv:
        def __init__(self, cfg, **kwargs):
            created.append(cfg)
            assert Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=root)).is_file()

    module = sys.modules[type(cpu_task_registry).__module__]
    monkeypatch.setattr(module, "LEGGED_GYM_ROOT_DIR", str(root))
    monkeypatch.setattr(module, "parse_sim_params", lambda *_args: None)
    monkeypatch.setattr(module, "set_seed", lambda _seed: None)
    cpu_task_registry.register(
        "asset_test", AssetCheckingEnv, env_cfg, train_namespace(config, checkpoint),
    )
    args = restore_args(mode, checkpoint)
    if asset_kind in ("missing", "external_missing"):
        with pytest.raises(FileNotFoundError, match="Checkpoint asset does not exist") as error:
            cpu_task_registry.make_env("asset_test", args=args)
        assert configured_asset in str(error.value)
        if asset_kind == "missing":
            assert str(local_asset) in str(error.value)
        assert not created
    else:
        _, restored = cpu_task_registry.make_env("asset_test", args=args)
        expected = str(local_asset) if asset_kind == "stale" else configured_asset
        assert restored.asset.file == expected
        assert len(created) == 1
    assert checkpoint.read_bytes() == original_checkpoint
    assert cpu_task_registry.env_cfgs["asset_test"].asset.file == configured_asset


def comparison_config(key):
    experiment = get_experiment(key)
    factors = experiment.RUNNER["comparison_factors"]
    config = all_moe_config(factors["residual"], factors["temporal"])
    config["runner"].update(copy.deepcopy(experiment.RUNNER))
    return config


@pytest.mark.parametrize("source_key,target_key", [
    ("g8_all_moe_control", "g10_all_moe_temporal"),
    ("g10_all_moe_temporal", "g8_all_moe_control"),
    ("g9_all_moe_residual", "g11_all_moe_residual_temporal"),
    ("g11_all_moe_residual_temporal", "g9_all_moe_residual"),
])
def test_resume_rejects_different_comparison_labels_with_identical_policy(
        cpu_task_registry, source_key, target_key):
    source, target = comparison_config(source_key), comparison_config(target_key)
    checkpoint = {"policy_config": source["policy"], "train_config": source}
    with pytest.raises(ValueError, match="same experiment_key"):
        cpu_task_registry._validate_experiment_checkpoint(
            train_namespace(target, "unused.pt"), checkpoint,
        )


@pytest.mark.parametrize("key", [
    "g8_all_moe_control", "g9_all_moe_residual",
    "g10_all_moe_temporal", "g11_all_moe_residual_temporal",
])
def test_resume_rejects_temporal_factor_that_disagrees_with_saved_label(cpu_task_registry, key):
    target = comparison_config(key)
    source = copy.deepcopy(target)
    expected_temporal = target["runner"]["comparison_factors"]["temporal"]
    source["algorithm"]["temporal_loss_coef"] = 0.0 if expected_temporal else 0.025
    checkpoint = {"policy_config": source["policy"], "train_config": source}
    with pytest.raises(ValueError, match="disagrees with target comparison factor"):
        cpu_task_registry._validate_experiment_checkpoint(
            train_namespace(target, "unused.pt"), checkpoint,
        )


@pytest.mark.parametrize("key", [
    "g8_all_moe_control", "g9_all_moe_residual",
    "g10_all_moe_temporal", "g11_all_moe_residual_temporal",
])
def test_matching_comparison_resume_preserves_a_tuned_positive_temporal_coefficient(
        tmp_path, cpu_task_registry, key):
    target = comparison_config(key)
    source = copy.deepcopy(target)
    temporal = target["runner"]["comparison_factors"]["temporal"]
    source["algorithm"]["temporal_loss_coef"] = 0.025 if temporal else 0.0
    checkpoint = tmp_path / "matching.pt"
    MoECTSOnPolicyRunner(FakeVectorEnv(), source).save(checkpoint)
    cpu_task_registry.register(
        "comparison", FakeVectorEnv, SimpleNamespace(), train_namespace(target, checkpoint),
    )
    restored, config = cpu_task_registry.make_alg_runner(
        FakeVectorEnv(), name="comparison", args=restore_args("resume", checkpoint), log_root=None,
    )
    assert config.runner.experiment_key == key
    assert restored.alg.temporal_loss_coef == (0.025 if temporal else 0.0)
    assert load_cts_checkpoint(checkpoint)["train_config"]["runner"]["experiment_key"] == key


def test_original_g7_keeps_legacy_resume_label_compatibility(cpu_task_registry):
    target = all_moe_config()
    target["runner"]["experiment_key"] = "g7_all_moe"
    source = copy.deepcopy(target)
    source["runner"].pop("experiment_key")
    cpu_task_registry._validate_experiment_checkpoint(
        train_namespace(target, "unused.pt"),
        {"policy_config": source["policy"], "train_config": source},
    )
