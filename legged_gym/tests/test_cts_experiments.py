"""Exercise all eight CTS ablations through configuration, resume and export."""

import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.registry import EXPERIMENT_KEYS, configure_experiment, export_path, get_experiment
from experiments.common import export_main
from deploy.deploy_mujoco.runtime import NezhaCTSPolicyRuntime
from legged_gym.scripts import export_nezha_moe_cts as export_script
from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils import cts_checkpoint
from test_cts_runner import (
    FakeVectorEnv, assert_nested_equal, cpu_task_registry, runner_config,
)


GROUPS = {
    "g0_mlp": ("mlp", "mlp", "mlp"),
    "g1_student_moe": ("moe", "mlp", "mlp"),
    "g2_actor_moe": ("mlp", "moe", "mlp"),
    "g3_student_actor_moe": ("moe", "moe", "mlp"),
    "g4_critic_moe": ("mlp", "mlp", "moe"),
    "g5_student_critic_moe": ("moe", "mlp", "moe"),
    "g6_actor_critic_moe": ("mlp", "moe", "moe"),
    "g7_all_moe": ("moe", "moe", "moe"),
}
STRUCTURE_FIELDS = (
    "student_encoder_type", "expert_num", "actor_type", "actor_expert_num",
    "critic_type", "critic_expert_num",
)


def expected_policy(key):
    student, actor, critic = GROUPS[key]
    return {
        "student_encoder_type": student, "expert_num": 8 if student == "moe" else 0,
        "actor_type": actor, "actor_expert_num": 8 if actor == "moe" else 0,
        "critic_type": critic, "critic_expert_num": 8 if critic == "moe" else 0,
    }


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


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


def test_complete_factorial_only_changes_structure_and_run_metadata(actual_config_module, cpu_task_registry):
    assert tuple(EXPERIMENT_KEYS) == tuple(GROUPS)
    base = actual_config_module.NezhaMoECTSCfgPPO()
    initial = config_dict(cpu_task_registry, base)
    common = []
    for key in GROUPS:
        experiment = get_experiment(key)
        assert experiment.TASK_NAME == "nezha_cts_" + key
        assert experiment.POLICY == expected_policy(key)
        value = config_dict(cpu_task_registry, configure_experiment(base, key))
        assert {field: value["policy"][field] for field in STRUCTURE_FIELDS} == expected_policy(key)
        assert value["runner"]["experiment_key"] == key
        assert value["runner"]["experiment_name"] == "nezha_moe_cts/" + key
        assert export_path(key).endswith("/nezha/policy/" + key + "/policy.pt")
        for field in STRUCTURE_FIELDS:
            value["policy"].pop(field)
        for field in experiment.RUNNER:
            value["runner"].pop(field)
        common.append(value)
    assert all(value == common[0] for value in common)
    assert config_dict(cpu_task_registry, base) == initial
    assert common[0]["policy"]["latent_dim"] == 32
    assert common[0]["policy"]["teacher_encoder_hidden_dims"] == [512, 256]
    assert common[0]["runner"]["save_interval"] == 1000
    assert common[0]["algorithm"]["teacher_env_ratio"] == 0.75


def test_mutating_one_experiment_cannot_change_other_or_shared_defaults(actual_config_module):
    base = actual_config_module.NezhaMoECTSCfgPPO()
    configs = {key: configure_experiment(base, key) for key in GROUPS}
    changed = configs["g0_mlp"]
    for field in ("actor_hidden_dims", "critic_hidden_dims", "student_encoder_hidden_dims"):
        getattr(changed.policy, field).append(17)
    changed.runner.wandb_tags.append("temporary-test-tag")
    changed.algorithm.learning_rate = 0.2
    for key, config in configs.items():
        if key == "g0_mlp":
            continue
        assert config.policy.actor_hidden_dims == [512, 256, 128]
        assert config.policy.critic_hidden_dims == [512, 256, 128]
        assert config.policy.student_encoder_hidden_dims == [512, 256, 256]
        assert "temporary-test-tag" not in config.runner.wandb_tags
        assert config.algorithm.learning_rate == base.algorithm.learning_rate == 1e-3
    assert base.policy.actor_hidden_dims == [512, 256, 128]
    assert actual_config_module.NezhaMoECTSCfgPPO().policy.critic_hidden_dims == [512, 256, 128]
    assert "temporary-test-tag" not in get_experiment("g0_mlp").RUNNER["wandb_tags"]


def test_real_task_registration_preserves_environment_and_training_isolation(
        actual_config_module, cpu_task_registry, monkeypatch):
    """Execute envs/__init__, replacing only classes that import the simulator."""
    robot_module = ModuleType("legged_gym.envs.base.legged_robot")
    robot_module.LeggedRobot = FakeVectorEnv
    monkeypatch.setitem(sys.modules, robot_module.__name__, robot_module)
    nezha = sys.modules["legged_gym.envs.nezha"]
    nezha.NezhaMoECTSCfg = actual_config_module.NezhaMoECTSCfg
    nezha.NezhaMoECTSCfgPPO = actual_config_module.NezhaMoECTSCfgPPO
    nezha.NezhaMoECTSEnv = FakeVectorEnv
    registry_module = ModuleType("legged_gym.utils.task_registry")
    registry_module.task_registry = cpu_task_registry
    monkeypatch.setitem(sys.modules, registry_module.__name__, registry_module)
    init_path = Path(__file__).resolve().parents[1] / "envs" / "__init__.py"
    package = sys.modules["legged_gym.envs"]
    exec(compile(init_path.read_text(), str(init_path), "exec"), package.__dict__)
    tasks = {get_experiment(key).TASK_NAME for key in GROUPS}
    assert tasks <= set(cpu_task_registry.task_classes)
    unchanged = []
    for key in GROUPS:
        env, train = cpu_task_registry.get_cfgs(get_experiment(key).TASK_NAME)
        unchanged.append(config_dict(cpu_task_registry, env))
        assert {field: getattr(train.policy, field) for field in STRUCTURE_FIELDS} == expected_policy(key)
    assert all(env == unchanged[0] for env in unchanged)
    env, train = cpu_task_registry.get_cfgs(get_experiment("g0_mlp").TASK_NAME)
    env.commands.ranges.lin_vel_x[0] = -99.0
    env.terrain.wf_sub_terrains.clear()
    train.policy.student_encoder_hidden_dims.append(17)
    for key in GROUPS:
        fresh_env, fresh_train = cpu_task_registry.get_cfgs(get_experiment(key).TASK_NAME)
        assert config_dict(cpu_task_registry, fresh_env) == unchanged[0]
        assert fresh_train.policy.student_encoder_hidden_dims == [512, 256, 256]


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


@pytest.mark.parametrize("key", GROUPS)
def test_latest_resolution_stays_inside_selected_experiment(
        tmp_path, monkeypatch, cpu_task_registry, key):
    registry_module = sys.modules[type(cpu_task_registry).__module__]
    monkeypatch.setattr(registry_module, "LEGGED_GYM_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(export_script, "REPO_ROOT", str(tmp_path))
    expected = {}
    for i, candidate_key in enumerate(GROUPS):
        root = tmp_path / "logs" / get_experiment(candidate_key).RUNNER["experiment_name"]
        older = root / "20260101_000000_run"
        newest = root / "20260201_000000_run"
        incomplete = root / "20260301_000000_incomplete"
        for directory in (older, newest, incomplete):
            directory.mkdir(parents=True)
        (older / "model_9000.pt").touch()
        (newest / "model_2.pt").touch()
        (newest / "model_10.pt").touch()
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
    """Run real PPO and distillation before exercising persistence boundaries."""
    root = tmp_path_factory.mktemp("cts_experiment_checkpoints")
    result = {}
    specifications = {key: expected_policy(key) for key in GROUPS}
    for field in ("expert_num", "actor_expert_num", "critic_expert_num"):
        specifications[field + "_4"] = {**expected_policy("g7_all_moe"), field: 4}
    for i, (label, policy_config) in enumerate(specifications.items()):
        torch.manual_seed(100 + i)
        settings = runner_config()
        settings["policy"].update(policy_config)
        settings["runner"].update(
            run_name="original_" + label,
            wandb_tags=["saved-checkpoint", label], wandb_notes="Prior run metadata",
        )
        runner = MoECTSOnPolicyRunner(FakeVectorEnv(), settings)
        runner.learn(1)
        path = root / (label + ".pt")
        runner.save(path)
        result[label] = (path, runner)
    for key in ("g1_student_moe", "g3_student_actor_moe"):
        source, runner = result[key]
        checkpoint = cts_checkpoint.load_cts_checkpoint(source)
        for section in (checkpoint["policy_config"], checkpoint["train_config"]["policy"]):
            for field in ("student_encoder_type", "critic_type", "critic_expert_num"):
                section.pop(field, None)
            if key == "g1_student_moe":
                section.pop("actor_type", None)
                section.pop("actor_expert_num", None)
        path = root / ("legacy_" + key + ".pt")
        torch.save(checkpoint, path)
        result["legacy_" + key] = (path, runner)
    return result


MISMATCHES = (
    ("g0_mlp", "g1_student_moe", "student_encoder_type"),
    ("g1_student_moe", "g0_mlp", "student_encoder_type"),
    ("g0_mlp", "g2_actor_moe", "actor_type"),
    ("g2_actor_moe", "g0_mlp", "actor_type"),
    ("g0_mlp", "g4_critic_moe", "critic_type"),
    ("g4_critic_moe", "g0_mlp", "critic_type"),
    ("g7_all_moe", "expert_num_4", "expert_num"),
    ("g7_all_moe", "actor_expert_num_4", "actor_expert_num"),
    ("g7_all_moe", "critic_expert_num_4", "critic_expert_num"),
)


@pytest.mark.parametrize("key,checkpoint_name,error", MISMATCHES)
@pytest.mark.parametrize("entry", ["make_env", "make_alg_runner"])
def test_explicit_resume_rejects_mismatched_structure_before_allocating_or_restoring(
        tmp_path, cpu_task_registry, experiment_checkpoints, key, checkpoint_name, error, entry):
    path, _ = experiment_checkpoints[checkpoint_name]
    config = training_config(key, str(path))
    original = config_dict(cpu_task_registry, config)
    cpu_task_registry.register("test", FakeVectorEnv, SimpleNamespace(), config)
    args = training_args()
    output_root = tmp_path / "new_run"
    with pytest.raises(ValueError, match=error):
        if entry == "make_env":
            cpu_task_registry.make_env("test", args=args)
        else:
            cpu_task_registry.make_alg_runner(
                FakeVectorEnv(), "test", args=args, log_root=str(output_root),
            )
    assert config_dict(cpu_task_registry, config) == original
    assert not output_root.exists()
    assert not hasattr(args, "cts_resume_path")


@pytest.mark.parametrize("key", GROUPS)
def test_each_group_trains_resumes_both_optimizers_and_exports(
        tmp_path, monkeypatch, cpu_task_registry, experiment_checkpoints, key):
    path, original = experiment_checkpoints[key]
    checkpoint = cts_checkpoint.load_cts_checkpoint(path)
    expected = expected_policy(key)
    assert {field: checkpoint["policy_config"][field] for field in STRUCTURE_FIELDS} == expected
    cts_checkpoint.validate_checkpoint_policy(checkpoint, expected)
    config = training_config(key, str(path))
    config.policy.actor_hidden_dims = [64]  # Saved widths are authoritative on resume.
    cpu_task_registry.register("test", FakeVectorEnv, SimpleNamespace(), config)
    restored, restored_config = cpu_task_registry.make_alg_runner(
        FakeVectorEnv(), "test", args=training_args(), log_root=str(tmp_path / "resume"),
    )
    assert restored_config.runner.experiment_key == key
    assert restored_config.runner.experiment_name == get_experiment(key).RUNNER["experiment_name"]
    for field in ("wandb_group", "wandb_tags", "wandb_notes"):
        assert getattr(restored_config.runner, field) == get_experiment(key).RUNNER[field]
    assert restored.current_learning_iteration == original.current_learning_iteration == 1
    assert_nested_equal(original.alg.actor_critic.state_dict(), restored.alg.actor_critic.state_dict())
    assert_nested_equal(original.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert_nested_equal(original.alg.optimizer_stu_enc.state_dict(), restored.alg.optimizer_stu_enc.state_dict())
    restored.learn(1)
    assert restored.current_learning_iteration == 2
    for optimizer in (restored.alg.optimizer, restored.alg.optimizer_stu_enc):
        assert optimizer.state
        assert all(state["step"].item() == 4 for state in optimizer.state.values())
    events = EventAccumulator(restored.log_dir).Reload()
    tags = events.Tags()["scalars"]
    assert "Loss/mean_latent_loss" in tags
    for prefix, count in (
        ("Experts/", expected["expert_num"]),
        ("ActorExperts/", expected["actor_expert_num"]),
        ("CriticExperts/", expected["critic_expert_num"]),
    ):
        if not count:
            assert not any(tag.startswith(prefix) for tag in tags)
            continue
        entropy = events.Scalars(prefix + "gate_entropy")[-1].value
        assert 0 <= entropy <= float(torch.log(torch.tensor(float(count)))) + 1e-6
        usage = [events.Scalars(prefix + "expert_%d_usage" % i)[-1].value for i in range(count)]
        assert all(0 <= value <= 1 for value in usage)
        assert sum(usage) == pytest.approx(1.0, abs=1e-6)
    saved = tmp_path / "continued.pt"
    restored.save(saved)
    output = tmp_path / "deployment" / "policy.pt"
    monkeypatch.setattr(sys, "argv", ["export.py", "--checkpoint", str(saved), "--output", str(output)])
    assert export_main(key) == str(output)
    deployed = torch.jit.load(str(output))
    history = torch.randn(3, FakeVectorEnv.num_obs)
    privileged = torch.randn(3, FakeVectorEnv.num_privileged_obs)
    policy = restored.alg.actor_critic
    assert deployed.get_student_encoder_type() == expected["student_encoder_type"]
    assert deployed.get_num_experts() == expected["expert_num"]
    assert deployed.get_actor_type() == expected["actor_type"]
    assert deployed.get_num_actor_experts() == expected["actor_expert_num"]
    assert not any(name.startswith(("critic.", "teacher_encoder.")) for name, _ in deployed.named_parameters())
    with torch.no_grad():
        actions = policy.act_inference(history)
        assert actions.shape == (3, FakeVectorEnv.num_actions)
        assert torch.isfinite(actions).all()
        assert torch.allclose(deployed(history), actions, atol=1e-6, rtol=1e-5)
        for teacher in (True, False):
            values = policy.evaluate(history, privileged, is_teacher=teacher)
            assert values.shape == (3, 1)
            assert torch.isfinite(values).all()
        for actual, reference, count in (
            (deployed.get_expert_probabilities(history), policy.get_expert_probabilities(history), expected["expert_num"]),
            (deployed.get_actor_expert_probabilities(history), policy.get_actor_expert_probabilities(history), expected["actor_expert_num"]),
        ):
            assert actual.shape == (3, count)
            assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
            if count:
                assert torch.allclose(actual.sum(dim=-1), torch.ones(3), atol=1e-6)
    runtime_kwargs = {
        "default_dof_pos": [0.0] * 16, "p_gains": [10.0] * 16,
        "d_gains": [2.0] * 16, "torque_limits": [100.0] * 16,
    }
    runtime = NezhaCTSPolicyRuntime(str(output), expected_policy=expected, **runtime_kwargs)
    assert {field: runtime.policy_config[field] for field in STRUCTURE_FIELDS} == expected
    for step in range(3):
        if step == 2:
            runtime.reset()
        actions, torques = runtime.step(
            [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.5, 0.0, 0.0],
            [0.0] * 16, [0.1 * step] * 16,
        )
        assert actions.shape == torques.shape == (16,)
        assert torch.isfinite(actions).all() and torch.isfinite(torques).all()
        assert runtime.expert_probabilities().shape == (expected["expert_num"],)
    for type_field, count_field in (
        ("student_encoder_type", "expert_num"),
        ("actor_type", "actor_expert_num"),
        ("critic_type", "critic_expert_num"),
    ):
        wrong = dict(expected)
        wrong[type_field] = "mlp" if expected[type_field] == "moe" else "moe"
        wrong[count_field] = 0 if wrong[type_field] == "mlp" else 8
        with pytest.raises(ValueError, match=type_field):
            NezhaCTSPolicyRuntime(str(output), expected_policy=wrong, **runtime_kwargs)
        if expected[type_field] == "moe":
            wrong = {**expected, count_field: 4}
            with pytest.raises(ValueError, match=count_field):
                NezhaCTSPolicyRuntime(str(output), expected_policy=wrong, **runtime_kwargs)


@pytest.mark.parametrize("key", ["g1_student_moe", "g3_student_actor_moe"])
def test_legacy_student_moe_mlp_critic_checkpoint_keeps_weights_and_optimizers(
        tmp_path, cpu_task_registry, experiment_checkpoints, key):
    path, original = experiment_checkpoints["legacy_" + key]
    checkpoint = cts_checkpoint.load_cts_checkpoint(path)
    assert {field: checkpoint["policy_config"][field] for field in STRUCTURE_FIELDS} == expected_policy(key)
    config = training_config(key, str(path))
    cpu_task_registry.register("test", FakeVectorEnv, SimpleNamespace(), config)
    restored, _ = cpu_task_registry.make_alg_runner(
        FakeVectorEnv(), "test", args=training_args(), log_root=str(tmp_path),
    )
    assert_nested_equal(original.alg.actor_critic.state_dict(), restored.alg.actor_critic.state_dict())
    assert_nested_equal(original.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert_nested_equal(original.alg.optimizer_stu_enc.state_dict(), restored.alg.optimizer_stu_enc.state_dict())
    restored.learn(1)
    assert restored.current_learning_iteration == 2


@pytest.mark.parametrize("key,checkpoint_name,error", MISMATCHES)
@pytest.mark.parametrize("existing_output", [False, True])
def test_wrong_group_export_cannot_create_or_overwrite_policy(
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


@pytest.mark.parametrize("module_field,count_field,prefix", [
    ("student_encoder_type", "expert_num", "student"),
    ("actor_type", "actor_expert_num", "actor"),
    ("critic_type", "critic_expert_num", "critic"),
])
def test_checkpoint_cannot_relabel_saved_moe_weights_as_mlp(
        experiment_checkpoints, module_field, count_field, prefix):
    path, _ = experiment_checkpoints["g7_all_moe"]
    checkpoint = cts_checkpoint.load_cts_checkpoint(path)
    checkpoint["policy_config"][module_field] = "mlp"
    checkpoint["policy_config"][count_field] = 0
    with pytest.raises(ValueError, match=prefix):
        cts_checkpoint.validate_cts_checkpoint(checkpoint)
