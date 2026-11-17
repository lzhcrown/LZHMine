"""Fair checkpoint initialization for the residual/temporal 2x2 study."""

from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint
from test_cts_runner import (
    CurriculumVectorEnv, FakeVectorEnv, assert_nested_equal, cpu_task_registry,
    runner_config,
)


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def all_moe_config(residual=False, temporal=False):
    cfg = runner_config("moe")
    cfg["seed"] = 42
    cfg["policy"].update(critic_type="moe", critic_expert_num=8,
                         residual_enabled=residual, residual_hidden_dims=[12, 8],
                         residual_action_scales=[.1, .1, .1, .05] * 4 if residual else None)
    cfg["algorithm"]["temporal_loss_coef"] = .01 if temporal else 0.
    return cfg


@pytest.fixture
def starting_point(tmp_path):
    torch.manual_seed(12)
    source = MoECTSOnPolicyRunner(CurriculumVectorEnv(), all_moe_config())
    source.learn(1)
    source.current_learning_iteration = 20000
    source.env.common_step_counter = 20000 * source.num_steps_per_env
    source.env.command_ranges["lin_vel_x"] = [-1.4, 1.4]
    source.env.wf_curriculum.update_command_ranges()
    path = tmp_path / "g7.pt"
    source.save(path)
    return source, path


@pytest.mark.parametrize("residual,temporal", [(False, False), (True, False),
                                               (False, True), (True, True)])
def test_initialize_preserves_function_clock_and_ranges_but_resets_adam(
        starting_point, tmp_path, residual, temporal):
    source, path = starting_point
    cfg = all_moe_config(residual, temporal)
    target = MoECTSOnPolicyRunner(CurriculumVectorEnv(), cfg)
    info = target.initialize_from(path)
    assert target.current_learning_iteration == 20000
    assert target.env.common_step_counter == source.env.common_step_counter
    assert target.env.command_ranges == source.env.command_ranges
    assert not target.alg.optimizer.state
    assert not target.alg.optimizer_stu_enc.state
    assert target.alg.optimizer.param_groups[0]["lr"] == cfg["algorithm"]["learning_rate"]
    assert target.alg.temporal_loss_coef == (.01 if temporal else 0.)
    assert info["optimizer"] == "reset_both_adam"
    for name, tensor in source.alg.actor_critic.state_dict().items():
        torch.testing.assert_close(target.alg.actor_critic.state_dict()[name], tensor,
                                   rtol=0, atol=0)
    probe = torch.randn(5, 228)
    torch.testing.assert_close(target.alg.actor_critic.act_inference(probe),
                               source.alg.actor_critic.act_inference(probe), rtol=0, atol=0)
    target.learn(1)
    assert target.current_learning_iteration == 20001
    assert target.alg.optimizer.state
    updated = tmp_path / "updated.pt"
    target.save(updated)
    restored = MoECTSOnPolicyRunner(CurriculumVectorEnv(), cfg)
    restored.load(updated)
    assert restored.initialization == info
    assert_nested_equal(target.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert_nested_equal(target.alg.actor_critic.state_dict(), restored.alg.actor_critic.state_dict())
    with pytest.raises(ValueError, match="fresh runner"):
        target.initialize_from(path)


def test_initialization_rejects_other_architectures_and_residual_sources(starting_point, tmp_path):
    source, path = starting_point
    wrong = MoECTSOnPolicyRunner(FakeVectorEnv(), runner_config("mlp"))
    with pytest.raises(ValueError, match="all-MoE"):
        wrong.initialize_from(path)
    with_residual = MoECTSOnPolicyRunner(CurriculumVectorEnv(), all_moe_config(True))
    with_residual.initialize_from(path)
    residual_path = tmp_path / "residual.pt"
    with_residual.save(residual_path)
    target = MoECTSOnPolicyRunner(CurriculumVectorEnv(), all_moe_config(True))
    with pytest.raises(ValueError, match="non-residual"):
        target.initialize_from(residual_path)
    checkpoint = load_cts_checkpoint(path)
    checkpoint["model_state_dict"].pop("std")
    broken = tmp_path / "broken.pt"
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match="beyond the action residual"):
        target.initialize_from(broken)


def test_registry_initialization_restores_shared_settings_and_keeps_temporal_factor(
        starting_point, tmp_path, cpu_task_registry):
    source, path = starting_point
    defaults = all_moe_config(True, True)
    defaults["policy"]["actor_hidden_dims"] = [64]
    defaults["algorithm"]["learning_rate"] = .02
    defaults["runner"].update(resume=False, resume_path=None, run_name="residual",
                              experiment_name="comparison", load_run=-1, checkpoint=-1)
    config = SimpleNamespace(seed=42, runner_class_name="MoECTSOnPolicyRunner",
                             **{key: SimpleNamespace(**value) for key, value in defaults.items()
                                if isinstance(value, dict)})
    cpu_task_registry.register("comparison", FakeVectorEnv, SimpleNamespace(), config)
    args = SimpleNamespace(seed=17, max_iterations=None, resume=False, from_scratch=False,
                           init_from=str(path), experiment_name=None, run_name=None,
                           load_run=None, checkpoint=None, rl_device="cpu")
    target, cfg = cpu_task_registry.make_alg_runner(
        CurriculumVectorEnv(), name="comparison", args=args, log_root=str(tmp_path))
    assert cfg.policy.actor_hidden_dims == source.policy_config["actor_hidden_dims"]
    assert cfg.policy.residual_enabled
    assert cfg.algorithm.learning_rate == source.train_cfg["algorithm"]["learning_rate"]
    assert cfg.algorithm.temporal_loss_coef == .01
    assert target.initialization["seed"] == 17
    assert not target.alg.optimizer.state
    assert target.current_learning_iteration == 20000
    args.resume = True
    with pytest.raises(ValueError, match="cannot be combined"):
        cpu_task_registry.make_alg_runner(CurriculumVectorEnv(), name="comparison",
                                          args=args, log_root=str(tmp_path))

