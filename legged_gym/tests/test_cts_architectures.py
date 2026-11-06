"""All eight CTS architectures preserve probability and gradient ownership."""

from itertools import product

import pytest
import torch
from torch.distributions import Normal

from rsl_rl.algorithms.moe_cts import MoECTS
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS, StudentMLPEncoder
from rsl_rl.modules.moe import MLP


ARCHITECTURES = list(product(("mlp", "moe"), repeat=3))


def make_policy(student, actor, critic):
    return ActorCriticMoECTS(
        228, 295, 57, 16, latent_dim=32,
        student_encoder_type=student, actor_type=actor, critic_type=critic,
        expert_num=8, actor_expert_num=8, critic_expert_num=8,
        actor_hidden_dims=[24, 16], critic_hidden_dims=[24, 16],
        teacher_encoder_hidden_dims=[24], student_encoder_hidden_dims=[24, 16],
    )


def has_gradient(module):
    return any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in module.parameters())


def snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def assert_unchanged(module, before):
    assert all(torch.equal(before[name], parameter)
               for name, parameter in module.named_parameters())


@pytest.mark.parametrize("student,actor,critic", ARCHITECTURES)
def test_eight_architectures_shapes_probabilities_and_gradient_boundaries(student, actor, critic):
    torch.manual_seed(417)
    policy = make_policy(student, actor, critic)
    history, privileged = torch.randn(5, 228), torch.randn(5, 295)
    latent, student_weights = policy.student_moe_encoder(history)
    assert latent.shape == (5, 32)
    assert torch.allclose(latent.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert policy.expert_num == (8 if student == "moe" else 0)
    assert policy.actor_expert_num == (8 if actor == "moe" else 0)
    assert policy.critic_expert_num == (8 if critic == "moe" else 0)
    assert student_weights.shape == (5, policy.expert_num)
    assert torch.equal(policy.get_expert_probabilities(history), student_weights)
    assert isinstance(policy.teacher_encoder[0], MLP)
    if student == "mlp":
        assert isinstance(policy.student_moe_encoder, StudentMLPEncoder)
        assert not any("gating" in key or "experts" in key
                       for key in policy.student_moe_encoder.state_dict())

    for is_teacher in (True, False):
        policy.zero_grad(set_to_none=True)
        actions = policy.act(history, privileged, is_teacher=is_teacher)
        expected = Normal(policy.action_mean, policy.action_std).log_prob(actions).sum(-1)
        assert torch.allclose(policy.get_actions_log_prob(actions), expected)
        assert policy.actor_gate_weights.shape == (5, policy.actor_expert_num)
        if actor == "moe":
            assert torch.all(policy.actor_gate_weights > 0)
            assert torch.allclose(policy.actor_gate_weights.sum(-1), torch.ones(5))
        if not is_teacher:
            assert torch.allclose(policy.act_inference(history), policy.action_mean)
        policy.action_mean.square().mean().backward()
        assert has_gradient(policy.actor)
        assert has_gradient(policy.teacher_encoder) == is_teacher
        assert not has_gradient(policy.student_moe_encoder)
        assert not has_gradient(policy.critic)

        policy.zero_grad(set_to_none=True)
        value = policy.evaluate(history, privileged, is_teacher=is_teacher)
        assert value.shape == (5, 1)
        assert policy.critic_gate_weights.shape == (5, policy.critic_expert_num)
        value.square().mean().backward()
        assert has_gradient(policy.critic)
        assert not has_gradient(policy.actor)
        assert not has_gradient(policy.teacher_encoder)
        assert not has_gradient(policy.student_moe_encoder)
        if critic == "moe":
            assert has_gradient(policy.critic.moe.gating_network)

    # Same-state distillation only updates the selected student architecture.
    policy.zero_grad(set_to_none=True)
    latent, weights = policy.student_moe_encoder(history)
    with torch.no_grad():
        target = policy.teacher_encoder(privileged)
    loss = (latent - target).square().mean()
    if student == "moe":
        loss = loss + 0.01 * (weights.mean(0) - 1 / 8).square().mean()
    loss.backward()
    assert has_gradient(policy.student_moe_encoder)
    assert not has_gradient(policy.teacher_encoder)
    assert not has_gradient(policy.actor)
    assert not has_gradient(policy.critic)


def test_critic_moe_combines_unbounded_scalar_values_with_its_own_gate():
    torch.manual_seed(97)
    policy = make_policy("moe", "moe", "moe")
    history, privileged = torch.randn(5, 228), torch.randn(5, 295)
    assert policy.critic.moe.gating_network[0].network[0].in_features == 327
    gates = [policy.student_moe_encoder.moe.gating_network,
             policy.actor.moe.gating_network, policy.critic.moe.gating_network]
    parameter_ids = [{id(p) for p in gate.parameters()} for gate in gates]
    assert not parameter_ids[0].intersection(parameter_ids[1] | parameter_ids[2])
    assert not parameter_ids[1].intersection(parameter_ids[2])
    with torch.no_grad():
        policy.critic.moe.experts.experts.weight.zero_()
        policy.critic.moe.experts.experts.bias.copy_(torch.arange(2.0, 10.0))
    for is_teacher in (True, False):
        with torch.no_grad():
            latent = (policy.teacher_encoder(privileged) if is_teacher
                      else policy.student_moe_encoder(history)[0])
            inputs = torch.cat((latent, privileged), dim=-1)
            weights = policy.critic.get_expert_probabilities(inputs)
            expected = (weights * torch.arange(2.0, 10.0)).sum(-1, keepdim=True)
        value = policy.evaluate(history, privileged, is_teacher=is_teacher)
        assert torch.allclose(value, expected)
        assert torch.all(value > 1.0)
        assert torch.allclose(policy.critic_gate_weights, weights)


@pytest.mark.parametrize("student,actor,critic", ARCHITECTURES)
def test_all_architectures_update_ppo_then_student_without_spurious_expert_metrics(
        monkeypatch, student, actor, critic):
    torch.manual_seed(31)
    policy = make_policy(student, actor, critic)
    algorithm = MoECTS(policy, 8, num_learning_epochs=1, num_mini_batches=2,
                       schedule="fixed")
    ppo_ids = {id(p) for group in algorithm.optimizer.param_groups for p in group["params"]}
    student_ids = {id(p) for group in algorithm.optimizer_stu_enc.param_groups for p in group["params"]}
    assert not ppo_ids.intersection(student_ids)
    assert ppo_ids | student_ids == {id(p) for p in policy.parameters()}
    algorithm.init_storage(8, 2, [228], [295], [16])
    for _ in range(2):
        algorithm.act(torch.randn(8, 228), torch.randn(8, 295))
        algorithm.process_env_step(torch.randn(8), torch.zeros(8, dtype=torch.bool), {})
    algorithm.compute_returns(torch.randn(8, 228), torch.randn(8, 295))
    before = {name: snapshot(getattr(policy, name))
              for name in ("teacher_encoder", "student_moe_encoder", "actor", "critic")}
    phases = []
    ppo_step, student_step = algorithm.optimizer.step, algorithm.optimizer_stu_enc.step
    frozen_after_ppo = {}

    def check_ppo(*args, **kwargs):
        assert "student" not in phases
        assert all(p.grad is None for p in policy.student_moe_encoder.parameters())
        assert_unchanged(policy.student_moe_encoder, before["student_moe_encoder"])
        phases.append("ppo")
        return ppo_step(*args, **kwargs)

    def check_student(*args, **kwargs):
        assert phases.count("ppo") == 2
        assert all(p.grad is None for p in algorithm.ppo_parameters)
        if not frozen_after_ppo:
            frozen_after_ppo.update({name: snapshot(getattr(policy, name))
                                     for name in ("teacher_encoder", "actor", "critic")})
        for name, weights in frozen_after_ppo.items():
            assert_unchanged(getattr(policy, name), weights)
        phases.append("student")
        return student_step(*args, **kwargs)

    monkeypatch.setattr(algorithm.optimizer, "step", check_ppo)
    monkeypatch.setattr(algorithm.optimizer_stu_enc, "step", check_student)
    metrics = algorithm.update()
    assert phases == ["ppo", "ppo", "student", "student"]
    assert algorithm.storage.step == 0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    for name, old in before.items():
        assert any(not torch.equal(old[key], parameter)
                   for key, parameter in getattr(policy, name).named_parameters())
    for network, kind in (("actor", actor), ("critic", critic)):
        if kind == "mlp":
            assert not any(key.startswith(network + "_") for key in metrics)
        else:
            for group in ("", "teacher_", "student_"):
                prefix = network + "_" + group
                assert metrics[prefix + "gate_entropy"] > 0
                assert sum(metrics[prefix + "expert_%d_usage" % i] for i in range(8)) == pytest.approx(1.0)
            changed_gates = [not torch.equal(before[network][key], parameter)
                             for key, parameter in getattr(policy, network).named_parameters()
                             if "gating_network" in key]
            assert any(changed_gates)
    if student == "mlp":
        assert "mean_load_balance_loss" not in metrics
        assert "gate_entropy" not in metrics
        assert not any(key.startswith("expert_") for key in metrics)
    else:
        assert metrics["mean_load_balance_loss"] >= 0
        assert metrics["gate_entropy"] > 0
        assert sum(metrics["expert_%d_usage" % i] for i in range(8)) == pytest.approx(1.0)


def test_legacy_defaults_keep_the_existing_moe_state_keys():
    kwargs = dict(num_actor_obs=228, num_critic_obs=295, num_one_step_obs=57,
                  num_actions=16, actor_hidden_dims=[24, 16], critic_hidden_dims=[24, 16],
                  teacher_encoder_hidden_dims=[24], student_encoder_hidden_dims=[24, 16])
    torch.manual_seed(98)
    legacy = ActorCriticMoECTS(**kwargs)
    torch.manual_seed(98)
    explicit = ActorCriticMoECTS(**kwargs, student_encoder_type="moe", critic_type="mlp")
    assert legacy.student_encoder_type == "moe"
    assert legacy.critic_type == "mlp"
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    assert any(key.startswith("student_moe_encoder.moe.") for key in legacy.state_dict())
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    history = torch.randn(3, 228)
    assert torch.equal(legacy.act_inference(history), explicit.act_inference(history))
