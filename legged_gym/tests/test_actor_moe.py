"""Validate action-space mixing and the unchanged CTS distribution contract."""

import pytest
import torch
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.modules.moe import MLP


def make_policy(actor_type="moe"):
    return ActorCriticMoECTS(
        228, 295, 57, 16, actor_type=actor_type, actor_expert_num=8,
        actor_hidden_dims=[32, 16], critic_hidden_dims=[32],
        teacher_encoder_hidden_dims=[32], student_encoder_hidden_dims=[32, 16],
    )


def test_eight_action_experts_mix_unbounded_actions_without_latent_normalization():
    torch.manual_seed(43)
    policy = make_policy()
    inputs = torch.randn(5, 89)
    assert policy.actor_type == "moe"
    assert policy.actor_expert_num == 8
    assert isinstance(policy.critic, MLP)
    assert policy.critic.network[0].in_features == 327
    experts = policy.actor.moe.experts(inputs)
    weights = policy.actor.get_expert_probabilities(inputs)
    assert experts.shape == (5, 8, 16)
    assert weights.shape == (5, 8)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5))
    assert torch.allclose(policy.actor(inputs), (experts * weights.unsqueeze(-1)).sum(dim=1))

    # Large, distinct constant expert actions make accidental L2 normalization
    # or tanh action bounding observable, independently of random initialization.
    with torch.no_grad():
        heads = policy.actor.moe.experts.experts
        heads.weight.zero_()
        heads.bias.copy_(torch.arange(2.0, 10.0).repeat_interleave(16))
    actions = policy.actor(inputs)
    expected = (weights * torch.arange(2.0, 10.0)).sum(dim=-1, keepdim=True).expand(-1, 16)
    assert torch.allclose(actions, expected)
    assert torch.all(actions > 1)
    assert torch.all(actions.norm(dim=-1) > 1)


@pytest.mark.parametrize("is_teacher", [True, False])
def test_mixed_action_mean_uses_one_gaussian_with_matching_log_probability(is_teacher):
    torch.manual_seed(71)
    policy = make_policy()
    history, privileged = torch.randn(7, 228), torch.randn(7, 295)
    with torch.no_grad():
        if is_teacher:
            latent = policy.teacher_encoder(privileged)
        else:
            latent, _ = policy.student_moe_encoder(history)
        inputs = torch.cat((latent, policy.get_current_observations(history)), dim=-1)
        expert_actions = policy.actor.moe.experts(inputs)
        weights = policy.actor.get_expert_probabilities(inputs)
        mean = (weights.unsqueeze(-1) * expert_actions).sum(dim=1)
        expected_distribution = Normal(mean, policy.std.clamp_min(1.0e-6).expand_as(mean))
    torch.manual_seed(97)
    expected_actions = expected_distribution.sample()
    torch.manual_seed(97)
    actions = policy.act(history, privileged, is_teacher=is_teacher)
    assert torch.allclose(policy.action_mean, mean)
    assert torch.allclose(actions, expected_actions)
    assert torch.allclose(policy.get_actions_log_prob(actions),
                          expected_distribution.log_prob(actions).sum(dim=-1))
    assert torch.allclose(policy.actor_gate_weights, weights)
    assert torch.allclose(policy.get_actor_expert_probabilities(
        history, privileged, is_teacher=is_teacher), weights)
    if not is_teacher:
        assert torch.allclose(policy.act_inference(history), mean)
        # Privileged observations cannot change student deployment routing.
        changed = policy.get_actor_expert_probabilities(history, privileged + 1000)
        assert torch.allclose(changed, weights)


def test_mlp_baseline_keeps_empty_actor_expert_metrics():
    policy = make_policy("mlp")
    assert policy.actor_type == "mlp"
    assert policy.actor_expert_num == 0
    history = torch.randn(3, 228)
    assert isinstance(policy.actor, MLP)
    assert policy.get_actor_expert_probabilities(history).shape == (3, 0)
