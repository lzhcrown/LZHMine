"""Temporal objectives must compare real neighboring states, not shuffled rows."""

import math

import pytest
import torch

from rsl_rl.algorithms.moe_cts import MoECTS
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS


def make_policy(residual=False):
    return ActorCriticMoECTS(
        num_actor_obs=12, num_critic_obs=5, num_one_step_obs=3,
        num_actions=2, history_length=4, observation_dims=[1, 2],
        actor_hidden_dims=[8], critic_hidden_dims=[8],
        teacher_encoder_hidden_dims=[8], student_encoder_hidden_dims=[8],
        latent_dim=4, expert_num=2, actor_type="moe", actor_expert_num=2,
        critic_type="moe", critic_expert_num=2,
        residual_enabled=residual, residual_hidden_dims=[8],
        residual_action_scales=[0.1, 0.05] if residual else None,
    )


def append_marker_transition(storage, global_time, done_envs=()):
    transition = RolloutStorageCTS.Transition()
    marker = (torch.arange(storage.num_envs) + 100 * global_time).float().unsqueeze(1)
    transition.observations = marker
    transition.critic_observations = marker + 1000
    transition.actions = marker + 10000
    transition.action_mean = marker + 20000
    transition.action_sigma = torch.ones_like(marker)
    transition.rewards = torch.zeros(storage.num_envs)
    transition.values = torch.zeros_like(marker)
    transition.actions_log_prob = torch.zeros_like(marker)
    transition.dones = torch.zeros(storage.num_envs, dtype=torch.bool)
    transition.dones[list(done_envs)] = True
    storage.add_transitions(transition)


def test_temporal_pairs_keep_environment_time_and_reset_boundaries_after_shuffle():
    storage = RolloutStorageCTS(4, 2, 3, [1], [1], [1], collect_temporal_pairs=True)
    append_marker_transition(storage, 0, done_envs=(1,))
    append_marker_transition(storage, 1)
    append_marker_transition(storage, 2, done_envs=(2,))
    seen = []
    for batch in storage.mini_batch_generator(2):
        current = batch["observations"][:, 0]
        previous = batch["previous_observations"][:, 0]
        privileged = batch["previous_critic_observations"][:, 0]
        valid = batch["temporal_mask"]
        for now, before, before_privileged, is_valid in zip(current, previous, privileged, valid):
            marker = int(now)
            time, env = divmod(marker, 100)
            seen.append(marker)
            expected_valid = time > 0 and not (time == 1 and env == 1)
            assert bool(is_valid) == expected_valid
            if time > 0:
                assert before.item() == marker - 100
                assert before_privileged.item() == marker - 100 + 1000
    assert sorted(seen) == [100 * time + env for time in range(3) for env in range(4)]

    # clear() preserves only the final inputs and their done masks, before
    # the main rollout tensor is overwritten by the next collection.
    storage.clear()
    for time in range(3, 6):
        append_marker_transition(storage, time)
    for batch in storage.mini_batch_generator(2):
        for current, previous, valid in zip(batch["observations"][:, 0],
                                            batch["previous_observations"][:, 0],
                                            batch["temporal_mask"]):
            marker = int(current)
            time, env = divmod(marker, 100)
            assert previous.item() == marker - 100
            assert bool(valid) == (not (time == 3 and env == 2))

    # Loading a model resets physics. Old carry states must not reconnect the
    # newly initialized simulator to the old rollout.
    storage.clear()
    storage.reset_temporal_history()
    for time in range(6, 9):
        append_marker_transition(storage, time)
    for batch in storage.mini_batch_generator(2):
        first = batch["observations"][:, 0] < 700
        assert not batch["temporal_mask"][first].any()


def test_disabled_temporal_storage_keeps_legacy_batch_and_no_carry_allocation():
    storage = RolloutStorageCTS(2, 1, 1, [1], [1], [1])
    append_marker_transition(storage, 0)
    batch = next(storage.mini_batch_generator(1))
    assert "previous_observations" not in batch
    assert "temporal_mask" not in batch
    assert not hasattr(storage, "_carry_observations")
    storage.clear()
    storage.reset_temporal_history()


def test_temporal_loss_recomputes_both_endpoints_and_averages_only_valid_dofs(monkeypatch):
    algorithm = MoECTS(make_policy(), 4, temporal_loss_coef=0.1)
    scale = torch.nn.Parameter(torch.tensor(2.0))
    branches = []

    def current_policy_mean(observations, critic_obs=None, is_teacher=False):
        branches.append(is_teacher)
        assert torch.equal(critic_obs[:, 0], observations[:, 0] + 1000)
        mean = observations[:, :1].expand(-1, 2) * scale
        return mean, None, None, None

    monkeypatch.setattr(algorithm.actor_critic, "action_mean_with_diagnostics", current_policy_mean)
    current = torch.tensor([[2.0], [4.0], [100.0], [8.0]])
    previous = torch.tensor([[1.0], [2.0], [float("nan")], [4.0]])
    means = current.expand(-1, 2) * scale
    batch = {
        "teacher_count": 2,
        "previous_observations": previous,
        "previous_critic_observations": previous + 1000,
        "temporal_mask": torch.tensor([True, True, False, True]),
        # Deliberately wrong old values: the objective must not consume them.
        "mu": torch.full((4, 2), 1.0e8),
        "actions": torch.full((4, 2), -1.0e8),
    }
    loss, count = algorithm._temporal_loss(batch, means)
    assert count.item() == 3
    # Differences are 2, 4, 8, each repeated across two action dimensions.
    assert loss.item() == pytest.approx((4 + 16 + 64) / 3)
    loss.backward()
    # Both endpoints depend on scale. Detaching the predecessor gives a
    # different gradient even though the loss's numerical value stays equal.
    assert scale.grad.item() == pytest.approx(28.0)
    assert branches == [True, False]


def test_no_valid_pairs_returns_finite_zero_without_evaluating_predecessors(monkeypatch):
    algorithm = MoECTS(make_policy(), 4, temporal_loss_coef=0.1)

    def must_not_run(*args, **kwargs):
        raise AssertionError("Invalid predecessor must not enter the policy")

    monkeypatch.setattr(algorithm.actor_critic, "action_mean_with_diagnostics", must_not_run)
    means = torch.randn(4, 2, requires_grad=True)
    batch = {"teacher_count": 2, "temporal_mask": torch.zeros(4, dtype=torch.bool)}
    loss, count = algorithm._temporal_loss(batch, means)
    assert loss.item() == 0.0
    assert count.item() == 0
    loss.backward()
    assert torch.equal(means.grad, torch.zeros_like(means))


@pytest.mark.parametrize("teacher_count", [0, 3])
def test_temporal_gradient_preserves_cts_encoder_boundaries(teacher_count):
    torch.manual_seed(42)
    policy = make_policy(residual=True)
    algorithm = MoECTS(policy, 4, temporal_loss_coef=0.1)
    observations, previous = torch.randn(3, 12), torch.randn(3, 12)
    privileged, previous_privileged = torch.randn(3, 5), torch.randn(3, 5)
    teacher = teacher_count > 0
    means, _, _, _ = policy.action_mean_with_diagnostics(observations, privileged, is_teacher=teacher)
    batch = {
        "teacher_count": teacher_count,
        "previous_observations": previous,
        "previous_critic_observations": previous_privileged,
        "temporal_mask": torch.ones(3, dtype=torch.bool),
    }
    loss, _ = algorithm._temporal_loss(batch, means)
    loss.backward()
    has_gradient = lambda module: any(p.grad is not None and p.grad.abs().sum() > 0
                                      for p in module.parameters())
    assert has_gradient(policy.actor)
    assert not has_gradient(policy.critic)
    assert not has_gradient(policy.student_moe_encoder)
    assert bool(has_gradient(policy.teacher_encoder)) == teacher


@pytest.mark.parametrize("residual", [False, True])
def test_temporal_ppo_update_has_finite_losses_and_retains_rollout_carry(residual, monkeypatch):
    torch.manual_seed(11)
    algorithm = MoECTS(make_policy(residual=residual), 4, num_learning_epochs=2,
                      num_mini_batches=2, temporal_loss_coef=0.01, schedule="fixed")
    algorithm.init_storage(4, 3, [12], [5], [2])
    for _ in range(3):
        algorithm.act(torch.randn(4, 12), torch.randn(4, 5))
        algorithm.process_env_step(torch.randn(4), torch.zeros(4, dtype=torch.bool), {})
    algorithm.compute_returns(torch.randn(4, 12), torch.randn(4, 5))
    original_step = algorithm.optimizer.step

    def ppo_step(*args, **kwargs):
        assert all(p.grad is None for p in algorithm.actor_critic.student_moe_encoder.parameters())
        return original_step(*args, **kwargs)

    monkeypatch.setattr(algorithm.optimizer, "step", ppo_step)
    losses = algorithm.update()
    assert all(math.isfinite(value) for value in losses.values())
    assert losses["temporal"] > 0
    assert losses["temporal_valid_pairs"] == 4 * (3 - 1)
    if residual:
        assert 0 < losses["residual_rms"] <= losses["residual_abs_max"]
        assert 0 <= losses["residual_saturation_fraction"] <= 1
    else:
        assert "residual_rms" not in losses
    assert algorithm.storage.step == 0
    assert algorithm.storage._carry_valid.all()
    assert losses["surrogate"] == pytest.approx(
        losses["teacher_surrogate"] + losses["student_surrogate"])


@pytest.mark.parametrize("coefficient", [-1.0, float("inf"), float("nan")])
def test_invalid_temporal_coefficients_are_rejected(coefficient):
    with pytest.raises(ValueError, match="temporal_loss_coef"):
        MoECTS(make_policy(), 4, temporal_loss_coef=coefficient)
